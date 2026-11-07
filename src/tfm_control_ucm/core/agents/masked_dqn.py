"""
masked_dqn.py

Double DQN with invalid-action masking, the DQN counterpart of PPOAgent with
TrainerConfig(action_masking=True) (test 31). DQNAgent and GridAgent are not
changed; this agent fixes what keeps DQNAgent from that comparison:

- action masking: act() takes the env's info['action_mask']; epsilon-greedy
  explores only among valid actions and the greedy action is the best valid one;
- the bootstrap target maximises over the valid actions of the next state
  (its mask is stored with the transition), not over moves into walls;
- the target network is copied every `target_update` gradient updates (DQNAgent
  checks the global step only on the steps it trains, so it updated irregularly);
- linear epsilon decay over `eps_decay` training steps, no training before
  `learning_starts` transitions, evaluation (training=False) does not move epsilon;
- a preallocated numpy replay buffer (the shared ReplayBuffer rebuilds arrays from
  dicts on every sample).

Without masks (action_mask=None) it is plain Double DQN.

Episode ends: terminated or truncated are both treated as terminal (no bootstrap),
as in DQNAgent; the step limit (400) is long compared with the episodes, so the
time-limit bias is small. Collisions are "truncated" in these environments, so
treating truncation as terminal is what makes a crash final.

Author: Martin
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_rl_agent import BaseRLAgent
from .utils import DQNConfig


@dataclass
class MaskedDQNConfig(DQNConfig):
    """
    DQNConfig for MaskedDQNAgent.

    eps_decay : int
        Training steps over which epsilon goes linearly from eps_start to eps_end
        (DQNAgent uses it as the time constant of an exponential).
    target_update : int
        Gradient updates between two copies of the online network to the target network.
    learning_starts : int
        Transitions in the buffer before the first gradient update.
    soft_target_update : bool
        If True the target network follows the online one with Polyak averaging
        (target = (1 - tau) * target + tau * online) after every update, instead of
        a hard copy every target_update updates.
    bootstrap_truncated : bool
        If True a truncated transition (step limit) bootstraps from the next state, as
        it is not a real end of the task. Only correct when truncation cannot be a
        collision (masked runs without slips); False treats it as terminal.
    """
    eps_decay: int = 500_000
    learning_starts: int = 5_000
    soft_target_update: bool = False
    bootstrap_truncated: bool = False


class MaskedReplayBuffer:
    """
    Circular replay buffer of preallocated numpy arrays:
    (state, action, reward, next_state, done, valid actions of next_state).
    """
    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = int(capacity)
        self.states = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.next_masks = np.ones((self.capacity, action_dim), dtype=bool)
        self.ptr = 0
        self.size = 0

    def __len__(self):
        return self.size

    def clear(self):
        self.ptr = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done, next_mask):
        i = self.ptr
        self.states[i] = np.asarray(state, dtype=np.float32).ravel()
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.next_states[i] = np.asarray(next_state, dtype=np.float32).ravel()
        self.dones[i] = float(done)
        self.next_masks[i] = next_mask
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return (self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx],
                self.dones[idx], self.next_masks[idx])


class MaskedDQNAgent(BaseRLAgent):
    """
    Double DQN with invalid-action masking (see the module docstring).

    Parameters
    ----------
    config : MaskedDQNConfig
        Hyperparameters (algorithm=RLAlgorithm.DQN).
    q_network : nn.Module
        Q-network obs -> one value per action (e.g. SimpleQNetwork); the target
        network is a copy of it.
    """
    train_on_step = True # The trainer calls train_step every TrainerConfig.eps_step steps

    def __init__(self, config: MaskedDQNConfig, q_network: nn.Module):
        super().__init__(config=config, policy_network=q_network)
        self.target_net = copy.deepcopy(self.policy_net).to(self.device)
        self.target_net.eval()
        for p in self.target_net.parameters():
            p.requires_grad_(False)
        self.buffer = MaskedReplayBuffer(config.buffer_size, int(config.obs_dim), int(config.action_dim)) # type: ignore
        self.epsilon = config.eps_start
        self.updates = 0
        self.last_loss = 0.0
        self.last_q = 0.0
        self._pending: Optional[tuple] = None # Last transition, waiting for the mask of its next state

    def _initialize_networks(self, **kwargs):
        raise ValueError("MaskedDQNAgent needs a q_network")

    # ------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------
    def _valid(self, action_mask) -> np.ndarray:
        """Boolean mask of valid actions (all valid without a mask, or if the mask has none)."""
        if action_mask is None: return np.ones(self.config.action_dim, dtype=bool)
        valid = np.asarray(action_mask, dtype=bool)
        return valid if valid.any() else np.ones(self.config.action_dim, dtype=bool)

    def _epsilon(self) -> float:
        cfg: MaskedDQNConfig = self.config # type: ignore
        fraction = min(1.0, self.global_step / max(1, cfg.eps_decay))
        return cfg.eps_start + fraction * (cfg.eps_end - cfg.eps_start)

    def select_action(self, state: np.ndarray, training: bool = True, action_mask: Optional[np.ndarray] = None) -> int:
        """Epsilon-greedy among the valid actions when training, greedy among them otherwise."""
        valid = self._valid(action_mask)
        if training:
            self.global_step += 1
            self.epsilon = self._epsilon()
            if np.random.random() < self.epsilon:
                return int(np.random.choice(np.flatnonzero(valid)))
        with torch.no_grad():
            q = self.policy_net(self._to_tensor(state).reshape(1, -1))[0] # type: ignore
        q = q.masked_fill(~torch.as_tensor(valid, device=q.device), float("-inf"))
        return int(q.argmax().item())

    # ------------------------------------------------------------
    # Episode hooks
    # ------------------------------------------------------------
    def on_episode_start(self):
        self._pending = None

    def act(self, state: np.ndarray, training: bool = True, action_mask: Optional[np.ndarray] = None) -> int:
        if training and self._pending is not None:
            # The mask of this state is the next-state mask of the previous transition
            self.buffer.add(*self._pending, self._valid(action_mask))
            self._pending = None
        return self.select_action(state, training=training, action_mask=action_mask)

    def observe(self, state, action, reward, next_state, terminated, truncated):
        cfg: MaskedDQNConfig = self.config # type: ignore
        done = bool(terminated or (truncated and not cfg.bootstrap_truncated))
        transition = (state, action, reward, next_state, done)
        if done: # Not bootstrapped: its next-state mask is not used
            self.buffer.add(*transition, np.ones(self.config.action_dim, dtype=bool))
        elif terminated or truncated:
            # Episode over but bootstrapped (step limit): no next act() will bring the mask; any valid action counts
            self.buffer.add(*transition, np.ones(self.config.action_dim, dtype=bool))
        else:
            self._pending = transition

    def on_episode_end(self, last_state: np.ndarray) -> Optional[float]:
        self._pending = None
        self.episode_count += 1
        return None

    def episode_metrics(self) -> Dict[str, float]:
        return {"Policy/Epsilon/Episode": float(self.epsilon), "Agent/Q/Mean": self.last_q,
                "Agent/DQN/Loss": self.last_loss, "Agent/DQN/Updates": float(self.updates)}

    # ------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------
    def train_step(self) -> Optional[float]:
        cfg: MaskedDQNConfig = self.config # type: ignore
        if len(self.buffer) < max(cfg.batch_size, cfg.learning_starts):
            return None
        loss = self._compute_loss(self.buffer.sample(cfg.batch_size))
        self.optimizer.zero_grad()
        loss.backward()
        self._clip_grad_norm()
        self.optimizer.step()

        self.updates += 1
        if cfg.soft_target_update:
            self.soft_update(self.policy_net, self.target_net, cfg.tau) # type: ignore
        elif self.updates % cfg.target_update == 0:
            self.hard_update(self.policy_net, self.target_net) # type: ignore
        self.last_loss = float(loss.item())
        return self.last_loss

    def _compute_loss(self, batch: Tuple) -> torch.Tensor:
        """Huber loss between Q(s, a) and r + gamma * (1 - done) * Q_target(s', a*), a* = best valid action of s'."""
        cfg: MaskedDQNConfig = self.config # type: ignore
        states, actions, rewards, next_states, dones, next_masks = batch
        states_t = self._to_tensor(states)
        actions_t = self._to_tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards_t = self._to_tensor(rewards)
        next_states_t = self._to_tensor(next_states)
        dones_t = self._to_tensor(dones)
        invalid_t = ~torch.as_tensor(next_masks, dtype=torch.bool, device=self.device)

        q = self.policy_net(states_t).gather(1, actions_t).squeeze(1) # type: ignore
        with torch.no_grad():
            if cfg.double_dqn: # Online network picks the action, target network evaluates it
                next_actions = self.policy_net(next_states_t).masked_fill(invalid_t, float("-inf")).argmax(dim=1, keepdim=True) # type: ignore
                next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(next_states_t).masked_fill(invalid_t, float("-inf")).max(dim=1)[0]
            next_q = torch.where(dones_t > 0, torch.zeros_like(next_q), next_q)
            target = rewards_t + cfg.gamma * next_q
        self.last_q = float(q.mean().item())
        return F.smooth_l1_loss(q, target)

    # ------------------------------------------------------------
    # Saving and loading
    # ------------------------------------------------------------
    def load(self, path: str):
        super().load(path)
        self.hard_update(self.policy_net, self.target_net) # type: ignore
        self.epsilon = self._epsilon()
