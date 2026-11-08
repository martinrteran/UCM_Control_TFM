"""
standard_ppo.py

Textbook PPO (Schulman et al. 2017, clipped surrogate; the usual implementation
of Stable-Baselines3 / CleanRL), next to PPOAgent, which is left unchanged.

PPOAgent (tests 14-35) is an off-policy variant: after each episode it takes one
gradient step on 64 transitions sampled from a 100k buffer of past episodes, with
stale log-probabilities and advantages (§2i.2 of the briefing). StandardPPOAgent
does what PPO prescribes:

1. Collect a rollout of `rollout_steps` environment steps with the current policy
   (it may span several episodes and stop mid-episode).
2. Compute GAE(lambda) advantages and returns over the rollout with the value
   function of that same policy:
       delta_t = r_t + gamma * V(s'_t) * (1 - terminal_t) - V(s_t)
       A_t     = delta_t + gamma * lambda * (1 - episode_end_t) * A_{t+1}
   A terminal state (goal, collision) does not bootstrap; an episode cut by the
   step limit, or a rollout that stops mid-episode, bootstraps from V(s'_t).
3. Run `n_epochs` passes over the rollout in shuffled minibatches of
   `batch_size`: clipped surrogate on per-minibatch normalised advantages,
   value loss (optionally clipped), entropy bonus, gradient clipping;
   stop the epochs early if the approximate KL exceeds `target_kl`.
4. Discard the rollout (on-policy) and start the next one.

Collisions: the environments end a collision as `truncated` (like the step limit),
so the flags alone cannot tell them apart. With a trainer that relays the step
info (core/training/step_info_trainer.py) the agent reads info['hit_obstacle']
and treats a collision as terminal. Without it, every truncation bootstraps
(`truncation_is_terminal=False`) or none does (True).

Action masking: act() takes info['action_mask']; the masks are stored with the
rollout and the update uses the log-probabilities and entropy of the masked policy.

The update runs inside observe() when the rollout is full, so it does not depend
on TrainerConfig.eps_step; on_episode_end returns the loss of the last update
done during the episode, if any. Checkpoints have the PPOAgent format
(algorithm "ppo", policy and value state dicts), so scripts/eval_agent.py loads them.

Author: Martin
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .base_rl_agent import BaseRLAgent
from .utils import PPOConfig


@dataclass
class StandardPPOConfig(PPOConfig):
    """
    PPOConfig for StandardPPOAgent (rollout_steps, clip_eps, gae_lambda, entropy_coeff,
    value_coeff, lr, gamma, max_grad_norm and batch_size come from PPOConfig / RLConfig).

    n_epochs : int
        Passes over each rollout.
    batch_size : int
        Minibatch size (inherited; default 64).
    normalize_advantages : bool
        Normalise the advantages of each minibatch to mean 0, std 1.
    clip_value : float or None
        If set, clip the new value to old_value ± clip_value in the value loss.
    target_kl : float or None
        Stop the epochs of an update when the approximate KL exceeds it.
    truncation_is_terminal : bool
        What a truncation without step info means: False bootstraps (time limit),
        True does not. With step info, a collision is always terminal and a time limit never is.
    """
    n_epochs: int = 10
    normalize_advantages: bool = True
    clip_value: Optional[float] = None
    target_kl: Optional[float] = None
    truncation_is_terminal: bool = False


class RolloutBuffer:
    """Fixed-size on-policy rollout (numpy arrays), filled step by step and cleared after each update."""

    def __init__(self, size: int, obs_dim: int, action_dim: int):
        self.size = int(size)
        self.states = np.zeros((self.size, obs_dim), dtype=np.float32)
        self.next_states = np.zeros((self.size, obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.size, dtype=np.int64)
        self.rewards = np.zeros(self.size, dtype=np.float32)
        self.log_probs = np.zeros(self.size, dtype=np.float32)
        self.values = np.zeros(self.size, dtype=np.float32)
        self.terminals = np.zeros(self.size, dtype=np.float32) # No bootstrap from s'
        self.episode_ends = np.zeros(self.size, dtype=np.float32) # GAE does not flow across
        self.masks = np.ones((self.size, action_dim), dtype=bool)
        self.n = 0

    def add(self, state, action, reward, next_state, log_prob, value, terminal, episode_end, mask):
        i = self.n
        self.states[i] = np.asarray(state, dtype=np.float32).ravel()
        self.next_states[i] = np.asarray(next_state, dtype=np.float32).ravel()
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.log_probs[i] = float(log_prob)
        self.values[i] = float(value)
        self.terminals[i] = float(terminal)
        self.episode_ends[i] = float(episode_end)
        self.masks[i] = mask
        self.n += 1

    @property
    def full(self) -> bool:
        return self.n >= self.size

    def clear(self):
        self.n = 0


def gae(rewards, values, next_values, terminals, episode_ends, gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    GAE(lambda) over a rollout that may contain several episodes.

    next_values[t] is V(s'_t); it is ignored where terminals[t] is 1. The
    recursion is cut where episode_ends[t] is 1 (terminal or truncated); the
    last step of the rollout bootstraps through next_values when it is not terminal.

    Returns
    -------
    (advantages, returns), returns = advantages + values.
    """
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    running = 0.0
    for t in reversed(range(n)):
        delta = rewards[t] + gamma * next_values[t] * (1.0 - terminals[t]) - values[t]
        running = delta + gamma * lam * (1.0 - episode_ends[t]) * running
        advantages[t] = running
    return advantages, advantages + values


class StandardPPOAgent(BaseRLAgent):
    """
    Textbook PPO with rollouts, GAE, several epochs of minibatch updates and
    optional action masking (see the module docstring).

    Parameters
    ----------
    config : StandardPPOConfig
    policy_network : nn.Module
        obs -> logits, one per action (e.g. SimpleQNetwork).
    value_network : nn.Module
        obs -> V(s), shape (..., 1) (e.g. ValueNetwork).
    """
    train_on_step = False # Updates happen in observe() when the rollout is full

    def __init__(self, config: StandardPPOConfig, policy_network: nn.Module, value_network: nn.Module):
        super().__init__(config=config, policy_network=policy_network, value_network=value_network)
        self.rollout = RolloutBuffer(config.rollout_steps, int(config.obs_dim), int(config.action_dim))
        self.updates = 0 # Completed PPO updates (rollouts)
        self.gradient_steps = 0
        self._pending: Optional[tuple] = None # (log_prob, value, mask) of the last training act()
        self._step_info: Optional[dict] = None # Info of the last env step, if a trainer relays it
        self._episode_loss: Optional[float] = None
        self.metrics: Dict[str, float] = {"Agent/Entropy": 0.0, "Agent/Policy/Loss": 0.0, "Agent/Value/Loss": 0.0,
                                          "Agent/PPO/ApproxKL": 0.0, "Agent/PPO/ClipFraction": 0.0,
                                          "Agent/PPO/ExplainedVariance": 0.0, "Agent/PPO/Updates": 0.0}

    def _initialize_networks(self, **kwargs):
        raise ValueError("StandardPPOAgent needs a policy_network and a value_network")

    # ------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------
    def _mask_tensor(self, action_mask, like: torch.Tensor) -> Optional[torch.Tensor]:
        if action_mask is None: return None
        mask = torch.as_tensor(np.asarray(action_mask), dtype=torch.bool, device=like.device)
        return mask if bool(mask.any()) else None # A mask with no valid action is ignored

    def _distribution(self, states: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.distributions.Categorical:
        logits = self.policy_net(states) # type: ignore
        if masks is not None:
            logits = logits.masked_fill(~masks, torch.finfo(logits.dtype).min)
        return torch.distributions.Categorical(logits=logits)

    def select_action(self, state: np.ndarray, training: bool = True, action_mask: Optional[np.ndarray] = None) -> int:
        """Sample from the (masked) policy when training, take its most likely action otherwise."""
        state_t = self._to_tensor(state).reshape(1, -1)
        with torch.no_grad():
            logits = self.policy_net(state_t) # type: ignore
            mask_t = self._mask_tensor(action_mask, logits)
            dist = self._distribution(state_t, None if mask_t is None else mask_t.reshape(1, -1))
            action = dist.sample() if training else dist.probs.argmax(dim=-1)
            if training:
                self.global_step += 1
                value = self.value_net(state_t).reshape(-1)[0].item() # type: ignore
                mask = np.ones(self.config.action_dim, dtype=bool) if mask_t is None else mask_t.cpu().numpy()
                self._pending = (float(dist.log_prob(action).item()), float(value), mask)
        return int(action.item())

    # ------------------------------------------------------------
    # Episode hooks
    # ------------------------------------------------------------
    def on_episode_start(self):
        self._episode_loss = None
        self._step_info = None

    def act(self, state: np.ndarray, training: bool = True, action_mask: Optional[np.ndarray] = None) -> int:
        return self.select_action(state, training=training, action_mask=action_mask)

    def on_step_info(self, info: dict):
        """Step info of the env, relayed by StepInfo trainers before observe()."""
        self._step_info = info

    def _is_terminal(self, terminated: bool, truncated: bool) -> bool:
        """True if the next state must not be bootstrapped (goal or collision, not a time limit)."""
        if terminated: return True
        if not truncated: return False
        info = self._step_info
        if info is not None and ("hit_obstacle" in info or "max_steps_reached" in info):
            return bool(info.get("hit_obstacle", False))
        return bool(self.config.truncation_is_terminal) # type: ignore

    def observe(self, state, action, reward, next_state, terminated, truncated):
        if self._pending is None: return # act() was not called in training mode
        log_prob, value, mask = self._pending
        self._pending = None
        terminal = self._is_terminal(bool(terminated), bool(truncated))
        self.rollout.add(state, action, reward, next_state, log_prob, value, terminal, bool(terminated or truncated), mask)
        self._step_info = None
        if self.rollout.full:
            self._episode_loss = self.train_step()

    def on_episode_end(self, last_state: np.ndarray) -> Optional[float]:
        self.episode_count += 1
        loss, self._episode_loss = self._episode_loss, None
        return loss

    def episode_metrics(self) -> Dict[str, float]:
        return dict(self.metrics)

    # ------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------
    def train_step(self) -> Optional[float]:
        """One PPO update on the current rollout (any length > 0), then clear it."""
        cfg: StandardPPOConfig = self.config # type: ignore
        n = self.rollout.n
        if n == 0: return None
        r = self.rollout
        with torch.no_grad():
            next_values = self.value_net(self._to_tensor(r.next_states[:n])).reshape(-1).cpu().numpy() # type: ignore
        advantages, returns = gae(r.rewards[:n], r.values[:n], next_values, r.terminals[:n], r.episode_ends[:n],
                                  cfg.gamma, cfg.gae_lambda)

        states = self._to_tensor(r.states[:n])
        actions = self._to_tensor(r.actions[:n], dtype=torch.int64)
        old_log_probs = self._to_tensor(r.log_probs[:n])
        old_values = self._to_tensor(r.values[:n])
        masks = torch.as_tensor(r.masks[:n], dtype=torch.bool, device=self.device)
        advantages_t = self._to_tensor(advantages)
        returns_t = self._to_tensor(returns)

        losses, entropies, policy_losses, value_losses, kls, clip_fractions = [], [], [], [], [], []
        stop = False
        for _ in range(cfg.n_epochs):
            for idx in torch.randperm(n, device=self.device).split(cfg.batch_size):
                dist = self._distribution(states[idx], masks[idx])
                log_probs = dist.log_prob(actions[idx])
                entropy = dist.entropy().mean()
                adv = advantages_t[idx]
                if cfg.normalize_advantages and len(idx) > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                log_ratio = log_probs - old_log_probs[idx]
                ratio = log_ratio.exp()
                policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv).mean()

                values = self.value_net(states[idx]).reshape(-1) # type: ignore
                if cfg.clip_value is not None:
                    clipped = old_values[idx] + (values - old_values[idx]).clamp(-cfg.clip_value, cfg.clip_value)
                    value_loss = 0.5 * torch.max((values - returns_t[idx]) ** 2, (clipped - returns_t[idx]) ** 2).mean()
                else:
                    value_loss = 0.5 * ((values - returns_t[idx]) ** 2).mean()

                loss = policy_loss + cfg.value_coeff * value_loss - cfg.entropy_coeff * entropy
                self.optimizer.zero_grad()
                loss.backward()
                self._clip_grad_norm()
                self.optimizer.step()
                self.gradient_steps += 1

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item() # Schulman's low-variance estimator
                    clip_fractions.append(((ratio - 1).abs() > cfg.clip_eps).float().mean().item())
                losses.append(loss.item()); entropies.append(entropy.item())
                policy_losses.append(policy_loss.item()); value_losses.append(value_loss.item()); kls.append(approx_kl)
                if cfg.target_kl is not None and approx_kl > 1.5 * cfg.target_kl:
                    stop = True
                    break
            if stop: break

        explained_variance = 1.0 - np.var(returns - r.values[:n]) / np.var(returns) if np.var(returns) > 0 else 0.0
        self.updates += 1
        self.rollout.clear()
        self.metrics = {"Agent/Entropy": float(np.mean(entropies)), "Agent/Policy/Loss": float(np.mean(policy_losses)),
                        "Agent/Value/Loss": float(np.mean(value_losses)), "Agent/PPO/ApproxKL": float(np.mean(kls)),
                        "Agent/PPO/ClipFraction": float(np.mean(clip_fractions)),
                        "Agent/PPO/ExplainedVariance": float(explained_variance), "Agent/PPO/Updates": float(self.updates)}
        self.log_training_step(loss=float(np.mean(losses)))
        return float(np.mean(losses))

    def _compute_loss(self, batch):
        raise NotImplementedError("StandardPPOAgent computes its losses in train_step")
