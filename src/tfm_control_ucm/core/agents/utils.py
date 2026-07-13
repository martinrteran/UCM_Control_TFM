from __future__ import annotations

from collections import deque
from os import name
import random
import numpy as np
import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.modules import (
    ProbabilisticActor,
    ValueOperator,
    MLP,
    OneHotCategorical
)
from dataclasses import dataclass, asdict
from collections import deque
from enum import Enum
import torch
import torch.nn as nn
from typing import Dict, Any, Union


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class RLAlgorithm(Enum):
    """Supported RL algorithms."""
    DQN = "dqn"
    DDQN = "ddqn"  # Double DQN
    DUELING_DQN = "dueling_dqn"
    PPO = "ppo"
    A2C = "a2c"
    A3C = "a3c"
    SAC = "sac"
    TRPO = "trpo"
    TD3 = "td3"
    GRID_AGENT = "grid_agent"  # Custom agent for grid environments with recovery mechanisms
    SIMPLE_Q_NETWORK = "simple_q_network"  # A simple Q-network for grid environments
    CUSTOM = "custom"  # Placeholder for user-defined algorithms


@dataclass
class RLConfig:
    """Configuration dataclass for RL agents."""
    name: str # Name of the agent (for logging and checkpointing)

    obs_dim: int # Observation dimension
    action_dim: int # Number of discrete actions (for DQN-family) or action dimension (for continuous)
    algorithm: RLAlgorithm # Algorithm to use (DQN, PPO, SAC, etc.)
    device: Union[str, torch.device] = "cpu" # "cpu", "cuda" or "mps"
    gamma: float = 0.99 # Discount factor
    lr: float = 1e-3 # Learning rate
    batch_size: int = 64 # Batch size for training
    buffer_size: int = 100_000 # Replay buffer size
    
    # Exploration
    eps_start: float = 1.0 # Initial epsilon for epsilon-greedy exploration (DQN-family)
    eps_end: float = 0.05 # Final epsilon after decay
    eps_decay: int = 100_000 # Steps over which to decay epsilon
    
    # Target network (DQN-family)
    target_update: int = 1000 # Steps between target network updates
    tau: float = 0.005  # Soft update coefficient
    
    # PPO-specific
    gae_lambda: float = 0.95 # GAE lambda for advantage estimation
    clip_ratio: float = 0.2 # PPO clipping ratio
    entropy_coeff: float = 0.01 # Entropy coefficient for exploration in policy gradient methods
    
    # SAC-specific
    alpha: float = 0.2 # Initial temperature parameter for SAC
    auto_entropy_tuning: bool = True # Whether to automatically tune alpha in SAC
    
    # General
    max_grad_norm: float = 10.0 # Max gradient norm for clipping
    double_dqn: bool = True # Whether to use Double DQN (if algorithm is DQN)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling non-serializable types."""
        cfg_dict = asdict(self)
        cfg_dict["algorithm"] = self.algorithm.value
        if isinstance(cfg_dict["device"], torch.device):
            cfg_dict["device"] = str(cfg_dict["device"])
        return cfg_dict

@dataclass
class BaseRLConfig:
    name: str
    obs_dim: int
    action_dim: int
    algorithm: RLAlgorithm
    device: Union[str, torch.device] = "cpu"
    gamma: float = 0.99
    lr: float = 1e-3
    max_grad_norm: float = 10.0
    batch_size: int = 64
    buffer_size: int = 100_000
    

    def __post_init__(self):
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

@dataclass
class PPOConfig(BaseRLConfig):
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coeff: float = 0.01
    rollout_steps: int = 2048


class ReplayBuffer:
    """Simple replay buffer for off-policy algorithms."""
    
    def __init__(self, max_size: int = 100_000, device: Union[str, torch.device] = "cpu"):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.device = device if isinstance(device, torch.device) else torch.device(device)
    
    def add(self, state, action, reward, next_state, done):
        """Add transition to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample random batch from buffer."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])

        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        rewards = np.asarray(rewards, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.bool_)
        
        if self.device != torch.device("cpu"):
            return (
                torch.from_numpy(states).to(self.device, torch.float32),
                torch.from_numpy(actions).to(self.device, torch.float32),
                torch.from_numpy(rewards).to(self.device, torch.float32),
                torch.from_numpy(next_states).to(self.device, torch.float32),
                torch.from_numpy(dones).to(self.device, torch.float32),
            )
        else:
            return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()


## A more specialized replay buffer for grid-based environments, if needed.

class GridReplayBuffer:
    def __init__(self, capacity = 50_000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)
        return_value = (np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones))
        return return_value

    def __len__(self): return len(self.buffer)
    def capacity(self): return self.buffer.maxlen



def build_ppo_actor_critic(obs_dim: int, n_actions: int, hidden_sizes=(256, 256), device="cpu"):
    """
    Returns (actor, critic) as TorchRL modules.

    Actor  : observation → logits → OneHotCategorical → action + log_prob
    Critic : observation → state value  V(s)
    """

    # ── Actor ────────────────────────────────────────────────────────────────
    actor_mlp = MLP(
        in_features=obs_dim,
        out_features=n_actions,
        num_cells=list(hidden_sizes),
        activation_class=nn.Tanh,
    ).to(device)

    actor_module = TensorDictModule(
        actor_mlp,
        in_keys=["observation"],
        out_keys=["logits"],
    )

    actor = ProbabilisticActor(
        module=actor_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=OneHotCategorical,
        return_log_prob=True,
        log_prob_key="sample_log_prob",
    )

    # ── Critic ───────────────────────────────────────────────────────────────
    critic_mlp = MLP(
        in_features=obs_dim,
        out_features=1,
        num_cells=list(hidden_sizes),
        activation_class=nn.Tanh,
    ).to(device)

    critic = ValueOperator(
        module=critic_mlp,
        in_keys=["observation"],
    )

    return actor, critic


def build_sac_networks(obs_dim: int, n_actions: int, hidden_sizes=(256, 256), device="cpu"):
    """
    SAC-style discrete actor + two Q-networks for critic.
    """
    from torchrl.modules import QValueActor

    # Actor
    actor_mlp = MLP(
        in_features=obs_dim,
        out_features=n_actions,
        num_cells=list(hidden_sizes),
        activation_class=nn.ReLU,
    ).to(device)

    actor_module = TensorDictModule(
        actor_mlp,
        in_keys=["observation"],
        out_keys=["logits"],
    )

    actor = ProbabilisticActor(
        module=actor_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=OneHotCategorical,
        return_log_prob=True,
        log_prob_key="sample_log_prob",
    )

    # Q-networks (two for clipped double-Q)
    def make_qnet():
        return TensorDictModule(
            MLP(
                in_features=obs_dim,
                out_features=n_actions,
                num_cells=list(hidden_sizes),
                activation_class=nn.ReLU,
            ).to(device),
            in_keys=["observation"],
            out_keys=["action_value"],
        )

    qnet1 = make_qnet()
    qnet2 = make_qnet()

    return actor, qnet1, qnet2

