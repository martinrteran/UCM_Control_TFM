"""
base_rl_agent.py

A flexible, extensible base reinforcement learning agent that supports multiple
algorithms (DQN, PPO, SAC, A3C, etc.) through inheritance and configurable
neural network components.

Architecture:
    - BaseRLAgent: Abstract base class defining the RL interface
    - RLAlgorithm: Enum for supported algorithms
    - Network components are passed as parameters (strategies pattern)
    - Child classes implement algorithm-specific logic

Author: Generated for flexible RL framework
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod
from .utils import RLConfig,ReplayBuffer
from typing import Dict, Any, Optional, SupportsFloat, Tuple, List, Union


# ============================================================================
# BASE RL AGENT
# ============================================================================

class BaseRLAgent(ABC):
    """
    Abstract base class for reinforcement learning agents.
    
    Subclasses should implement:
        - _initialize_networks()
        - select_action()
        - train_step()
        - _compute_loss()
    
    Parameters
    ----------
    config : RLConfig
        Configuration object with all hyperparameters
    policy_network : nn.Module, optional
        Policy network (actor). If None, must be created in _initialize_networks()
    value_network : nn.Module, optional
        Value/critic network. If None, must be created in _initialize_networks()
    """
    
    def __init__(
        self,
        config: RLConfig,
        policy_network: Optional[nn.Module] = None,
        value_network: Optional[nn.Module] = None,
        **kwargs
    ) -> None:
        assert policy_network is not None or value_network is not None, "At least one network must be provided"

        if isinstance(config.device, str):
            config.device = torch.device(config.device)
        
        self.config = config
        self.device = config.device
        self.algorithm = config.algorithm
        
        # Networks
        self.policy_net = policy_network
        self.value_net = value_network
        
        # Initialize networks if not provided
        if self.policy_net is None and self.value_net is None:
            self._initialize_networks(**kwargs)
        
        assert self.policy_net is not None, "The policy networks must be initialized"

        # Move networks to device
        self.policy_net.to(self.device)
        if self.value_net is not None:
            self.value_net.to(self.device)
        
        # Optimizer
        self._initialize_optimizer()
        #from torchrl.data import LazyTensorStorage, ReplayBuffer
        # Replay buffer (for off-policy methods)
        self.buffer =  ReplayBuffer(max_size=config.buffer_size,device=self.device) # ReplayBuffer(
        #     storage=LazyTensorStorage(
        #         max_size=self.buffer.storage.max_size, device=self.device
        #     ),
        #     batch_size=self.config.batch_size,
        # )# 
        # Tracking
        self.global_step = 0
        self.episode_count = 0
        self.train_history = {
            "loss": [],
            "reward": [],
            "episode_length": [],
        }

    @abstractmethod
    def _initialize_networks(self, **kwargs):
        """
        Initialize policy and value networks.
        
        Called during __init__ if networks not provided.
        Subclasses should override to create algorithm-specific architectures.
        """
        raise NotImplementedError
    
    def _initialize_optimizer(self):
        """Initialize optimizer for policy network."""
        params = list(self.policy_net.parameters()) # type: ignore
        if self.value_net is not None:
            params.extend(self.value_net.parameters())
        
        self.optimizer = optim.Adam(params, lr=self.config.lr)
    
    @abstractmethod
    def select_action(self, state: np.ndarray, training: bool = True) -> Union[int, np.ndarray]:
        """
        Select action based on current policy.
        
        Parameters
        ----------
        state : np.ndarray
            Observation/state
        training : bool
            Whether in training mode (affects exploration)
        
        Returns
        -------
        int or np.ndarray
            Selected action
        """
        raise NotImplementedError
    
    @abstractmethod
    def train_step(self) -> Optional[float]:
        """
        Perform one training step.
        
        Returns
        -------
        float or None
            Loss value, or None if training not performed
        """
        raise NotImplementedError
    
    @abstractmethod
    def _compute_loss(self, batch: Tuple) -> torch.Tensor:
        """
        Compute loss for the batch.
        
        Algorithm-specific implementation.
        """
        raise NotImplementedError
    
    # ========================================================================
    # Common utilities
    # ========================================================================
    
    def store(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: Union[float, SupportsFloat],
        next_state: np.ndarray,
        done: bool,
    ):
        """Store transition in replay buffer."""
        self.buffer.add(state, action, reward, next_state, done)
    
    def _to_tensor(
        self,
        data: np.ndarray,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Convert numpy array to tensor on device."""
        return torch.tensor(data, dtype=dtype, device=self.device)
    
    def _clip_grad_norm(self):
        """Clip gradients for stability."""
        if self.policy_net is not None:
            nn.utils.clip_grad_norm_(
                self.policy_net.parameters(),
                self.config.max_grad_norm
            )
        if self.value_net is not None:
            nn.utils.clip_grad_norm_(
                self.value_net.parameters(),
                self.config.max_grad_norm
            )
    
    def soft_update(self, source: nn.Module, target: nn.Module, tau: Optional[float] = None):
        """Soft update target network: target = (1-tau)*target + tau*source"""
        if tau is None:
            tau = self.config.tau
        
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                tau * source_param.data + (1 - tau) * target_param.data
            )
    
    def hard_update(self, source: nn.Module, target: nn.Module):
        """Hard update target network: target = source"""
        target.load_state_dict(source.state_dict())
    
    # ========================================================================
    # State management
    # ========================================================================
    
    def reset_exploration(self):
        """Reset exploration parameters."""
        self.global_step = 0
        if hasattr(self, "epsilon"):
            self.epsilon = self.config.eps_start
    
    def clear_buffer(self):
        """Clear replay buffer."""
        self.buffer.clear()

    
    # ========================================================================
    # Saving and loading
    # ========================================================================
    
    def save(self, path: str):
        """Save agent checkpoint."""
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        checkpoint = {
            "algorithm": self.algorithm.value,
            "config": self.config.to_dict(),
            "policy_state_dict": self.policy_net.state_dict(), # type: ignore
            "global_step": self.global_step,
            "episode_count": self.episode_count,
            "train_history": self.train_history,
        }
        
        if self.value_net is not None:
            checkpoint["value_state_dict"] = self.value_net.state_dict()
        
        if hasattr(self, "epsilon"):
            checkpoint["epsilon"] = self.epsilon
        
        if hasattr(self, "optimizer"):
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
        
        torch.save(checkpoint, path)
        print(f"✅ Saved checkpoint to {path}")
    
    def load(self, path: str):
        """Load agent checkpoint."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint found at {path}")
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"]) # type: ignore
        
        if "value_state_dict" in checkpoint and self.value_net is not None:
            self.value_net.load_state_dict(checkpoint["value_state_dict"])
        
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        self.global_step = checkpoint.get("global_step", 0)
        self.episode_count = checkpoint.get("episode_count", 0)
        self.train_history = checkpoint.get("train_history", {})
        
        if "epsilon" in checkpoint:
            self.epsilon = checkpoint["epsilon"]
        
        print(f"✅ Loaded checkpoint from {path}")
        print(f"   - Algorithm: {checkpoint['algorithm']}")
        print(f"   - Steps: {self.global_step}")
        print(f"   - Episodes: {self.episode_count}")
    
    # ========================================================================
    # Diagnostics
    # ========================================================================
    
    def get_q_value_stats(self, states: List[np.ndarray]) -> Dict[str, Any]:
        """
        Get Q-value statistics (for DQN-based agents).
        
        Parameters
        ----------
        states : list of np.ndarray
            Sample states to analyze
        
        Returns
        -------
        dict
            Statistics about network outputs
        """
        self.policy_net.eval() # type: ignore
        outputs_list = []
        
        with torch.no_grad():
            for state in states:
                state_t = self._to_tensor(state).unsqueeze(0)
                output = self.policy_net(state_t).cpu().numpy()[0] # type: ignore
                outputs_list.append(output)
        
        outputs_array = np.array(outputs_list)
        
        return {
            "mean": outputs_array.mean(axis=0),
            "std": outputs_array.std(axis=0),
            "min": outputs_array.min(axis=0),
            "max": outputs_array.max(axis=0),
            "all_negative": np.all(outputs_array < 0),
            "all_similar": np.all(outputs_array.std(axis=0) < 0.01),
        }
    
    def log_training_step(self, loss: Optional[float] = None, reward: Optional[float] = None):
        """Log training metrics."""
        if loss is not None:
            self.train_history["loss"].append(loss)
        if reward is not None:
            self.train_history["reward"].append(reward)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        summary = {}
        
        if self.train_history["loss"]:
            losses = self.train_history["loss"]
            summary["avg_loss"] = np.mean(losses[-100:])  # Last 100 steps
            summary["min_loss"] = np.min(losses)
            summary["max_loss"] = np.max(losses)
        
        if self.train_history["reward"]:
            rewards = self.train_history["reward"]
            summary["avg_reward"] = np.mean(rewards[-100:])  # Last 100 episodes
            summary["max_reward"] = np.max(rewards)
        
        summary["global_step"] = self.global_step
        summary["episode_count"] = self.episode_count
        
        return summary
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  algorithm={self.algorithm.value},\n"
            f"  obs_dim={self.config.obs_dim},\n"
            f"  action_dim={self.config.action_dim},\n"
            f"  device={self.device},\n"
            f"  global_step={self.global_step}\n"
            f")"
        )

