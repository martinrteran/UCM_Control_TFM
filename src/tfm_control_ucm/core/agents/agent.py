"""
simple_agent.py

A simple reactive agent that uses LIDAR and robot orientation
to choose actions intelligently.

Actions:
    0 = forward
    1 = turn left
    2 = turn right

Author: Martin
"""

import os
from typing import Optional, Tuple
import numpy as np
import torch.nn as nn
import torch
from torch import optim

from .utils import RLAlgorithm

from .networks import  GridAgentNet, PolicyNetwork, ValueNetwork, SimpleQNetwork

from .base_rl_agent import BaseRLAgent, RLConfig, compute_gae
import random

from torchrl.data import ListStorage, ReplayBuffer, LazyTensorStorage
from tensordict import TensorDict
import torch.nn.functional as F

class GridAgent_0:
    def __init__(
        self,
        obs_dim,
        action_dim,
        device: str | torch.device = "cpu",
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=50_000,
        target_update=1000,
        buffer_size=100_000,
    ):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update

        # Networks
        self.policy_net = GridAgentNet(obs_dim=obs_dim, action_dim=action_dim).to(
            device
        )
        self.target_net = GridAgentNet(obs_dim=obs_dim, action_dim=action_dim).to(
            device
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer + loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Replay buffer
        self.buffer = ReplayBuffer(storage=ListStorage(max_size=buffer_size))
        # self.buffer = GridReplayBuffer()

        # Exploration
        self.epsilon = eps_start
        self.global_step = 0

        self.action_dim = action_dim

    # -----------------------------------------------------
    # Epsilon-greedy action selection
    # -----------------------------------------------------
    def select_action(self, state):
        self.global_step += 1

        # Update epsilon
        self.epsilon = self.eps_end + (self.eps_start - self.eps_end) * np.exp(
            -1.0 * self.global_step / self.eps_decay
        )

        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(q_values.argmax(dim=1).item())

    # -----------------------------------------------------
    # Store transition
    # -----------------------------------------------------
    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done) # type: ignore

    # -----------------------------------------------------
    # One training step
    # -----------------------------------------------------
    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(
            actions, dtype=torch.int64, device=self.device
        ).unsqueeze(1)
        rewards_t = torch.tensor(
            rewards, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        next_states_t = torch.tensor(
            next_states, dtype=torch.float32, device=self.device
        )
        dones_t = torch.tensor(
            dones, dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        # Q(s,a)
        q_values = self.policy_net(states_t).gather(1, actions_t)

        # max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
            target = rewards_t + self.gamma * (1 - dones_t) * next_q_values

        loss = self.criterion(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        if self.global_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    # -----------------------------------------------------
    # Save agent checkpoint
    # -----------------------------------------------------
    def save(self, path):
        checkpoint = {
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "global_step": self.global_step,
        }
        dir = os.path.dirname(path)
        if dir and not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
        torch.save(checkpoint, path)
        # print(f"[GridAgent] Saved checkpoint to {path}")

    # -----------------------------------------------------
    # Load agent checkpoint
    # -----------------------------------------------------
    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint found at {path}")

        checkpoint = torch.load(path, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.epsilon = checkpoint.get("epsilon", 1.0)
        self.global_step = checkpoint.get("global_step", 0)

        print(f"[GridAgent] Loaded checkpoint from {path}")


class GridAgent_2_FIXED:
    """
    DQN Agent with ALL fixes applied:
    - Larger network (128→64 instead of 10→5)
    - Better epsilon decay (100k instead of 50k)
    - Methods to reset when stuck
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: str | torch.device = "cpu",
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 64,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: int = 100_000,  # ← CHANGED: Was 50_000
        target_update: int = 1000,
        buffer_size: int = 100_000,
        double_dqn: bool = True,
    ):
        if isinstance(device, str):
            device = torch.device(device)

        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.double_dqn = double_dqn

        # Networks - USING FIXED NETWORK
        self.policy_net = GridAgentNet(obs_dim=obs_dim, action_dim=action_dim).to(
            device
        )
        self.target_net = GridAgentNet(obs_dim=obs_dim, action_dim=action_dim).to(
            device
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Replay Buffer
        self.buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=buffer_size, device=device),
            batch_size=batch_size,
        )

        # Exploration tracking
        self.epsilon = eps_start
        self.global_step = 0

    def select_action(self, state: np.ndarray):
        """Select action using epsilon-greedy policy."""
        self.global_step += 1

        # Exponential decay of epsilon
        self.epsilon = self.eps_end + (self.eps_start - self.eps_end) * np.exp(
            -1.0 * self.global_step / self.eps_decay
        )

        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            return np.argmax(np.random.randint(0, 10, self.action_dim))

        # Greedy action selection
        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store transition in replay buffer."""
        td = TensorDict(
            {
                "state": torch.tensor(state, dtype=torch.float32, device=self.device),
                "action": torch.tensor([action], dtype=torch.int64, device=self.device),
                "reward": torch.tensor(
                    [reward], dtype=torch.float32, device=self.device
                ),
                "next_state": torch.tensor(
                    next_state, dtype=torch.float32, device=self.device
                ),
                "done": torch.tensor([done], dtype=torch.bool, device=self.device),
            },
            batch_size=[],
        )
        self.buffer.add(td)

    def train_step(self) -> float | None:
        """Perform one training step."""
        if len(self.buffer) < self.batch_size:
            return None

        # Sample batch
        batch = self.buffer.sample(self.batch_size)

        states = batch["state"]
        actions = batch["action"]
        rewards = batch["reward"]
        next_states = batch["next_state"]
        dones = batch["done"].float()

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions)

        # Target Q values
        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions)
            else:
                next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]

            target_q = rewards + self.gamma * (1 - dones) * next_q

        # Loss and optimization
        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update target network
        if self.global_step % self.target_update == 0:
            self.update_target_network()

        return loss.item()

    def update_target_network(self):
        """Update target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # ========================================================================
    # NEW: Recovery Methods
    # ========================================================================


    def get_q_value_stats(self, states):
        """
        Get Q-value statistics for diagnostics.

        Parameters
        ----------
        states : list of np.ndarray
            Sample states to analyze

        Returns
        -------
        dict
            Statistics about Q-values
        """
        self.policy_net.eval()
        q_values_list = []

        with torch.no_grad():
            for state in states:
                state_t = torch.tensor(
                    state, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                q_values = self.policy_net(state_t).cpu().numpy()[0]
                q_values_list.append(q_values)

        q_values_array = np.array(q_values_list)

        return {
            "mean": q_values_array.mean(axis=0),
            "std": q_values_array.std(axis=0),
            "min": q_values_array.min(axis=0),
            "max": q_values_array.max(axis=0),
            "all_negative": np.all(q_values_array < 0),
            "all_similar": np.all(q_values_array.std(axis=0) < 0.01),
        }

    # ========================================================================
    # Save/Load (unchanged)
    # ========================================================================

    def save(self, path: str):
        """Save agent checkpoint."""
        checkpoint = {
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "global_step": self.global_step,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
        }

        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        torch.save(checkpoint, path)

    def load(self, path: str):
        """Load agent checkpoint."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint found at {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.epsilon = checkpoint.get("epsilon", 1.0)
        self.global_step = checkpoint.get("global_step", 0)

        print(f"[GridAgent] Loaded checkpoint from {path}")
        print(f"  - Epsilon: {self.epsilon:.4f}")
        print(f"  - Global step: {self.global_step}")



class GridAgent(BaseRLAgent):
    """
    DQN Agent for grid-based environments with recovery mechanisms.
    
    Inherits from BaseRLAgent and implements:
    - Double DQN with target network
    - Epsilon-greedy exploration with extended decay
    - Soft and hard reset for recovery from stuck states
    - Q-value diagnostics
    
    Parameters
    ----------
    config : RLConfig
        Configuration object containing:
        - obs_dim: observation dimension
        - action_dim: action dimension
        - gamma: discount factor (default 0.99)
        - lr: learning rate (default 1e-3)
        - batch_size: batch size (default 64)
        - eps_start: initial epsilon (default 1.0)
        - eps_end: final epsilon (default 0.05)
        - eps_decay: epsilon decay steps (default 100_000)
        - target_update: target network update frequency (default 1000)
        - buffer_size: replay buffer size (default 100_000)
        - double_dqn: use Double DQN (default True)
    """
    
    def __init__(
        self,
        config: RLConfig,  # RLConfig object
        policy_net: nn.Module,
        target_net: nn.Module,
        **kwargs
    ):
        """
        Initialize GridAgent.
        
        Parameters
        ----------
        config : RLConfig
            Configuration object
        hidden_dim : int, optional
            Hidden layer dimension for Q-network (default 128)
        num_hidden_layers : int, optional
            Number of hidden layers for Q-network (default 1)
        """
        # Store DQN-specific hyperparameters
        self.double_dqn = getattr(config, 'double_dqn', True)
        self.target_update_freq = getattr(config, 'target_update', 1000)
        self.eps_start = getattr(config, 'eps_start', 1.0)
        self.eps_end = getattr(config, 'eps_end', 0.05)
        self.eps_decay = getattr(config, 'eps_decay', 100_000)
        self.device = config.device
        # Initialize epsilon before calling parent __init__
        self.epsilon = self.eps_start
        # Create target network for DQN
        
        
        # self.value_net = target_net ## To avoid getting the initialization called

        # Call parent constructor (creates networks, optimizer, buffer)
        super().__init__(config, policy_net ,**kwargs)

        # Initialize target network with policy network weights
        self.target_net = target_net.to(config.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        
    def _initialize_optimizer(self):
        """Initialize optimizer."""
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.lr)
    
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Parameters
        ----------
        state : np.ndarray
            Current observation
        training : bool
            Whether in training mode (affects exploration)
        
        Returns
        -------
        int
            Selected action
        """
        self.global_step += 1
        
        # Exponential decay of epsilon
        self.epsilon = self.eps_end + (self.eps_start - self.eps_end) * np.exp(
            -1.0 * self.global_step / self.eps_decay
        )
        
        # Epsilon-greedy exploration (only during training)
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.config.action_dim)
        
        # Greedy action selection
        state_t = self._to_tensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        
        return int(q_values.argmax(dim=1).item())
    
    def train_step(self) -> Optional[float]:
        """
        Perform one DQN training step.
        
        Samples a batch from the replay buffer, computes target Q-values
        using the target network, and updates the policy network.
        
        Returns
        -------
        float or None
            Loss value, or None if buffer is too small
        """
        if len(self.buffer) < self.config.batch_size:
            return None
        
        # Sample batch from replay buffer
        batch = self.buffer.sample(self.config.batch_size)
        
        # Unpack batch
        states, actions, rewards, next_states, dones = batch
        
        # Convert to tensors
        states_t = self._to_tensor(states)
        actions_t = self._to_tensor(actions, dtype=torch.int64)
        rewards_t = self._to_tensor(rewards)
        next_states_t = self._to_tensor(next_states)
        dones_t = self._to_tensor(dones).float()
        
        # Compute current Q-values
        current_q = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1))
        
        # Compute target Q-values
        loss = self._compute_loss(
            current_q,
            next_states_t,
            rewards_t,
            dones_t
        )
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self._clip_grad_norm()
        self.optimizer.step()
        
        # Update target network periodically
        if self.global_step % self.target_update_freq == 0:
            self.update_target_network()
        
        return loss.item()
    
    def _compute_loss(
        self,
        current_q: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute DQN loss (Double DQN if enabled).
        
        Parameters
        ----------
        current_q : torch.Tensor
            Current Q-values from policy network
        next_states : torch.Tensor
            Next state observations
        rewards : torch.Tensor
            Rewards received
        dones : torch.Tensor
            Episode termination flags
        
        Returns
        -------
        torch.Tensor
            Smooth L1 loss
        """
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: use policy network to select actions,
                # target network to evaluate them
                next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions)
            else:
                # Standard DQN: use target network for both selection and evaluation
                next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            
            target_q = rewards.unsqueeze(1) + self.config.gamma * (1 - dones.unsqueeze(1)) * next_q
        
        # Smooth L1 loss for stability
        loss = F.smooth_l1_loss(current_q, target_q)
        
        return loss
    
    def update_target_network(self):
        """Update target network weights from policy network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Log update if training verbose
        if hasattr(self.config, 'verbose') and self.config.verbose:
            print(f"[Step {self.global_step}] Target network updated")
    
    # ========================================================================
    #  reset methods with DQN-specific behavior
    # ========================================================================

    
    def _initialize_networks(self, **kwargs):
        """
        Initialize policy and value networks.
        
        Called during __init__ if networks not provided.
        Subclasses should  to create algorithm-specific architectures.
        """
        # self.policy_net.apply(self.policy_net._init_weights)
        self.policy_net.reset_parameters() # type: ignore
        self.target_net.reset_parameters() # type: ignore
    

        self.policy_net.to(self.device)
        self.target_net.to(self.device)

        self.policy_net.load_state_dict(self.target_net.state_dict())
        self.target_net.eval()

        
    
    # ========================================================================
    #  save/load to include target network and epsilon
    # ========================================================================
    
    def save(self, path: str):
        """
        Save agent checkpoint including target network.
        
        Parameters
        ----------
        path : str
            Path to save checkpoint
        """
        from tqdm import tqdm
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        checkpoint = {
            "algorithm": self.config.algorithm.value,
            "config": self.config.to_dict(),
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "global_step": self.global_step,
            "episode_count": self.episode_count,
            "train_history": self.train_history,
        }
        
        torch.save(checkpoint, path)
        tqdm.write(f"✅ Saved GridAgent checkpoint to {path}")
        tqdm.write(f"   - Policy net weights saved")
        tqdm.write(f"   - Target net weights saved")
        tqdm.write(f"   - Epsilon: {self.epsilon:.4f}")
        tqdm.write(f"   - Global step: {self.global_step}")
    
    def load(self, path: str):
        """
        Load agent checkpoint including target network.
        
        Parameters
        ----------
        path : str
            Path to load checkpoint from
        """
        from tqdm import tqdm
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint found at {path}")
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        self.epsilon = checkpoint.get("epsilon", self.eps_start)
        self.global_step = checkpoint.get("global_step", 0)
        self.episode_count = checkpoint.get("episode_count", 0)
        self.train_history = checkpoint.get("train_history", {})
        
        tqdm.write(f"✅ Loaded GridAgent checkpoint from {path}")
        tqdm.write(f"   - Algorithm: {checkpoint.get('algorithm', 'DQN')}")
        tqdm.write(f"   - Epsilon: {self.epsilon:.4f}")
        tqdm.write(f"   - Global step: {self.global_step}")
        tqdm.write(f"   - Episodes: {self.episode_count}")
    
    # ========================================================================
    # Diagnostics
    # ========================================================================
    
    def get_q_value_stats(self, states):
        """
        Get Q-value statistics for diagnostics.
        
        Analyzes the network outputs to detect issues like:
        - All negative Q-values (potential reward issue)
        - All similar Q-values (network not learning properly)
        
        Parameters
        ----------
        states : list of np.ndarray
            Sample states to analyze
        
        Returns
        -------
        dict
            Statistics including mean, std, min, max, and flags
        """
        self.policy_net.eval()
        q_values_list = []
        
        with torch.no_grad():
            for state in states:
                state_t = self._to_tensor(state).unsqueeze(0)
                q_values = self.policy_net(state_t).cpu().numpy()[0]
                q_values_list.append(q_values)
        
        q_values_array = np.array(q_values_list)
        
        stats = {
            "mean": q_values_array.mean(axis=0),
            "std": q_values_array.std(axis=0),
            "min": q_values_array.min(axis=0),
            "max": q_values_array.max(axis=0),
            "all_negative": np.all(q_values_array < 0),
            "all_similar": np.all(q_values_array.std(axis=0) < 0.01),
        }
        
        return stats
    
    def __repr__(self) -> str:
        return (
            f"GridAgent(\n"
            f"  obs_dim={self.config.obs_dim},\n"
            f"  action_dim={self.config.action_dim},\n"
            f"  device={self.device},\n"
            f"  double_dqn={self.double_dqn},\n"
            f"  epsilon={self.epsilon:.4f},\n"
            f"  global_step={self.global_step}\n"
            f")"
        )


# ============================================================================
# EXAMPLE: DQN IMPLEMENTATION
# ============================================================================

class DQNAgent(BaseRLAgent):
    """
    DQN (Deep Q-Network) agent implementation.
    
    Parameters
    ----------
    config : RLConfig
        Configuration with algorithm=RLAlgorithm.DQN or DDQN
    q_network : nn.Module, optional
        Q-network. If None, creates SimpleQNetwork
    """
    
    def __init__(
        self,
        config: RLConfig,
        q_network: Optional[nn.Module] = None,
        **kwargs
    ):
        self.q_network = q_network
        super().__init__(
            config=config,
            policy_network=q_network,
            value_network=None,
            **kwargs
        )
        
        # Target network for DQN
        self.target_net = SimpleQNetwork(
            config.obs_dim,
            config.action_dim,
            hidden_dim=kwargs.get("hidden_dim", 128)
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Exploration
        self.epsilon = config.eps_start
    
    def _initialize_networks(self, hidden_dim: int = 128, **kwargs):
        """Initialize Q-networks if not provided."""
        self.policy_net = SimpleQNetwork(
            self.config.obs_dim,
            self.config.action_dim,
            hidden_dim=hidden_dim
        )
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        self.global_step += 1
        
        # Epsilon decay
        self.epsilon = self.config.eps_end + (
            self.config.eps_start - self.config.eps_end
        ) * np.exp(-1.0 * self.global_step / self.config.eps_decay)
        
        # Epsilon-greedy
        if training and np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.config.action_dim))
        
        state_t = self._to_tensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(q_values.argmax(dim=1).item())
    
    def train_step(self) -> Optional[float]:
        """Perform DQN training step."""
        if len(self.buffer) < self.config.batch_size:
            return None
        
        batch = self.buffer.sample(self.config.batch_size)
        loss = self._compute_loss(batch)
        
        self.optimizer.zero_grad()
        loss.backward()
        self._clip_grad_norm()
        self.optimizer.step()
        
        # Update target network
        if self.global_step % self.config.target_update == 0:
            self.hard_update(self.policy_net, self.target_net)
        
        return loss.item()
    
    def _compute_loss(self, batch: Tuple) -> torch.Tensor:
        """Compute DQN loss."""
        states, actions, rewards, next_states, dones = batch
        
        states_t = self._to_tensor(states)
        actions_t = self._to_tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards_t = self._to_tensor(rewards).unsqueeze(1)
        next_states_t = self._to_tensor(next_states)
        dones_t = self._to_tensor(dones.astype(np.float32)).unsqueeze(1)
        
        # Current Q values
        q_values = self.policy_net(states_t).gather(1, actions_t)
        
        # Target Q values
        with torch.no_grad():
            if self.config.use_double_dqn:
                # Double DQN: use policy network to select action, target network to evaluate
                next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
                next_q_values = self.target_net(next_states_t).gather(1, next_actions)
            else:
                # Standard DQN
                next_q_values = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
            
            target_q = rewards_t + self.config.gamma * (1 - dones_t) * next_q_values
        
        # Loss
        loss = F.smooth_l1_loss(q_values, target_q)
        return loss


# ============================================================================
# EXAMPLE: Policy Gradient Network
# ============================================================================

class A2CAgent(BaseRLAgent):
    """
    Advantage Actor-Critic (A2C) agent implementation.
    
    Parameters
    ----------
    config : RLConfig
        Configuration with algorithm=RLAlgorithm.A2C
    policy_network : nn.Module, optional
        Policy (actor) network
    value_network : nn.Module, optional
        Value (critic) network
    """
    
    def __init__(
        self,
        config: RLConfig,
        policy_network: Optional[nn.Module] = None,
        value_network: Optional[nn.Module] = None,
        **kwargs
    ):
        super().__init__(
            config=config,
            policy_network=policy_network,
            value_network=value_network,
            **kwargs
        )
        self.episode_transitions = []
    
    def _initialize_networks(self, hidden_dim: int = 128, **kwargs):
        """Initialize policy and value networks if not provided."""
        self.policy_net = PolicyNetwork(
            self.config.obs_dim,
            self.config.action_dim,
            hidden_dim=hidden_dim
        )
        self.value_net = ValueNetwork(
            self.config.obs_dim,
            hidden_dim=hidden_dim
        )
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action from policy distribution."""
        self.global_step += 1
        
        state_t = self._to_tensor(state).unsqueeze(0)
        with torch.no_grad():
            probs = self.policy_net(state_t)
        
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.last_log_prob = dist.log_prob(action)
        
        return int(action.item())
    
    def train_step(self) -> Optional[float]:
        """Perform A2C training step on episode."""
        if len(self.buffer) < self.config.batch_size:
            return None
        
        batch = self.buffer.sample(self.config.batch_size)
        loss = self._compute_loss(batch)
        
        self.optimizer.zero_grad()
        loss.backward()
        self._clip_grad_norm()
        self.optimizer.step()
        
        return loss.item()
    
    def _compute_loss(self, batch: Tuple) -> torch.Tensor:
        """Compute A2C loss (actor + critic)."""
        states, actions, rewards, next_states, dones = batch
        
        states_t = self._to_tensor(states)
        actions_t = self._to_tensor(actions, dtype=torch.int64)
        rewards_t = self._to_tensor(rewards)
        next_states_t = self._to_tensor(next_states)
        dones_t = self._to_tensor(dones.astype(np.float32))
        
        # Compute advantages
        with torch.no_grad():
            values = self.value_net(states_t).squeeze()
            next_values = self.value_net(next_states_t).squeeze()
            td_target = rewards_t + self.config.gamma * next_values * (1 - dones_t)
            advantages = td_target - values
        
        # Actor loss
        probs = self.policy_net(states_t)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions_t)
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic loss
        values = self.value_net(states_t).squeeze()
        critic_loss = F.mse_loss(values, td_target)
        
        # Entropy bonus
        entropy = dist.entropy().mean()
        
        # Total loss
        total_loss = actor_loss + 0.5 * critic_loss - self.config.entropy_coeff * entropy
        
        return total_loss
    


class PPOAgent(BaseRLAgent):
    """
    Proximal Policy Optimization (PPO) agent implementation.

    Parameters
    ----------
    config : RLConfig
        Configuration with algorithm=RLAlgorithm.PPO
    policy_network : nn.Module, optional
        Policy (actor) network
    value_network : nn.Module, optional
        Value (critic) network
    """

    def __init__(
        self,
        config: RLConfig,
        policy_network: Optional[nn.Module] = None,
        value_network: Optional[nn.Module] = None,
        **kwargs
    ):
        super().__init__(
            config=config,
            policy_network=policy_network,
            value_network=value_network,
            **kwargs
        )

        self.clip_eps = config.clip_eps
        self.entropy_coeff = config.entropy_coeff
        self.value_coeff = getattr(config, "value_coeff", 0.5)
        self.gamma = config.gamma
        self.lam = getattr(config, "gae_lambda", 0.95)

    def _initialize_networks(self, hidden_dim: int = 128, **kwargs):
        """Initialize policy and value networks if not provided."""
        self.policy_net = PolicyNetwork(
            self.config.obs_dim,
            self.config.action_dim,
            hidden_dim=hidden_dim
        )
        self.value_net = ValueNetwork(
            self.config.obs_dim,
            hidden_dim=hidden_dim
        )

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action from policy distribution (stochastic in training, greedy in eval)."""
        self.global_step += 1

        state_t = self._to_tensor(state).unsqueeze(0)
        logits_or_probs = self.policy_net(state_t)

        # Asumiendo que PolicyNetwork devuelve logits; si devuelve probs, cambia a probs=...
        dist = torch.distributions.Categorical(logits=logits_or_probs)

        if training:
            action = dist.sample()
        else:
            action = torch.argmax(logits_or_probs, dim=-1)

        self.last_log_prob = dist.log_prob(action).detach()
        return int(action.item())

    def _compute_loss(self, batch: Tuple) -> torch.Tensor:
        """
        Compute PPO loss from batch:
        batch = (states, actions, rewards, next_states, dones, old_log_probs, advantages, returns)
        """
        (states, actions, rewards, next_states,
         dones, old_log_probs, advantages, returns) = batch

        states_t = self._to_tensor(states)
        actions_t = self._to_tensor(actions, dtype=torch.int64)
        old_log_probs_t = self._to_tensor(old_log_probs)
        advantages_t = self._to_tensor(advantages)
        returns_t = self._to_tensor(returns)

        logits_or_probs = self.policy_net(states_t)
        dist = torch.distributions.Categorical(logits=logits_or_probs)
        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()

        # ratio
        ratio = torch.exp(log_probs - old_log_probs_t)
        surr1 = ratio * advantages_t
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages_t
        policy_loss = -torch.min(surr1, surr2).mean()

        # value loss
        values = self.value_net(states_t)
        value_loss = (returns_t - values).pow(2).mean()

        loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy
        return loss

    def train_step(self) -> Optional[float]:
        """Perform PPO training step."""
        if len(self.buffer) < self.config.batch_size:
            return None

        batch = self.buffer.sample(self.config.batch_size)
        loss = self._compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self._clip_grad_norm()
        self.optimizer.step()

        self.log_training_step(loss=float(loss.item()))
        return float(loss.item())


class SACDiscreteAgent(BaseRLAgent):
    """
    Soft Actor-Critic (Discrete) agent.
    Compatible with your BaseRLAgent and ReplayBuffer.
    """

    def __init__(
        self,
        config: RLConfig,
        policy_network: Optional[nn.Module] = None,
        value_network: Optional[nn.Module] = None,
        **kwargs
    ):
        super().__init__(
            config=config,
            policy_network=policy_network,
            value_network=value_network,
            **kwargs
        )

        # Two Q networks (SAC requirement)
        hidden_dim = kwargs.get("hidden_dim", 128)
        self.q1 = SimpleQNetwork(config.obs_dim, config.action_dim, hidden_dim).to(self.device)
        self.q2 = SimpleQNetwork(config.obs_dim, config.action_dim, hidden_dim).to(self.device)

        self.q1_target = SimpleQNetwork(config.obs_dim, config.action_dim, hidden_dim).to(self.device)
        self.q2_target = SimpleQNetwork(config.obs_dim, config.action_dim, hidden_dim).to(self.device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.alpha = config.alpha  # entropy temperature

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=config.lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=config.lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=config.lr)

    def _initialize_networks(self, hidden_dim: int = 128, **kwargs):
        self.policy_net = PolicyNetwork(self.config.obs_dim, self.config.action_dim, hidden_dim)
        self.value_net = None  # SAC does not use a value network

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        state_t = self._to_tensor(state).unsqueeze(0)
        logits = self.policy_net(state_t)
        dist = torch.distributions.Categorical(logits=logits)

        if training:
            action = dist.sample()
        else:
            action = torch.argmax(logits, dim=-1)

        return int(action.item())

    def _compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch

        states_t = self._to_tensor(states)
        actions_t = self._to_tensor(actions, dtype=torch.long)
        rewards_t = self._to_tensor(rewards)
        next_states_t = self._to_tensor(next_states)
        dones_t = self._to_tensor(dones)

        # Policy distribution
        logits = self.policy_net(states_t)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()

        # Q-values
        q1_vals = self.q1(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()
        q2_vals = self.q2(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()

        # Next-state policy
        next_logits = self.policy_net(next_states_t)
        next_dist = torch.distributions.Categorical(logits=next_logits)
        next_actions = next_dist.sample()
        next_log_probs = next_dist.log_prob(next_actions)

        q1_next = self.q1_target(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze()
        q2_next = self.q2_target(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze()
        q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs

        target = rewards_t + self.config.gamma * (1 - dones_t) * q_next.detach()

        # Q losses
        q1_loss = F.mse_loss(q1_vals, target)
        q2_loss = F.mse_loss(q2_vals, target)

        # Policy loss
        policy_loss = (self.alpha * log_probs - torch.min(q1_vals, q2_vals)).mean()

        # Update networks
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Soft update
        self.soft_update(self.q1, self.q1_target)
        self.soft_update(self.q2, self.q2_target)

        return (q1_loss + q2_loss + policy_loss).item()

    def train_step(self):
        if len(self.buffer) < self.config.batch_size:
            return None
        batch = self.buffer.sample(self.config.batch_size)
        return self._compute_loss(batch)


class AgentFactory:
    @staticmethod
    def create_agent(config: RLConfig, **kwargs) -> BaseRLAgent:
        if config.algorithm == RLAlgorithm.DQN or config.algorithm == RLAlgorithm.DDQN:
            return DQNAgent(config, **kwargs)
        elif config.algorithm == RLAlgorithm.A2C:
            return A2CAgent(config, **kwargs)
        elif config.algorithm == RLAlgorithm.GRID_AGENT:
            return GridAgent(config, **kwargs)
        else:
            raise ValueError(f"Unsupported algorithm: {config.algorithm}")
        

if __name__ == "__main__":
    # Example usage
    print("=" * 70)
    print("Base RL Agent Framework - Example Usage")
    print("=" * 70)
    
    # Create DQN agent
    config = RLConfig(
        name ="test",
        obs_dim=10,
        action_dim=3,
        algorithm=RLAlgorithm.DDQN,
        device="cpu",
        gamma=0.99,
        lr=1e-3,
        batch_size=32,
    )
    
    print("\n📦 Creating DQN Agent...")
    dqn_agent = DQNAgent(config, hidden_dim=128)
    print(dqn_agent)
    
    # Example interaction
    state = np.random.randn(10).astype(np.float32)
    action = dqn_agent.select_action(state)
    print(f"\n🎯 Selected action: {action}")
    
    # Create A2C agent
    print("\n" + "=" * 70)
    print("Creating A2C Agent...")
    config.algorithm = RLAlgorithm.A2C
    a2c_agent = A2CAgent(config, hidden_dim=128)
    print(a2c_agent)
    
    action = a2c_agent.select_action(state)
    print(f"\n🎯 Selected action: {action}")
    
    print("\n" + "=" * 70)
    print("✅ Framework ready for DQN, PPO, SAC, A3C extensions!")
    print("=" * 70)