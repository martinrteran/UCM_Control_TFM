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
import numpy as np
import torch.nn as nn
import torch
from torch import optim
from .utils import GridReplayBuffer
import random


class GridAgentNet(nn.Module):
    def __init__(self, *, obs_dim, action_dim):
        """
        Parameters
        ----------
        turn_threshold : float
            If the front LIDAR rays detect an obstacle closer than this
            distance, the agent will turn instead of moving forward.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 10),
            nn.ReLU(),
            nn.Linear(10,5),
            nn.ReLU(),
            nn.Linear(5, action_dim)
        )

    def forward(self, x):
        return self.net(x)
    

class GridAgent:
    def __init__(
        self,
        obs_dim,
        action_dim,
        device:str|torch.device="cpu",
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=50_000,
        target_update=1000
    ):
        if isinstance(device, str): device = torch.device(device)
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update

        # Networks
        self.policy_net = GridAgentNet(obs_dim=obs_dim, action_dim=action_dim).to(device)
        self.target_net = GridAgentNet(obs_dim=obs_dim, action_dim=action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer + loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Replay buffer
        self.buffer = GridReplayBuffer()

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
        self.epsilon = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1.0 * self.global_step / self.eps_decay)

        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(q_values.argmax(dim=1).item())

    # -----------------------------------------------------
    # Store transition
    # -----------------------------------------------------
    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    # -----------------------------------------------------
    # One training step
    # -----------------------------------------------------
    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

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
            "global_step": self.global_step
        }
        dir = os.path.dirname(path)
        if dir and not os.path.exists(dir): os.makedirs(dir,exist_ok=True)
        torch.save(checkpoint, path)
        print(f"[GridAgent] Saved checkpoint to {path}")

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


