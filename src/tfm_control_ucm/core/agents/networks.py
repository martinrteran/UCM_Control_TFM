from dataclasses import dataclass
from os import name

import torch.nn.functional as F
import torch.nn as nn


class GridAgentNet(nn.Module):
    name = "grid_agent_network"
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
            nn.Linear(obs_dim, 253),
            nn.ReLU(),
            nn.Linear(253, 56),
            nn.ReLU(),
            nn.Linear(56, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimpleQNetwork(nn.Module):
    """Simple Q-network for DQN."""
    name = "simple_q_network"

    def __init__(self, obs_dim: int, action_dim: int, num_hidden_layers: int = 1, hidden_dim: int = 10):
        super().__init__()
        capas = [nn.Linear(obs_dim, hidden_dim), nn.ReLU()]
        if num_hidden_layers > 0:
            for _ in range(num_hidden_layers):
                capas.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        capas.append(nn.Linear(hidden_dim, action_dim))

        self.net = nn.Sequential(*capas)
    
    def forward(self, x):
        return self.net(x)
    
    def reset_parameters(self):
        """Reset network parameters (useful for soft resets)."""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()                
    

class PolicyNetwork(nn.Module):
    """Policy network for policy gradient methods (PPO, A2C)."""
    name = "policy_network"
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
    
    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)


class ValueNetwork(nn.Module):
    """Value network for policy gradient methods."""
    name = "value_network"
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    
    def forward(self, x):
        return self.net(x)


def CreateNetwork(name:str, **kwargs)->nn.Module:
    if name == "grid_agent_network":
        return  GridAgentNet(**kwargs)
    elif name == "simple_q_network":
        return SimpleQNetwork(**kwargs)
    elif name == "policy_network":
        return PolicyNetwork(**kwargs)
    elif name == "value_network":
        return ValueNetwork(**kwargs)
    else:
        raise ValueError(f"Unknown network type: {name}")