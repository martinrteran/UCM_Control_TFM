"""DQNAgent fixes: target-network schedule and evaluation not moving the exploration."""
import numpy as np
import torch

from tfm_control_ucm.core.agents.agent import DQNAgent
from tfm_control_ucm.core.agents.networks import SimpleQNetwork
from tfm_control_ucm.core.agents.utils import DQNConfig, RLAlgorithm

OBS, ACT = 6, 4


def make_agent(**cfg):
    torch.manual_seed(0)
    np.random.seed(0)
    config = DQNConfig(name="dqn", obs_dim=OBS, action_dim=ACT, algorithm=RLAlgorithm.DQN, device="cpu", batch_size=4, **cfg)
    return DQNAgent(config, q_network=SimpleQNetwork(OBS, ACT, 1, 16))


def same_networks(agent):
    return all(torch.equal(a, b) for a, b in zip(agent.policy_net.parameters(), agent.target_net.parameters()))


def test_evaluation_does_not_move_epsilon_or_the_step_counter():
    agent = make_agent(eps_decay=10)
    for _ in range(5):
        agent.select_action(np.zeros(OBS), training=True)
    epsilon, steps = agent.epsilon, agent.global_step
    for _ in range(50):
        agent.select_action(np.zeros(OBS), training=False)
    assert agent.epsilon == epsilon and agent.global_step == steps == 5


def test_target_network_is_copied_every_target_update_updates_whatever_the_step():
    agent = make_agent(target_update=3, lr=0.1)
    for i in range(8):
        agent.store(np.random.rand(OBS), i % ACT, float(i), np.random.rand(OBS), True)
    agent.global_step = 7  # Not a multiple of target_update: the old check on global_step never fired here
    agent.train_step(); agent.train_step()
    assert not same_networks(agent)
    agent.train_step()
    assert same_networks(agent) and agent.updates == 3
