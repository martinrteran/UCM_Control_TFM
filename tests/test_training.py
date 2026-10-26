"""
One training loop for every agent: smoke tests with tiny maps and few episodes,
plus the episode hooks of the on-policy agents.
"""
import numpy as np
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from tfm_control_ucm.core.agents.agent import DQNAgent, PPOAgent, RecurrentPPOAgent
from tfm_control_ucm.core.agents.networks import (RecurrentGRUPolicy, RecurrentGRUValue, SimpleQNetwork,
                                                  ValueNetwork)
from tfm_control_ucm.core.agents.utils import DQNConfig, PPOConfig, PPORNNConfig, RLAlgorithm
from tfm_control_ucm.core.grid_env.environment import Grid_Robot_Sections_Env
from tfm_control_ucm.core.grid_env.map import GridMap
from tfm_control_ucm.core.grid_env.robot import GridRobotNotTurning
from tfm_control_ucm.core.rewards import ProgressReward
from tfm_control_ucm.core.training.TrainingManager import MultiEnvironmentTrainer, SingleTrainer, TrainerConfig

LIDAR = {"num_rays": 24, "max_range": 10, "fov": float(2 * np.pi), "noise_std": 0.0, "with_cache": True, "device": "cpu"}
EPISODES = 4


def make_env(name="m", size=(25, 25)):
    return Grid_Robot_Sections_Env(map=GridMap(name, size), robot=GridRobotNotTurning(), reward_fn=ProgressReward(),
                                   lidar_config=LIDAR, num_sections=8, max_iteration_steps=30)


def ppo_agent(env):
    obs, act = int(env.get_observation_dim()), int(env.get_action_dim())
    config = PPOConfig(name="ppo", obs_dim=obs, action_dim=act, algorithm=RLAlgorithm.PPO, device="cpu", batch_size=8)
    return PPOAgent(config, policy_network=SimpleQNetwork(obs, act, 1, 16), value_network=ValueNetwork(obs, 16))


def dqn_agent(env):
    obs, act = int(env.get_observation_dim()), int(env.get_action_dim())
    config = DQNConfig(name="dqn", obs_dim=obs, action_dim=act, algorithm=RLAlgorithm.DQN, device="cpu", batch_size=8)
    return DQNAgent(config, q_network=SimpleQNetwork(obs, act, 1, 16))


def recurrent_agent(env):
    obs, act = int(env.get_observation_dim()), int(env.get_action_dim())
    config = PPORNNConfig(name="gru", obs_dim=obs, action_dim=act, algorithm=RLAlgorithm.PPO, device="cpu", seq_len=8)
    return RecurrentPPOAgent(config, RecurrentGRUPolicy(obs, 16, act), RecurrentGRUValue(obs, 16))


def scalars(log_dir, tag):
    acc = EventAccumulator(str(log_dir))
    acc.Reload()
    return [e.value for e in acc.Scalars(tag)] if tag in acc.Tags()["scalars"] else None


@pytest.mark.parametrize("make_agent", [ppo_agent, dqn_agent, recurrent_agent])
def test_every_agent_trains_with_the_same_loop_and_tags(tmp_path, make_agent):
    torch.manual_seed(0)
    env = make_env()
    agent = make_agent(env)
    trainer = SingleTrainer(str(tmp_path / "runs"), str(tmp_path / "ckpt"), TrainerConfig(num_episodes=1001, useTQDM=False, eps_step=2))
    trainer.add_agents(agent).add_environments(env)
    trainer.train(num_episodes=EPISODES)

    log_dir = tmp_path / "runs" / agent.config.name
    for tag in ("Reward/Total", "Success/Episode", "Steps/Episode", "Steps/Ratio/Episode", "Done/Hit Obstacle", "Distances/Change"):
        values = scalars(log_dir, tag)
        assert values is not None and len(values) == EPISODES, tag
    assert (tmp_path / "ckpt" / agent.config.name / "final.pth").exists()


def test_multi_environment_trainer_alternates_maps(tmp_path):
    envs = [make_env("a"), make_env("b")]
    agent = ppo_agent(envs[0])
    trainer = MultiEnvironmentTrainer(str(tmp_path / "runs"), str(tmp_path / "ckpt"),
                                      TrainerConfig(num_episodes=1001, useTQDM=False, onePerEnv=False))
    trainer.add_agents(agent).add_environments(*envs)
    trainer.train(num_episodes=EPISODES)
    log_dir = tmp_path / "runs" / agent.config.name
    assert len(scalars(log_dir, "Success/Map/a")) == EPISODES // 2
    assert len(scalars(log_dir, "Success/Map/b")) == EPISODES // 2


def test_ppo_hooks_fill_the_buffer_with_gae_at_episode_end():
    env = make_env()
    agent = ppo_agent(env)
    state, _ = env.reset(seed=0)
    agent.on_episode_start()
    steps, done = 0, False
    while not done:
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        agent.observe(state, action, reward, next_state, terminated, truncated)
        state, done, steps = next_state, terminated or truncated, steps + 1
    assert len(agent.buffer) == 0  # nothing is stored before the episode ends
    agent.on_episode_end(state)
    assert len(agent.buffer) == steps
    assert agent._episode["states"] == []


def test_ppo_value_loss_compares_each_return_with_its_own_value():
    env = make_env()
    agent = ppo_agent(env)
    states = np.random.rand(8, int(env.get_observation_dim())).astype(np.float32)
    returns = np.arange(8, dtype=np.float32)
    batch = (states, np.zeros(8, dtype=np.int64), None, None, None, np.zeros(8, dtype=np.float32),
             np.zeros(8, dtype=np.float32), returns)
    agent._compute_loss(batch)
    with torch.no_grad():
        values = agent.value_net(torch.as_tensor(states)).squeeze(-1).numpy()
    assert agent.last_value_loss == pytest.approx(float(np.mean((returns - values) ** 2)), rel=1e-5)


def test_greedy_act_does_not_record():
    env = make_env()
    agent = ppo_agent(env)
    state, _ = env.reset(seed=0)
    agent.on_episode_start()
    agent.act(state, training=False)
    assert not hasattr(agent, "_pending")


# ------------------------------------------------------------
# Recurrent PPO: gradient clipping and padding mask
# ------------------------------------------------------------
def fill_recurrent_episode(agent, env, steps):
    state, _ = env.reset(seed=0)
    agent.on_episode_start()
    for _ in range(steps):
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        agent.observe(state, action, reward, next_state, terminated, truncated)
        state = next_state
    agent.store(*(agent._episode[k] for k in ("states", "actions", "rewards", "dones", "log_probs", "values")))


def test_recurrent_agent_clips_gradients(monkeypatch):
    env = Grid_Robot_Sections_Env(map=GridMap("m", (25, 25)), robot=GridRobotNotTurning(), reward_fn=ProgressReward(),
                                  lidar_config=LIDAR, num_sections=8, max_iteration_steps=1000)
    agent = recurrent_agent(env)
    calls = []
    monkeypatch.setattr(agent, "_clip_grad_norm", lambda: calls.append(1))
    fill_recurrent_episode(agent, env, 12)
    agent.train_step()
    assert calls == [1]


def test_recurrent_value_loss_ignores_padding(monkeypatch):
    import copy
    env = Grid_Robot_Sections_Env(map=GridMap("m", (25, 25)), robot=GridRobotNotTurning(), reward_fn=ProgressReward(),
                                  lidar_config=LIDAR, num_sections=8, max_iteration_steps=1000)
    torch.manual_seed(0)
    agent = recurrent_agent(env)  # seq_len 8
    fill_recurrent_episode(agent, env, 11)  # chunks of 8 and 3: the second is padded with 5 steps
    buffer = copy.deepcopy(agent.buffer)
    monkeypatch.setattr(agent.optimizer, "step", lambda: None)  # keep the weights to recompute the loss
    agent.train_step()

    seq_states, _, _, _, seq_ret = buffer.build_sequences(agent._value_fn)
    errors = []
    with torch.no_grad():
        for states, returns in zip(seq_states, seq_ret):
            h = torch.zeros(1, 1, agent.value_net.hidden_dim)
            values, _ = agent.value_net(states.unsqueeze(0), h)
            errors.append((values.squeeze(0) - returns).pow(2))
    expected = torch.cat(errors).mean().item()  # mean over the 11 real steps only
    assert agent.last_value_loss == pytest.approx(expected, rel=1e-5)
