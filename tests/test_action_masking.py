"""Invalid-action masking: robots report valid moves, the env publishes them, PPO acts and learns with them."""
import numpy as np
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from tfm_control_ucm.core.agents.agent import DQNAgent, PPOAgent
from tfm_control_ucm.core.agents.networks import SimpleQNetwork, ValueNetwork
from tfm_control_ucm.core.agents.utils import DQNConfig, PPOConfig, RLAlgorithm
from tfm_control_ucm.core.grid_env.environment import Grid_Robot_Sections_Env
from tfm_control_ucm.core.grid_env.map import GridMap, GridObstacle
from tfm_control_ucm.core.grid_env.robot import GridRobot, GridRobotNotTurning
from tfm_control_ucm.core.grid_env.utils import Orientation
from tfm_control_ucm.core.rewards import ProgressReward
from tfm_control_ucm.core.training.TrainingManager import SingleTrainer, TrainerConfig

LIDAR = {"num_rays": 24, "max_range": 10, "fov": float(2 * np.pi), "noise_std": 0.0, "with_cache": True, "device": "cpu"}


# ------------------------------------------------------------
# Robots
# ------------------------------------------------------------
def test_not_turning_robot_masks_walls_and_obstacles():
    mapa = GridMap("m", (10, 10), grid_obstacles=[GridObstacle((5, 4), (1, 1))])  # obstacle right of (4, 4)
    robot = GridRobotNotTurning(position=(4, 4))
    assert robot.valid_actions(mapa).tolist() == [True, False, True, True]  # up, right, down, left
    robot = GridRobotNotTurning(position=(0, 0))
    assert robot.valid_actions(mapa).tolist() == [False, True, True, False]


def test_grid_robot_can_always_turn():
    mapa = GridMap("m", (10, 10))
    robot = GridRobot(position=(5, 0), orientation=Orientation.north)  # facing the top wall
    assert robot.valid_actions(mapa).tolist() == [False, True, True]


def test_env_publishes_the_mask():
    env = Grid_Robot_Sections_Env(map=GridMap("m", (25, 25)), robot=GridRobotNotTurning(), reward_fn=ProgressReward(),
                                  lidar_config=LIDAR, num_sections=8)
    _, info = env.reset(seed=0)
    assert info["action_mask"].tolist() == env.robot.valid_actions(env.map).tolist()
    env.robot.position = (0, 0)
    *_, info = env.step(1)  # move right to (1, 0)
    assert info["action_mask"].tolist() == [False, True, True, True]


# ------------------------------------------------------------
# PPO with masks
# ------------------------------------------------------------
def ppo_agent(obs_dim=18, action_dim=4):
    config = PPOConfig(name="ppo", obs_dim=obs_dim, action_dim=action_dim, algorithm=RLAlgorithm.PPO, device="cpu", batch_size=4)
    return PPOAgent(config, policy_network=SimpleQNetwork(obs_dim, action_dim, 1, 16), value_network=ValueNetwork(obs_dim, 16))


def test_ppo_never_picks_a_masked_action():
    torch.manual_seed(0)
    agent = ppo_agent()
    agent.on_episode_start()
    mask = np.array([False, True, False, False])
    state = np.random.rand(18).astype(np.float32)
    assert {agent.act(state, training=True, action_mask=mask) for _ in range(50)} == {1}
    assert agent.act(state, training=False, action_mask=mask) == 1
    assert agent.last_log_prob.item() == pytest.approx(0.0, abs=1e-6)  # the only valid action has probability 1


def test_ppo_stores_masks_and_uses_them_in_the_loss():
    torch.manual_seed(0)
    agent = ppo_agent()
    agent.on_episode_start()
    mask = np.array([True, True, False, False])
    for _ in range(6):
        state = np.random.rand(18).astype(np.float32)
        action = agent.act(state, training=True, action_mask=mask)
        agent.observe(state, action, 0.0, state, False, False)
    agent.on_episode_end(state)
    batch = agent.buffer.sample(4)
    assert len(batch) == 9 and batch[8].shape == (4, 4) and not batch[8][:, 2:].any()
    agent._compute_loss(batch)
    # Entropy of a 2-action distribution is at most ln 2, below the ln 4 of the unmasked policy
    assert agent.last_entropy <= np.log(2) + 1e-6


def test_agents_without_masking_support_refuse_a_mask():
    config = DQNConfig(name="dqn", obs_dim=18, action_dim=4, algorithm=RLAlgorithm.DQN, device="cpu")
    agent = DQNAgent(config, q_network=SimpleQNetwork(18, 4, 1, 8))
    with pytest.raises(NotImplementedError):
        agent.act(np.zeros(18, dtype=np.float32), action_mask=np.ones(4, dtype=bool))


def test_trainer_with_masking_has_no_collisions(tmp_path):
    torch.manual_seed(0)
    walls = GridMap("walls", (25, 25), grid_obstacles=[GridObstacle((5, 5), (15, 1)), GridObstacle((5, 15), (15, 1))])
    env = Grid_Robot_Sections_Env(map=walls, robot=GridRobotNotTurning(), reward_fn=ProgressReward(),
                                  lidar_config=LIDAR, num_sections=8, max_iteration_steps=60)
    agent = ppo_agent(int(env.get_observation_dim()), int(env.get_action_dim()))
    trainer = SingleTrainer(str(tmp_path / "runs"), str(tmp_path / "ckpt"),
                            TrainerConfig(num_episodes=1001, useTQDM=False, log_steps=False, action_masking=True))
    trainer.add_agents(agent).add_environments(env)
    trainer.train(num_episodes=15)
    acc = EventAccumulator(str(tmp_path / "runs" / "ppo"))
    acc.Reload()
    assert sum(e.value for e in acc.Scalars("Done/Hit Obstacle")) == 0
