"""MaskedDQNAgent: Double DQN with invalid-action masking (the DQN counterpart of test 31)."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from tfm_control_ucm.core.agents.masked_dqn import MaskedDQNAgent, MaskedDQNConfig, MaskedReplayBuffer
from tfm_control_ucm.core.agents.networks import SimpleQNetwork
from tfm_control_ucm.core.agents.utils import RLAlgorithm
from tfm_control_ucm.core.grid_env.environment import Grid_Robot_Sections_Env
from tfm_control_ucm.core.grid_env.map import GridMap
from tfm_control_ucm.core.grid_env.robot import GridRobotNotTurning
from tfm_control_ucm.core.rewards import ProgressReward
from tfm_control_ucm.core.training.TrainingManager import SingleTrainer, TrainerConfig

OBS, ACT = 6, 4


def make_agent(**cfg):
    torch.manual_seed(0)
    np.random.seed(0)
    config = MaskedDQNConfig(name="dqn", obs_dim=OBS, action_dim=ACT, algorithm=RLAlgorithm.DQN, device="cpu",
                             batch_size=4, learning_starts=4, **cfg)
    return MaskedDQNAgent(config, SimpleQNetwork(OBS, ACT, 1, 16))


class FixedQ(nn.Module):
    """Q-network that returns the same values for every state."""
    def __init__(self, values):
        super().__init__()
        self.values = nn.Parameter(torch.tensor(values, dtype=torch.float32))

    def forward(self, x):
        return self.values.expand(x.shape[0], -1)


# ------------------------------------------------------------
# Buffer
# ------------------------------------------------------------
def test_buffer_wraps_around_and_samples_arrays():
    buffer = MaskedReplayBuffer(3, OBS, ACT)
    for i in range(5):
        buffer.add(np.full(OBS, i), i % ACT, float(i), np.full(OBS, i + 1), i == 4, np.array([True, False, True, False]))
    assert len(buffer) == 3
    states, actions, rewards, next_states, dones, masks = buffer.sample(10)
    assert states.shape == (10, OBS) and masks.shape == (10, ACT) and masks.dtype == bool
    assert set(rewards.tolist()) <= {2.0, 3.0, 4.0}  # the oldest two were overwritten


# ------------------------------------------------------------
# Acting
# ------------------------------------------------------------
def test_greedy_action_is_the_best_valid_one():
    agent = make_agent()
    agent.policy_net = FixedQ([5.0, 1.0, 3.0, 0.0])
    assert agent.act(np.zeros(OBS), training=False) == 0
    assert agent.act(np.zeros(OBS), training=False, action_mask=np.array([False, True, True, True])) == 2


def test_exploration_only_picks_valid_actions():
    agent = make_agent(eps_start=1.0, eps_end=1.0)
    mask = np.array([False, True, False, True])
    picks = {agent.act(np.zeros(OBS), training=True, action_mask=mask) for _ in range(200)}
    assert picks == {1, 3}


def test_epsilon_decays_linearly_and_evaluation_does_not_move_it():
    agent = make_agent(eps_start=1.0, eps_end=0.0, eps_decay=10)
    for _ in range(5):
        agent.act(np.zeros(OBS), training=True)
    assert agent.epsilon == pytest.approx(0.5)
    for _ in range(5):
        agent.act(np.zeros(OBS), training=False)
    assert agent.epsilon == pytest.approx(0.5) and agent.global_step == 5
    for _ in range(20):
        agent.act(np.zeros(OBS), training=True)
    assert agent.epsilon == pytest.approx(0.0)


# ------------------------------------------------------------
# Transitions and next-state masks
# ------------------------------------------------------------
def test_next_state_mask_comes_from_the_next_act():
    agent = make_agent()
    s0, s1 = np.zeros(OBS), np.ones(OBS)
    agent.on_episode_start()
    agent.act(s0, training=True, action_mask=np.array([True, True, True, True]))
    agent.observe(s0, 1, 0.5, s1, False, False)
    assert len(agent.buffer) == 0  # waiting for the mask of s1
    agent.act(s1, training=True, action_mask=np.array([False, True, False, False]))
    assert len(agent.buffer) == 1 and agent.buffer.next_masks[0].tolist() == [False, True, False, False]


def test_terminal_transitions_are_stored_at_once():
    agent = make_agent()
    agent.observe(np.zeros(OBS), 2, 20.0, np.ones(OBS), True, False)
    agent.observe(np.zeros(OBS), 3, -5.0, np.ones(OBS), False, True)  # collision / time limit
    assert len(agent.buffer) == 2 and agent.buffer.dones[:2].tolist() == [1.0, 1.0]


def test_target_maximises_over_the_valid_next_actions():
    agent = make_agent(double_dqn=False, gamma=0.5)
    agent.policy_net = FixedQ([0.0, 0.0, 0.0, 0.0])
    agent.target_net = FixedQ([100.0, 2.0, 50.0, 4.0])
    batch = (np.zeros((1, OBS), np.float32), np.array([0]), np.array([1.0], np.float32), np.zeros((1, OBS), np.float32),
             np.array([0.0], np.float32), np.array([[False, True, False, True]]))
    loss = agent._compute_loss(batch)
    # target = 1 + 0.5 * max(2, 4) = 3; Q = 0 -> Huber(3) = 3 - 0.5
    assert loss.item() == pytest.approx(2.5)


def test_double_dqn_uses_the_online_choice_among_valid_actions():
    agent = make_agent(double_dqn=True, gamma=1.0)
    agent.policy_net = FixedQ([9.0, 1.0, 0.0, 3.0])  # best valid (1 or 3) for the online net: 3
    agent.target_net = FixedQ([100.0, 7.0, 0.0, 2.0])
    batch = (np.zeros((1, OBS), np.float32), np.array([2]), np.array([0.0], np.float32), np.zeros((1, OBS), np.float32),
             np.array([0.0], np.float32), np.array([[False, True, False, True]]))
    # target = 0 + Q_target(a=3) = 2; Q(s, 2) = 0 -> Huber(2) = 1.5
    assert agent._compute_loss(batch).item() == pytest.approx(1.5)


def test_done_transitions_are_not_bootstrapped():
    agent = make_agent(gamma=1.0)
    agent.policy_net = FixedQ([0.0, 0.0, 0.0, 0.0])
    agent.target_net = FixedQ([100.0, 100.0, 100.0, 100.0])
    batch = (np.zeros((1, OBS), np.float32), np.array([0]), np.array([0.5], np.float32), np.zeros((1, OBS), np.float32),
             np.array([1.0], np.float32), np.ones((1, ACT), bool))
    assert agent._compute_loss(batch).item() == pytest.approx(0.5 * 0.5 ** 2)  # Huber(0.5) = 0.125


# ------------------------------------------------------------
# Updates
# ------------------------------------------------------------
def test_no_update_before_learning_starts():
    agent = make_agent()
    agent.config.learning_starts = 10
    for _ in range(9):
        agent.observe(np.zeros(OBS), 0, 0.0, np.zeros(OBS), True, False)
    assert agent.train_step() is None
    agent.observe(np.zeros(OBS), 0, 0.0, np.zeros(OBS), True, False)
    assert agent.train_step() is not None


def test_target_network_is_copied_every_target_update_updates():
    agent = make_agent(target_update=3, lr=0.1)
    for i in range(8):
        agent.observe(np.random.rand(OBS), i % ACT, float(i), np.random.rand(OBS), True, False)
    same = lambda: all(torch.equal(a, b) for a, b in zip(agent.policy_net.parameters(), agent.target_net.parameters()))
    agent.train_step(); agent.train_step()
    assert not same()
    agent.train_step()
    assert same() and agent.updates == 3


def test_save_and_load_restore_networks_and_epsilon(tmp_path):
    agent = make_agent(eps_decay=10, eps_end=0.1)
    for _ in range(20):
        agent.act(np.zeros(OBS), training=True)
    agent.save(str(tmp_path / "dqn.pth"))
    loaded = make_agent(eps_decay=10, eps_end=0.1)
    loaded.load(str(tmp_path / "dqn.pth"))
    assert loaded.epsilon == pytest.approx(0.1) and loaded.global_step == 20
    for a, b in zip(loaded.policy_net.parameters(), loaded.target_net.parameters()):
        assert torch.equal(a, b)


# ------------------------------------------------------------
# With the trainer and the environment
# ------------------------------------------------------------
def test_trains_in_the_shared_loop_with_action_masking(tmp_path):
    env = Grid_Robot_Sections_Env(map=GridMap("m", (15, 15)), robot=GridRobotNotTurning(), reward_fn=ProgressReward(),
                                  lidar_config={"num_rays": 24, "max_range": 10, "fov": float(2 * np.pi), "noise_std": 0.0,
                                                "with_cache": True, "device": "cpu"},
                                  num_sections=8, max_iteration_steps=40)
    obs, act = int(env.get_observation_dim()), int(env.get_action_dim())
    config = MaskedDQNConfig(name="dqn", obs_dim=obs, action_dim=act, algorithm=RLAlgorithm.DQN, device="cpu",
                             batch_size=8, learning_starts=16)
    agent = MaskedDQNAgent(config, SimpleQNetwork(obs, act, 1, 16))
    trainer = SingleTrainer(str(tmp_path / "runs"), str(tmp_path / "ckpt"),
                            TrainerConfig(num_episodes=1001, useTQDM=False, eps_step=2, action_masking=True))
    trainer.add_agents(agent).add_environments(env)
    trainer.train(num_episodes=4)
    assert agent.updates > 0 and len(agent.buffer) > 0
    assert (tmp_path / "ckpt" / "dqn" / "final.pth").exists()


def test_eval_agent_loads_masked_dqn_checkpoints(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("eval_agent", Path(__file__).resolve().parent.parent / "scripts" / "eval_agent.py")
    eval_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_agent)

    agent = make_agent()
    agent.save(str(tmp_path / "final.pth"))
    loaded, obs_dim, action_dim = eval_agent.load_agent_from_checkpoint(str(tmp_path / "final.pth"), torch.device("cpu"))
    assert isinstance(loaded, MaskedDQNAgent) and (obs_dim, action_dim) == (OBS, ACT)
    assert loaded.act(np.zeros(OBS, np.float32), training=False, action_mask=np.array([False, False, True, False])) == 2


def test_masked_agent_never_moves_into_a_wall():
    # Big enough for the env's minimum start-goal separation (10 cells for the Sections env)
    env = Grid_Robot_Sections_Env(map=GridMap("m", (25, 25)), robot=GridRobotNotTurning(), reward_fn=ProgressReward(),
                                  lidar_config={"num_rays": 24, "max_range": 10, "fov": float(2 * np.pi), "noise_std": 0.0,
                                                "with_cache": True, "device": "cpu"},
                                  num_sections=8, max_iteration_steps=200)
    obs, act = int(env.get_observation_dim()), int(env.get_action_dim())
    agent = MaskedDQNAgent(MaskedDQNConfig(name="dqn", obs_dim=obs, action_dim=act, algorithm=RLAlgorithm.DQN,
                                           device="cpu", eps_start=1.0, eps_end=1.0), SimpleQNetwork(obs, act, 1, 16))
    state, info = env.reset(seed=0)
    for _ in range(200):
        state, _, terminated, truncated, info = env.step(agent.act(state, training=True, action_mask=info["action_mask"]))
        assert not info["hit_obstacle"]
        if terminated or truncated:
            state, info = env.reset()
