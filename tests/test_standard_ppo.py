"""StandardPPOAgent (textbook PPO) and the trainers that relay the step info."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from tfm_control_ucm.core.agents.networks import SimpleQNetwork, ValueNetwork
from tfm_control_ucm.core.agents.standard_ppo import StandardPPOAgent, StandardPPOConfig, gae
from tfm_control_ucm.core.agents.utils import RLAlgorithm
from tfm_control_ucm.core.grid_env.environment import Grid_Robot_Sections_Env
from tfm_control_ucm.core.grid_env.map import GridMap
from tfm_control_ucm.core.grid_env.robot import GridRobotNotTurning
from tfm_control_ucm.core.rewards import ProgressReward
from tfm_control_ucm.core.training.step_info_trainer import StepInfoRelay, StepInfoSingleTrainer
from tfm_control_ucm.core.training.TrainingManager import SingleTrainer, TrainerConfig

OBS, ACT = 6, 4
LIDAR = {"num_rays": 24, "max_range": 10, "fov": float(2 * np.pi), "noise_std": 0.0, "with_cache": True, "device": "cpu"}


def make_agent(**cfg):
    torch.manual_seed(0)
    np.random.seed(0)
    params = {"rollout_steps": 8, "batch_size": 4, "n_epochs": 2, **cfg}
    config = StandardPPOConfig(name="ppo", obs_dim=OBS, action_dim=ACT, algorithm=RLAlgorithm.PPO, device="cpu", **params)
    return StandardPPOAgent(config, SimpleQNetwork(OBS, ACT, 1, 16), ValueNetwork(OBS, 16))


class ConstantValue(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(value)))

    def forward(self, x):
        return self.value.expand(x.shape[0], 1)


def make_env(size=(25, 25), max_steps=30):
    return Grid_Robot_Sections_Env(map=GridMap("m", size), robot=GridRobotNotTurning(), reward_fn=ProgressReward(),
                                   lidar_config=LIDAR, num_sections=8, max_iteration_steps=max_steps)


# ------------------------------------------------------------
# GAE
# ------------------------------------------------------------
def test_gae_terminal_step_does_not_bootstrap():
    adv, ret = gae(np.array([1.0]), np.array([0.5]), np.array([10.0]), np.array([1.0]), np.array([1.0]), gamma=0.9, lam=0.95)
    assert adv[0] == pytest.approx(1.0 - 0.5) and ret[0] == pytest.approx(1.0)


def test_gae_time_limit_bootstraps_but_does_not_flow_into_the_next_episode():
    # step 0 ends an episode by time limit (bootstrap V(s')=10), step 1 starts the next one
    rewards, values = np.array([0.0, 1.0]), np.array([0.0, 0.0])
    next_values, terminals, ends = np.array([10.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])
    adv, _ = gae(rewards, values, next_values, terminals, ends, gamma=0.5, lam=1.0)
    assert adv.tolist() == pytest.approx([0.5 * 10.0, 1.0])  # step 0 does not receive step 1's advantage


def test_gae_matches_the_recursion_inside_an_episode():
    rewards, values = np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.4, 0.3])
    next_values = np.array([0.4, 0.3, 7.0])  # the rollout stops mid-episode: last step bootstraps V(s')=7
    terminals, ends = np.zeros(3), np.zeros(3)
    g, l = 0.9, 0.8
    d = rewards + g * next_values - values
    expected = [d[0] + g * l * (d[1] + g * l * d[2]), d[1] + g * l * d[2], d[2]]
    adv, ret = gae(rewards, values, next_values, terminals, ends, gamma=g, lam=l)
    assert adv.tolist() == pytest.approx(expected) and ret.tolist() == pytest.approx((np.array(expected) + values).tolist())


# ------------------------------------------------------------
# Collisions vs time limit
# ------------------------------------------------------------
def test_collision_is_terminal_when_the_step_info_is_relayed():
    agent = make_agent()
    for info, expected in [({"hit_obstacle": True, "max_steps_reached": False}, 1.0),
                           ({"hit_obstacle": False, "max_steps_reached": True}, 0.0)]:
        agent.act(np.zeros(OBS), training=True)
        agent.on_step_info(info)
        agent.observe(np.zeros(OBS), 0, -5.0, np.ones(OBS), False, True)
        assert agent.rollout.terminals[agent.rollout.n - 1] == expected
        assert agent.rollout.episode_ends[agent.rollout.n - 1] == 1.0


@pytest.mark.parametrize("truncation_is_terminal,expected", [(False, 0.0), (True, 1.0)])
def test_truncation_without_step_info_follows_the_config(truncation_is_terminal, expected):
    agent = make_agent(truncation_is_terminal=truncation_is_terminal)
    agent.act(np.zeros(OBS), training=True)
    agent.observe(np.zeros(OBS), 0, 0.0, np.ones(OBS), False, True)
    assert agent.rollout.terminals[0] == expected


# ------------------------------------------------------------
# Rollout and update
# ------------------------------------------------------------
def test_update_runs_when_the_rollout_is_full_and_clears_it():
    agent = make_agent(rollout_steps=8, batch_size=4, n_epochs=3)
    for t in range(7):
        agent.act(np.random.rand(OBS), training=True)
        agent.observe(np.random.rand(OBS), t % ACT, 1.0, np.random.rand(OBS), False, False)
    assert agent.updates == 0 and agent.rollout.n == 7
    agent.act(np.random.rand(OBS), training=True)
    agent.observe(np.random.rand(OBS), 0, 1.0, np.random.rand(OBS), False, False)
    assert agent.updates == 1 and agent.rollout.n == 0
    assert agent.gradient_steps == 3 * 2  # 3 epochs x 2 minibatches of 4
    assert agent.on_episode_end(np.zeros(OBS)) is not None  # the loss of that update


def test_target_kl_stops_the_epochs_early():
    agent = make_agent(rollout_steps=8, batch_size=4, n_epochs=50, target_kl=1e-12, lr=0.5)
    for t in range(8):
        agent.act(np.random.rand(OBS), training=True)
        agent.observe(np.random.rand(OBS), t % ACT, float(t), np.random.rand(OBS), False, False)
    assert 0 < agent.gradient_steps < 50 * 2


def test_masked_policy_never_samples_invalid_actions_and_greedy_picks_a_valid_one():
    agent = make_agent()
    mask = np.array([False, True, False, True])
    picks = {agent.act(np.random.rand(OBS), training=True, action_mask=mask) for _ in range(200)}
    assert picks <= {1, 3}
    assert agent.act(np.random.rand(OBS), training=False, action_mask=mask) in (1, 3)


def test_update_uses_the_masked_log_probabilities():
    agent = make_agent(rollout_steps=4, batch_size=4, n_epochs=1, lr=0.0)
    mask = np.array([True, True, False, False])
    for _ in range(4):
        action = agent.act(np.zeros(OBS), training=True, action_mask=mask)
        agent.observe(np.zeros(OBS), action, 0.0, np.zeros(OBS), False, False)  # the action act() chose, as the trainer does
    # lr 0: the policy did not change, so with consistent masks the ratio is exactly 1 (KL 0, no clipping)
    assert agent.metrics["Agent/PPO/ApproxKL"] == pytest.approx(0.0, abs=1e-6)
    assert agent.metrics["Agent/PPO/ClipFraction"] == 0.0


def test_evaluation_records_nothing():
    agent = make_agent()
    agent.act(np.zeros(OBS), training=False)
    agent.observe(np.zeros(OBS), 0, 1.0, np.ones(OBS), False, False)
    assert agent.rollout.n == 0 and agent.global_step == 0


def test_value_learns_a_constant_return():
    torch.manual_seed(0)
    config = StandardPPOConfig(name="ppo", obs_dim=OBS, action_dim=ACT, algorithm=RLAlgorithm.PPO, device="cpu",
                               rollout_steps=16, batch_size=16, n_epochs=300, lr=0.05, entropy_coeff=0.0)
    agent = StandardPPOAgent(config, SimpleQNetwork(OBS, ACT, 1, 16), ConstantValue(0.0))
    for _ in range(16):  # one-step episodes with reward 2: V should go to 2
        agent.act(np.zeros(OBS), training=True)
        agent.observe(np.zeros(OBS), 0, 2.0, np.zeros(OBS), True, False)
    assert agent.value_net.value.item() == pytest.approx(2.0, abs=0.1)


# ------------------------------------------------------------
# Variants from the literature
# ------------------------------------------------------------
def _cfg(**kw):
    return StandardPPOConfig(name="p", obs_dim=OBS, action_dim=ACT, algorithm=RLAlgorithm.PPO, device="cpu", clip_eps=0.2, **kw)


def test_clip_surrogate_is_the_ppo_objective():
    from tfm_control_ucm.core.agents.standard_ppo import surrogate
    ratio, adv = torch.tensor([0.5, 1.0, 1.5, 1.5]), torch.tensor([1.0, 1.0, 1.0, -1.0])
    # min(rA, clip(r)A): 0.5, 1.0, min(1.5, 1.2) = 1.2, min(-1.5, -1.2) = -1.5
    assert surrogate(ratio, adv, _cfg()).tolist() == pytest.approx([0.5, 1.0, 1.2, -1.5])


def test_dual_clip_bounds_negative_advantages_from_below():
    from tfm_control_ucm.core.agents.standard_ppo import surrogate
    ratio, adv = torch.tensor([5.0, 5.0, 1.5]), torch.tensor([-1.0, 1.0, -1.0])
    # A < 0, r = 5: clip gives min(-5, -1.2) = -5, dual clip raises it to c·A = -3; A > 0 is unchanged (1.2)
    assert surrogate(ratio, adv, _cfg(clip_mode="dual_clip", dual_clip_c=3.0)).tolist() == pytest.approx([-3.0, 1.2, -1.5])


def test_rollback_reverses_the_slope_outside_the_clip_range():
    from tfm_control_ucm.core.agents.standard_ppo import surrogate
    cfg = _cfg(clip_mode="rollback", rollback_alpha=0.3)
    ratio = torch.tensor([1.0, 1.2, 1.5, 0.5], requires_grad=True)
    adv = torch.tensor([1.0, 1.0, 1.0, -1.0])
    out = surrogate(ratio, adv, cfg)
    # F(1.5) = -0.3·1.5 + 1.3·1.2 = 1.11 < 1.5 -> 1.11; F(0.5) = -0.3·0.5 + 1.3·0.8 = 0.89, min(-0.5, -0.89) = -0.89
    assert out.tolist() == pytest.approx([1.0, 1.2, 1.11, -0.89])
    out.sum().backward()
    assert ratio.grad[2].item() == pytest.approx(-0.3)  # pushes the ratio back (clip would give 0)


def test_adaptive_kl_coefficient_follows_schulman_rule():
    agent = make_agent(rollout_steps=8, batch_size=8, n_epochs=1, lr=0.0, kl_coef=1.0, kl_target=0.01)
    for _ in range(8):
        state = np.random.rand(OBS)
        action = agent.act(state, training=True)
        agent.observe(state, action, 1.0, np.random.rand(OBS), False, False)
    assert agent.kl_coef == pytest.approx(0.5)  # lr 0: KL = 0 < target / 1.5 -> halved


def test_learning_rate_decays_linearly_over_episodes():
    agent = make_agent(lr=1e-3, anneal_lr_episodes=10)
    for _ in range(5):
        agent.on_episode_end(np.zeros(OBS))
    assert agent.optimizer.param_groups[0]["lr"] == pytest.approx(5e-4)
    for _ in range(10):
        agent.on_episode_end(np.zeros(OBS))
    assert agent.optimizer.param_groups[0]["lr"] == 0.0


def test_unknown_clip_mode_is_rejected():
    with pytest.raises(ValueError):
        _cfg(clip_mode="spline")


# ------------------------------------------------------------
# Trainers
# ------------------------------------------------------------
def test_relay_hands_the_step_info_to_the_agent():
    received = []
    class Spy:
        def on_step_info(self, info): received.append(info)
    env = make_env()
    relay = StepInfoRelay(env, Spy())
    relay.reset(seed=0)
    relay.step(0)
    assert received and "hit_obstacle" in received[0]
    assert relay.get_observation_dim() == env.get_observation_dim()  # other attributes pass through


@pytest.mark.parametrize("trainer_cls", [StepInfoSingleTrainer, SingleTrainer])
def test_trains_in_the_shared_loop(tmp_path, trainer_cls):
    env = make_env()
    obs, act = int(env.get_observation_dim()), int(env.get_action_dim())
    config = StandardPPOConfig(name="ppo", obs_dim=obs, action_dim=act, algorithm=RLAlgorithm.PPO, device="cpu",
                               rollout_steps=32, batch_size=8, n_epochs=2)
    agent = StandardPPOAgent(config, SimpleQNetwork(obs, act, 1, 16), ValueNetwork(obs, 16))
    trainer = trainer_cls(str(tmp_path / "runs"), str(tmp_path / "ckpt"),
                          TrainerConfig(num_episodes=1001, useTQDM=False, action_masking=True, eps_step=4))
    trainer.add_agents(agent).add_environments(env)
    trainer.train(num_episodes=5)
    assert agent.updates >= 1 and (tmp_path / "ckpt" / "ppo" / "final.pth").exists()


def test_eval_agent_loads_standard_ppo_checkpoints(tmp_path):
    import importlib.util
    from pathlib import Path
    from tfm_control_ucm.core.agents.agent import PPOAgent
    spec = importlib.util.spec_from_file_location("eval_agent", Path(__file__).resolve().parent.parent / "scripts" / "eval_agent.py")
    eval_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_agent)

    agent = make_agent()
    agent.save(str(tmp_path / "final.pth"))
    loaded, obs_dim, action_dim = eval_agent.load_agent_from_checkpoint(str(tmp_path / "final.pth"), torch.device("cpu"))
    assert isinstance(loaded, PPOAgent) and (obs_dim, action_dim) == (OBS, ACT)
    state = np.random.rand(OBS).astype(np.float32)
    mask = np.array([False, True, True, False])
    with torch.no_grad():
        logits = agent.policy_net(torch.as_tensor(state)).numpy()
    expected = int(np.argmax(np.where(mask, logits, -np.inf)))
    assert loaded.act(state, training=False, action_mask=mask) == expected  # same greedy action
