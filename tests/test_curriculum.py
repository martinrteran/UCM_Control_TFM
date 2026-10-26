import json

import numpy as np
import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from tfm_control_ucm.core.agents.agent import PPOAgent
from tfm_control_ucm.core.agents.networks import SimpleQNetwork, ValueNetwork
from tfm_control_ucm.core.agents.utils import PPOConfig, RLAlgorithm
from tfm_control_ucm.core.curriculum import CurriculumStage, SuccessCurriculum
from tfm_control_ucm.core.grid_env.environment import Grid_Robot_Sections_Env_MultiMap
from tfm_control_ucm.core.grid_env.map import GridMap
from tfm_control_ucm.core.grid_env.robot import GridRobotNotTurning
from tfm_control_ucm.core.training.TrainingManager import SingleTrainer, TrainerConfig

LIDAR = {"num_rays": 24, "max_range": 10, "fov": float(2 * np.pi), "noise_std": 0.0, "with_cache": True, "device": "cpu"}


def two_stage(window=4, threshold=0.75, min_episodes=0):
    return SuccessCurriculum([CurriculumStage(maps=["a"], max_goal_distance=12.0, success_threshold=threshold, min_episodes=min_episodes),
                              CurriculumStage(maps=["a", "b"])], window=window)


# ------------------------------------------------------------
# Stage logic
# ------------------------------------------------------------
@pytest.mark.parametrize("kwargs", [dict(maps=[]), dict(maps=["a"], max_goal_distance=0),
                                    dict(maps=["a"], success_threshold=0.0), dict(maps=["a"], min_episodes=-1)])
def test_invalid_stage(kwargs):
    with pytest.raises(ValueError):
        CurriculumStage(**kwargs)


def test_advances_only_with_a_full_window_above_threshold():
    curriculum = two_stage(window=4, threshold=0.75)
    assert [curriculum.record(s) for s in (True, True, True)] == [False, False, False]  # window not full
    assert curriculum.record(False) is True  # 3/4 = 0.75
    assert curriculum.stage_idx == 1 and curriculum.stage_started_at == [0, 4]
    assert curriculum.success_rate() == 0.0  # window restarts in the new stage


def test_does_not_advance_below_threshold_or_before_min_episodes():
    curriculum = two_stage(window=2, threshold=1.0, min_episodes=5)
    for _ in range(4):
        assert curriculum.record(True) is False
    assert curriculum.record(True) is True


def test_max_episodes_forces_the_next_stage():
    curriculum = SuccessCurriculum([CurriculumStage(maps=["a"], success_threshold=1.0, min_episodes=0, max_episodes=3),
                                    CurriculumStage(maps=["a"])], window=2)
    assert [curriculum.record(False) for _ in range(3)] == [False, False, True]
    assert curriculum.stage_idx == 1


def test_max_episodes_below_min_episodes_is_rejected():
    with pytest.raises(ValueError):
        CurriculumStage(maps=["a"], min_episodes=10, max_episodes=5)


def test_last_stage_is_kept():
    curriculum = two_stage(window=1, threshold=0.5)
    curriculum.record(True)
    assert curriculum.is_last_stage
    assert not any(curriculum.record(True) for _ in range(10))


def test_config_is_json_serialisable():
    config = two_stage().get_config()
    assert json.loads(json.dumps(config))["stages"][0]["maps"] == ["a"]


# ------------------------------------------------------------
# Environment applies the current stage
# ------------------------------------------------------------
def make_env(curriculum):
    maps = [GridMap("a", (30, 30)), GridMap("b", (30, 30))]
    return Grid_Robot_Sections_Env_MultiMap(maps=maps, robot=GridRobotNotTurning(), reward_fn=lambda ctx: 0.0,
                                            lidar_config=LIDAR, num_sections=8, max_iteration_steps=5, curriculum=curriculum)


def test_env_rejects_unknown_curriculum_maps():
    with pytest.raises(ValueError):
        make_env(SuccessCurriculum([CurriculumStage(maps=["missing"])]))


def test_env_uses_only_the_stage_maps_and_goal_distance():
    env = make_env(two_stage(window=1000))
    env.reset(seed=0)
    for _ in range(100):
        _, info = env.reset()
        assert info["map"] == "a" and info["curriculum_stage"] == 0
        assert env._dist_to_goal() <= 12.0


def test_env_reports_outcomes_and_moves_to_the_next_stage():
    env = make_env(two_stage(window=2, threshold=1.0))
    for _ in range(2):  # two successes in a row
        env.reset(seed=0)
        env.robot.position = (5, 5)
        env.goal_pos = np.array([6, 5])
        _, _, terminated, _, info = env.step(1)
        assert terminated
    assert info["curriculum_advanced"] and env.curriculum.stage_idx == 1
    maps_seen = {env.reset()[1]["map"] for _ in range(60)}
    assert maps_seen == {"a", "b"}


def test_trainer_logs_the_stage(tmp_path):
    env = make_env(two_stage(window=1000))
    obs, act = int(env.get_observation_dim()), int(env.get_action_dim())
    agent = PPOAgent(PPOConfig(name="ppo", obs_dim=obs, action_dim=act, algorithm=RLAlgorithm.PPO, device="cpu", batch_size=4),
                     policy_network=SimpleQNetwork(obs, act, 1, 8), value_network=ValueNetwork(obs, 8))
    trainer = SingleTrainer(str(tmp_path / "runs"), str(tmp_path / "ckpt"),
                            TrainerConfig(num_episodes=1001, useTQDM=False, log_steps=False))
    trainer.add_agents(agent).add_environments(env)
    trainer.train(num_episodes=3)
    acc = EventAccumulator(str(tmp_path / "runs" / "ppo"))
    acc.Reload()
    assert [e.value for e in acc.Scalars("Curriculum/Stage")] == [0, 0, 0]
    assert not any(tag.startswith("Info/") for tag in acc.Tags()["scalars"])  # log_steps=False
    assert "Config/Curriculum/text_summary" in acc.Tags()["tensors"]
