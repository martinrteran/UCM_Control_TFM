from collections import Counter

import numpy as np
import pytest

from tfm_control_ucm.core.grid_env.environment import (
    Grid_Robot_Env,
    Grid_Robot_Sections_Env,
    Grid_Robot_Sections_Env_MultiMap,
)
from tfm_control_ucm.core.grid_env.map import GridMap
from tfm_control_ucm.core.grid_env.robot import GridRobot, GridRobotNotTurning
from tfm_control_ucm.core.rewards import ProgressReward, StepContext


def zero_reward(ctx):
    return 0.0


LIDAR = {"num_rays": 12, "max_range": 20, "fov": float(2 * np.pi), "noise_std": 0.0, "with_cache": True, "device": "cpu"}


def make_maps(sizes):
    return [GridMap(f"m{i}", size) for i, size in enumerate(sizes)]


# ------------------------------------------------------------
# Smoke tests: every env resets and steps with an injected reward
# ------------------------------------------------------------
@pytest.mark.parametrize("robot_cls", [GridRobotNotTurning, GridRobot])
def test_sections_env_reset_and_step(robot_cls):
    env = Grid_Robot_Sections_Env(map=GridMap("m", (25, 25)), robot=robot_cls(), reward_fn=zero_reward, lidar_config=LIDAR)
    obs, _ = env.reset(seed=0)
    assert np.all(np.isfinite(obs))
    for action in range(env.get_action_dim()):
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (env.get_observation_dim(),)
        assert reward == 0.0


def test_multimap_env_reset_and_step():
    env = Grid_Robot_Sections_Env_MultiMap(maps=make_maps([(25, 25), (30, 30)]), robot=GridRobotNotTurning(),
                                           reward_fn=zero_reward, lidar_config=LIDAR)
    obs, _ = env.reset(seed=0)
    obs, *_ = env.step(0)
    assert obs.shape == (env.get_observation_dim(),)


def test_grid_robot_env_reset_and_step():
    env = Grid_Robot_Env(map=GridMap("m", (25, 25)), robot=GridRobotNotTurning(), reward_fn=zero_reward, lidar_config=LIDAR)
    env.reset(seed=0)
    obs, *_ = env.step(0)
    assert np.all(np.isfinite(np.asarray(obs, dtype=np.float64)))


def test_reward_fn_receives_a_step_context():
    calls = []

    def spy(ctx):
        calls.append(ctx)
        return 1.5, {"a": 1.0, "b": 0.5}

    env = Grid_Robot_Sections_Env(map=GridMap("m", (25, 25)), robot=GridRobotNotTurning(), reward_fn=spy, lidar_config=LIDAR)
    env.reset(seed=0)
    start_dist = env._dist_to_goal()
    _, reward, terminated, truncated, info = env.step(1)

    assert reward == 1.5
    assert info["reward_components"] == {"a": 1.0, "b": 0.5}
    assert info["componente_reward_diff"] == pytest.approx(0.5)
    ctx = calls[0]
    assert isinstance(ctx, StepContext)
    assert ctx.action == 1 and ctx.previous_action == -1 and ctx.steps == 1
    assert ctx.previous_dist_to_goal == pytest.approx(start_dist)
    assert ctx.dist_to_goal == pytest.approx(env._dist_to_goal())


def test_previous_dist_is_carried_between_steps():
    calls = []
    env = Grid_Robot_Sections_Env(map=GridMap("m", (25, 25)), robot=GridRobotNotTurning(),
                                  reward_fn=lambda ctx: calls.append(ctx) or 0.0, lidar_config=LIDAR)
    env.reset(seed=0)
    env.step(1)
    env.step(2)
    assert calls[1].previous_dist_to_goal == pytest.approx(calls[0].dist_to_goal)
    assert calls[1].previous_action == 1


def test_non_callable_reward_fn_is_rejected():
    with pytest.raises(TypeError):
        Grid_Robot_Sections_Env(map=GridMap("m", (25, 25)), robot=GridRobotNotTurning(), reward_fn=3.0, lidar_config=LIDAR)


# ------------------------------------------------------------
# The environment, not the reward, decides when an episode ends
# ------------------------------------------------------------
def make_sections_env(**kwargs):
    env = Grid_Robot_Sections_Env(map=GridMap("m", (25, 25)), robot=GridRobotNotTurning(),
                                  reward_fn=zero_reward, lidar_config=LIDAR, **kwargs)
    env.reset(seed=0)
    return env


def test_reaching_the_goal_terminates():
    env = make_sections_env()
    env.robot.position = (5, 5)
    env.goal_pos = np.array([6, 5])
    _, _, terminated, truncated, info = env.step(1)  # move right onto the goal
    assert terminated and not truncated
    assert info["reached_goal"]


def test_collision_truncates():
    env = make_sections_env()
    env.robot.position = (0, 0)
    env.goal_pos = np.array([20, 20])
    _, _, terminated, truncated, info = env.step(3)  # move left into the wall
    assert truncated and not terminated
    assert info["hit_obstacle"] and not info["max_steps_reached"]


def test_step_limit_truncates():
    env = make_sections_env(max_iteration_steps=1)
    env.robot.position = (10, 10)
    env.goal_pos = np.array([20, 20])
    _, _, terminated, truncated, info = env.step(0)
    assert truncated and not terminated
    assert info["max_steps_reached"] and not info["hit_obstacle"]


def test_progress_reward_plugs_into_env():
    env = make_sections_env()
    env.reward_fn = ProgressReward()
    env.robot.position = (5, 5)
    env.goal_pos = np.array([20, 5])
    env._start_episode(env._get_observation())
    _, reward, *_ , info = env.step(1)  # one cell closer to the goal
    assert info["reward_components"]["progress"] > 0
    assert reward == pytest.approx(sum(info["reward_components"].values()))


# ------------------------------------------------------------
# Previous observation must not alias the returned observation
# ------------------------------------------------------------
@pytest.mark.parametrize("env_cls", [Grid_Robot_Sections_Env, Grid_Robot_Sections_Env_MultiMap])
def test_previous_obs_is_a_copy(env_cls):
    kwargs = {"maps": make_maps([(25, 25)])} if env_cls is Grid_Robot_Sections_Env_MultiMap else {"map": GridMap("m", (25, 25))}
    env = env_cls(robot=GridRobotNotTurning(), reward_fn=zero_reward, lidar_config=LIDAR, **kwargs)
    env.reset(seed=0)
    obs, *_ = env.step(0)
    obs[:] = 999.0
    assert not np.all(env._previous_obs == 999.0)


# ------------------------------------------------------------
# MultiMap map selection
# ------------------------------------------------------------
def test_random_map_selection_is_uniform():
    env = Grid_Robot_Sections_Env_MultiMap(maps=make_maps([(25, 25)] * 4), robot=GridRobotNotTurning(),
                                           reward_fn=zero_reward, lidar_config=LIDAR)
    env.reset(seed=0)
    counts = Counter(env._get_random_map().name for _ in range(4000))
    assert set(counts) == {"m0", "m1", "m2", "m3"}
    for n in counts.values():
        assert 800 < n < 1200  # 25 % ± 5 %


def test_random_map_selection_is_reproducible_with_seed():
    def sequence(seed):
        env = Grid_Robot_Sections_Env_MultiMap(maps=make_maps([(25, 25)] * 4), robot=GridRobotNotTurning(),
                                               reward_fn=zero_reward, lidar_config=LIDAR)
        env.reset(seed=seed)
        return [env._get_random_map().name for _ in range(20)]

    assert sequence(7) == sequence(7)


def test_sequential_map_selection_cycles_in_order():
    env = Grid_Robot_Sections_Env_MultiMap(maps=make_maps([(25, 25)] * 3), robot=GridRobotNotTurning(),
                                           reward_fn=zero_reward, lidar_config=LIDAR, random_map=False)
    names = []
    for _ in range(4):
        env.reset()
        names.append(env.map.name)
    assert names == ["m0", "m1", "m2", "m0"]


def test_map_diagonal_uses_largest_map():
    env = Grid_Robot_Sections_Env_MultiMap(maps=make_maps([(25, 25), (40, 30)]), robot=GridRobotNotTurning(),
                                           reward_fn=zero_reward, lidar_config=LIDAR)
    assert env._map_diagonal == pytest.approx(50.0)  # sqrt(40² + 30²)
