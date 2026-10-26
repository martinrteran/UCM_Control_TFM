import dataclasses

import numpy as np
import pytest

from tfm_control_ucm.core.rewards import (
    ProgressReward,
    RewardFunction,
    RewardResult,
    StepContext,
    evaluate_reward,
    summarize_components,
)


def make_ctx(**overrides):
    base = dict(action=0, previous_action=-1, obs=np.zeros(4), previous_obs=np.zeros(4),
                dist_to_goal=9.0, previous_dist_to_goal=10.0, min_obstacle_dist=10.0,
                hit_obstacle=False, reached_goal=False, max_steps_reached=False,
                steps=1, max_steps=100, robot_pos=np.array([0, 0]), goal_pos=np.array([5, 5]),
                map_diagonal=100.0, lidar_max_range=10.0)
    base.update(overrides)
    return StepContext(**base)


def test_progress_toward_goal_is_rewarded():
    result = ProgressReward()(make_ctx())
    assert result.components["progress"] == pytest.approx(10.0 * 1.0 / 100.0)
    assert result.reward == pytest.approx(sum(result.components.values()))


def test_moving_away_is_penalised_unless_clipped():
    away = make_ctx(dist_to_goal=11.0)
    assert ProgressReward()(away).components["progress"] < 0
    assert ProgressReward(clip_negative_progress=True)(away).components["progress"] == 0.0


def test_proximity_penalty_grows_linearly_inside_the_margin():
    reward = ProgressReward(safety_margin=4.0, max_proximity_penalty=2.0)
    assert reward(make_ctx(min_obstacle_dist=5.0)).components["proximity_penalty"] == 0.0
    assert reward(make_ctx(min_obstacle_dist=2.0)).components["proximity_penalty"] == pytest.approx(-1.0)
    assert reward(make_ctx(min_obstacle_dist=0.0)).components["proximity_penalty"] == pytest.approx(-2.0)


def test_terminal_terms():
    reward = ProgressReward(goal_reward=20.0, collision_penalty=5.0)
    assert reward(make_ctx(reached_goal=True)).components["terminal_reward"] == 20.0
    assert reward(make_ctx(hit_obstacle=True)).components["terminal_reward"] == -5.0
    assert reward(make_ctx()).components["terminal_reward"] == 0.0


def test_get_config_lists_name_and_parameters():
    config = ProgressReward(step_penalty=0.1).get_config()
    assert config["name"] == "ProgressReward"
    assert config["step_penalty"] == 0.1
    assert set(config) == {"name"} | {f.name for f in dataclasses.fields(ProgressReward)}


def test_custom_reward_function_subclass():
    class Sparse(RewardFunction):
        def __call__(self, ctx):
            return RewardResult(1.0 if ctx.reached_goal else 0.0, {})

    assert Sparse()(make_ctx(reached_goal=True)).reward == 1.0
    assert Sparse().get_config() == {"name": "Sparse"}


@pytest.mark.parametrize("output, expected", [
    (2.0, RewardResult(2.0, {})),
    ((2.0, {"a": 2.0}), RewardResult(2.0, {"a": 2.0})),
    (RewardResult(2.0, {"a": 2.0}), RewardResult(2.0, {"a": 2.0})),
])
def test_evaluate_reward_accepts_plain_callables(output, expected):
    assert evaluate_reward(lambda ctx: output, make_ctx()) == expected


def test_summarize_components():
    summary = summarize_components({"a": -3.0, "b": 1.0, "c": 0.5})
    assert summary == {"componente_reward_min": 0.5, "componente_reward_max": 3.0, "componente_reward_diff": 2.5}
    assert summarize_components({})["componente_reward_diff"] == 0.0


# ------------------------------------------------------------
# Geodesic (BFS) progress
# ------------------------------------------------------------
def test_geodesic_progress_uses_the_shortest_path_distance():
    # Detour around a wall: straight-line distance grows, path distance shrinks
    ctx = make_ctx(previous_dist_to_goal=5.0, dist_to_goal=6.0,
                   previous_geodesic_dist_to_goal=20.0, geodesic_dist_to_goal=19.0)
    assert ProgressReward()(ctx).components["progress"] < 0
    assert ProgressReward(distance="geodesic")(ctx).components["progress"] == pytest.approx(10.0 * 1.0 / 100.0)


def test_geodesic_progress_falls_back_to_euclidean_when_unreachable():
    ctx = make_ctx(previous_geodesic_dist_to_goal=float("inf"), geodesic_dist_to_goal=float("inf"))
    assert ProgressReward(distance="geodesic")(ctx) == ProgressReward()(ctx)


def test_invalid_distance_is_rejected():
    with pytest.raises(ValueError):
        ProgressReward(distance="manhattan")
