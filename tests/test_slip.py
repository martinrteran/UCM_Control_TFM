"""Terrain (floor grip) on the map, the slip model and how the environment applies it."""
import json
from pathlib import Path

import numpy as np
import pytest

from tfm_control_ucm.core.grid_env.environment import Grid_Robot_Env, Grid_Robot_Sections_Env
from tfm_control_ucm.core.grid_env.map import GridMap, GridObstacle, GridTerrain
from tfm_control_ucm.core.grid_env.robot import GridRobot, GridRobotNotTurning
from tfm_control_ucm.core.slip import SlipModel, classify_maneuver

LIDAR = {"num_rays": 12, "max_range": 20, "fov": float(2 * np.pi), "noise_std": 0.0, "with_cache": True, "device": "cpu"}
MAPS_DIR = Path(__file__).resolve().parent.parent / "maps"

# GridRobotNotTurning actions: 0 = north (0,-1), 1 = east (1,0), 2 = south (0,1), 3 = west (-1,0)
N, E, S, W = 0, 1, 2, 3


def zero_reward(ctx):
    return 0.0


def always_slips(**kwargs):
    """A model that always slips on mu=0 floors (gain large, no cap)."""
    return SlipModel(gain=100.0, max_prob=1.0, **kwargs)


def ice_env(slip_model=None, size=(25, 25), obstacles=None, robot=None, **kwargs):
    """Sections env whose whole floor is ice (mu = 0)."""
    mapa = GridMap("ice", size, grid_obstacles=obstacles, terrains=[GridTerrain((0, 0), size, 0.0)])
    env = Grid_Robot_Sections_Env(map=mapa, robot=robot or GridRobotNotTurning(), reward_fn=zero_reward,
                                  lidar_config=LIDAR, slip_model=slip_model, **kwargs)
    env.reset(seed=0)
    env.goal_pos = np.array([size[0] - 1, size[1] - 1])
    return env


def place(env, col, row, momentum=(0, 0)):
    env.robot.position = (col, row)
    env._momentum = momentum


# ------------------------------------------------------------
# Map terrain
# ------------------------------------------------------------
def test_default_friction_is_one_and_terrain_paints_its_cells():
    mapa = GridMap("m", (10, 8), terrains=[GridTerrain((2, 3), (4, 2), 0.2)])  # cols 2-5, rows 3-4
    assert mapa.friction_at(0, 0) == 1.0
    assert mapa.friction_at(3, 2) == pytest.approx(0.2) and mapa.friction_at(4, 5) == pytest.approx(0.2)
    assert mapa.friction_at(5, 2) == 1.0
    assert mapa.friction_at(-1, 0) == 1.0  # outside the map


def test_later_terrain_overrides_earlier_one():
    mapa = GridMap("m", (10, 10), terrains=[GridTerrain((0, 0), (5, 5), 0.2), GridTerrain((2, 2), (5, 5), 0.5)])
    assert mapa.friction_at(1, 1) == pytest.approx(0.2)
    assert mapa.friction_at(3, 3) == pytest.approx(0.5)


def test_terrain_does_not_change_occupancy():
    plain = GridMap("m", (10, 10), grid_obstacles=[GridObstacle((1, 1), (2, 2))])
    slippery = GridMap("m", (10, 10), grid_obstacles=[GridObstacle((1, 1), (2, 2))],
                       terrains=[GridTerrain((0, 0), (10, 10), 0.1)])
    assert np.array_equal(plain.to_numpy(), slippery.to_numpy())


@pytest.mark.parametrize("friction", [-0.1, 1.5])
def test_friction_outside_unit_interval_is_rejected(friction):
    with pytest.raises(ValueError):
        GridTerrain((0, 0), (2, 2), friction)


def test_terrain_out_of_bounds_is_rejected():
    with pytest.raises(IndexError):
        GridMap("m", (5, 5), terrains=[GridTerrain((4, 4), (2, 2), 0.5)])


def test_str_marks_slippery_free_cells():
    mapa = GridMap("m", (3, 2), grid_obstacles=[GridObstacle((1, 0), (1, 1))], terrains=[GridTerrain((0, 0), (3, 1), 0.3)])
    assert str(mapa) == "~#~\n..."


def test_save_and_load_round_trip(tmp_path):
    mapa = GridMap("rt", (12, 9), grid_obstacles=[GridObstacle((1, 1), (2, 3))],
                   terrains=[GridTerrain((4, 4), (3, 2), 0.25, "ice")])
    mapa.save(str(tmp_path))
    loaded = GridMap.load(str(tmp_path / "rt.json"))
    assert np.array_equal(loaded.to_numpy(), mapa.to_numpy())
    assert np.allclose(loaded.friction.numpy(), mapa.friction.numpy())
    assert loaded.get_terrains()[0].name == "ice"


def test_map_files_without_terrains_still_load(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"name": "old", "size": [5, 5], "obstacles": []}))
    mapa = GridMap.load(str(path))
    assert not mapa.has_terrain and float(mapa.friction.min()) == 1.0


@pytest.mark.parametrize("name", ["map_obstacle_free", "map_1_obstacle", "map_4_square_obstacle", "map_simple"])
def test_slippery_maps_keep_the_occupancy_of_the_originals(name):
    original = GridMap.load(str(MAPS_DIR / f"{name}.json"))
    slippery = GridMap.load(str(MAPS_DIR / f"{name}_slippery.json"))
    assert np.array_equal(original.to_numpy(), slippery.to_numpy())
    assert slippery.has_terrain and slippery.name != original.name


# ------------------------------------------------------------
# Slip model
# ------------------------------------------------------------
@pytest.mark.parametrize("momentum,intended,expected", [
    ((0, 0), (1, 0), "start"), ((1, 0), (1, 0), "straight"),
    ((1, 0), (-1, 0), "reverse"), ((1, 0), (0, -1), "turn"), ((0, 1), (1, 0), "turn"),
])
def test_classify_maneuver(momentum, intended, expected):
    assert classify_maneuver(momentum, intended) == expected


def test_probability_is_zero_when_the_floor_gives_enough_grip():
    model = SlipModel()
    for momentum, intended in [((0, 0), (1, 0)), ((1, 0), (1, 0)), ((1, 0), (0, 1)), ((1, 0), (-1, 0))]:
        assert model.slip_probability(1.0, momentum, intended) == 0.0


def test_probability_grows_with_the_demand_of_the_maneuver():
    model = SlipModel()  # d = 0.3 / 0.4 / 0.6 / 0.8, gain 2, cap 0.9
    ice = 0.2
    assert model.slip_probability(ice, (1, 0), (1, 0)) == pytest.approx(0.2)   # straight
    assert model.slip_probability(ice, (0, 0), (1, 0)) == pytest.approx(0.4)   # start
    assert model.slip_probability(ice, (1, 0), (0, 1)) == pytest.approx(0.8)   # turn
    assert model.slip_probability(ice, (1, 0), (-1, 0)) == pytest.approx(0.9)  # reverse, capped
    assert model.slip_probability(0.5, (1, 0), (0, 1)) == pytest.approx(0.2)   # turn on wet floor


def test_disabled_kind_never_slips():
    model = always_slips(kinds=("overshoot",))
    rng = np.random.default_rng(0)
    assert all(model.sample(0.0, (1, 0), (0, 1), rng).kind == "none" for _ in range(50))  # turn -> diagonal disabled


@pytest.mark.parametrize("momentum,intended,kind,path,new_momentum", [
    ((1, 0), (1, 0), "overshoot", ((1, 0), (1, 0)), (1, 0)),
    ((0, 0), (1, 0), "stay", (), (0, 0)),
    ((1, 0), (0, -1), "diagonal", ((1, 0), (0, -1)), (0, -1)),
    ((1, 0), (-1, 0), "drag", ((1, 0),), (1, 0)),
])
def test_sample_paths(momentum, intended, kind, path, new_momentum):
    event = always_slips().sample(0.0, momentum, intended, np.random.default_rng(0))
    assert (event.kind, event.path, event.momentum) == (kind, path, new_momentum)


def test_lateral_drift_is_perpendicular():
    model = always_slips(kinds=("lateral",), lateral_demand=1.0)
    rng = np.random.default_rng(0)
    sides = {model.sample(0.0, (1, 0), (1, 0), rng).path for _ in range(100)}
    assert sides == {((0, 1),), ((0, -1),)}


def test_invalid_model_parameters_are_rejected():
    with pytest.raises(ValueError): SlipModel(kinds=("teleport",))
    with pytest.raises(ValueError): SlipModel(into_obstacle="bounce")


# ------------------------------------------------------------
# Environment dynamics
# ------------------------------------------------------------
def test_overshoot_slides_one_more_cell():
    env = ice_env(always_slips())
    place(env, 5, 5, momentum=(1, 0))
    _, _, terminated, truncated, info = env.step(E)
    assert tuple(env.robot.position) == (7, 5)
    assert info["slip_kind"] == "overshoot" and info["slipped"] and not (terminated or truncated)


def test_stay_is_not_a_collision():
    env = ice_env(always_slips())
    place(env, 5, 5)
    _, _, _, truncated, info = env.step(E)
    assert tuple(env.robot.position) == (5, 5)
    assert info["slip_kind"] == "stay" and not info["hit_obstacle"] and not truncated


def test_diagonal_slip_when_turning():
    env = ice_env(always_slips())
    place(env, 5, 5, momentum=(1, 0))
    env.step(N)
    assert tuple(env.robot.position) == (6, 4)
    assert env._momentum == (0, -1)


def test_drag_when_reversing():
    env = ice_env(always_slips())
    place(env, 5, 5, momentum=(1, 0))
    env.step(W)
    assert tuple(env.robot.position) == (6, 5)


def test_slide_into_a_wall_is_a_collision_by_default():
    env = ice_env(always_slips())
    place(env, 23, 5, momentum=(1, 0))  # one cell from the east border
    _, _, terminated, truncated, info = env.step(E)
    assert truncated and not terminated
    assert info["hit_obstacle"] and info["slip_collision"]
    assert tuple(env.robot.position) == (24, 5)  # stays on the last free cell


def test_slide_into_a_wall_can_just_stop():
    env = ice_env(always_slips(into_obstacle="stop"))
    place(env, 23, 5, momentum=(1, 0))
    _, _, _, truncated, info = env.step(E)
    assert not truncated and not info["hit_obstacle"] and info["slipped"]
    assert tuple(env.robot.position) == (24, 5) and env._momentum == (0, 0)


def test_moving_into_a_wall_is_a_plain_collision():
    env = ice_env(always_slips())
    place(env, 0, 5)
    _, _, _, truncated, info = env.step(W)
    assert truncated and info["hit_obstacle"] and not info["slip_collision"] and not info["slipped"]


def test_normal_floor_never_slips():
    env = Grid_Robot_Sections_Env(map=GridMap("m", (25, 25)), robot=GridRobotNotTurning(), reward_fn=zero_reward,
                                  lidar_config=LIDAR, slip_model=always_slips())
    env.reset(seed=0)
    env.goal_pos = np.array([24, 24])
    for action in [E, E, N, W, S, E]:
        place(env, 10, 10, momentum=env._momentum)
        _, _, _, _, info = env.step(action)
        assert not info["slipped"]


def test_without_slip_model_momentum_is_tracked_and_info_has_no_slip_keys():
    env = ice_env(None)
    place(env, 5, 5)
    _, _, _, _, info = env.step(E)
    assert tuple(env.robot.position) == (6, 5) and env._momentum == (1, 0)
    assert "slipped" not in info


def test_grid_robot_turning_in_place_stops_the_momentum():
    env = ice_env(always_slips(), robot=GridRobot())
    env.robot.position = (10, 10)
    env._momentum = (0, -1)
    env.step(1)  # turn left: no translation, no slip
    assert tuple(env.robot.position) == (10, 10) and env._momentum == (0, 0)


def test_step_context_carries_the_slip():
    calls = []
    env = ice_env(always_slips())
    env.reward_fn = lambda ctx: calls.append(ctx) or 0.0
    place(env, 5, 5)
    env.step(E)
    assert calls[0].slipped and calls[0].slip_kind == "stay" and calls[0].friction == 0.0


def test_slips_do_not_change_the_maps_and_positions_of_an_evaluation():
    """With the same seed, the start/goal sequence must not depend on the slips (paired evaluations)."""
    def starts(slip_model):
        mapa = GridMap("ice", (25, 25), terrains=[GridTerrain((0, 0), (25, 25), 0.0)])
        env = Grid_Robot_Sections_Env(map=mapa, robot=GridRobotNotTurning(), reward_fn=zero_reward,
                                      lidar_config=LIDAR, slip_model=slip_model, max_iteration_steps=5)
        positions = []
        for episode in range(5):
            env.reset(seed=3 if episode == 0 else None)
            positions.append((tuple(env.robot.position), tuple(env.goal_pos)))
            for action in [E, E, N, S, W]:
                *_, terminated, truncated, _ = env.step(action)
                if terminated or truncated: break
        return positions

    assert starts(None) == starts(SlipModel())


# ------------------------------------------------------------
# Terrain observation
# ------------------------------------------------------------
@pytest.mark.parametrize("mode,extra", [("none", 0), ("proprio", 4), ("neighbors", 8)])
def test_terrain_obs_extends_the_observation(mode, extra):
    plain = Grid_Robot_Sections_Env(map=GridMap("m", (25, 25)), robot=GridRobotNotTurning(), reward_fn=zero_reward, lidar_config=LIDAR)
    env = ice_env(SlipModel(), terrain_obs=mode)
    assert env.get_observation_dim() == plain.get_observation_dim() + extra
    obs, _ = env.reset(seed=0)
    assert obs.shape == (env.get_observation_dim(),)
    if mode != "none":  # The plain Sections space keeps its historical (N+1, 2) shape
        assert env.observation_space.shape == obs.shape
        assert np.all(obs[-extra:] >= env.observation_space.low[-extra:]) and np.all(obs[-extra:] <= env.observation_space.high[-extra:])


def test_proprio_reports_momentum_slip_and_grip():
    env = ice_env(always_slips(), terrain_obs="proprio", terrain_obs_noise=0.0)
    place(env, 5, 5, momentum=(1, 0))
    obs, *_ = env.step(E)  # overshoot east
    assert obs[-4:].tolist() == pytest.approx([1.0, 0.0, 1.0, 0.0])  # forward = east for this robot, slipped, mu = 0
    place(env, 5, 5)
    obs, *_ = env.step(N)  # starting from rest: stays
    assert obs[-4:].tolist() == pytest.approx([0.0, 0.0, 1.0, 0.0])


def test_neighbors_are_egocentric():
    mapa = GridMap("m", (25, 25), terrains=[GridTerrain((10, 9), (1, 1), 0.3)])  # cell north of (10, 10)
    env = Grid_Robot_Sections_Env(map=mapa, robot=GridRobot(), reward_fn=zero_reward, lidar_config=LIDAR,
                                  terrain_obs="neighbors", terrain_obs_noise=0.0)
    env.reset(seed=0)
    from tfm_control_ucm.core.grid_env.utils import Orientation
    env.robot.reset((10, 10), Orientation.north)
    assert env._terrain_features()[4:].tolist() == pytest.approx([0.3, 1.0, 1.0, 1.0])  # front = north
    env.robot.reset((10, 10), Orientation.east)
    assert env._terrain_features()[4:].tolist() == pytest.approx([1.0, 0.3, 1.0, 1.0])  # north is on the left


def test_terrain_obs_does_not_change_the_obstacle_distance():
    wall = [GridObstacle((14, 0), (1, 25))]
    def min_dist(mode):
        env = Grid_Robot_Sections_Env(map=GridMap("m", (25, 25), grid_obstacles=wall), robot=GridRobotNotTurning(),
                                      reward_fn=zero_reward, lidar_config=LIDAR, terrain_obs=mode)
        env.reset(seed=0)
        env.robot.position = (10, 10)
        env.goal_pos = np.array([2, 2])
        return env.step(E)[-1]["min_dist"]
    assert min_dist("neighbors") == min_dist("none")


def test_terrain_obs_needs_a_flat_observation():
    with pytest.raises(ValueError):
        Grid_Robot_Env(map=GridMap("m", (25, 25)), robot=GridRobotNotTurning(), reward_fn=zero_reward,
                       lidar_config=LIDAR, terrain_obs="proprio")
