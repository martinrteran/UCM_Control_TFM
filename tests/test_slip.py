"""Slippery floors: SlipperyGridMap, the slip model, the slippery robots and their environment."""
import json
from pathlib import Path

import numpy as np
import pytest

from tfm_control_ucm.core.grid_env.environment import Grid_Robot_Sections_Env_MultiMap
from tfm_control_ucm.core.grid_env.map import GridMap, GridObstacle
from tfm_control_ucm.core.grid_env.robot import GridRobotNotTurning
from tfm_control_ucm.core.grid_env.slippery_environment import Slippery_Grid_Robot_Sections_Env_MultiMap
from tfm_control_ucm.core.grid_env.slippery_map import GridTerrain, SlipperyGridMap, friction_at
from tfm_control_ucm.core.grid_env.slippery_robot import SlipperyGridRobot, SlipperyGridRobotNotTurning
from tfm_control_ucm.core.grid_env.utils import Orientation
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


def ice_map(size=(25, 25), obstacles=None):
    return SlipperyGridMap("ice", size, grid_obstacles=obstacles, terrains=[GridTerrain((0, 0), size, 0.0)])


def ice_env(slip_model=None, robot=None, mapa=None, **kwargs):
    """Slippery env on a map whose whole floor is ice (mu = 0)."""
    robot = robot or SlipperyGridRobotNotTurning(slip_model=slip_model)
    env = Slippery_Grid_Robot_Sections_Env_MultiMap(maps=[mapa or ice_map()], robot=robot, reward_fn=zero_reward,
                                                    lidar_config=LIDAR, **kwargs)
    env.reset(seed=0)
    env.goal_pos = np.array([24, 24])
    return env


def place(env, col, row, momentum=(0, 0)):
    env.robot.position = (col, row)
    env.robot.momentum = momentum


# ------------------------------------------------------------
# SlipperyGridMap
# ------------------------------------------------------------
def test_default_friction_is_one_and_terrain_paints_its_cells():
    mapa = SlipperyGridMap("m", (10, 8), terrains=[GridTerrain((2, 3), (4, 2), 0.2)])  # cols 2-5, rows 3-4
    assert mapa.friction_at(0, 0) == 1.0
    assert mapa.friction_at(3, 2) == pytest.approx(0.2) and mapa.friction_at(4, 5) == pytest.approx(0.2)
    assert mapa.friction_at(5, 2) == 1.0
    assert mapa.friction_at(-1, 0) == 1.0  # outside the map


def test_plain_grid_map_has_the_default_grip():
    assert friction_at(GridMap("m", (5, 5)), 2, 2) == 1.0


def test_later_terrain_overrides_earlier_one():
    mapa = SlipperyGridMap("m", (10, 10), terrains=[GridTerrain((0, 0), (5, 5), 0.2), GridTerrain((2, 2), (5, 5), 0.5)])
    assert mapa.friction_at(1, 1) == pytest.approx(0.2)
    assert mapa.friction_at(3, 3) == pytest.approx(0.5)


def test_terrain_does_not_change_occupancy():
    plain = GridMap("m", (10, 10), grid_obstacles=[GridObstacle((1, 1), (2, 2))])
    slippery = SlipperyGridMap("m", (10, 10), grid_obstacles=[GridObstacle((1, 1), (2, 2))],
                               terrains=[GridTerrain((0, 0), (10, 10), 0.1)])
    assert np.array_equal(plain.to_numpy(), slippery.to_numpy())


@pytest.mark.parametrize("friction", [-0.1, 1.5])
def test_friction_outside_unit_interval_is_rejected(friction):
    with pytest.raises(ValueError):
        GridTerrain((0, 0), (2, 2), friction)


def test_terrain_out_of_bounds_is_rejected():
    with pytest.raises(IndexError):
        SlipperyGridMap("m", (5, 5), terrains=[GridTerrain((4, 4), (2, 2), 0.5)])


def test_str_marks_slippery_free_cells():
    mapa = SlipperyGridMap("m", (3, 2), grid_obstacles=[GridObstacle((1, 0), (1, 1))], terrains=[GridTerrain((0, 0), (3, 1), 0.3)])
    assert str(mapa) == "~#~\n..."


def test_save_and_load_round_trip(tmp_path):
    mapa = SlipperyGridMap("rt", (12, 9), grid_obstacles=[GridObstacle((1, 1), (2, 3))],
                           terrains=[GridTerrain((4, 4), (3, 2), 0.25, "ice")])
    mapa.save(str(tmp_path))
    loaded = SlipperyGridMap.load(str(tmp_path / "rt.json"))
    assert np.array_equal(loaded.to_numpy(), mapa.to_numpy())
    assert np.allclose(loaded.friction.numpy(), mapa.friction.numpy())
    assert loaded.get_terrains()[0].name == "ice"


def test_plain_map_files_load_as_slippery_maps(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"name": "old", "size": [5, 5], "obstacles": []}))
    mapa = SlipperyGridMap.load(str(path))
    assert not mapa.has_terrain and float(mapa.friction.min()) == 1.0


@pytest.mark.parametrize("name", ["map_obstacle_free", "map_1_obstacle", "map_4_square_obstacle", "map_simple"])
def test_slippery_maps_keep_the_occupancy_of_the_originals(name):
    original = GridMap.load(str(MAPS_DIR / f"{name}.json"))
    slippery = SlipperyGridMap.load(str(MAPS_DIR / f"{name}_slippery.json"))
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
# Slippery robots
# ------------------------------------------------------------
def test_robot_without_slip_model_moves_like_the_parent_and_tracks_momentum():
    robot = SlipperyGridRobotNotTurning(position=(5, 5))
    assert robot.move(ice_map(), E) and tuple(robot.position) == (6, 5) and robot.momentum == (1, 0)
    robot.position = (0, 0)
    assert not robot.move(ice_map(), W) and robot.momentum == (0, 0)  # wall: collision


def test_robot_reset_clears_the_momentum():
    robot = SlipperyGridRobotNotTurning(position=(5, 5))
    robot.move(ice_map(), E)
    robot.random_reset(ice_map(), np.random.default_rng(0))
    assert robot.momentum == (0, 0)


def test_grid_robot_turning_in_place_stops_the_momentum():
    robot = SlipperyGridRobot(position=(10, 10), orientation=Orientation.north, slip_model=always_slips())
    robot.momentum = (0, -1)
    assert robot.move(ice_map(), 1)  # turn left: no translation, no slip
    assert tuple(robot.position) == (10, 10) and robot.momentum == (0, 0) and robot.last_slip.kind == "none"


def test_plain_robots_are_unchanged():
    robot = GridRobotNotTurning(position=(5, 5))
    assert not hasattr(robot, "momentum")


# ------------------------------------------------------------
# Slippery environment dynamics
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
    assert env.robot.momentum == (0, -1)


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
    assert tuple(env.robot.position) == (24, 5) and env.robot.momentum == (0, 0)


def test_moving_into_a_wall_is_a_plain_collision():
    env = ice_env(always_slips())
    place(env, 0, 5)
    _, _, _, truncated, info = env.step(W)
    assert truncated and info["hit_obstacle"] and not info["slip_collision"] and not info["slipped"]


def test_normal_floor_never_slips():
    env = ice_env(always_slips(), mapa=SlipperyGridMap("m", (25, 25)))
    for action in [E, E, N, W, S, E]:
        place(env, 10, 10, momentum=env.robot.momentum)
        _, _, _, _, info = env.step(action)
        assert not info["slipped"]


def test_without_slip_model_info_has_no_slip_keys():
    env = ice_env(None)
    place(env, 5, 5)
    _, _, _, _, info = env.step(E)
    assert tuple(env.robot.position) == (6, 5) and "slipped" not in info


def test_reward_sees_the_lidar_and_goal_part_of_the_observation():
    calls = []
    env = ice_env(always_slips(), terrain_obs="proprio")
    env.reward_fn = lambda ctx: calls.append(ctx) or 0.0
    place(env, 5, 5)
    obs, *_ = env.step(E)
    assert calls[0].obs.shape == calls[0].previous_obs.shape == (obs.shape[0] - 4,)


def test_slips_do_not_change_the_maps_and_positions_of_an_evaluation():
    """With the same seed, the start/goal sequence must not depend on the slips (paired evaluations)."""
    def starts(slip_model):
        env = Slippery_Grid_Robot_Sections_Env_MultiMap(maps=[ice_map()], robot=SlipperyGridRobotNotTurning(slip_model=slip_model),
                                                        reward_fn=zero_reward, lidar_config=LIDAR, max_iteration_steps=5)
        positions = []
        for episode in range(5):
            env.reset(seed=3 if episode == 0 else None)
            positions.append((tuple(env.robot.position), tuple(env.goal_pos)))
            for action in [E, E, N, S, W]:
                *_, terminated, truncated, _ = env.step(action)
                if terminated or truncated: break
        return positions

    assert starts(None) == starts(SlipModel())


def test_slips_are_reproducible_with_the_seed():
    def kinds():
        env = ice_env(SlipModel())
        env.reset(seed=1)
        out = []
        for action in [E, E, N, N, W, S, E, E]:
            place(env, 10, 10, momentum=env.robot.momentum)
            out.append(env.step(action)[-1]["slip_kind"])
        return out
    assert kinds() == kinds()


def test_slippery_env_needs_a_slippery_robot():
    with pytest.raises(TypeError):
        Slippery_Grid_Robot_Sections_Env_MultiMap(maps=[ice_map()], robot=GridRobotNotTurning(), reward_fn=zero_reward, lidar_config=LIDAR)


# ------------------------------------------------------------
# Terrain observation
# ------------------------------------------------------------
@pytest.mark.parametrize("mode,extra", [("none", 0), ("proprio", 4), ("neighbors", 8)])
def test_terrain_obs_extends_the_observation(mode, extra):
    plain = Grid_Robot_Sections_Env_MultiMap(maps=[GridMap("m", (25, 25))], robot=GridRobotNotTurning(),
                                             reward_fn=zero_reward, lidar_config=LIDAR)
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
    mapa = SlipperyGridMap("m", (25, 25), terrains=[GridTerrain((10, 9), (1, 1), 0.3)])  # cell north of (10, 10)
    env = ice_env(None, robot=SlipperyGridRobot(), mapa=mapa, terrain_obs="neighbors", terrain_obs_noise=0.0)
    env.robot.reset((10, 10), Orientation.north)
    assert env._terrain_features()[4:].tolist() == pytest.approx([0.3, 1.0, 1.0, 1.0])  # front = north
    env.robot.reset((10, 10), Orientation.east)
    assert env._terrain_features()[4:].tolist() == pytest.approx([1.0, 0.3, 1.0, 1.0])  # north is on the left


# ------------------------------------------------------------
# Slip trainer
# ------------------------------------------------------------
def _train(tmp_path, trainer_cls, env, episodes=3):
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tfm_control_ucm.core.agents.agent import PPOAgent
    from tfm_control_ucm.core.agents.networks import SimpleQNetwork, ValueNetwork
    from tfm_control_ucm.core.agents.utils import PPOConfig, RLAlgorithm
    from tfm_control_ucm.core.training.TrainingManager import TrainerConfig

    torch.manual_seed(0)
    obs, act = int(env.get_observation_dim()), int(env.get_action_dim())
    agent = PPOAgent(PPOConfig(name="ppo", obs_dim=obs, action_dim=act, algorithm=RLAlgorithm.PPO, device="cpu", batch_size=8),
                     policy_network=SimpleQNetwork(obs, act, 1, 16), value_network=ValueNetwork(obs, 16))
    trainer = trainer_cls(str(tmp_path / "runs"), str(tmp_path / "ckpt"), TrainerConfig(num_episodes=1001, useTQDM=False, eps_step=2))
    trainer.add_agents(agent).add_environments(env)
    trainer.train(num_episodes=episodes)
    acc = EventAccumulator(str(tmp_path / "runs" / "ppo"))
    acc.Reload()
    return acc


def _slippery_training_env(slip_model):
    return Slippery_Grid_Robot_Sections_Env_MultiMap(maps=[ice_map()], robot=SlipperyGridRobotNotTurning(slip_model=slip_model),
                                                     reward_fn=zero_reward, lidar_config=LIDAR, max_iteration_steps=30)


def test_slip_trainer_logs_the_slips_of_every_episode(tmp_path):
    from tfm_control_ucm.core.training.slip_trainer import SlipSingleTrainer
    acc = _train(tmp_path, SlipSingleTrainer, _slippery_training_env(SlipModel()))
    scalars = acc.Tags()["scalars"]
    for tag in ("Slip/Count", "Slip/Rate", "Slip/Collision", "Slip/Kind/overshoot", "Slip/Kind/stay", "Slip/Kind/diagonal", "Slip/Kind/drag"):
        assert tag in scalars and len(acc.Scalars(tag)) == 3, tag
    counts = [e.value for e in acc.Scalars("Slip/Count")]
    by_kind = [sum(acc.Scalars(f"Slip/Kind/{k}")[i].value for k in ("overshoot", "stay", "diagonal", "drag")) for i in range(3)]
    assert counts == by_kind and sum(counts) > 0  # all-ice floor: the robot slips
    assert "Config/Slip/text_summary" in acc.Tags()["tensors"]
    assert "Success/Episode" in scalars  # the parent's tags are still there


def test_slip_trainer_logs_nothing_extra_without_slips(tmp_path):
    from tfm_control_ucm.core.training.slip_trainer import SlipSingleTrainer
    acc = _train(tmp_path, SlipSingleTrainer, _slippery_training_env(None))
    assert not any(tag.startswith("Slip/") for tag in acc.Tags()["scalars"])


def test_plain_trainer_does_not_log_slips(tmp_path):
    from tfm_control_ucm.core.training.TrainingManager import SingleTrainer
    acc = _train(tmp_path, SingleTrainer, _slippery_training_env(SlipModel()))
    assert not any(tag.startswith("Slip/") for tag in acc.Tags()["scalars"])


def test_terrain_obs_does_not_change_the_obstacle_distance():
    wall = [GridObstacle((14, 0), (1, 25))]
    def min_dist(mode):
        env = ice_env(None, mapa=SlipperyGridMap("m", (25, 25), grid_obstacles=wall), terrain_obs=mode)
        env.robot.position = (10, 10)
        env.goal_pos = np.array([2, 2])
        return env.step(E)[-1]["min_dist"]
    assert min_dist("neighbors") == min_dist("none")
