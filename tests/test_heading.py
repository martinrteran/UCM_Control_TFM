"""
Egocentric observation: lidar and goal angles share one convention
(0 = where the robot faces, counter-clockwise positive) for every robot.
"""
import numpy as np
import pytest

from tfm_control_ucm.core.grid_env.environment import Grid_Robot_Sections_Env, Grid_Robot_Sections_Env_MultiMap
from tfm_control_ucm.core.grid_env.map import GridMap, GridObstacle
from tfm_control_ucm.core.grid_env.robot import GridRobot, GridRobotNotTurning
from tfm_control_ucm.core.grid_env.sensor import GridLidar
from tfm_control_ucm.core.grid_env.utils import Orientation

LIDAR = {"num_rays": 36, "max_range": 20, "fov": float(2 * np.pi), "noise_std": 0.0, "with_cache": True, "device": "cpu"}


def zero_reward(ctx):
    return 0.0


def nearest_ray_angle(scan):
    """Circular mean of the angles of the rays that share the shortest distance."""
    scan = scan.cpu().numpy()
    nearest = scan[np.isclose(scan[:, 0], scan[:, 0].min()), 1]
    return np.angle(np.exp(1j * nearest).mean())


def angular_gap(a, b):
    return abs(np.angle(np.exp(1j * (a - b))))


ROBOT = (15, 15)


def wall_north_map():
    """30x30 map with a horizontal wall on row 12, 3 cells north of ROBOT; borders are 15 cells away."""
    return GridMap("wall", (30, 30), grid_obstacles=[GridObstacle((0, 12), (30, 1))])


@pytest.mark.parametrize("orientation, heading", [
    (Orientation.east, 0.0), (Orientation.north, np.pi / 2), (Orientation.west, np.pi), (Orientation.south, -np.pi / 2),
])
def test_grid_robot_heading(orientation, heading):
    robot = GridRobot(position=(0, 0), orientation=orientation)
    assert np.isclose(np.cos(robot.heading), np.cos(heading)) and np.isclose(np.sin(robot.heading), np.sin(heading))


def test_robot_without_orientation_faces_east():
    assert GridRobotNotTurning().heading == 0.0


@pytest.mark.parametrize("with_cache", [True, False])
@pytest.mark.parametrize("heading, expected", [(0.0, np.pi / 2), (np.pi / 2, 0.0), (np.pi, -np.pi / 2)])
def test_lidar_angles_are_relative_to_heading(with_cache, heading, expected):
    lidar = GridLidar(**{**LIDAR, "with_cache": with_cache})
    angle = nearest_ray_angle(lidar.scan(wall_north_map(), ROBOT, heading))
    assert angular_gap(angle, expected) < 0.2


def test_cached_and_uncached_lidar_agree():
    mapa = wall_north_map()
    cached = GridLidar(**LIDAR).scan(mapa, ROBOT, 0.0).cpu().numpy()
    direct = GridLidar(**{**LIDAR, "with_cache": False}).scan(mapa, ROBOT, 0.0).cpu().numpy()
    assert np.allclose(cached[:, 0], direct[:, 0])
    assert np.all(angular_gap(cached[:, 1], direct[:, 1]) < 1e-5)


@pytest.mark.parametrize("robot", [GridRobotNotTurning(), GridRobot(orientation=Orientation.north)])
def test_goal_and_obstacle_in_the_same_direction_get_the_same_angle(robot):
    env = Grid_Robot_Sections_Env(map=wall_north_map(), robot=robot, reward_fn=zero_reward,
                                  lidar_config=LIDAR, num_sections=8)
    env.reset(seed=0)
    env.robot.position = ROBOT
    if isinstance(robot, GridRobot):
        env.robot.reset(ROBOT, Orientation.north)
    env.goal_pos = np.array([15, 5])  # straight north, beyond the wall
    goal_angle = env._angle_to_goal()
    wall_angle = nearest_ray_angle(env.lidar.scan(env.map, env.robot.position, env.robot.heading))
    assert angular_gap(goal_angle, wall_angle) < 0.2
    assert np.isclose(goal_angle, 0.0 if isinstance(robot, GridRobot) else np.pi / 2)


def test_grid_robot_random_reset_varies_orientation():
    env = Grid_Robot_Sections_Env(map=GridMap("m", (25, 25)), robot=GridRobot(), reward_fn=zero_reward, lidar_config=LIDAR)
    orientations = set()
    env.reset(seed=1)
    for _ in range(40):
        env.reset()
        orientations.add(env.robot.orientation)
    assert orientations == set(GridRobot.ORIENTATIONS_LIST)


def test_seeded_reset_is_reproducible():
    def episode_start(seed):
        env = Grid_Robot_Sections_Env(map=GridMap("m", (25, 25)), robot=GridRobot(), reward_fn=zero_reward, lidar_config=LIDAR)
        obs, _ = env.reset(seed=seed)
        return tuple(env.robot.position), env.robot.orientation, tuple(env.goal_pos), obs

    a, b = episode_start(3), episode_start(3)
    assert a[:3] == b[:3] and np.allclose(a[3], b[3])


def test_reset_info_has_map_and_optimal_steps():
    env = Grid_Robot_Sections_Env(map=GridMap("m", (25, 25)), robot=GridRobotNotTurning(), reward_fn=zero_reward, lidar_config=LIDAR)
    _, info = env.reset(seed=0)
    (c, r), (gc, gr) = env.robot.position, env.goal_pos
    assert info["map"] == "m"
    assert info["optimal_steps"] == abs(gc - c) + abs(gr - r)  # empty map: Manhattan distance


def test_renderer_is_dropped_when_the_map_changes():
    maps = [GridMap("a", (25, 25)), GridMap("b", (25, 25))]
    env = Grid_Robot_Sections_Env_MultiMap(maps=maps, robot=GridRobotNotTurning(), reward_fn=zero_reward,
                                           lidar_config=LIDAR, random_map=False)
    env.reset()
    env.renderer = object()  # stands in for a pygame window
    env.reset()
    assert env.renderer is None


@pytest.mark.parametrize("num_sections", [2, 6])
def test_invalid_number_of_sections_is_rejected(num_sections):
    with pytest.raises(ValueError):
        Grid_Robot_Sections_Env(map=GridMap("m", (25, 25)), robot=GridRobotNotTurning(), reward_fn=zero_reward,
                                lidar_config=LIDAR, num_sections=num_sections)
