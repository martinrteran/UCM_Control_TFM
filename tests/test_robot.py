import numpy as np
import pytest

from tfm_control_ucm.core.grid_env.map import GridMap, GridObstacle
from tfm_control_ucm.core.grid_env.robot import GridRobot, GridRobotNotTurning
from tfm_control_ucm.core.grid_env.utils import Orientation


@pytest.fixture
def mapa():
    return GridMap("m", (10, 10))


# ------------------------------------------------------------
# GridRobot (egocentric: forward, turn left, turn right)
# ------------------------------------------------------------
def test_four_left_turns_return_to_start():
    robot = GridRobot(position=(5, 5), orientation=Orientation.east)
    for _ in range(4):
        robot.turn_left()
    assert robot.orientation == Orientation.east


def test_left_then_right_is_identity():
    robot = GridRobot(position=(5, 5))
    robot.turn_left()
    robot.turn_right()
    assert robot.orientation == Orientation.north


def test_turn_left_from_north_faces_west():
    robot = GridRobot(position=(5, 5))
    robot.turn_left()
    assert robot.orientation == Orientation.west


def test_forward_north_decreases_row(mapa):
    robot = GridRobot(position=(5, 5))
    assert robot.move(mapa, 0) is True
    c, r = robot.position
    assert (c, r) == (5, 4)


def test_forward_into_wall_is_blocked(mapa):
    robot = GridRobot(position=(5, 0))  # top row, facing north
    assert robot.move(mapa, 0) is False
    assert tuple(robot.position) == (5, 0)


def test_forward_into_obstacle_is_blocked(mapa):
    mapa.append_obstacle(GridObstacle((5, 4), (1, 1)))
    robot = GridRobot(position=(5, 5))
    assert robot.move(mapa, 0) is False


def test_move_accepts_numpy_integer_actions(mapa):
    robot = GridRobot(position=(5, 5))
    assert robot.move(mapa, np.int8(1)) is True
    assert robot.orientation == Orientation.west


@pytest.mark.parametrize("action", [-1, 3, 7])
def test_move_rejects_out_of_range_actions(mapa, action):
    robot = GridRobot(position=(5, 5))
    with pytest.raises(ValueError):
        robot.move(mapa, action)


def test_invalid_orientation_raises_value_error():
    with pytest.raises(ValueError):
        GridRobot(position=(0, 0), orientation=Orientation.north_east)


# ------------------------------------------------------------
# GridRobotNotTurning (absolute: up, right, down, left)
# ------------------------------------------------------------
@pytest.mark.parametrize("action, expected", [(0, (5, 4)), (1, (6, 5)), (2, (5, 6)), (3, (4, 5))])
def test_not_turning_moves_in_each_direction(mapa, action, expected):
    robot = GridRobotNotTurning(position=(5, 5))
    assert robot.move(mapa, action) is True
    assert tuple(robot.position) == expected


def test_not_turning_accepts_numpy_integer_actions(mapa):
    robot = GridRobotNotTurning(position=(5, 5))
    assert robot.move(mapa, np.int8(1)) is True
    assert tuple(robot.position) == (6, 5)


@pytest.mark.parametrize("action", [-1, 4])
def test_not_turning_rejects_out_of_range_actions(mapa, action):
    robot = GridRobotNotTurning(position=(5, 5))
    with pytest.raises(ValueError):
        robot.move(mapa, action)


def test_not_turning_blocked_by_wall(mapa):
    robot = GridRobotNotTurning(position=(0, 0))
    assert robot.move(mapa, 3) is False
    assert tuple(robot.position) == (0, 0)
