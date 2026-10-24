import pytest

from tfm_control_ucm.core.grid_env.map import GridMap, GridObstacle


def test_empty_map_is_free():
    mapa = GridMap("empty", (10, 8))
    assert mapa.get_width() == 10 and mapa.get_height() == 8
    assert int(mapa.grid.sum()) == 0


def test_append_obstacle_marks_cells():
    mapa = GridMap("m", (10, 10))
    mapa.append_obstacle(GridObstacle((2, 3), (4, 2)))  # x=2..5, y=3..4
    assert mapa.is_obstacle(3, 2) and mapa.is_obstacle(4, 5)
    assert mapa.is_free(5, 2) and mapa.is_free(3, 6)
    assert int(mapa.grid.sum()) == 8


def test_append_obstacle_rejects_overlap():
    mapa = GridMap("m", (10, 10))
    mapa.append_obstacle(GridObstacle((2, 2), (3, 3)))
    with pytest.raises(ValueError):
        mapa.append_obstacle(GridObstacle((4, 4), (2, 2)))


def test_append_obstacle_rejects_out_of_bounds():
    mapa = GridMap("m", (10, 10))
    with pytest.raises(IndexError):
        mapa.append_obstacle(GridObstacle((8, 8), (3, 3)))


def test_obstacle_touching_bottom_right_edge_is_in_bounds():
    mapa = GridMap("m", (10, 10))
    mapa.append_obstacle(GridObstacle((8, 8), (2, 2)))  # cells 8..9, last valid index
    assert mapa.is_obstacle(9, 9)


def test_pop_obstacle_frees_cells():
    mapa = GridMap("m", (10, 10))
    mapa.append_obstacle(GridObstacle((1, 1), (2, 2)))
    mapa.append_obstacle(GridObstacle((5, 5), (2, 2)))
    popped = mapa.pop_obstacle(0)
    assert popped is not None
    assert mapa.is_free(1, 1)
    assert mapa.is_obstacle(5, 5)
    assert int(mapa.grid.sum()) == 4


def test_clearing_one_of_two_overlapping_obstacles_keeps_the_other():
    a = GridObstacle((0, 0), (4, 4))
    b = GridObstacle((2, 2), (4, 4))
    mapa = GridMap("m", (10, 10), grid_obstacles=[a, b])
    mapa.clear_obstacle(a)
    assert mapa.is_obstacle(2, 2)  # shared cell still belongs to b
    assert mapa.is_free(0, 0)


def test_str_draws_free_cells_and_obstacles():
    mapa = GridMap("m", (3, 2), grid_obstacles=[GridObstacle((1, 0), (1, 1))])
    assert str(mapa) == ".#.\n..."


def test_constructor_does_not_alias_caller_list():
    obstacles = [GridObstacle((0, 0), (2, 2))]
    mapa = GridMap("m", (10, 10), grid_obstacles=obstacles)
    mapa.append_obstacle(GridObstacle((5, 5), (2, 2)))
    assert len(obstacles) == 1
