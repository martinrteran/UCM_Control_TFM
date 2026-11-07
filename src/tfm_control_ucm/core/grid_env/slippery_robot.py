"""
slippery_robot.py

Robots that can slip on low-grip floors (SlipperyGridMap + core/slip.py).

The robots of robot.py are not changed. SlipperyRobotMixin adds, on top of any
grid robot, the momentum (last displacement, (0, 0) when stopped) and a move()
that asks a SlipModel what happens:

- turning in place, moving into a wall or having no slip model: the parent's
  move(), unchanged (moving into a wall is a plain collision);
- otherwise the slip model picks the outcome (overshoot, stay, diagonal, drag,
  lateral or none) and each displacement of its path must land on a free cell.
  A slide that hits something is a collision when into_obstacle is "collide"
  (the robot stays on the last free cell either way).

move() returns False only when the robot hit a wall or an obstacle, so the
environment's "hit_obstacle = not moved" still means collision: a wheel spin
("stay") returns True although the robot did not move.

The outcome of the last move is in last_slip, last_slip_collision and
last_friction. Slips are drawn from the robot's own generator (set_rng): the
slippery environment seeds it from reset(seed=...), apart from the generator
that picks maps and positions.

Author: Martin
"""
from __future__ import annotations

from typing import Any, Optional, Tuple, Union, Iterable

import numpy as np

from tfm_control_ucm.core.slip import NO_SLIP, SlipEvent, SlipModel
from .map import GridMap
from .robot import GridRobot, GridRobotNotTurning
from .slippery_map import friction_at


class SlipperyRobotMixin:
    """
    Momentum and slipping for a grid robot (put it before the robot class in the bases).

    Parameters
    ----------
    slip_model : SlipModel, optional
        None keeps the parent's deterministic moves but still tracks the momentum
        (useful to give the same observation without slips).
    rng : np.random.Generator, optional
        Generator of the slips (unseeded by default; the slippery environment sets it at reset).
    **kwargs
        Arguments of the robot class (e.g. position).
    """
    momentum: Tuple[int, int]
    last_slip: SlipEvent
    last_slip_collision: bool
    last_friction: float

    def __init__(self, *, slip_model: Optional[SlipModel] = None, rng: Optional[np.random.Generator] = None, **kwargs):
        super().__init__(**kwargs)
        if slip_model is not None and not isinstance(slip_model, SlipModel): raise TypeError("The slip_model must be of type SlipModel")
        self.slip_model = slip_model
        self._rng = rng if rng is not None else np.random.default_rng()
        self._clear_dynamics()

    def _clear_dynamics(self):
        self.momentum = (0, 0) # Stopped
        self.last_slip = NO_SLIP
        self.last_slip_collision = False
        self.last_friction = 1.0

    def set_rng(self, rng: np.random.Generator):
        """Generator the slips are drawn from."""
        self._rng = rng

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs) # type: ignore
        self._clear_dynamics()

    def random_reset(self, map: GridMap, rng: np.random.Generator):
        super().random_reset(map, rng) # type: ignore
        self._clear_dynamics()

    def displacement(self, action: Any) -> Tuple[int, int]:
        """World displacement (dc, dr) the action intends; (0, 0) if it does not translate the robot."""
        raise NotImplementedError

    def _cell(self) -> Tuple[int, int]:
        c, r = (int(v) for v in self.position) # type: ignore
        return c, r

    def _translate(self, map: GridMap, dc: int, dr: int) -> bool:
        """Move by (dc, dr) without turning if the target cell is free."""
        c, r = self._cell()
        if not map.is_free(r + dr, c + dc): return False
        self.position = (c + dc, r + dr)
        return True

    def move(self, map: GridMap, direction: Any) -> bool:
        """
        Execute an action, slipping depending on the grip of the current cell.

        Returns
        -------
        bool
            False if the robot hit a wall or an obstacle (also a slide into one with
            into_obstacle="collide"), True otherwise, even if a wheel spin kept it in place.
        """
        c, r = self._cell()
        self.last_friction = friction_at(map, r, c)
        self.last_slip, self.last_slip_collision = NO_SLIP, False

        intended = self.displacement(direction) if isinstance(direction, (int, np.integer)) else None
        if (self.slip_model is None or intended is None or intended == (0, 0)
                or not map.is_free(r + intended[1], c + intended[0])):
            moved = super().move(map, direction) # type: ignore
            nc, nr = self._cell()
            self.momentum = (nc - c, nr - r) # (0, 0) after a collision or a turn in place
            return moved

        event = self.slip_model.sample(self.last_friction, self.momentum, intended, self._rng)
        self.last_slip = event
        for dc, dr in event.path:
            if not self._translate(map, dc, dr):
                self.momentum = (0, 0) # Stopped against the obstacle
                self.last_slip_collision = self.slip_model.into_obstacle == "collide"
                return not self.last_slip_collision
        self.momentum = event.momentum
        return True


class SlipperyGridRobotNotTurning(SlipperyRobotMixin, GridRobotNotTurning):
    """
    GridRobotNotTurning (absolute N/S/E/W moves, heading 0) that can slip.

    Parameters
    ----------
    position : sequence of 2 ints, optional
        Initial (col, row).
    slip_model : SlipModel, optional
        See SlipperyRobotMixin.
    rng : np.random.Generator, optional
        See SlipperyRobotMixin.
    """
    def __init__(self, *, position: Union[Iterable[int], np.ndarray] = (0, 0),
                 slip_model: Optional[SlipModel] = None, rng: Optional[np.random.Generator] = None):
        super().__init__(position=position, slip_model=slip_model, rng=rng)

    def displacement(self, action: int) -> Tuple[int, int]:
        if not isinstance(action, (int, np.integer)) or not 0 <= action < self.NUM_ACTIONS:
            raise ValueError(f"Invalid direction {action}. Must be an integer in [0, {self.NUM_ACTIONS})")
        dc, dr = self.s_dirs_pos[int(action)]
        return (int(dc), int(dr))

    def __repr__(self):
        return f"SlipperyRobot(pos={self.position}, momentum={self.momentum})"


class SlipperyGridRobot(SlipperyRobotMixin, GridRobot):
    """
    GridRobot (forward / turn left / turn right) that can slip.

    Turning in place stops the robot (momentum (0, 0)), so its next forward move is
    a "start"; consecutive forward moves are "straight". Slides keep the orientation.

    Parameters
    ----------
    position : sequence of 2 ints, optional
        Initial (col, row).
    orientation : Orientation, optional
        Initial orientation.
    slip_model : SlipModel, optional
        See SlipperyRobotMixin.
    rng : np.random.Generator, optional
        See SlipperyRobotMixin.
    """
    def __init__(self, *, position: Union[Iterable[int], np.ndarray] = (0, 0), slip_model: Optional[SlipModel] = None,
                 rng: Optional[np.random.Generator] = None, **kwargs):
        super().__init__(position=position, slip_model=slip_model, rng=rng, **kwargs)

    def displacement(self, action: int) -> Tuple[int, int]:
        if not isinstance(action, (int, np.integer)) or not 0 <= action < self.NUM_ACTIONS:
            raise ValueError(f"Invalid direction {action}. Must be an integer in [0, {self.NUM_ACTIONS})")
        if action != 0: return (0, 0)
        dc, dr = self.orientation.get_value()
        return (int(dc), int(dr))

    def __repr__(self):
        return f"SlipperyRobot(pos={self.position}, orient='{self.orientation}', momentum={self.momentum})"
