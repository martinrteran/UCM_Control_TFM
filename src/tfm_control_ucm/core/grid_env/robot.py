"""
robot.py

Defines the Robot classes used for navigation, RL, and simulation.
The agents interact with a GridMap for state transitions but do not handle rendering.

Author: Martin
"""

from __future__ import annotations
from typing import Any, Tuple, Optional, Union, Iterable

from .utils import Direction, DirectionType, Orientation, OrientationType
from .map import GridMap
import numpy as np
from abc import ABC, abstractmethod

class Robot(ABC):
    """
    Abstract base class for a robot agent in a grid environment.

    Manages the internal spatial state (position) of the agent and enforces
    the implementation of a movement dynamic.
    """
    position: Union[Iterable[int], np.ndarray]

    def __init__(self, position: Union[Iterable[int], np.ndarray]):
        """
        Initialize the robot's spatial state.

        Parameters
        ----------
        position : Union[Iterable[int], np.ndarray]
            The initial (col, row) or (x, y) position of the robot.

        Raises
        ------
        ValueError
            If the position array does not have exactly 2 elements, or if any element is negative.
        TypeError
            If the elements of the position are not numeric.
        """
        position = np.asarray(position)
        if position.size != 2: raise ValueError("Invalid size of position, it must be of size 2")
        if not np.issubdtype(position.dtype, np.number): raise TypeError("Invalid value type of the elements of the position")
        if any(position < 0): raise ValueError("All the elements of the position must be positive or equal to zero")

        self.position = position
    
    def reset(self, position: Union[Iterable[int], np.ndarray]):
        """
        Reset the robot to a specific position.

        Parameters
        ----------
        position : Union[Iterable[int], np.ndarray]
            The new position coordinates.
        """
        self.position = position

    @property
    def heading(self) -> float:
        """
        Direction the robot faces, in radians, in the lidar convention:
        0 = east (+col), pi/2 = north (-row), counter-clockwise positive.
        Robots without orientation face east (0).
        """
        return 0.0

    def random_reset(self, map: GridMap, rng: np.random.Generator):
        """
        Place the robot on a uniformly random free cell of the map.

        Parameters
        ----------
        map : GridMap
            Map to place the robot on.
        rng : np.random.Generator
            Random generator (the environment's, so reset(seed=...) is reproducible).
        """
        height, width = map.grid.shape
        while True:
            r, c = int(rng.integers(height)), int(rng.integers(width))
            if map.is_free(r, c):
                self.reset((c, r))
                return

    @abstractmethod
    def move(self, map: GridMap, direction: Any) -> bool:
        """
        Execute a movement step within the given map.

        Parameters
        ----------
        map : GridMap
            The environment map used for collision checking.
        direction : Any
            The direction or action command to execute.

        Returns
        -------
        bool
            True if the movement was successful, False if blocked or invalid.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    # Declaración del tipo para linters (mypy)
    NUM_ACTIONS: int 

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, 'NUM_ACTIONS'):
            raise TypeError(f"La subclase {cls.__name__} debe definir el atributo de clase 'NUM_ACTIONS'.")


class GridRobot(Robot):
    """
    A grid-based robot that maintains an internal orientation.

    Uses egocentric actions (e.g., move forward, turn left, turn right).

    Parameters
    ----------
    position : Union[Iterable[int], np.ndarray], optional
        Initial position of the robot. Default is (0, 0).
    orientation : Orientation, optional
        Initial orientation. Default is Orientation.north.
    """
    
    ORIENTATIONS_LIST = [Orientation.north, Orientation.west, Orientation.south, Orientation.east]
    NUM_ACTIONS = 3  # Forward, Turn Left, Turn Right
    
    def __init__(
        self,
        *,
        position: Union[Iterable[int], np.ndarray] = (0, 0),
        orientation: Orientation = Orientation.north,
    ):
        super().__init__(position)
        if orientation not in self.ORIENTATIONS_LIST:
            raise ValueError(f"Invalid orientation {orientation}. Must be one of {', '.join(o.name for o in self.ORIENTATIONS_LIST)}")
        
        self.orientation = orientation
        self._orientation_idx = self.ORIENTATIONS_LIST.index(orientation)
    
    def reset(self, position, orientation: Orientation = Orientation.north):
        """
        Reset the robot's position and orientation.

        Parameters
        ----------
        position : Union[Iterable[int], np.ndarray]
            The new position coordinates.
        orientation : Orientation, optional
            The new orientation. Default is Orientation.north.
        """
        self.position = position
        self.orientation = orientation
        self._orientation_idx = self.ORIENTATIONS_LIST.index(orientation)

    def random_reset(self, map: GridMap, rng: np.random.Generator):
        """Place the robot on a random free cell, facing a random cardinal direction."""
        super().random_reset(map, rng)
        self.reset(self.position, self.ORIENTATIONS_LIST[int(rng.integers(len(self.ORIENTATIONS_LIST)))])

    @property
    def heading(self) -> float:
        dc, dr = self.orientation.get_value()
        return float(np.arctan2(-dr, dc)) # Rows grow downwards, the lidar convention is counter-clockwise

    # ------------------------------------------------------------
    # Orientation control
    # ------------------------------------------------------------
    def turn_left(self):
        """Rotate the robot 90 degrees to the left (counter-clockwise)."""
        self._orientation_idx = (self._orientation_idx + 1) % 4
        self.orientation = self.ORIENTATIONS_LIST[self._orientation_idx]

    def turn_right(self):
        """Rotate the robot 90 degrees to the right (clockwise)."""
        self._orientation_idx = (self._orientation_idx - 1) % 4
        self.orientation = self.ORIENTATIONS_LIST[self._orientation_idx]

    # ------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------
    def forward(self, map: GridMap) -> bool:
        """
        Move forward one cell in the current orientation if the cell is free.

        Parameters
        ----------
        map : GridMap
            The environment map used for collision checking.

        Returns
        -------
        bool
            True if the movement succeeded, False if blocked by an obstacle.
        """
        dc, dr = self.orientation.get_value()
        c, r = self.position
        nr, nc = r + dr, c + dc
        moved = False

        if map.is_free(nr, nc):
            self.position = (nc, nr)
            moved = True

        return moved  # collision

    # ------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------
    def move(self, map: GridMap, direction: int) -> bool:
        """
        Execute an egocentric action.

        Parameters
        ----------
        map : GridMap
            The environment map used for collision checking.
        direction : int
            The discrete action to perform:
            - 0: Move forward
            - 1: Turn left
            - 2: Turn right

        Returns
        -------
        bool
            True if an action was executed (always True for turns; based on collision for forward).

        Raises
        ------
        ValueError
            If the direction is not an integer in [0, NUM_ACTIONS).
        """
        if not isinstance(direction, (int, np.integer)) or not 0 <= direction < self.NUM_ACTIONS:
            raise ValueError(f"Invalid direction {direction}. Must be an integer in [0, {self.NUM_ACTIONS})")

        if direction == 0:
            return self.forward(map)
        elif direction == 1:
            self.turn_left()
        else:
            self.turn_right()
        
        return True

    # ------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------
    def __repr__(self):
        return f"Robot(pos={self.position}, orient='{self.orientation}')"


class GridRobotNotTurning(Robot):
    """
    A grid-based robot that moves absolutely without maintaining orientation.

    Uses absolute directional actions (e.g., North, South, East, West).

    Parameters
    ----------
    position : Union[Iterable[int], np.ndarray], optional
        Initial position of the robot. Default is (0, 0).
    """
    
    NUM_ACTIONS = 4  # North, South, East, West

    def __init__(
        self,
        *,
        position: Union[Iterable[int], np.ndarray] = (0, 0),
    ):
        super().__init__(position)

    s_directions = list(Direction.__members__.values())
    s_dirs_pos = [dir.value for dir in s_directions]
    s_dirs_names = list(Direction.__members__.keys())
    
    # ------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------
    def move(self, map: GridMap, direction: DirectionType) -> bool:
        """
        Translate the robot one cell in the specified absolute direction.

        Parameters
        ----------
        map : GridMap
            The environment map used for collision checking.
        direction : DirectionType
            The target direction to move. Supports string names, Direction enums,
            integer indices, or coordinate tuples.

        Returns
        -------
        bool
            True if the movement succeeded, False if blocked by an obstacle.

        Raises
        ------
        ValueError
            If the provided direction cannot be parsed or is out of bounds.
        """
        if direction and isinstance(direction, str) and direction.lower() not in self.s_dirs_names: 
            raise ValueError(f"Invalid direction {direction}. Must be one of {', '.join(self.s_dirs_names)}")
        if direction and isinstance(direction, Direction) and direction not in Direction: 
            raise ValueError(f"Invalid direction {direction}. Must be one of {', '.join(self.s_dirs_names)}")
        if isinstance(direction, np.integer): direction = int(direction) # Actions sampled from gym spaces are numpy integers
        if isinstance(direction, int) and (direction < 0 or direction >= len(self.s_directions)):
            raise ValueError(f"Invalid direction {direction}. Must be an integer in [0, {len(self.s_directions)})")

        dc, dr  = direction.value if isinstance(direction, Direction) else self.s_dirs_pos[direction] if isinstance(direction, int) else Direction[direction].value if isinstance(direction, str) else direction
        c, r = self.position
        
        # The 0,0 is on the top left corner, so we need to invert the row direction. X is the column, Y is the row
        nr, nc = r + dr, c + dc 
        moved = False

        if map.is_free(nr, nc):
            self.position = (nc, nr)
            moved = True

        return moved  # collision

    # ------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------
    def __repr__(self):
        return f"Robot(pos={self.position})"