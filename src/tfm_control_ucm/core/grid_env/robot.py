"""
robot.py

Defines the Robot class used for navigation, RL, and simulation.
The robot interacts with a GridMap but does not handle rendering.

Author: Martin
"""

from __future__ import annotations
from typing import Tuple, Optional, Union, Iterable

from .utils import Direction
from .map import GridMap
import numpy as np

class GridRobot:
    """
    A simple robot that moves inside a GridMap.

    Parameters
    ----------
    gridmap : GridMap
        The map where the robot moves.
    position : (row, col)
        Initial position of the robot.
    orientation : str
        One of: "N", "S", "E", "W".
    """
    
    ORIENTATIONS = {"N":{"angle":np.pi/2,"vector":(0,1)},"E":{"angle":0.0,"vector":(1,0)},"S":{"angle":-np.pi/2,"vector":(0,-1)},"W":{"angle":np.pi,"vector":(-1,0)}} #["N", "E", "S", "W"]  # clockwise order
    ORIENTATIONS_LIST = ["N","W","S","E"]

    @classmethod
    def get_orientations(cls):
        return cls.ORIENTATIONS

    def __init__(
        self,
        *,
        position: Union[Iterable[int],np.ndarray] = (0, 0),
        orientation: str = "N",
    ):
        if not isinstance(orientation, str) or orientation not in self.ORIENTATIONS: raise ValueError(f"Invalid orientation {orientation}. Must be one of {', '.join(self.ORIENTATIONS.keys())}")
        position = np.asarray(position)
        if position.size != 2: raise ValueError("Invalid size of position, it must be of size 2")
        if not np.issubdtype(position.dtype,np.number): raise TypeError("Invalid value type of the elements of the position")
        if any(position < 0): raise ValueError("All the elements of the position must be positive or equal to zero")

        self.position = position
        self.orientation = orientation
    
    def reset(self, position, orientation: str):
        self.position = position
        self.orientation = orientation

    # ------------------------------------------------------------
    # Orientation control
    # ------------------------------------------------------------
    def turn_left(self):
        idx = self.ORIENTATIONS_LIST.index(self.orientation)
        self.orientation = self.ORIENTATIONS_LIST[(idx + 1) % 4]

    def turn_right(self):
        idx = self.ORIENTATIONS_LIST.index(self.orientation) - 1
        idx = idx if idx >= 0 else 3
        self.orientation = self.ORIENTATIONS_LIST[idx]

    # ------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------
    def forward(self, map: GridMap) -> bool:
        """
        Move forward one cell if possible.
        Returns True if movement succeeded, False if blocked.
        """
        dc, dr = self.ORIENTATIONS[self.orientation]['vector']
        c, r = self.position
        nr, nc = r - dr, c + dc
        moved = False

        if map.is_free(nr, nc):
            self.position = (nc, nr)
            moved = True

        return moved  # collision

    def backward(self, map: GridMap) -> bool:
        """
        Move backward one cell if possible.
        """
        dc, dr = self.ORIENTATIONS[self.orientation]['vector']
        c, r = self.position
        nr, nc = r + dr, c - dc
        moved = False

        if map.is_free(nr, nc):
            self.position = (nc, nr)
            moved = True

        return moved

    # ------------------------------------------------------------
    # Sensors (basic version)
    # ------------------------------------------------------------
    def front_cell(self) -> Tuple[int, int]:
        """Return the coordinates of the cell in front of the robot."""
        dc, dr = self.ORIENTATIONS[self.orientation]['vector']
        c, r = self.position
        return (c + dc, r + dr)

    def is_front_free(self, map: GridMap) -> bool:
        """Check if the cell in front is free."""
        dc, dr = self.ORIENTATIONS[self.orientation]['vector']
        c, r = self.position
        return map.is_free(r+dr, c+dc)

    # ------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------
    def __repr__(self):
        return f"Robot(pos={self.position}, orient='{self.orientation}')"

class GridRobotNotTurning:
    """
    A simple robot that moves inside a GridMap.

    Parameters
    ----------
    gridmap : GridMap
        The map where the robot moves.
    position : (row, col)
        Initial position of the robot.
    orientation : str
        One of: "N", "S", "E", "W".
    """
    
    @classmethod
    def get_orientations(cls):
        return list(Direction.__members__.values())

    def __init__(
        self,
        *,
        position: Union[Iterable[int],np.ndarray] = (0, 0),
    ):
        position = np.asarray(position)
        if position.size != 2: raise ValueError("Invalid size of position, it must be of size 2")
        if not np.issubdtype(position.dtype,np.number): raise TypeError("Invalid value type of the elements of the position")
        if any(position < 0): raise ValueError("All the elements of the position must be positive or equal to zero")

        self.position = position
    
    def reset(self, position):
        self.position = position

    s_directions = list(Direction.__members__.values())
    s_dirs_pos = [dir.value for dir in s_directions]
    s_dirs_names = list(Direction.__members__.keys())
    # ------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------
    def move(self, map: GridMap, direction: str|Direction|int) -> bool:

        if direction and isinstance(direction, str) and direction.lower() not in self.s_dirs_names: raise ValueError(f"Invalid direction {direction}. Must be one of {', '.join(self.s_dirs_names)}")
        if direction and isinstance(direction, Direction) and direction not in Direction: raise ValueError(f"Invalid direction {direction}. Must be one of {', '.join(self.s_dirs_names)}")
        if direction and isinstance(direction, int) and( direction < 0 or direction >= len(self.s_directions)): raise ValueError(f"Invalid direction {direction}. Must be one of {', '.join(str(VAL) for VAL in self.s_dirs_pos)}")

        dc, dr  = direction.value if isinstance(direction, Direction) else self.s_dirs_pos[direction] if isinstance(direction, int) else Direction[direction].value
        c, r = self.position
        nr, nc = r - dr, c + dc # The 0,0 is on the top left corner, so we need to invert the row direction. X is the column, Y is the row
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

