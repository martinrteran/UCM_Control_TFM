"""
robot.py

Defines the Robot class used for navigation, RL, and simulation.
The robot interacts with a GridMap but does not handle rendering.

Author: Martin
"""

from __future__ import annotations
from typing import Tuple, Optional
from .gridmap import GridMap


class Grid_Robot:
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

    ORIENTATIONS = ["N", "E", "S", "W"]  # clockwise order

    # Movement vectors for each orientation
    MOVE_VECTOR = {
        "N": (-1, 0),
        "E": (0, 1),
        "S": (1, 0),
        "W": (0, -1),
    }

    def __init__(
        self,
        *,
        gridmap: GridMap,
        position: Tuple[int, int] = (0, 0),
        orientation: str = "N",
    ):
        self.gridmap = gridmap
        self.position = position
        self.orientation = orientation

        if orientation not in self.ORIENTATIONS:
            raise ValueError(f"Invalid orientation {orientation}. Must be one of {self.ORIENTATIONS}")

        if not self.gridmap.is_free(*position):
            raise ValueError(f"Initial position {position} is not free")

    # ------------------------------------------------------------
    # Orientation control
    # ------------------------------------------------------------
    def turn_left(self):
        idx = self.ORIENTATIONS.index(self.orientation)
        self.orientation = self.ORIENTATIONS[(idx - 1) % 4]

    def turn_right(self):
        idx = self.ORIENTATIONS.index(self.orientation)
        self.orientation = self.ORIENTATIONS[(idx + 1) % 4]

    # ------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------
    def forward(self) -> bool:
        """
        Move forward one cell if possible.
        Returns True if movement succeeded, False if blocked.
        """
        dr, dc = self.MOVE_VECTOR[self.orientation]
        r, c = self.position
        nr, nc = r + dr, c + dc

        if self.gridmap.is_free(nr, nc):
            self.position = (nr, nc)
            return True

        return False  # collision

    def backward(self) -> bool:
        """
        Move backward one cell if possible.
        """
        dr, dc = self.MOVE_VECTOR[self.orientation]
        r, c = self.position
        nr, nc = r - dr, c - dc

        if self.gridmap.is_free(nr, nc):
            self.position = (nr, nc)
            return True

        return False

    # ------------------------------------------------------------
    # Sensors (basic version)
    # ------------------------------------------------------------
    def front_cell(self) -> Tuple[int, int]:
        """Return the coordinates of the cell in front of the robot."""
        dr, dc = self.MOVE_VECTOR[self.orientation]
        r, c = self.position
        return (r + dr, c + dc)

    def is_front_free(self) -> bool:
        """Check if the cell in front is free."""
        return self.gridmap.is_free(*self.front_cell())

    # ------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------
    def __repr__(self):
        return f"Robot(pos={self.position}, orient='{self.orientation}')"