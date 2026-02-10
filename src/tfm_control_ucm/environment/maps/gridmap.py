"""
gridmap.py

Defines the GridMap class used to represent a 2D discrete environment
for navigation, reinforcement learning, and robot simulation.

This class is intentionally lightweight and modular so it can be used
inside a Gymnasium environment or independently for planning, sensing,
or visualization.

Author: Martin
"""

from __future__ import annotations
from optparse import Option
from typing import Iterable, List, Tuple, Optional, Union
import json
import numpy as np

SizeType = Union[Iterable[int],np.ndarray]

class GridMap:
    """
    A simple 2D grid map representation.

    Parameters
    ----------
    width : int, optional
        Number of columns.
    height : int, optional
        Number of rows.
    size : tuple/list/np.ndarray of two ints, optional
        Alternative way to specify (width, height).
        Overrides width/height if provided.
    obstacles : list of (row, col), optional
        Coordinates of obstacles.
    """


    FREE = 0
    OBSTACLE = 1

    def __init__(self, *, width: Optional[int]=None, height: Optional[int]=None, size:Optional[SizeType]=None, obstacles: Optional[List[SizeType]]) -> None:
        self.width, self.height = self._resolve_size(width, height, size)
        self.size = np.array([self.height, self.width])
        self.grid = np.ones(self.size,dtype=np.int8) * self.FREE

        if obstacles:
            for r, c in obstacles:
                self.set_obstacle(r, c)

    
    def _resolve_size(
        self,
        width: Optional[int],
        height: Optional[int],
        size: Optional[SizeType],
    ) -> Tuple[int, int]:
        """
        Resolve width and height from either:
        - width + height
        - size = (width, height)
        """

        if size is not None:
            arr = np.array(size, dtype=int)
            if arr.size != 2:
                raise ValueError("size must contain exactly two integers: (width, height)")
            w, h = int(arr[0]), int(arr[1])
            if w <= 0 or h <= 0:
                raise ValueError("width and height must be positive integers")
            return w, h

        # If size is not provided, use width/height
        if width is None or height is None:
            raise ValueError("You must specify either size=(w,h) or both width and height")

        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive integers")

        return width, height

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def is_free(self, row: int, col: int) -> bool:
        return self.in_bounds(row, col) and self.grid[row, col] == self.FREE

    def is_obstacle(self, row: int, col: int) -> bool:
        return self.in_bounds(row, col) and self.grid[row, col] == self.OBSTACLE

    def set_obstacle(self, row: int, col: int):
        if self.in_bounds(row, col):
            self.grid[row, col] = self.OBSTACLE

    def clear_obstacle(self, row: int, col: int):
        if self.in_bounds(row, col):
            self.grid[row, col] = self.FREE
    
    def to_numpy(self) -> np.ndarray:
        return self.grid.copy()

    def __repr__(self) -> str:
        return f"GridMap(width={self.width}, height={self.height})"

    def __str__(self) -> str:
        symbols = {self.FREE: ".", self.OBSTACLE: "#"}
        return "\n".join("".join(symbols[cell] for cell in row) for row in self.grid) # type: ignore

    # ------------------------------------------------------------
    # Saving and loading maps
    # ------------------------------------------------------------
    def save(self, path: str):
        """
        Save the grid map to a JSON file.

        The JSON structure is:
        {
            "width": int,
            "height": int,
            "obstacles": [[r1, c1], [r2, c2], ...]
        }
        """
        obstacles = [
            [int(r), int(c)]
            for r in range(self.height)
            for c in range(self.width)
            if self.grid[r, c] == self.OBSTACLE
        ]

        data = {
            "width": self.width,
            "height": self.height,
            "obstacles": obstacles,
        }

        with open(path, "wt") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load(cls, path: str) -> GridMap:
        """
        Load a grid map from a JSON file.
        """
        with open(path, "r") as f:
            data = json.load(f)

        width = data["width"]
        height = data["height"]
        obstacles = [tuple(obs) for obs in data["obstacles"]]

        return cls(width=width, height=height, obstacles=obstacles) # type: ignore
