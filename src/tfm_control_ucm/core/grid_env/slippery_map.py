"""
slippery_map.py

A GridMap whose floor has a grip (friction) per cell, for robots that can slip.

GridMap is not changed: SlipperyGridMap adds a separate friction layer, so the
occupancy grid, the lidar and the path planner behave exactly as on a GridMap.

Author: Martin
"""
from __future__ import annotations

import json
import os
from typing import Iterable, List, Optional, Union

import numpy as np
import torch

from .map import GridMap, GridObstacle


class GridTerrain:
    """
    A rectangular patch of floor with its own grip.

    It covers the same cells a GridObstacle with the same corner and size would,
    but it does not block anything.

    Parameters
    ----------
    left_up_corner_position : sequence of 2 ints
        (x, y) = (col, row) of the top-left cell.
    size : sequence of 2 ints
        (width, height) in cells.
    friction : float
        Grip coefficient mu in [0, 1] (adimensional; 1 = normal floor, never slips;
        about 0.5 wet floor, about 0.2 ice). It is not a measured Coulomb coefficient:
        the slip model turns it into a probability (see core/slip.py).
    name : str, optional
        Label of the patch (e.g. "ice").
    """
    def __init__(self, left_up_corner_position: Union[Iterable[int], np.ndarray], size: Union[Iterable[int], np.ndarray],
                 friction: float, name: str = "") -> None:
        corner = np.asarray(left_up_corner_position)
        size_ = np.asarray(size)
        if corner.size != 2: raise ValueError("The input 'left_up_corner_position' must be of size 2")
        if size_.size != 2: raise ValueError("The input 'size' must be of size 2")
        if not np.issubdtype(corner.dtype, np.number) or not np.issubdtype(size_.dtype, np.number):
            raise ValueError("The corner and the size must be numeric")
        if any(size_ <= 0): raise ValueError("All the elements of size must be positive and greater than 0")
        if not 0.0 <= float(friction) <= 1.0: raise ValueError(f"The friction must be in [0, 1], got {friction}")

        self._corner = corner.reshape((2,)).astype(int)
        self._size = size_.reshape((2,)).astype(int)
        self.friction = float(friction)
        self.name = name

    def get_corner(self): return self._corner

    def get_size(self): return self._size

    def get_json_repr(self):
        return {'corner': self._corner.tolist(), 'size': self._size.tolist(), 'friction': self.friction, 'name': self.name}

    def __repr__(self):
        return f"GridTerrain(corner={self._corner.tolist()}, size={self._size.tolist()}, friction={self.friction}, name='{self.name}')"


class SlipperyGridMap(GridMap):
    """
    GridMap with a friction layer.

    Parameters
    ----------
    name, size, grid_obstacles
        As in GridMap.
    terrains : list of GridTerrain, optional
        Patches of floor with a different grip. They fill `friction` (float tensor,
        DEFAULT_FRICTION elsewhere); a later patch overrides an earlier one where
        they overlap. Patches on obstacle cells are allowed and have no effect.

    The JSON file is a GridMap file with an optional "terrains" list, so a plain
    map file loads as a SlipperyGridMap with the default grip everywhere.
    """
    DEFAULT_FRICTION = 1.0

    def __init__(self,
                 name: str,
                 size: Optional[Union[Iterable[int], np.ndarray]],
                 grid_obstacles: Optional[List[GridObstacle]] = None,
                 terrains: Optional[List[GridTerrain]] = None) -> None:
        super().__init__(name, size, grid_obstacles)
        self.friction = torch.full((int(self._height), int(self._width)), self.DEFAULT_FRICTION, dtype=torch.float32, device='cpu')
        self._terrains: List[GridTerrain] = []
        for terrain in terrains or []:
            self.add_terrain(terrain)

    # ------------------------------------------------------------
    # Terrain
    # ------------------------------------------------------------
    def add_terrain(self, terrain: GridTerrain):
        """Paint a patch of floor with its friction (overrides earlier patches where they overlap)."""
        if not isinstance(terrain, GridTerrain): raise TypeError("The terrain must be of type GridTerrain")
        x, y = terrain.get_corner()
        width, height = terrain.get_size()
        if not (x >= 0 and y >= 0 and x + width <= self._width and y + height <= self._height):
            raise IndexError(f"The terrain {terrain} is out of bounds")
        self.friction[y:y+height, x:x+width] = terrain.friction
        self._terrains.append(terrain)

    def get_terrains(self) -> List[GridTerrain]:
        return list(self._terrains)

    @property
    def has_terrain(self) -> bool:
        return bool(self._terrains)

    def friction_at(self, row: int, col: int) -> float:
        """Grip of the cell (DEFAULT_FRICTION outside the map or outside every terrain patch)."""
        if not self.in_bounds(row, col): return self.DEFAULT_FRICTION
        return float(self.friction[row, col])

    def __str__(self) -> str:
        """'.' free, '#' obstacle, '~' free cell with less grip than the default."""
        def symbol(r, c):
            if self.grid[r, c] == self.OBSTACLE: return "#"
            return "~" if float(self.friction[r, c]) < self.DEFAULT_FRICTION else "."
        return "\n".join("".join(symbol(r, c) for c in range(int(self._width))) for r in range(int(self._height)))

    # ------------------------------------------------------------
    # Saving and loading
    # ------------------------------------------------------------
    def save(self, path: str):
        """Save the map to <path>/<name>.json, with the obstacles and the terrain patches."""
        data = {
            "name": self.name,
            "description": "Grid map with terrain friction",
            "size": [int(v) for v in self._size],
            "obstacles": [{"corner": [int(v) for v in o.get_corner()], "size": [int(v) for v in o.get_size()]} for o in self._obstacles],
            "terrains": [terrain.get_json_repr() for terrain in self._terrains],
        }
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, self.name + ".json"), "wt", encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load(cls, path: str) -> SlipperyGridMap:
        """Load a map file; the "terrains" list is optional."""
        with open(path, "r") as f:
            data = json.load(f)
        obstacles = [GridObstacle(o['corner'], o['size']) for o in data["obstacles"]]
        terrains = [GridTerrain(t['corner'], t['size'], t['friction'], t.get('name', '')) for t in data.get("terrains", [])]
        return cls(name=data['name'], size=data["size"], grid_obstacles=obstacles, terrains=terrains)


def friction_at(map: GridMap, row: int, col: int) -> float:
    """Grip of a cell of any GridMap: SlipperyGridMap's friction, the default on a plain GridMap."""
    return map.friction_at(row, col) if isinstance(map, SlipperyGridMap) else SlipperyGridMap.DEFAULT_FRICTION
