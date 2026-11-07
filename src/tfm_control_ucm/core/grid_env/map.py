from __future__ import annotations
import numpy as np
import torch
from typing import Sequence, Iterable,Union, Optional, List
import json

class GridObstacle:
    """
    A simple 2D rectangular obstacle for grid maps, where the amount of spaces is specified by its left_up corner position and the width and height

    # Parameters
    - left_up_corner_position: sequential of dim (2,) with the position of the left-up corner of the obstacle
    - size: sequential of dim (2,) with the width and height
    """
    def __init__(self, left_up_corner_position:Union[Iterable[int],np.ndarray], size:Union[Iterable[int],np.ndarray]) -> None:
        corner = np.asarray(left_up_corner_position)
        size_ = np.asarray(size)
        
        if corner.size != 2: raise ValueError(f"The input 'left_up_corner_position' must be of size 2")
        if size_.size != 2: raise ValueError(f"The input 'size' must be of size 2")
        if not np.issubdtype(corner.dtype,np.number): raise ValueError(f"The data type of all the elements of 'left_up_corner_position' must be numeric")
        if not np.issubdtype(size_.dtype,np.number): raise ValueError(f"The data type of all the elements of 'size' must be numeric")
        if any(size_ <= 0): raise ValueError(f"All the elements of size must be positive and greater than 0")

        size_ = size_.reshape((2,))
        corner = corner.reshape((2,))

        self._corner = corner
        self._size = size_
        self.p_corners = np.array([self._corner, self._corner + [self._size[0], 0], self._corner + self._size, self._corner + [0, self._size[1]]])
        self.p_grid = np.ones(self._size)
    
    def get_corner(self): return self._corner

    def get_size(self): return self._size
    
    def get_corners(self):  return self.p_corners

    def get_json_repr(self):  return {'corner':self._corner.tolist(),'size':self._size.tolist()}

    def __repr__(self):  return f'{{"corner":[{self._corner[0]},{self._corner[1]}],"size":[{self._size[0]},{self._size[1]}]}}'

class GridTerrain:
    """
    A rectangular patch of floor with its own grip, for maps where the robot can slip.

    The patch covers the same cells a GridObstacle with the same corner and size
    would, but it does not block anything: the lidar and the path planner only see
    obstacles. Cells outside every patch have GridMap.DEFAULT_FRICTION.

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
        Label of the patch (e.g. "ice"), for the map file and plots.
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
    terrains : list of GridTerrain, optional
        Patches of floor with a different grip. They fill the `friction` layer
        (float, DEFAULT_FRICTION elsewhere); a later patch overrides an earlier one
        where they overlap. The occupancy `grid` is not affected.
    """

    FREE = 0
    OBSTACLE = 1
    DEFAULT_FRICTION = 1.0

    def __init__(self,
                 name: str,
                 size:Optional[Union[Iterable[int],np.ndarray]],
                 grid_obstacles: Optional[List[GridObstacle]]=None,
                 terrains: Optional[List[GridTerrain]]=None) -> None:
        size = np.asarray(size)
        if size.size != 2: raise ValueError("The size of 'size' must be 2")
        if not np.issubdtype(size.dtype, np.number): raise TypeError("The type of value of all the elements of 'size' must be numeric")
        if any(size<=0):raise ValueError("The value of all the elements of 'size' must be greater than zero")
        
        self._size = size
        self._width, self._height = self._size
        self.grid = torch.full((self._height, self._width), self.FREE, dtype=torch.int8, device='cpu', requires_grad=False)

        if grid_obstacles:
            for grid_obstacle in grid_obstacles:
                if not self._is_obstacle_in_bounds(grid_obstacle): 
                    raise IndexError(f"The grid_obstacle {grid_obstacle} is out of bounds")
                self._set_obstacle(grid_obstacle)
            self._obstacles = list(grid_obstacles)
        else:
            self._obstacles = []

        # Grip of the floor, separate from the occupancy grid so the lidar and the path planner are unchanged
        self.friction = torch.full((self._height, self._width), self.DEFAULT_FRICTION, dtype=torch.float32, device='cpu')
        self._terrains: List[GridTerrain] = []
        for terrain in terrains or []:
            self.add_terrain(terrain)

        self.name = name
    def get_size(self): return self._size
    def get_width(self): return self._width
    def get_height(self): return self._height

    def _is_obstacle_in_bounds(self, obstacle: Union[GridObstacle, GridTerrain]):
        # The rectangle covers [x, x+width) x [y, y+height): its far edge may touch the map border
        x, y = obstacle.get_corner()
        width, height = obstacle.get_size()
        return x >= 0 and y >= 0 and x + width <= self._width and y + height <= self._height

    # ------------------------------------------------------------
    # Terrain (floor grip)
    # ------------------------------------------------------------
    def add_terrain(self, terrain: GridTerrain):
        """Paint a patch of floor with its friction (overrides earlier patches where they overlap)."""
        if not isinstance(terrain, GridTerrain): raise TypeError("The terrain must be of type GridTerrain")
        if not self._is_obstacle_in_bounds(terrain): raise IndexError(f"The terrain {terrain} is out of bounds")
        x, y = terrain.get_corner()
        width, height = terrain.get_size()
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

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self._height and 0 <= col < self._width

    def is_free(self, row: int, col: int) -> bool:
        return self.in_bounds(row, col) and bool(self.grid[row, col] == self.FREE)

    def is_obstacle(self, row: int, col: int) -> bool:
        return self.in_bounds(row, col) and bool(self.grid[row, col] == self.OBSTACLE)

    def _set_obstacle(self, obstacle: GridObstacle):
        x,y = obstacle.get_corner()
        width, height = obstacle.get_size()
        self.grid[y:y+height, x:x+width] = self.OBSTACLE

    def extend_obstacles(self, obstacles: Iterable[GridObstacle]):
        for obstacle in obstacles:
            self.append_obstacle(obstacle)

    def append_obstacle(self, obstacle: GridObstacle):
        if not self._is_obstacle_in_bounds(obstacle): 
            raise IndexError(f"The grid_obstacle {obstacle} is out of bounds")

        x,y = obstacle.get_corner()
        width, height = obstacle.get_size()
        if any(self.is_obstacle(r, c) for r in range(y, y +height) for c in range(x, x + width)):
            raise ValueError(f"The grid_obstacle {obstacle} collides with existing obstacles")

        self._obstacles.append(obstacle)
        self._set_obstacle(obstacle)
    
    def pop_obstacle(self, index: int):
        if index >= len(self._obstacles): return
        obs = self._obstacles[index]
        self.clear_obstacle(obs)
        return obs

    def clear_obstacle(self, obstacle: GridObstacle):
        if obstacle in self._obstacles:
            x,y = obstacle.get_corner()
            width, height = obstacle.get_size()
            self.grid[y:y+height, x:x+width] = self.FREE
            self._obstacles.remove(obstacle)
            # Restore cells shared with obstacles that overlapped the removed one
            for remaining in self._obstacles:
                self._set_obstacle(remaining)
    
    def to_numpy(self) -> np.ndarray:
        return self.grid.detach().clone().cpu().numpy()

    def __repr__(self) -> str:
        return f"GridMap[{self.name}](width={self._width}, height={self._height})"

    def __str__(self) -> str:
        """'.' free, '#' obstacle, '~' free cell with less grip than the default."""
        def symbol(r, c):
            if self.grid[r, c] == self.OBSTACLE: return "#"
            return "~" if float(self.friction[r, c]) < self.DEFAULT_FRICTION else "."
        return "\n".join("".join(symbol(r, c) for c in range(self._width)) for r in range(self._height))

    # ------------------------------------------------------------
    # Saving and loading maps
    # ------------------------------------------------------------
    def save(self, path: str):
        """
        Save the grid map to a JSON file.

        The JSON structure is:
        {
            "size": [int, int],
            "obstacles": [{"corner":[int,int],"size":[int,int]}, ...],
            "terrains": [{"corner":[int,int],"size":[int,int],"friction":float,"name":str}, ...]  (only if any)
        }
        """

        data = {
            "name": self.name,
            "description": "Grid map representation",
            "size": [int(v) for v in self._size],
            "obstacles": [obstacle.get_json_repr() for obstacle in self._obstacles],
        }
        if self._terrains:
            data["terrains"] = [terrain.get_json_repr() for terrain in self._terrains]
        import os
        _path = os.path.join(path, self.name+".json")
        if not os.path.exists(path):
            os.makedirs(path)
        with open(_path, "wt", encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load(cls, path: str) -> GridMap:
        """
        Load a grid map from a JSON file.
        """
        with open(path, "r") as f:
            data = json.load(f)

        name = data['name']
        size = data["size"]
        obstacles = [GridObstacle(obs['corner'],obs['size']) for obs in data["obstacles"]]
        # Optional: maps without "terrains" have the default grip everywhere
        terrains = [GridTerrain(t['corner'], t['size'], t['friction'], t.get('name', '')) for t in data.get("terrains", [])]

        return cls(name=name, size = size, grid_obstacles=obstacles, terrains=terrains) # type: ignore
