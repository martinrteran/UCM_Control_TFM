from __future__ import annotations
import numpy as np
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

    def get_json_repr(self):  return {'corner':self._corner,'size':self._size}

    def __repr__(self):  return f'{{"corner":[{self._corner[0]},{self._corner[1]}],"size":[{self._size[0]},{self._size[1]}]}}'

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

    def __init__(self, 
                 size:Optional[Union[Iterable[int],np.ndarray]], 
                 grid_obstacles: Optional[List[GridObstacle]]=None) -> None:
        size = np.asarray(size)
        if size.size != 2: raise ValueError("The size of 'size' must be 2")
        if not np.issubdtype(size.dtype, np.number): raise TypeError("The type of value of all the elements of 'size' must be numeric")
        if any(size<=0):raise ValueError("The value of all the elements of 'size' must be greater than zero")
        
        self._size = size
        self._width, self._height = self._size
        self.grid = np.ones((self._height, self._width), dtype=np.int8) * self.FREE

        if grid_obstacles:
            for grid_obstacle in grid_obstacles:
                if not self.is_obstacle_in_bounds(grid_obstacle): 
                    raise IndexError(f"The grid_obstacle {grid_obstacle} is out of bounds")
                self.set_obstacle(grid_obstacle)
            self._obstacles = grid_obstacles
        else:
            self._obstacles = []
    def get_size(self): return self._size
    def get_width(self): return self._width
    def get_height(self): return self._height

    def is_obstacle_in_bounds(self, obstacle:GridObstacle):
        corners = obstacle.get_corners()
        in_bounds = all(self.in_bounds(r,c) for c, r in corners)
        return in_bounds

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self._height and 0 <= col < self._width

    def is_free(self, row: int, col: int) -> bool:
        return self.in_bounds(row, col) and self.grid[row, col] == self.FREE

    def is_obstacle(self, row: int, col: int) -> bool:
        return self.in_bounds(row, col) and self.grid[row, col] == self.OBSTACLE

    def set_obstacle(self, obstacle: GridObstacle):
        x,y = obstacle.get_corner()
        width, height = obstacle.get_size()
        self.grid[y:y+height, x:x+width] += self.OBSTACLE
    
    def pop_obstacle(self, index: int):
        if index >= len(self._obstacles): return
        return self._obstacles.pop(index)

    def clear_obstacle(self, obstacle: GridObstacle):
        if obstacle in self._obstacles:
            x,y = obstacle.get_corner()
            width, height = obstacle.get_size()
            self.grid[y:y+height, x:x+width] -= self.OBSTACLE
            self._obstacles.remove(obstacle)
    
    def to_numpy(self) -> np.ndarray:
        return self.grid.copy()

    def __repr__(self) -> str:
        return f"GridMap(width={self._width}, height={self._height})"

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
            "size": [int, int],
            "obstacles": [{"corner":[int,int],"size":[int,int]}, ...]
        }
        """

        data = {
            "size": self._size,
            "obstacles":  self._obstacles.copy(),
        }

        with open(path, "wt", encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load(cls, path: str) -> GridMap:
        """
        Load a grid map from a JSON file.
        """
        with open(path, "r") as f:
            data = json.load(f)

        size = data["size"]
        obstacles = [GridObstacle(obs['corner'],obs['size']) for obs in data["obstacles"]]

        return cls(size = size, grid_obstacles=obstacles) # type: ignore
