import enum

from typing import Tuple, Any, Union
import numpy as np


class BaseEnumeration(enum.Enum):
    @classmethod
    def get_names(cls) -> list[str]:
        """
        Get the names of the orientations.
        """
        return list(cls.__members__.keys())

    @classmethod
    def get_values(cls) -> list[Any]:
        """
        Get the vectors of the orientations.
        """
        return [dir.value for dir in cls.__members__.values()]

    def get_value(self) -> Any:
        """
        Get the vector of the orientation.
        """
        return self.value
    
    def get_name(self) -> str:
        """
        Get the name of the orientation.
        """
        return self.name

class Direction(BaseEnumeration):
    forward = (0,-1)
    right = (1,0)
    backward = (0,1)
    left = (-1,0)

class Orientation(BaseEnumeration):
    north = (0,-1)
    west = (-1,0)
    south = (0,1)
    east = (1,0)

    # Mixed orientations
    north_east = (1,-1)
    north_west = (-1,-1)
    south_east = (1,1)
    south_west = (-1,1)

    # Mixed orientations with diagonal movement
    north_north_east = (1,-2)
    north_north_west = (-1,-2)
    south_south_east = (1,2)
    south_south_west = (-1,2)
    east_north_east = (2,-1)
    east_north_west = (2,1)
    west_south_east = (-2,-1)
    west_south_west = (-2,1)

    def get_angle(self, radian: bool = True) -> float:
        """
        Get the angle of the orientation in radians.
        """
        x, y = self.value
        angle = np.arctan2(y, x)
        return angle if radian else np.degrees(angle)

    def get_direction(self) -> Direction:
        """
        Get the closest direction of the orientation.
        """
        x, y = self.value
        if abs(x) > abs(y):
            return Direction.right if x > 0 else Direction.left
        else:
            return Direction.forward if y < 0 else Direction.backward

    def get_opposite(self): # type: ignore
        """
        Get the opposite orientation.
        """
        x, y = self.value
        return Orientation((-x, -y))

    def is_diagonal(self) -> bool:
        """
        Check if the orientation is diagonal.
        """
        x, y = self.value
        return abs(x) == abs(y) and x != 0 and y != 0

    def is_mixed(self) -> bool:
        """
        Check if the orientation is mixed.
        """
        x, y = self.value
        return abs(x) != abs(y) and x != 0 and y != 0

    def is_cardinal(self) -> bool:
        """
        Check if the orientation is cardinal.
        """
        x, y = self.value
        return (x == 0 and y != 0) or (x != 0 and y == 0)


DirectionType = Union[str, Direction, int, Tuple[int, int]]
OrientationType = Union[str, Orientation, int, Tuple[int, int]]