import enum


class Direction(enum.Enum):
    forward = (0,1)
    right = (1,0)
    backward = (0,-1)
    left = (-1,0)