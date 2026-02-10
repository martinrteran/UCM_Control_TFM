"""
lidar.py

A configurable 2D LIDAR sensor for the Robot.
The robot does not know the map; only the LIDAR interacts with the GridMap.

Author: Martin
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np

from ..gridmap import GridMap


class Grid_Lidar:
    """
    A 2D LIDAR sensor that casts multiple rays in a configurable field of view.

    Parameters
    ----------
    num_rays : int
        Number of rays to cast.
    max_range : int
        Maximum number of grid cells each ray can travel.
    fov_deg : float
        Field of view in degrees (e.g., 180 for a semicircle).
    noise_std : float
        Optional Gaussian noise added to distance readings.
    """

    def __init__(
        self,
        *,
        num_rays: int = 9,
        max_range: int = 10,
        fov_deg: float = 180.0,
        noise_std: float = 0.0,
    ):
        self.num_rays = num_rays
        self.max_range = max_range
        self.fov_deg = fov_deg
        self.noise_std = noise_std

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def scan(
        self,
        gridmap: GridMap,
        position: Tuple[int, int],
        orientation: str,
    ) -> np.ndarray:
        """
        Perform a LIDAR scan from the robot's position.

        Returns
        -------
        distances : np.ndarray
            Array of size (num_rays,) with distances to obstacles.
        """
        angles = self._compute_ray_angles(orientation)
        distances = np.zeros(self.num_rays, dtype=float)

        for i, angle in enumerate(angles):
            distances[i] = self._cast_single_ray(gridmap, position, angle)

        if self.noise_std > 0:
            distances += np.random.normal(0, self.noise_std, size=self.num_rays)

        return distances

    # ------------------------------------------------------------
    # Ray angle computation
    # ------------------------------------------------------------
    def _compute_ray_angles(self, orientation: str) -> np.ndarray:
        """
        Compute absolute angles (in degrees) for each ray based on robot orientation.
        Orientation is one of: N, E, S, W.
        """

        orientation_to_angle = {
            "N": 90,
            "E": 0,
            "S": 270,
            "W": 180,
        }

        base_angle = orientation_to_angle[orientation]

        # Spread rays evenly across the FOV
        half_fov = self.fov_deg / 2
        return np.linspace(base_angle - half_fov, base_angle + half_fov, self.num_rays)

    # ------------------------------------------------------------
    # Raycasting
    # ------------------------------------------------------------
    def _cast_single_ray(
        self,
        gridmap: GridMap,
        position: Tuple[int, int],
        angle_deg: float,
    ) -> float:
        """
        Cast a single ray and return the distance to the nearest obstacle.
        """

        angle_rad = np.deg2rad(angle_deg)
        dr = np.sin(angle_rad)
        dc = np.cos(angle_rad)

        r, c = position

        for dist in range(1, self.max_range + 1):
            rr = int(round(r + dr * dist))
            cc = int(round(c + dc * dist))

            if not gridmap.in_bounds(rr, cc):
                return dist  # hit boundary

            if gridmap.is_obstacle(rr, cc):
                return dist  # hit obstacle

        return float(self.max_range)