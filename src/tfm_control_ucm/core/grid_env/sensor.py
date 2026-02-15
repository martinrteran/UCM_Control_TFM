import numpy as np
from typing import Iterable, Union
from tfm_control_ucm.core.grid_env.map import GridMap

class GridLidar:
    """
    A 2D LIDAR sensor that casts multiple rays in a configurable field of view.

    Parameters
    ----------
    num_rays : int
        Number of rays to cast.
    max_range : int
        Maximum number of grid cells each ray can travel.
    fov_rad : float
        Field of view in radians (e.g., pi for a semicircle).
    noise_std : float
        Optional Gaussian noise added to distance readings.
    """

    def __init__(
        self,
        *,
        num_rays: int = 10,
        max_range: int = 10,
        fov: float = np.pi,
        noise_std: float = 0.001,
    ):
        if not isinstance(num_rays, np.number|int): raise TypeError("The num_rays must be an integer")
        if not isinstance(max_range, np.number|int): raise TypeError("The max_range must be an integer")
        if not isinstance(fov, np.number|float): raise TypeError("The fov must be a float")
        if not isinstance(noise_std, np.number|float): raise TypeError("The noise_std must be a float")

        if num_rays <= 0: raise ValueError("The num_rays must be greater than zero")
        if max_range <= 0: raise ValueError("The max_range must be greater than zero")
        if fov <= 0: raise ValueError("The fov must be greater than zero")
        if noise_std <= 0: raise ValueError("The noise_std must be greater than zero")


        self.num_rays = num_rays
        self.max_range = max_range
        self.fov_rad = fov
        self.noise_std = noise_std
        self.last_results = np.ones((self.num_rays,2))*(-1.0)
    
    def get_config(self):
        return {'num_rays':self.num_rays, 'max_range':self.max_range, 'fov': self.fov_rad, 'noise_std': self.noise_std}
    
    @classmethod
    def config_keys(cls):
        return ['num_rays', 'max_range', 'fov', 'noise_std']
    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def scan(
        self,
        gridmap: GridMap,
        position: Union[Iterable[int],np.ndarray],
        orientation: Union[float,np.floating],
    ) :
        """
        Perform a LIDAR scan from the robot's position.

        Returns
        -------
        distances : np.ndarray
            Array of size (num_rays,) with distances to obstacles.
        """
        position = np.asarray(position)

        if position.size != 2: raise ValueError("The position must be of size 2")

        if not np.issubdtype(position.dtype,np.number): raise TypeError("The position must be of type number")

        
        angles = self._compute_ray_angles(orientation)
        # distances = np.zeros(self.num_rays, dtype=float)
        return_values = []

        for i, angle in enumerate(angles):
            distance = self._cast_single_ray(gridmap, position, angle)
            # if distance> 0 and self.noise_std > 0:
            #     distance += np.random.normal(0, self.noise_std) #, size=self.num_rays)
            return_values.append([distance, angle])

        self.last_results = np.array(return_values)

        return np.array(return_values)

    # ------------------------------------------------------------
    # Ray angle computation
    # ------------------------------------------------------------
    def _compute_ray_angles(self, orientation: Union[float, np.floating]) -> np.ndarray:
        """
        Compute absolute angles (in radians) for each ray based on robot orientation.
        Orientation is one of: N pi/2, E 0, S -pi/2, W pi.
        """

        base_angle = orientation

        # Spread rays evenly across the FOV
        half_fov = self.fov_rad / 2
        return np.linspace(base_angle - half_fov, base_angle + half_fov, self.num_rays)

    # ------------------------------------------------------------
    # Raycasting
    # ------------------------------------------------------------
    def _cast_single_ray(
        self,
        gridmap: GridMap,
        position: np.ndarray,
        angle_rad: float,
    ) -> float:
        """
        Cast a single ray and return the distance to the nearest obstacle.
        """

        dr = np.sin(angle_rad)
        dc = np.cos(angle_rad)

        c, r = position

        for dist in range(1, self.max_range + 1):
            rr = int(round(r - dr * dist))
            cc = int(round(c + dc * dist))

            if not gridmap.in_bounds(rr, cc):
                return dist  # hit boundary

            if gridmap.is_obstacle(rr, cc):
                return dist  # hit obstacle

        return -1.0