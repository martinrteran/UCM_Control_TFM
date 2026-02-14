"""
environment.py

Gymnasium environment integrating:
- GridMap
- Robot (with orientation)
- Lidar sensor

Author: Martin
"""

from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Union, Iterable, Dict

from .map import GridMap
from .robot import GridRobot
from .sensor import GridLidar

from src.tfm_control_ucm.renderers.grid_renderer import PygameRenderer

class Grid_Robot_Env(gym.Env):
    """
    Docstring for Robot_Grid_Env
    
    """
    metadata = {"render_modes": ["human"], "render_fps": 10}
    _min_num_rays = 8
    _max_noise_std = 0.5
    _max_range = 100
    
    def __init__(self, *,
                map: GridMap,
                robot: GridRobot,
                lidar_config: Optional[Dict] = None,
                max_iteration_steps: Union[int,np.integer] = 200,
                render_mode: Optional[str] = None
                ) -> None:
        if not isinstance(map, GridMap): raise TypeError("The map must be of type GridMap")
        if not isinstance(robot, GridRobot): raise TypeError("The robot must be of type GridRobot")
        if lidar_config: 
            if any(key not in GridLidar.config_keys() for key in lidar_config.keys()): raise ValueError(f"One of the configuration keys for the lidar is not correct. The configuration keys are: {', '.join(GridLidar.config_keys())}")
            elif lidar_config['num_rays'] < self._min_num_rays: raise ValueError(f"The field 'num_rays' (number of rays) must be greater or equal to {self._min_num_rays}")
            elif lidar_config['max_range'] > self._max_range: raise ValueError(f"The field 'max_range' (maximum detection distance) must be less or equal to {self._max_range}")
            elif lidar_config['noise_std'] > self._max_noise_std: raise ValueError(f"The field 'noise_std' (maximum noise standard deviation) must be less or equal to {self._max_noise_std}")

        super().__init__()
        self.map = map
        self.robot = robot
        if not lidar_config:
            self.lidar = GridLidar()
        else:
            self.lidar = GridLidar(**lidar_config)
        self.max_steps = max_iteration_steps
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(2,dtype=np.int8) # One for move foward and bakward and another one for changing orientation
        self.observation_space = spaces.Box( low=np.array([[0.0, -np.pi]] * 8), high=np.array([[self._max_range, np.pi]] * 8), dtype=np.float32 )

        self.steps = 0
   
    def _get_observation(self):
        return self.lidar.scan(self.map, self.robot.position, self.robot.ORIENTATIONS[self.robot.orientation]['vector']).astype(np.float32)
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        height, width = self.map.grid.shape
        isCorrect = False
        while not isCorrect:
            r = np.random.randint(0,height)
            c = np.random.randint(0,width)
            orien = self.robot.ORIENTATIONS_LIST[np.random.randint(0,4)]
            if self.map.is_free(r, c):
                self.robot.reset((c,r),orien)
                isCorrect = True
        
        self.steps = 0

        obs = self._get_observation()
        info = {}

        return obs, info
    
    def step(self, action):
        
        reward = 0.0
        terminated = False
        

        truncated = self.steps >= self.max_steps

        obs = self._get_observation()
        info = {}

        return obs, reward, terminated, truncated, info


class RobotEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def step(self, action):
        self.steps += 1

        reward = 0
        terminated = False
        truncated = False

        # Execute action
        if action == 0:  # forward
            moved = self.robot.forward()
            reward += 1 if moved else -1
            if not moved:
                terminated = True  # collision
        elif action == 1:  # turn left
            self.robot.turn_left()
        elif action == 2:  # turn right
            self.robot.turn_right()

        # Check goal
        if self.goal_pos and self.robot.position == self.goal_pos:
            reward += 10
            terminated = True

        # Step penalty
        reward -= 0.01

        # Max steps
        if self.steps >= self.max_steps:
            truncated = True

        obs = self._get_observation()
        info = {}

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------
    def _get_observation(self):
        return self.lidar.scan(
            self.gridmap,
            self.robot.position,
            self.robot.orientation,
        ).astype(np.float32)

    # ------------------------------------------------------------
    # Rendering (placeholder)
    # ------------------------------------------------------------
    def render(self):
        if self.render_mode != "human":
            return

        if not hasattr(self, "renderer"):
            self.renderer = PygameRenderer(self.gridmap)

        distances = self._get_observation()
        self.renderer.handle_events()
        self.renderer.render(self.robot, self.lidar, distances)