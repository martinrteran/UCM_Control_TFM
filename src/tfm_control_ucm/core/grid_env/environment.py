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

from tfm_control_ucm.renderers.grid_renderer import PygameRenderer

class Grid_Robot_Env(gym.Env):
    """
    Docstring for Robot_Grid_Env
    
    """
    metadata = {"render_modes": ["human"], "render_fps": 10}
    _min_num_rays = 8
    _max_noise_std = 0.5
    _max_range = 100
    _min_separation_distance = 5
    
    def __init__(self, *,
                map: GridMap,
                robot: GridRobot,
                lidar_config: Optional[Dict] = None,
                max_iteration_steps: Union[int,np.integer] = 200,
                render_mode: Optional[str] = None,
                cell_size: int = 32
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

        max_map_distance = np.sqrt(np.sum(np.array(self.map.grid.shape) **2))
        self._map_diagonal = max_map_distance

        self.action_space = spaces.Discrete(1,dtype=np.int8) # One for move foward and bakward and another one for changing orientation
        self.observation_space = spaces.Box( low=np.array([[-1, -np.pi]] * self._min_num_rays + [[0.0, -np.pi]]), high=np.array([[self._max_range, np.pi]] * self._min_num_rays + [[max_map_distance, np.pi]]))

        self.steps = 0
        self.cell_size = cell_size
   
    def _get_observation(self):
        scanning = self.lidar.scan(self.map, self.robot.position, self.robot.ORIENTATIONS[self.robot.orientation]['angle'])
        
        idx = np.argsort(scanning[:,0])
        scanning = scanning[idx[::-1]] # descending order
        
        top_scanning = scanning[0:self._min_num_rays]
        robot_obs = np.array([self._dist_to_goal(),self._angle_to_goal()])
        
        concated = np.vstack([top_scanning, robot_obs])
        return concated

    def _dist_to_goal(self):
        return np.sqrt(np.sum((self.goal_pos - self.robot.position)**2))    
    
    def _angle_to_goal(self):
        x_diff, y_diff = self.goal_pos - self.robot.position
        return np.atan2(y_diff, x_diff)
    
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
        
        isCorrect = False
        while not isCorrect:
            r = np.random.randint(0,height)
            c = np.random.randint(0,width)
            self.goal_pos = np.array([c,r])
            dist_to_robot = self._dist_to_goal()
            if self.map.is_free(r, c) and dist_to_robot > self._min_separation_distance:
                isCorrect = True

        self.steps = 0
        self.previous_action = -1 # Do nothing

        obs = self._get_observation()
        info = {}

        return obs, info
    
    def step(self, action: int):
        self.steps += 1
    
        terminated = False
        if action == 0: # Move foward
            moved = self.robot.forward(self.map)
            terminated = not moved
        # elif action == [-1,0]: # Move backward
        #     moved = self.robot.backward(self.map)
        #     terminated = not moved
        elif action == 1: # Rotate left
            self.robot.turn_left()
        else:# elif action == 3: # Rotate right
            self.robot.turn_right()            

        obs = self._get_observation()
        info = {}
        
        dist2goal = obs[-1][0]
        
        mean_dists = np.mean(obs[:-1,0])

        reward = -0.2 if self.previous_action > 0 and action > 0 else -0.01
        reward -= dist2goal/self._map_diagonal
        reward -= mean_dists/self._map_diagonal

        if not terminated and self._dist_to_goal() < 1:
            terminated = True
            reward += 10.0


        truncated = bool(self.steps >= self.max_steps)

        self.previous_action = action
        

        return obs, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode != 'human': return
        if not hasattr(self, "renderer"): self.renderer = PygameRenderer(self.map, cell_size=self.cell_size)
        dists = self._get_observation()
        self.renderer.handle_events()
        self.renderer.render(self.robot, self.lidar,dists[:-1,:], self.goal_pos)
        