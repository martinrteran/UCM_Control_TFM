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
from typing import Callable, Optional, Union, Iterable, Dict, List

from .map import GridMap
from .robot import GridRobot, GridRobotNotTurning, Robot
from .sensor import GridLidar
import torch
from torchrl.envs import EnvBase
from torchrl.data import (
    Bounded,        # replaces BoundedTensorSpec
    Categorical,    # replaces DiscreteTensorSpec
    Composite,      # replaces CompositeSpec
    Unbounded,      # replaces UnboundedContinuousTensorSpec
)
from tensordict import TensorDict

from tfm_control_ucm.renderers.grid_renderer import PygameRenderer, PygameRenderer_Image_Lidar
from tfm_control_ucm.core.rewards import RewardCallable, StepContext, evaluate_reward, summarize_components

class RewardedStepMixin:
    """
    End-of-step logic shared by the grid environments.

    The environment decides whether the episode ended (goal reached, collision,
    step limit) and the injected reward function only scores the step, so the
    success rate means the same thing whatever reward an experiment uses.

    Expects the environment to define robot, goal_pos, steps, max_steps, lidar,
    reward_fn, _map_diagonal and _dist_to_goal().
    """
    _goal_tolerance = 1.0 # The goal is reached when the robot is closer than this (cells)

    @staticmethod
    def _check_reward_fn(reward_fn: RewardCallable):
        if not callable(reward_fn): raise TypeError("The reward_fn must be a RewardFunction or a callable taking a StepContext")

    def _start_episode(self, obs: np.ndarray):
        """Reset the per-episode memory used to build the next StepContext."""
        self._previous_action = -1 # Do nothing
        self._previous_obs = obs.copy()
        self._previous_dist_to_goal = float(self._dist_to_goal()) # type: ignore

    def _finish_step(self, action: int, obs: np.ndarray, hit_obstacle: bool):
        """
        Decide termination, score the step and update the per-episode memory.

        Parameters
        ----------
        action : int
            Action taken this step.
        obs : np.ndarray
            Observation after the step.
        hit_obstacle : bool
            True if the robot could not move because of a wall or an obstacle.

        Returns
        -------
        tuple
            (reward, terminated, truncated, info) as expected by gymnasium's step().
        """
        dist_to_goal = float(self._dist_to_goal()) # type: ignore
        rays = np.asarray(obs).reshape((-1, 2))[:-1, 0]
        detected = rays[rays >= 0]
        min_obstacle_dist = float(detected.min()) if detected.size else float(self.lidar.max_range) # type: ignore

        reached_goal = dist_to_goal < self._goal_tolerance
        max_steps_reached = bool(self.steps >= self.max_steps) # type: ignore
        terminated = reached_goal
        truncated = not terminated and (hit_obstacle or max_steps_reached)

        ctx = StepContext(action=int(action), previous_action=int(self._previous_action),
                          obs=obs, previous_obs=self._previous_obs, # type: ignore
                          dist_to_goal=dist_to_goal, previous_dist_to_goal=self._previous_dist_to_goal,
                          min_obstacle_dist=min_obstacle_dist, hit_obstacle=hit_obstacle,
                          reached_goal=reached_goal, max_steps_reached=max_steps_reached,
                          steps=self.steps, max_steps=int(self.max_steps), # type: ignore
                          robot_pos=np.asarray(self.robot.position), goal_pos=np.asarray(self.goal_pos), # type: ignore
                          map_diagonal=float(self._map_diagonal), lidar_max_range=float(self.lidar.max_range)) # type: ignore
        reward, components = evaluate_reward(self.reward_fn, ctx) # type: ignore

        info = {'reached_goal': reached_goal, 'hit_obstacle': hit_obstacle, 'max_steps_reached': max_steps_reached,
                'dist2goal': dist_to_goal, 'min_dist': min_obstacle_dist, 'steps': self.steps, # type: ignore
                'goal_pos': self.goal_pos, 'robot_pos': self.robot.position, # type: ignore
                'reward': reward, 'reward_components': components, **summarize_components(components)}

        self._previous_action = action
        self._previous_obs = obs.copy()
        self._previous_dist_to_goal = dist_to_goal
        return reward, terminated, truncated, info

class Grid_Robot_Env(RewardedStepMixin, gym.Env):
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
                robot: Robot,
                reward_fn: RewardCallable,
                lidar_config: Optional[Dict] = None,
                max_iteration_steps: Union[int,np.integer] = 10_000,
                render_mode: Optional[str] = None,
                cell_size: int = 32
                ) -> None:
        if not isinstance(map, GridMap): raise TypeError("The map must be of type GridMap")
        if not isinstance(robot, Robot): raise TypeError("The robot must be of type Robot")
        self._check_reward_fn(reward_fn)
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

        self.action_space = spaces.Discrete(robot.NUM_ACTIONS, dtype=np.int8) # One for move foward and bakward and another one for changing orientation
        self.observation_space = spaces.Box( low=np.array([[-1, -np.pi]] * self._min_num_rays + [[0.0, -np.pi]]), high=np.array([[self._max_range, np.pi]] * self._min_num_rays + [[max_map_distance, np.pi]]))

        self.steps = 0
        self.cell_size = cell_size
        self.reward_fn = reward_fn
        self._previous_action = -1 # Do nothing
        self._previous_obs = None
    
    def get_observation_shape(self):
        return self.observation_space.shape
    
    def get_observation_dim(self):
        return np.prod(self.observation_space.shape) # type: ignore
    
    def get_action_dim(self):
        return self.action_space.n # type: ignore
    
    def get_action_shape(self):
        return self.action_space.shape

    def _get_observation(self):
        scanning = self.lidar.scan(self.map, self.robot.position, np.asarray(0.0)).cpu().numpy()

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
            if self.map.is_free(r, c):
                self.robot.reset((c,r))
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

        obs = self._get_observation()
        self._start_episode(obs)
        info = {}

        return obs, info

    def step(self, action: int):
        self.steps += 1

        moved = self.robot.move(self.map, action)

        obs = self._get_observation()
        reward, terminated, truncated, info = self._finish_step(action, obs, hit_obstacle=not moved)
        return obs, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode != 'human': return
        if not hasattr(self, "renderer"): self.renderer = PygameRenderer(self.map, cell_size=self.cell_size)
        dists = self._previous_obs.reshape((-1, 2)) # type: ignore
        self.renderer.handle_events()
        self.renderer.render(self.robot, self.lidar,dists[:-1,:], self.goal_pos)

class Grid_Robot_Sections_Env(RewardedStepMixin, gym.Env):
    """
    Docstring for Robot_Grid_Env
    
    """
    metadata = {"render_modes": ["human"], "render_fps": 10}
    _min_num_rays = 12
    _min_num_sections = 4
    _max_noise_std = 0.5
    _max_range = 100
    _min_separation_distance = 10
    _max_previous_actions = 5
    
    
    def __init__(self, *,
                map: GridMap,
                robot: Robot,
                reward_fn: RewardCallable,
                lidar_config: Optional[Dict] = None,
                max_iteration_steps: Union[int,np.integer] = 10_000,
                render_mode: Optional[str] = None,
                cell_size: int = 32,
                num_sections: int = _min_num_sections
                ) -> None:
        if not isinstance(map, GridMap): raise TypeError("The map must be of type GridMap")
        if not isinstance(robot, Robot): raise TypeError("The robot must be of type Robot")
        self._check_reward_fn(reward_fn)
        if num_sections < self._min_num_sections and num_sections%4 != 0: raise ValueError(f"The number of sections must be greater or equal to {self._min_num_sections} and a multiple of 4")

        if lidar_config: 
            if any(key not in GridLidar.config_keys() for key in lidar_config.keys()): raise ValueError(f"One of the configuration keys for the lidar is not correct. The configuration keys are: {', '.join(GridLidar.config_keys())}")
            elif lidar_config['num_rays'] < self._min_num_rays: raise ValueError(f"The field 'num_rays' (number of rays) must be greater or equal to {self._min_num_rays}")
            elif lidar_config['max_range'] > self._max_range: raise ValueError(f"The field 'max_range' (maximum detection distance) must be less or equal to {self._max_range}")
            elif lidar_config['noise_std'] > self._max_noise_std: raise ValueError(f"The field 'noise_std' (maximum noise standard deviation) must be less or equal to {self._max_noise_std}")
        
        super().__init__()
        self.map = map
        self.robot = robot
        self.reward_fn = reward_fn
        if not lidar_config:
            self.lidar = GridLidar(num_rays=self._min_num_rays, max_range=self._max_range, noise_std=self._max_noise_std, with_cache=True)
        else:
            self.lidar = GridLidar(**lidar_config)
        self.max_steps = max_iteration_steps
        self.render_mode = render_mode
        
        max_map_distance = np.sqrt(np.sum(np.array(self.map.grid.shape) **2))
        self._map_diagonal = max_map_distance

        self.action_space = spaces.Discrete(robot.NUM_ACTIONS,start=0,dtype=np.int8) # One for move foward and bakward and another one for changing orientation
        self.observation_space = spaces.Box( low=np.array([[-1, -np.pi]] * num_sections + [[0.0, -np.pi]],dtype=np.float32),
                        high=np.array([[self._max_range, np.pi]] * num_sections + [[max_map_distance, np.pi]], dtype=np.float32))

        self.steps = 0
        self.cell_size = cell_size
        self.num_sections = num_sections

        self._angle_per_section = self.lidar.fov_rad/num_sections;
        self._start_angle = -self._angle_per_section/2;
        self._previous_actions: np.ndarray = np.array([])
        self._previous_action = -1 # Do nothing
        self._previous_obs = None
    
    def get_observation_shape(self):
        return self.observation_space.shape
    
    def get_observation_dim(self):
        return np.prod(self.observation_space.shape) # type: ignore
    
    def get_action_dim(self):
        return self.action_space.n # type: ignore
    
    def get_action_shape(self):
        return self.action_space.shape

    def _get_observation(self):
        # scanning is (N, 2) tensor: [dist, angle]
        scanning = self.lidar.scan(self.map, self.robot.position, np.asarray(0.0)) # We don't need the robot orientation for this environment, we just need the angles of the rays

        angles = scanning[:, 1]
        # Normalize angles to (-pi, pi]
        angles = torch.atan2(torch.sin(angles), torch.cos(angles))
        
        device = scanning.device
        top_scanning = torch.full((self.num_sections, 2), -1.0, device=device)
        
        # Robust sectioning logic using modular arithmetic
        for i in range(self.num_sections):
            section_center = self._start_angle + (i + 0.5) * self._angle_per_section
            
            # Difference between ray angle and section center, wrapped to (-pi, pi]
            angle_diff = torch.atan2(torch.sin(angles - section_center), torch.cos(angles - section_center))
            in_section = torch.abs(angle_diff) <= (self._angle_per_section / 2.0)
            
            if torch.any(in_section):
                section_rays = scanning[in_section]
                closest_idx = torch.argmin(section_rays[:, 0])
                top_scanning[i, :] = section_rays[closest_idx, :]
            else:
                top_scanning[i, 0] = float(self.lidar.max_range)
                top_scanning[i, 1] = float(section_center)
        
        robot_obs = torch.tensor([self._dist_to_goal(), self._angle_to_goal()], device=device)
        
        # If the top_scanning is in the max_range +- the noise_std, then set the angle to the middle of the section
        for i in range(self.num_sections):
            if top_scanning[i, 0] >= self.lidar.max_range - self.lidar.noise_std:
                top_scanning[i, 1] = self._start_angle + (i + 0.5) * self._angle_per_section        

        concated = torch.cat([top_scanning.flatten(), robot_obs])
        return concated.detach().cpu().numpy()

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
            if self.map.is_free(r, c):
                self.robot.reset((c,r))
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

        obs = self._get_observation()
        self._start_episode(obs)
        info = {}
        self._previous_actions = np.array([]) # Empty it

        return obs, info
    
    def step(self, action: int):
        self.steps += 1

        moved = self.robot.move(self.map, action)

        obs_flat = self._get_observation()
        reward, terminated, truncated, info = self._finish_step(action, obs_flat, hit_obstacle=not moved)
        return obs_flat, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode != 'human': return
        if not hasattr(self, "renderer"): self.renderer = PygameRenderer(self.map, cell_size=self.cell_size)
        dists = self._previous_obs.reshape((-1, 2)) # type: ignore
        self.renderer.handle_events()
        self.renderer.render(self.robot,dists[:-1,:],self.num_sections, self.goal_pos)


class Grid_Robot_Sections_Env_MultiMap(RewardedStepMixin, gym.Env):
    """
    Docstring for Robot_Grid_Env
    
    """
    metadata = {"render_modes": ["human"], "render_fps": 10}
    _min_num_rays = 12
    _min_num_sections = 4
    _max_noise_std = 0.5
    _max_range = 100
    _min_separation_distance = 10
    _max_previous_actions = 5
    
    
    def __init__(self, *,
                maps: Iterable[GridMap],
                robot: Robot,
                reward_fn: RewardCallable,
                lidar_config: Optional[Dict] = None,
                max_iteration_steps: Union[int,np.integer] = 10_000,
                render_mode: Optional[str] = None,
                cell_size: int = 32,
                num_sections: int = _min_num_sections,
                random_map: bool = True
                ) -> None:
        if not isinstance(maps, Iterable): raise TypeError("The maps must be an iterable of GridMap instances")
        maps = list(maps)
        if not maps: raise ValueError("The maps iterable cannot be empty")
        if not all(isinstance(m, GridMap) for m in maps): raise TypeError("All elements in the maps iterable must be of type GridMap")
        
        if not isinstance(robot, Robot): raise TypeError("The robot must be of type Robot")
        self._check_reward_fn(reward_fn)
        if num_sections < self._min_num_sections and num_sections%4 != 0: raise ValueError(f"The number of sections must be greater or equal to {self._min_num_sections} and a multiple of 4")

        if lidar_config: 
            if any(key not in GridLidar.config_keys() for key in lidar_config.keys()): raise ValueError(f"One of the configuration keys for the lidar is not correct. The configuration keys are: {', '.join(GridLidar.config_keys())}")
            elif lidar_config['num_rays'] < self._min_num_rays: raise ValueError(f"The field 'num_rays' (number of rays) must be greater or equal to {self._min_num_rays}")
            elif lidar_config['max_range'] > self._max_range: raise ValueError(f"The field 'max_range' (maximum detection distance) must be less or equal to {self._max_range}")
            elif lidar_config['noise_std'] > self._max_noise_std: raise ValueError(f"The field 'noise_std' (maximum noise standard deviation) must be less or equal to {self._max_noise_std}")
        
        super().__init__()
        self.maps = maps
        self._random_map = random_map
        self._map_idx: Optional[int] = None # Set on reset; the first map is a placeholder until then
        self.map = self.maps[0]
        self.robot = robot
        self.reward_fn = reward_fn
        if not lidar_config:
            self.lidar = GridLidar(num_rays=self._min_num_rays, max_range=self._max_range, noise_std=self._max_noise_std, with_cache=True)
        else:
            self.lidar = GridLidar(**lidar_config)
        self.max_steps = max_iteration_steps
        self.render_mode = render_mode
        
        # Largest diagonal over all maps, so the observation bounds and reward normalisation hold for every map
        max_map_distance = max(np.sqrt(np.sum(np.array(m.grid.shape) **2)) for m in self.maps)
        self._map_diagonal = max_map_distance

        self.action_space = spaces.Discrete(robot.NUM_ACTIONS,start=0,dtype=np.int8) # One for move foward and bakward and another one for changing orientation
        self.observation_space = spaces.Box( low=np.array([[-1, -np.pi]] * num_sections + [[0.0, -np.pi]],dtype=np.float32),
                        high=np.array([[self._max_range, np.pi]] * num_sections + [[max_map_distance, np.pi]], dtype=np.float32))

        self.steps = 0
        self.cell_size = cell_size
        self.num_sections = num_sections

        self._angle_per_section = self.lidar.fov_rad/num_sections;
        self._start_angle = -self._angle_per_section/2;
        self._previous_actions: np.ndarray = np.array([])
        self._previous_action = -1 # Do nothing
        self._previous_obs = None

    def _get_random_map(self)->GridMap:
        """
        Select the map for the next episode: uniformly at random (seeded through reset(seed=...)),
        or cycling through the maps in order when random_map is False.
        """
        if self._random_map:
            self._map_idx = int(self.np_random.integers(len(self.maps)))
        else:
            self._map_idx = 0 if self._map_idx is None else (self._map_idx + 1) % len(self.maps)
        return self.maps[self._map_idx]

    def get_observation_shape(self):
        return self.observation_space.shape
    
    def get_observation_dim(self):
        return np.prod(self.observation_space.shape) # type: ignore
    
    def get_action_dim(self):
        return self.action_space.n # type: ignore
    
    def get_action_shape(self):
        return self.action_space.shape

    def _get_observation(self):
        # scanning is (N, 2) tensor: [dist, angle]
        scanning = self.lidar.scan(self.map, self.robot.position, np.asarray(0.0)) # We don't need the robot orientation for this environment, we just need the angles of the rays

        angles = scanning[:, 1]
        # Normalize angles to (-pi, pi]
        angles = torch.atan2(torch.sin(angles), torch.cos(angles))
        
        device = scanning.device
        top_scanning = torch.full((self.num_sections, 2), -1.0, device=device)
        
        # Robust sectioning logic using modular arithmetic
        for i in range(self.num_sections):
            section_center = self._start_angle + (i + 0.5) * self._angle_per_section
            
            # Difference between ray angle and section center, wrapped to (-pi, pi]
            angle_diff = torch.atan2(torch.sin(angles - section_center), torch.cos(angles - section_center))
            in_section = torch.abs(angle_diff) <= (self._angle_per_section / 2.0)
            
            if torch.any(in_section):
                section_rays = scanning[in_section]
                closest_idx = torch.argmin(section_rays[:, 0])
                top_scanning[i, :] = section_rays[closest_idx, :]
            else:
                top_scanning[i, 0] = float(self.lidar.max_range)
                top_scanning[i, 1] = float(section_center)
        
        robot_obs = torch.tensor([self._dist_to_goal(), self._angle_to_goal()], device=device)
        
        # If the top_scanning is in the max_range +- the noise_std, then set the angle to the middle of the section
        for i in range(self.num_sections):
            if top_scanning[i, 0] >= self.lidar.max_range - self.lidar.noise_std:
                top_scanning[i, 1] = self._start_angle + (i + 0.5) * self._angle_per_section        

        concated = torch.cat([top_scanning.flatten(), robot_obs])
        return concated.detach().cpu().numpy()

    def _dist_to_goal(self):
        return np.sqrt(np.sum((self.goal_pos - self.robot.position)**2))    
    
    def _angle_to_goal(self):
        x_diff, y_diff = self.goal_pos - self.robot.position
        return np.atan2(y_diff, x_diff)
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.map = self._get_random_map()

        height, width = self.map.grid.shape
        isCorrect = False
        while not isCorrect:
            r = np.random.randint(0,height)
            c = np.random.randint(0,width)
            if self.map.is_free(r, c):
                self.robot.reset((c,r))
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

        obs = self._get_observation()
        self._start_episode(obs)
        info = {}
        self._previous_actions = np.array([]) # Empty it

        return obs, info
    
    def step(self, action: int):
        self.steps += 1

        moved = self.robot.move(self.map, action)

        obs_flat = self._get_observation()
        reward, terminated, truncated, info = self._finish_step(action, obs_flat, hit_obstacle=not moved)
        return obs_flat, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode != 'human': return
        if not hasattr(self, "renderer"): self.renderer = PygameRenderer(self.map, cell_size=self.cell_size)
        dists = self._previous_obs.reshape((-1, 2)) # type: ignore
        self.renderer.handle_events()
        self.renderer.render(self.robot,dists[:-1,:],self.num_sections, self.goal_pos)

# class Grid_Robot_Env_Image_lidar(gym.Env):
#     """
#     Docstring for Robot_Grid_Env
    
#     """
#     metadata = {"render_modes": ["human"], "render_fps": 10}
#     _min_num_rays = 50
#     _max_noise_std = 0.5
#     _max_range = 100
#     _min_separation_distance = 5
#     _previous_actions: list = []
#     _max_previous_actions = 5
    
#     def __init__(self, *,
#                 map: GridMap,
#                 robot: GridRobot,
#                 lidar_config: Optional[Dict] = None,
#                 max_iteration_steps: Union[int,np.integer] = 10_000,
#                 render_mode: Optional[str] = None,
#                 cell_size: int = 32
#                 ) -> None:
#         if not isinstance(map, GridMap): raise TypeError("The map must be of type GridMap")
#         if not isinstance(robot, GridRobot): raise TypeError("The robot must be of type GridRobot")
#         if lidar_config: 
#             if any(key not in GridLidar.config_keys() for key in lidar_config.keys()): raise ValueError(f"One of the configuration keys for the lidar is not correct. The configuration keys are: {', '.join(GridLidar.config_keys())}")
#             elif lidar_config['num_rays'] < self._min_num_rays: raise ValueError(f"The field 'num_rays' (number of rays) must be greater or equal to {self._min_num_rays}")
#             elif lidar_config['max_range'] > self._max_range: raise ValueError(f"The field 'max_range' (maximum detection distance) must be less or equal to {self._max_range}")
#             elif lidar_config['noise_std'] > self._max_noise_std: raise ValueError(f"The field 'noise_std' (maximum noise standard deviation) must be less or equal to {self._max_noise_std}")

#         super().__init__()
#         self.map = map
#         self.robot = robot
#         if not lidar_config:
#             self.lidar = GridLidar_2D()
#         else:
#             self.lidar = GridLidar_2D(**lidar_config)
#         self.max_steps = max_iteration_steps
#         self.render_mode = render_mode
        
#         max_map_distance = np.sqrt(np.sum(np.array(self.map.grid.shape) **2))
#         self._map_diagonal = max_map_distance

#         self.action_space = spaces.Discrete(3,dtype=np.int8) # One for move foward and bakward and another one for changing orientation
#         grid_size = 2 * self.lidar.max_range + 1;
#         self.observation_space = spaces.Dict({
#             "lidar": spaces.Box(low=0, high=self.lidar.max_range,
#                                  shape=(grid_size, grid_size), dtype=np.int8),
#             "robot": spaces.Box(low=np.array([0.0, -np.pi]), 
#                     high=np.array([max_map_distance, np.pi]), dtype=np.float32)
#         })
#         self.observation_space = spaces.Box( low=np.array([[-1, -np.pi]] * self._min_num_rays + [[0.0, -np.pi]]), high=np.array([[self._max_range, np.pi]] * self._min_num_rays + [[max_map_distance, np.pi]]))

#         self.steps = 0
#         self.cell_size = cell_size
    
#     def get_observation_shape(self):
#         return self.observation_space.shape
    
#     def get_observation_dim(self):
#         return np.prod(self.observation_space.shape) # type: ignore
    
#     def get_action_dim(self):
#         return self.action_space.n # type: ignore
    
#     def get_action_shape(self):
#         return self.action_space.shape

#     def _get_observation(self):
#         scanning = self.lidar.scan(self.map, self.robot.position, self.robot.ORIENTATIONS[self.robot.orientation]['angle'])
        
#         robot_obs = np.array([self._dist_to_goal(),self._angle_to_goal()])
        
#         return {"lidar": scanning, "robot": robot_obs}

#     def _dist_to_goal(self):
#         return np.sqrt(np.sum((self.goal_pos - self.robot.position)**2))    
    
#     def _angle_to_goal(self):
#         x_diff, y_diff = self.goal_pos - self.robot.position
#         return np.atan2(y_diff, x_diff)
    
#     def reset(self, *, seed=None, options=None):
#         super().reset(seed=seed)

#         height, width = self.map.grid.shape
#         isCorrect = False
#         while not isCorrect:
#             r = np.random.randint(0,height)
#             c = np.random.randint(0,width)
#             orien = self.robot.ORIENTATIONS_LIST[np.random.randint(0,4)]
#             if self.map.is_free(r, c):
#                 self.robot.reset((c,r),orien)
#                 isCorrect = True
        
#         isCorrect = False
#         while not isCorrect:
#             r = np.random.randint(0,height)
#             c = np.random.randint(0,width)
#             self.goal_pos = np.array([c,r])
#             dist_to_robot = self._dist_to_goal()
#             if self.map.is_free(r, c) and dist_to_robot > self._min_separation_distance:
#                 isCorrect = True

#         self.steps = 0
#         self._previous_action = -1 # Do nothing
#         self._previous_actions.clear() # Empty it

#         obs = self._get_observation()
#         info = {}

#         return obs, info
    
#     def step(self, action: int):
#         self.steps += 1
    
#         previous_dist2goal = self._dist_to_goal()
#         truncated = False
#         if action == 0: # Move foward
#             moved = self.robot.forward(self.map)
#             truncated = not moved
#         elif action == 1: # Rotate left
#             self.robot.turn_left()
#         else:# elif action == 3: # Rotate right
#             self.robot.turn_right()            
#         if len(self._previous_actions) >= self._max_previous_actions: # Keep only the last 5 actions
#             self._previous_actions.pop(0)
#         self._previous_actions.append(action)

#         obs = self._get_observation()
#         scanning = obs["lidar"]
#         robot_obs = obs["robot"]
#         info = {}
        
#         dist2goal = robot_obs[0]
#         # Distances to LIDAR
#         centerLidar = self.lidar.max_range;
#         obstacle_coords = np.argwhere(scanning == 1)
#         if obstacle_coords.size > 0:
#             # Eucledian Distance from the center to each detected obstacle
#             dists_to_obstacles = np.linalg.norm(obstacle_coords - [centerLidar, centerLidar], axis=1)
#             dists_to_obstacles = np.clip(dists_to_obstacles, 0, self.lidar.max_range) # Clip to max range
#             dists_to_obstacles = np.power(dists_to_obstacles, 2) # Square the distances to emphasize closer obstacles
            
#             min_obstacle_dist = np.min(dists_to_obstacles)
#             mean_dists = np.mean(dists_to_obstacles)
#         else:
#             # If there are no obstacles in view, we assume the maximum safe distance
#             min_obstacle_dist = -float(self.lidar.max_range)
#             mean_dists = min_obstacle_dist

#         reward = -0.5 if all(action!=0 for action in self._previous_actions) else -0.01
#         reward -= (5*dist2goal)/self._map_diagonal
#         reward -= mean_dists/self._map_diagonal
#         reward -= mean_dists/self._map_diagonal
#         reward += ((previous_dist2goal - dist2goal)/self._map_diagonal) * 5

#         terminated = False
#         if dist2goal < 1:
#             terminated = True
#             reward = 100.0  # Big success reward

#         # Replace your reward calculation with:
#         truncated = not terminated and (truncated or bool(self.steps >= self.max_steps))
#         if truncated:
#             reward -= 10.0
        
        
#         return (scanning, robot_obs), reward, terminated, truncated, info
    
#     def render(self):
#         if self.render_mode != 'human': 
#             return
            
#         if not hasattr(self, "renderer"): 
#             self.renderer = PygameRenderer(self.map, cell_size=self.cell_size)
            
#         obs = self._get_observation()
#         lidar_image = obs["lidar"] 
        
#         self.renderer.handle_events()
#         self.renderer.render(self.robot, self.lidar, lidar_image, self.goal_pos)
