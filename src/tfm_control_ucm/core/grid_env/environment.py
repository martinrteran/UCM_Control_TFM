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
from .robot import GridRobot, GridRobotNotTurning
from .sensor import GridLidar, GridLidar_2D
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
                max_iteration_steps: Union[int,np.integer] = 10_000,
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

        self.action_space = spaces.Discrete(3,dtype=np.int8) # One for move foward and bakward and another one for changing orientation
        self.observation_space = spaces.Box( low=np.array([[-1, -np.pi]] * self._min_num_rays + [[0.0, -np.pi]]), high=np.array([[self._max_range, np.pi]] * self._min_num_rays + [[max_map_distance, np.pi]]))

        self.steps = 0
        self.cell_size = cell_size
    
    def get_observation_shape(self):
        return self.observation_space.shape
    
    def get_observation_dim(self):
        return np.prod(self.observation_space.shape) # type: ignore
    
    def get_action_dim(self):
        return self.action_space.n # type: ignore
    
    def get_action_shape(self):
        return self.action_space.shape

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
    
        previous_dist2goal = self._dist_to_goal()
        truncated = False
        if action == 0: # Move foward
            moved = self.robot.forward(self.map)
            truncated = not moved
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
        reward += (previous_dist2goal - dist2goal)/self._map_diagonal * 5

        terminated = False
        if dist2goal < 1:
            terminated = True
            reward += 100.0  # Big success reward

        # Replace your reward calculation with:
        truncated = truncated or bool(self.steps >= self.max_steps)
        if truncated:
            reward = -10.0
        
        self.previous_action = action
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode != 'human': return
        if not hasattr(self, "renderer"): self.renderer = PygameRenderer(self.map, cell_size=self.cell_size)
        dists = self._get_observation()
        self.renderer.handle_events()
        self.renderer.render(self.robot, self.lidar,dists[:-1,:], self.goal_pos)

class Grid_Robot_Sections_Env(gym.Env):
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
    _previous_actions: np.ndarray = np.array([])
    
    def __init__(self, *,
                map: GridMap,
                robot: GridRobot,
                lidar_config: Optional[Dict] = None,
                max_iteration_steps: Union[int,np.integer] = 10_000,
                render_mode: Optional[str] = None,
                cell_size: int = 32,
                num_sections: int = _min_num_sections
                ) -> None:
        if not isinstance(map, GridMap): raise TypeError("The map must be of type GridMap")
        if not isinstance(robot, GridRobot): raise TypeError("The robot must be of type GridRobot")
        if num_sections < self._min_num_sections and num_sections%4 != 0: raise ValueError(f"The number of sections must be greater or equal to {self._min_num_sections} and a multiple of 4")

        if lidar_config: 
            if any(key not in GridLidar.config_keys() for key in lidar_config.keys()): raise ValueError(f"One of the configuration keys for the lidar is not correct. The configuration keys are: {', '.join(GridLidar.config_keys())}")
            elif lidar_config['num_rays'] < self._min_num_rays: raise ValueError(f"The field 'num_rays' (number of rays) must be greater or equal to {self._min_num_rays}")
            elif lidar_config['max_range'] > self._max_range: raise ValueError(f"The field 'max_range' (maximum detection distance) must be less or equal to {self._max_range}")
            elif lidar_config['noise_std'] > self._max_noise_std: raise ValueError(f"The field 'noise_std' (maximum noise standard deviation) must be less or equal to {self._max_noise_std}")
        
        super().__init__()
        self.map = map
        self.robot = robot
        if not lidar_config:
            self.lidar = GridLidar(num_rays=self._min_num_rays, max_range=self._max_range, noise_std=self._max_noise_std, with_cache=True)
        else:
            self.lidar = GridLidar(**lidar_config)
        self.max_steps = max_iteration_steps
        self.render_mode = render_mode
        
        max_map_distance = np.sqrt(np.sum(np.array(self.map.grid.shape) **2))
        self._map_diagonal = max_map_distance

        self.action_space = spaces.Discrete(3,dtype=np.int8) # One for move foward and bakward and another one for changing orientation
        self.observation_space = spaces.Box( low=np.array([[-1, -np.pi]] * num_sections + [[0.0, -np.pi]],dtype=np.float32),
                        high=np.array([[self._max_range, np.pi]] * num_sections + [[max_map_distance, np.pi]], dtype=np.float32))

        self.steps = 0
        self.cell_size = cell_size
        self.num_sections = num_sections

        self._angle_per_section = 2*np.pi/num_sections;
        self._start_angle = -self._angle_per_section/2;
    
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
        scanning = self.lidar.scan(self.map, self.robot.position, self.robot.ORIENTATIONS[self.robot.orientation]['angle'])

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
        
        # If the top_scanning is in the max_range +- the noise_std, then set the distance to -1.0 to indicate that there is no obstacle detected in that section
        # top_scanning[top_scanning[:, 0] ==self.lidar.max_range, 0] = float(-1)
        

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
        self._previous_actions = np.array([]) # Empty it

        return obs, info
    
    _safety_margin = 3.0       # lidar distance below which we start penalizing proximity
    _max_proximity_penalty = 1.0
    _stagnation_window = 8     # how many recent actions to check for spinning

    def step(self, action: int):
        self.steps += 1

        previous_dist2goal = self._dist_to_goal()
        truncated = False
        if action == 0:            # forward
            moved = self.robot.forward(self.map)
            truncated = not moved
        elif action == 1:          # turn left
            self.robot.turn_left()
        else:                       # turn right
            self.robot.turn_right()

        obs_flat = self._get_observation()
        obs = obs_flat.reshape((-1, 2))
        info = {}

        dist2goal = obs[-1][0]
        min_dist = np.min(obs[:-1, 0][obs[:-1, 0] >= 0]) if np.any(obs[:-1, 0] >= 0) else self._max_range

        # 1. Progress toward goal — the dominant signal, bounded per-step
        progress = (previous_dist2goal - dist2goal) / self._map_diagonal
        reward = progress * 10.0

        # 2. Small constant step cost so idling/short paths are preferred over long ones
        reward -= 0.05

        # 3. Bounded obstacle-proximity penalty — no division, no blow-up
        if min_dist < self._safety_margin:
            closeness = (self._safety_margin - min_dist) / self._safety_margin  # in [0, 1]
            reward -= closeness * self._max_proximity_penalty

        # 4. Anti-spinning penalty: check if recent actions are all turns with no net progress
        self._previous_actions = np.append(self._previous_actions, action)
        if len(self._previous_actions) > self._stagnation_window:
            self._previous_actions = self._previous_actions[1:]

        if len(self._previous_actions) == self._stagnation_window:
            recent = self._previous_actions
            no_forward = np.all(recent != 0)          # never moved forward
            if no_forward:
                reward -= 0.5                          # flat penalty for pure turning stretches

        # 5. Terminal reward/penalty — clearly bigger than any step reward, but not extreme
        done = False
        if dist2goal < 1:
            done = True
            reward += 20.0

        truncated = truncated or bool(self.steps >= self.max_steps)
        if truncated:
            reward -= 5.0

        return obs_flat, reward, done, truncated, info
    
    def render(self):
        if self.render_mode != 'human': return
        if not hasattr(self, "renderer"): self.renderer = PygameRenderer(self.map, cell_size=self.cell_size)
        dists = self._get_observation()
        dists = dists.reshape((-1, 2))
        self.renderer.handle_events()
        self.renderer.render(self.robot,dists[:-1,:],self.num_sections, self.goal_pos)

class Grid_Robot_Env_Image_lidar(gym.Env):
    """
    Docstring for Robot_Grid_Env
    
    """
    metadata = {"render_modes": ["human"], "render_fps": 10}
    _min_num_rays = 50
    _max_noise_std = 0.5
    _max_range = 100
    _min_separation_distance = 5
    _previous_actions: list = []
    _max_previous_actions = 5
    
    def __init__(self, *,
                map: GridMap,
                robot: GridRobot,
                lidar_config: Optional[Dict] = None,
                max_iteration_steps: Union[int,np.integer] = 10_000,
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
            self.lidar = GridLidar_2D()
        else:
            self.lidar = GridLidar_2D(**lidar_config)
        self.max_steps = max_iteration_steps
        self.render_mode = render_mode
        
        max_map_distance = np.sqrt(np.sum(np.array(self.map.grid.shape) **2))
        self._map_diagonal = max_map_distance

        self.action_space = spaces.Discrete(3,dtype=np.int8) # One for move foward and bakward and another one for changing orientation
        grid_size = 2 * self.lidar.max_range + 1;
        self.observation_space = spaces.Dict({
            "lidar": spaces.Box(low=0, high=self.lidar.max_range,
                                 shape=(grid_size, grid_size), dtype=np.int8),
            "robot": spaces.Box(low=np.array([0.0, -np.pi]), 
                    high=np.array([max_map_distance, np.pi]), dtype=np.float32)
        })
        self.observation_space = spaces.Box( low=np.array([[-1, -np.pi]] * self._min_num_rays + [[0.0, -np.pi]]), high=np.array([[self._max_range, np.pi]] * self._min_num_rays + [[max_map_distance, np.pi]]))

        self.steps = 0
        self.cell_size = cell_size
    
    def get_observation_shape(self):
        return self.observation_space.shape
    
    def get_observation_dim(self):
        return np.prod(self.observation_space.shape) # type: ignore
    
    def get_action_dim(self):
        return self.action_space.n # type: ignore
    
    def get_action_shape(self):
        return self.action_space.shape

    def _get_observation(self):
        scanning = self.lidar.scan(self.map, self.robot.position, self.robot.ORIENTATIONS[self.robot.orientation]['angle'])
        
        robot_obs = np.array([self._dist_to_goal(),self._angle_to_goal()])
        
        return {"lidar": scanning, "robot": robot_obs}

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
        self._previous_actions.clear() # Empty it

        obs = self._get_observation()
        info = {}

        return obs, info
    
    def step(self, action: int):
        self.steps += 1
    
        previous_dist2goal = self._dist_to_goal()
        truncated = False
        if action == 0: # Move foward
            moved = self.robot.forward(self.map)
            truncated = not moved
        elif action == 1: # Rotate left
            self.robot.turn_left()
        else:# elif action == 3: # Rotate right
            self.robot.turn_right()            
        if len(self._previous_actions) >= self._max_previous_actions: # Keep only the last 5 actions
            self._previous_actions.pop(0)
        self._previous_actions.append(action)

        obs = self._get_observation()
        scanning = obs["lidar"]
        robot_obs = obs["robot"]
        info = {}
        
        dist2goal = robot_obs[0]
        # Distances to LIDAR
        centerLidar = self.lidar.max_range;
        obstacle_coords = np.argwhere(scanning == 1)
        if obstacle_coords.size > 0:
            # Eucledian Distance from the center to each detected obstacle
            dists_to_obstacles = np.linalg.norm(obstacle_coords - [centerLidar, centerLidar], axis=1)
            dists_to_obstacles = np.clip(dists_to_obstacles, 0, self.lidar.max_range) # Clip to max range
            dists_to_obstacles = np.power(dists_to_obstacles, 2) # Square the distances to emphasize closer obstacles
            
            min_obstacle_dist = np.min(dists_to_obstacles)
            mean_dists = np.mean(dists_to_obstacles)
        else:
            # If there are no obstacles in view, we assume the maximum safe distance
            min_obstacle_dist = -float(self.lidar.max_range)
            mean_dists = min_obstacle_dist

        reward = -0.5 if all(action!=0 for action in self._previous_actions) else -0.01
        reward -= (5*dist2goal)/self._map_diagonal
        reward -= mean_dists/self._map_diagonal
        reward -= mean_dists/self._map_diagonal
        reward += ((previous_dist2goal - dist2goal)/self._map_diagonal) * 5

        terminated = False
        if dist2goal < 1:
            terminated = True
            reward = 100.0  # Big success reward

        # Replace your reward calculation with:
        truncated = not terminated and (truncated or bool(self.steps >= self.max_steps))
        if truncated:
            reward -= 10.0
        
        
        return (scanning, robot_obs), reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode != 'human': 
            return
            
        if not hasattr(self, "renderer"): 
            self.renderer = PygameRenderer(self.map, cell_size=self.cell_size)
            
        obs = self._get_observation()
        lidar_image = obs["lidar"] 
        
        self.renderer.handle_events()
        self.renderer.render(self.robot, self.lidar, lidar_image, self.goal_pos)


class GridRobotSections_Env(gym.Env):
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
    _previous_actions: np.ndarray = np.array([])
    
    def __init__(self, *,
                map: GridMap,
                robot: GridRobotNotTurning,
                lidar_config: Optional[Dict] = None,
                max_iteration_steps: Union[int,np.integer] = 10_000,
                render_mode: Optional[str] = None,
                cell_size: int = 32,
                num_sections: int = _min_num_sections,
                device: Union[str,torch.device] = "cpu"
                ) -> None:
        if not isinstance(map, GridMap): raise TypeError("The map must be of type GridMap")
        if not isinstance(robot, GridRobotNotTurning): raise TypeError("The robot must be of type GridRobotNotTurning")
        if num_sections < self._min_num_sections and num_sections%4 != 0: raise ValueError(f"The number of sections must be greater or equal to {self._min_num_sections} and a multiple of 4")
        
        if lidar_config: 
            if any(key not in GridLidar.config_keys() for key in lidar_config.keys()): raise ValueError(f"One of the configuration keys for the lidar is not correct. The configuration keys are: {', '.join(GridLidar.config_keys())}")
            elif lidar_config['num_rays'] < self._min_num_rays: raise ValueError(f"The field 'num_rays' (number of rays) must be greater or equal to {self._min_num_rays}")
            elif lidar_config['max_range'] > self._max_range: raise ValueError(f"The field 'max_range' (maximum detection distance) must be less or equal to {self._max_range}")
            elif lidar_config['noise_std'] > self._max_noise_std: raise ValueError(f"The field 'noise_std' (maximum noise standard deviation) must be less or equal to {self._max_noise_std}")
        
        super().__init__()
        self.map = map
        self.robot = robot
        self.device = torch.device(device) if isinstance(device, str) else device
        
        if not lidar_config:
            self.lidar = GridLidar(num_rays=self._min_num_rays, max_range=self._max_range, noise_std=self._max_noise_std, with_cache=True, device=self.device)
        else:
            self.lidar = GridLidar(**lidar_config)
        self.max_steps = max_iteration_steps
        self.render_mode = render_mode
        
        max_map_distance = torch.tensor(np.sqrt(np.sum(np.array(self.map.grid.shape) **2)))
        self._map_diagonal = max_map_distance

        self.action_space = spaces.Discrete(4,dtype=np.int8) # One for move foward and bakward and another one for changing orientation
        self.observation_space = spaces.Box( low=np.array([[-1, -np.pi]] * num_sections + [[0.0, -np.pi]]), high=np.array([[self._max_range, np.pi]] * num_sections + [[max_map_distance, np.pi]]))

        self.steps = 0
        self.cell_size = cell_size
        self.num_sections = num_sections

        self._angle_per_section = 2*np.pi/num_sections;
        self._start_angle = -self._angle_per_section/2;
    
    def get_observation_shape(self):
        return self.observation_space.shape
    
    def get_observation_dim(self):
        return np.prod(self.observation_space.shape) # type: ignore
    
    def get_action_dim(self):
        return self.action_space.n # type: ignore
    
    def get_action_shape(self):
        return self.action_space.shape

    def _get_observation(self) -> Dict[str, torch.Tensor]:
        scanning = self.lidar.scan(self.map, self.robot.position, self.robot.ORIENTATIONS[self.robot.orientation]['angle'])

        angles = scanning[:, 1]
        # Normalize angles to (-pi, pi]
        angles = torch.atan2(torch.sin(angles), torch.cos(angles))
        
        top_scanning = torch.full((self.num_sections, 2), -1.0, device=self.device)
        
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
        
        robot_obs = torch.tensor([self._dist_to_goal(), self._angle_to_goal()], device=self.device)
        
        return {'lidar': top_scanning, 'robot': robot_obs}

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
        lidar = obs["lidar"]
        robot_obs = obs["robot"]
        info = {}
        self._previous_actions = np.array([]) # Empty it

        return torch.vstack([lidar, robot_obs]).reshape((self.get_observation_dim(),)), info
    
    def step(self, action: int):
        self.steps += 1
    
        previous_dist2goal = self._dist_to_goal()
        truncated = False
        direction = ""
        match action:
            case 0:
                direction = "forward"
            case 1:
                direction = "right"
            case 2:
                direction = "backward"
            case 3:
                direction = "left"

        moved = self.robot.move(self.map, direction)
        truncated = not moved

        obs = self._get_observation()
        lidar = obs["lidar"]
        robot_obs = obs["robot"]
        info = {}
        
        dist2goal = robot_obs[0]
        
        reward = -dist2goal/self._map_diagonal
        reward -= self._map_diagonal/(torch.min(lidar[:,0]) + 1e-6)
#        reward += (previous_dist2goal - dist2goal)/self._map_diagonal * 5

        done = False
        if dist2goal < 1:
            done = True
            reward = 100.0  # Big success reward

        truncated = truncated or bool(self.steps >= self.max_steps)
        if truncated:
            reward -= 50.0
        
        self._previous_actions = np.append(self._previous_actions, action)
        if len(self._previous_actions) > self._max_previous_actions: # Keep only the last 5 actions
            self._previous_actions = self._previous_actions[1:]
        
        return torch.vstack([lidar, robot_obs]).reshape((self.get_observation_dim(),)), reward, done, truncated, info
    
    def render(self):
        if self.render_mode != 'human': return
        if not hasattr(self, "renderer"): self.renderer = PygameRenderer(self.map, cell_size=self.cell_size)
        obs = self._get_observation()
        dists = obs["lidar"]
        self.renderer.handle_events()
        self.renderer.render(self.robot, self.lidar,dists, self.goal_pos)
