"""
environment.py

Gymnasium environment integrating:
- GridMap
- Robot (with or without orientation)
- Lidar sensor
- Injected reward function (see core/rewards.py)

Author: Martin
"""

from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from abc import ABC, abstractmethod
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
from tfm_control_ucm.core.curriculum import SuccessCurriculum
from tfm_control_ucm.utils.path_planner import PathPlanner

def _wrap_angle(angle):
    """Wrap an angle (radians) to (-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))

class BaseGridEnv(gym.Env, ABC):
    """
    Base class of the grid navigation environments.

    Composes the pieces injected by the training script (maps, robot, lidar
    configuration and reward function) and implements everything the
    environments share: map selection, start/goal sampling, the end-of-episode
    rules, the StepContext handed to the reward and rendering.

    Angles in the observation are egocentric: 0 is the direction the robot
    faces (robot.heading) and positive angles turn counter-clockwise, the same
    convention as the lidar. Robots without orientation have heading 0 (east).

    Subclasses only define the observation:
    _build_observation_space(), _get_observation() and _render_frame().

    Parameters
    ----------
    maps : Iterable[GridMap]
        Maps to train on. One is selected at each reset.
    robot : Robot
        Robot to move; its NUM_ACTIONS sets the action space.
    reward_fn : RewardCallable
        Reward function called with a StepContext after each step.
    lidar_config : dict, optional
        Keyword arguments of GridLidar. Defaults to _default_lidar().
    max_iteration_steps : int, optional
        Step limit of an episode.
    render_mode : str, optional
        "human" to draw the environment with pygame.
    cell_size : int, optional
        Pixels per cell when rendering.
    random_map : bool, optional
        Pick the map uniformly at random at each reset (seeded through reset(seed=...)),
        or cycle through the maps in order when False.
    curriculum : SuccessCurriculum, optional
        If given, its current stage limits the maps (picked at random among the
        stage's) and the start-goal distance, and it is told the outcome of every
        episode. info['curriculum_stage'] reports the stage.
    """
    metadata = {"render_modes": ["human"], "render_fps": 10}
    _min_num_rays = 8
    _max_noise_std = 0.5
    _max_range = 100
    _min_separation_distance = 5
    _goal_tolerance = 1.0 # The goal is reached when the robot is closer than this (cells)
    # Before test 25 the goal angle was atan2(row_diff, col_diff): mirrored with respect to the lidar
    # angles. Only for evaluating checkpoints trained that way (scripts/eval_agent.py sets it).
    _legacy_goal_angle = False

    def __init__(self, *,
                maps: Iterable[GridMap],
                robot: Robot,
                reward_fn: RewardCallable,
                lidar_config: Optional[Dict] = None,
                max_iteration_steps: Union[int,np.integer] = 10_000,
                render_mode: Optional[str] = None,
                cell_size: int = 32,
                random_map: bool = True,
                curriculum: Optional[SuccessCurriculum] = None
                ) -> None:
        if not isinstance(maps, Iterable): raise TypeError("The maps must be an iterable of GridMap instances")
        maps = list(maps)
        if not maps: raise ValueError("At least one map must be provided")
        if not all(isinstance(m, GridMap) for m in maps): raise TypeError("All the maps must be of type GridMap")
        if not isinstance(robot, Robot): raise TypeError("The robot must be of type Robot")
        if not callable(reward_fn): raise TypeError("The reward_fn must be a RewardFunction or a callable taking a StepContext")
        self._check_lidar_config(lidar_config)
        if curriculum is not None: curriculum.check_maps([m.name for m in maps])

        super().__init__()
        self.maps = maps
        self.map = maps[0] # Placeholder until the first reset selects one
        self._random_map = random_map
        self._map_idx: Optional[int] = None
        self.curriculum = curriculum
        self.robot = robot
        self.reward_fn = reward_fn
        self.lidar = GridLidar(**lidar_config) if lidar_config else self._default_lidar()
        self.max_steps = max_iteration_steps
        self.render_mode = render_mode
        self.cell_size = cell_size
        self.renderer = None

        # Largest diagonal over all maps, so the observation bounds and reward normalisation hold for every map
        self._map_diagonal = max(np.sqrt(np.sum(np.array(m.grid.shape) **2)) for m in self.maps)

        self.action_space = spaces.Discrete(robot.NUM_ACTIONS, start=0, dtype=np.int8)
        self.observation_space = self._build_observation_space()

        self.steps = 0
        self.goal_pos = np.array([0, 0])
        self._previous_action = -1 # Do nothing
        self._previous_obs = None
        self._previous_dist_to_goal = 0.0
        self._previous_geodesic_dist_to_goal = float("inf")
        self._goal_field = np.full(self.map.grid.shape, np.inf)

    @classmethod
    def _check_lidar_config(cls, lidar_config: Optional[Dict]):
        if not lidar_config: return
        if any(key not in GridLidar.config_keys() for key in lidar_config.keys()): raise ValueError(f"One of the configuration keys for the lidar is not correct. The configuration keys are: {', '.join(GridLidar.config_keys())}")
        if lidar_config.get('num_rays', cls._min_num_rays) < cls._min_num_rays: raise ValueError(f"The field 'num_rays' (number of rays) must be greater or equal to {cls._min_num_rays}")
        if lidar_config.get('max_range', 0) > cls._max_range: raise ValueError(f"The field 'max_range' (maximum detection distance) must be less or equal to {cls._max_range}")
        if lidar_config.get('noise_std', 0.0) > cls._max_noise_std: raise ValueError(f"The field 'noise_std' (maximum noise standard deviation) must be less or equal to {cls._max_noise_std}")

    def _default_lidar(self) -> GridLidar:
        return GridLidar(num_rays=self._min_num_rays, max_range=self._max_range, noise_std=self._max_noise_std, with_cache=True)

    # ------------------------------------------------------------
    # What subclasses define
    # ------------------------------------------------------------
    @abstractmethod
    def _build_observation_space(self) -> spaces.Box:
        """Observation space; called at the end of BaseGridEnv.__init__, after lidar and _map_diagonal exist."""
        raise NotImplementedError

    @abstractmethod
    def _get_observation(self) -> np.ndarray:
        """Observation of the current state; the last row is [dist_to_goal, angle_to_goal]."""
        raise NotImplementedError

    @abstractmethod
    def _render_frame(self, renderer: PygameRenderer, rays: np.ndarray):
        """Draw one frame; rays are the lidar rows of the last observation as [dist, world angle]."""
        raise NotImplementedError

    # ------------------------------------------------------------
    # Spaces
    # ------------------------------------------------------------
    def get_observation_shape(self):
        return self.observation_space.shape

    def get_observation_dim(self):
        return np.prod(self.observation_space.shape) # type: ignore

    def get_action_dim(self):
        return self.action_space.n # type: ignore

    def get_action_shape(self):
        return self.action_space.shape

    # ------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------
    def _dist_to_goal(self) -> float:
        return float(np.sqrt(np.sum((self.goal_pos - np.asarray(self.robot.position))**2)))

    def _angle_to_goal(self) -> float:
        """Angle to the goal relative to the robot heading, in the lidar convention (counter-clockwise, rows grow downwards)."""
        dc, dr = self.goal_pos - np.asarray(self.robot.position)
        if self._legacy_goal_angle: return float(np.arctan2(dr, dc))
        return float(_wrap_angle(np.arctan2(-dr, dc) - self.robot.heading))

    def _compute_goal_field(self):
        """BFS (wavefront) distance from every free cell to the goal, 4-connected; inf where unreachable."""
        gc, gr = (int(v) for v in self.goal_pos)
        self._goal_field = PathPlanner(self.map.to_numpy()).wavefront((gr, gc))

    def _geodesic_dist_to_goal(self) -> float:
        """Shortest-path distance from the robot to the goal around obstacles (turns not counted); inf if unreachable."""
        c, r = (int(v) for v in self.robot.position)
        return float(self._goal_field[r, c])

    def _optimal_steps(self) -> float:
        """Shortest path length from the robot to the goal (wavefront BFS, turns not counted); inf if unreachable."""
        return self._geodesic_dist_to_goal()

    # ------------------------------------------------------------
    # Episode
    # ------------------------------------------------------------
    def _select_map(self) -> GridMap:
        """
        Map for the next episode: uniformly at random among the curriculum stage's
        maps (all maps without curriculum), or the next one in order when random_map is False.
        """
        if self.curriculum is not None:
            allowed = [i for i, m in enumerate(self.maps) if m.name in self.curriculum.stage.maps]
            self._map_idx = allowed[int(self.np_random.integers(len(allowed)))]
        elif self._random_map:
            self._map_idx = int(self.np_random.integers(len(self.maps)))
        else:
            self._map_idx = 0 if self._map_idx is None else (self._map_idx + 1) % len(self.maps)
        return self.maps[self._map_idx]

    _max_goal_attempts = 10_000 # Then the curriculum's distance limit is dropped (e.g. no free cell close enough)

    def _sample_goal(self) -> np.ndarray:
        height, width = self.map.grid.shape
        max_distance = self.curriculum.stage.max_goal_distance if self.curriculum is not None else None
        attempts = 0
        while True:
            attempts += 1
            r, c = int(self.np_random.integers(height)), int(self.np_random.integers(width))
            goal = np.array([c, r])
            distance = np.sqrt(np.sum((goal - np.asarray(self.robot.position))**2))
            too_far = max_distance is not None and distance > max_distance and attempts < self._max_goal_attempts
            if self.map.is_free(r, c) and distance > self._min_separation_distance and not too_far:
                return goal

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        previous_map = self.map
        self.map = self._select_map()
        if self.map is not previous_map: self.renderer = None # The renderer draws a single map

        self.robot.random_reset(self.map, self.np_random)
        self.goal_pos = self._sample_goal()
        self._compute_goal_field()
        self.steps = 0

        obs = self._get_observation()
        self._start_episode(obs)
        info = {'map': self.map.name, 'optimal_steps': self._optimal_steps(), 'dist2goal': self._dist_to_goal(),
                'action_mask': self.robot.valid_actions(self.map)}
        if self.curriculum is not None: info['curriculum_stage'] = self.curriculum.stage_idx

        return obs, info

    def step(self, action: int):
        self.steps += 1

        moved = self.robot.move(self.map, action)

        obs = self._get_observation()
        reward, terminated, truncated, info = self._finish_step(action, obs, hit_obstacle=not moved)
        return obs, reward, terminated, truncated, info

    def _start_episode(self, obs: np.ndarray):
        """Reset the per-episode memory used to build the next StepContext."""
        self._previous_action = -1 # Do nothing
        self._previous_obs = obs.copy()
        self._previous_dist_to_goal = self._dist_to_goal()
        self._previous_geodesic_dist_to_goal = self._geodesic_dist_to_goal()

    def _finish_step(self, action: int, obs: np.ndarray, hit_obstacle: bool):
        """
        Decide termination, score the step and update the per-episode memory.

        The environment decides whether the episode ended (goal reached, collision,
        step limit) and the reward function only scores the step, so the success
        rate means the same thing whatever reward an experiment uses.

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
        dist_to_goal = self._dist_to_goal()
        geodesic_dist_to_goal = self._geodesic_dist_to_goal()
        rays = np.asarray(obs).reshape((-1, 2))[:-1, 0]
        detected = rays[rays >= 0]
        min_obstacle_dist = float(detected.min()) if detected.size else float(self.lidar.max_range)

        reached_goal = dist_to_goal < self._goal_tolerance
        max_steps_reached = bool(self.steps >= self.max_steps)
        terminated = reached_goal
        truncated = not terminated and (hit_obstacle or max_steps_reached)

        ctx = StepContext(action=int(action), previous_action=int(self._previous_action),
                          obs=obs, previous_obs=self._previous_obs, # type: ignore
                          dist_to_goal=dist_to_goal, previous_dist_to_goal=self._previous_dist_to_goal,
                          min_obstacle_dist=min_obstacle_dist, hit_obstacle=hit_obstacle,
                          reached_goal=reached_goal, max_steps_reached=max_steps_reached,
                          steps=self.steps, max_steps=int(self.max_steps),
                          robot_pos=np.asarray(self.robot.position), goal_pos=np.asarray(self.goal_pos),
                          map_diagonal=float(self._map_diagonal), lidar_max_range=float(self.lidar.max_range),
                          geodesic_dist_to_goal=geodesic_dist_to_goal,
                          previous_geodesic_dist_to_goal=self._previous_geodesic_dist_to_goal)
        reward, components = evaluate_reward(self.reward_fn, ctx)

        info = {'action_mask': self.robot.valid_actions(self.map), # Valid actions in the new state
                'reached_goal': reached_goal, 'hit_obstacle': hit_obstacle, 'max_steps_reached': max_steps_reached,
                'dist2goal': dist_to_goal, 'min_dist': min_obstacle_dist, 'steps': self.steps,
                'goal_pos': self.goal_pos, 'robot_pos': self.robot.position, 'map': self.map.name,
                'reward': reward, 'reward_components': components, **summarize_components(components)}
        if self.curriculum is not None:
            info['curriculum_stage'] = self.curriculum.stage_idx # Stage this episode was played in
            if terminated or truncated:
                info['curriculum_advanced'] = self.curriculum.record(reached_goal)

        self._previous_action = action
        self._previous_obs = obs.copy()
        self._previous_dist_to_goal = dist_to_goal
        self._previous_geodesic_dist_to_goal = geodesic_dist_to_goal
        return reward, terminated, truncated, info

    # ------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------
    def render(self):
        if self.render_mode != 'human' or self._previous_obs is None: return
        if self.renderer is None: self.renderer = PygameRenderer(self.map, cell_size=self.cell_size)
        rays = np.asarray(self._previous_obs, dtype=np.float64).reshape((-1, 2))[:-1].copy()
        rays[:, 1] += self.robot.heading # The renderer draws in the world frame
        self.renderer.handle_events()
        self._render_frame(self.renderer, rays)

class Grid_Robot_Env(BaseGridEnv):
    """
    Grid environment whose observation is the _min_num_rays farthest lidar
    readings [dist, angle] (descending) plus [dist_to_goal, angle_to_goal],
    as a (_min_num_rays + 1, 2) array.
    """
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
                cell_size: int = 32,
                curriculum: Optional[SuccessCurriculum] = None
                ) -> None:
        if not isinstance(map, GridMap): raise TypeError("The map must be of type GridMap")
        super().__init__(maps=[map], robot=robot, reward_fn=reward_fn, lidar_config=lidar_config,
                         max_iteration_steps=max_iteration_steps, render_mode=render_mode, cell_size=cell_size,
                         curriculum=curriculum)

    def _default_lidar(self) -> GridLidar:
        return GridLidar()

    def _build_observation_space(self) -> spaces.Box:
        return spaces.Box(low=np.array([[-1, -np.pi]] * self._min_num_rays + [[0.0, -np.pi]], dtype=np.float32),
                          high=np.array([[self._max_range, np.pi]] * self._min_num_rays + [[self._map_diagonal, np.pi]], dtype=np.float32))

    def _get_observation(self):
        scanning = self.lidar.scan(self.map, self.robot.position, np.asarray(self.robot.heading)).cpu().numpy()

        idx = np.argsort(scanning[:,0])
        scanning = scanning[idx[::-1]] # descending order

        top_scanning = scanning[0:self._min_num_rays]
        robot_obs = np.array([self._dist_to_goal(),self._angle_to_goal()])

        concated = np.vstack([top_scanning, robot_obs])
        return concated

    def _render_frame(self, renderer: PygameRenderer, rays: np.ndarray):
        renderer.render(self.robot, rays, len(rays), self.goal_pos)

class Grid_Robot_Sections_Env(BaseGridEnv):
    """
    Grid environment whose observation splits the lidar field of view into
    num_sections angular sections and keeps the nearest reading [dist, angle]
    of each (an empty section reports [max_range, section centre]), followed by
    [dist_to_goal, angle_to_goal]. The observation is flat: num_sections * 2 + 2 values.
    """
    _min_num_rays = 12
    _min_num_sections = 4
    _max_noise_std = 0.5
    _max_range = 100
    _min_separation_distance = 10

    def __init__(self, *,
                map: GridMap,
                robot: Robot,
                reward_fn: RewardCallable,
                lidar_config: Optional[Dict] = None,
                max_iteration_steps: Union[int,np.integer] = 10_000,
                render_mode: Optional[str] = None,
                cell_size: int = 32,
                num_sections: int = _min_num_sections,
                curriculum: Optional[SuccessCurriculum] = None
                ) -> None:
        if not isinstance(map, GridMap): raise TypeError("The map must be of type GridMap")
        self._setup(maps=[map], robot=robot, reward_fn=reward_fn, lidar_config=lidar_config,
                    max_iteration_steps=max_iteration_steps, render_mode=render_mode, cell_size=cell_size,
                    num_sections=num_sections, random_map=True, curriculum=curriculum)

    def _setup(self, *, num_sections: int, **kwargs):
        if num_sections < self._min_num_sections or num_sections % 4 != 0: raise ValueError(f"The number of sections must be greater or equal to {self._min_num_sections} and a multiple of 4")
        self.num_sections = num_sections # Needed by _build_observation_space
        BaseGridEnv.__init__(self, **kwargs)
        self._angle_per_section = self.lidar.fov_rad/num_sections
        self._start_angle = -self._angle_per_section/2

    def _build_observation_space(self) -> spaces.Box:
        return spaces.Box(low=np.array([[-1, -np.pi]] * self.num_sections + [[0.0, -np.pi]], dtype=np.float32),
                          high=np.array([[self._max_range, np.pi]] * self.num_sections + [[self._map_diagonal, np.pi]], dtype=np.float32))

    def _get_observation(self):
        # scanning is (N, 2) tensor: [dist, angle relative to the robot heading]
        scanning = self.lidar.scan(self.map, self.robot.position, np.asarray(self.robot.heading))

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

    def _render_frame(self, renderer: PygameRenderer, rays: np.ndarray):
        renderer.render(self.robot, rays, self.num_sections, self.goal_pos)

class Grid_Robot_Sections_Env_MultiMap(Grid_Robot_Sections_Env):
    """
    Grid_Robot_Sections_Env over several maps: one is picked at each reset,
    uniformly at random or in order (random_map=False).
    """
    def __init__(self, *,
                maps: Iterable[GridMap],
                robot: Robot,
                reward_fn: RewardCallable,
                lidar_config: Optional[Dict] = None,
                max_iteration_steps: Union[int,np.integer] = 10_000,
                render_mode: Optional[str] = None,
                cell_size: int = 32,
                num_sections: int = Grid_Robot_Sections_Env._min_num_sections,
                random_map: bool = True,
                curriculum: Optional[SuccessCurriculum] = None
                ) -> None:
        self._setup(maps=maps, robot=robot, reward_fn=reward_fn, lidar_config=lidar_config,
                    max_iteration_steps=max_iteration_steps, render_mode=render_mode, cell_size=cell_size,
                    num_sections=num_sections, random_map=random_map, curriculum=curriculum)

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
