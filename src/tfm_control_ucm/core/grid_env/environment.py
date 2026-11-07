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
from tfm_control_ucm.core.slip import SlipModel, SlipEvent
from tfm_control_ucm.utils.path_planner import PathPlanner

def _wrap_angle(angle):
    """Wrap an angle (radians) to (-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))

# Values appended to the observation for each terrain_obs mode (see BaseGridEnv._terrain_features)
TERRAIN_OBS_SIZES = {"none": 0, "proprio": 4, "neighbors": 8}
_NO_SLIP = SlipEvent("none", "", 0.0, (), (0, 0))

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
    slip_model : SlipModel, optional
        If given, moves can slip depending on the grip of the cell the robot stands
        on (GridMap.friction) and on the maneuver relative to its momentum
        (core/slip.py). Without it the dynamics are deterministic, as before.
        Slips use their own random generator, seeded from reset(seed=...), so the
        sequence of maps and start/goal positions does not depend on them.
    terrain_obs : str, optional
        Proprioceptive values appended to the (flat) observation:
        "none" (default); "proprio": [momentum forward, momentum left, slipped on the
        last step, estimated grip of the current cell] (what odometry + IMU give a real
        robot); "neighbors": proprio + estimated grip of the front, left, back and
        right cells (like a camera-based terrain classifier). Egocentric.
    terrain_obs_noise : float, optional
        Standard deviation of the Gaussian noise of the grip estimates (clipped to [0, 1]).
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
    # True if _get_observation returns a 1-D array (the declared space may still be (N+1, 2)); terrain_obs needs it
    _flat_observation = False

    def __init__(self, *,
                maps: Iterable[GridMap],
                robot: Robot,
                reward_fn: RewardCallable,
                lidar_config: Optional[Dict] = None,
                max_iteration_steps: Union[int,np.integer] = 10_000,
                render_mode: Optional[str] = None,
                cell_size: int = 32,
                random_map: bool = True,
                curriculum: Optional[SuccessCurriculum] = None,
                slip_model: Optional[SlipModel] = None,
                terrain_obs: str = "none",
                terrain_obs_noise: float = 0.05
                ) -> None:
        if not isinstance(maps, Iterable): raise TypeError("The maps must be an iterable of GridMap instances")
        maps = list(maps)
        if not maps: raise ValueError("At least one map must be provided")
        if not all(isinstance(m, GridMap) for m in maps): raise TypeError("All the maps must be of type GridMap")
        if not isinstance(robot, Robot): raise TypeError("The robot must be of type Robot")
        if not callable(reward_fn): raise TypeError("The reward_fn must be a RewardFunction or a callable taking a StepContext")
        self._check_lidar_config(lidar_config)
        if curriculum is not None: curriculum.check_maps([m.name for m in maps])
        if slip_model is not None and not isinstance(slip_model, SlipModel): raise TypeError("The slip_model must be of type SlipModel")
        if terrain_obs not in TERRAIN_OBS_SIZES: raise ValueError(f"terrain_obs must be one of {list(TERRAIN_OBS_SIZES)}")
        if terrain_obs_noise < 0: raise ValueError("terrain_obs_noise must be >= 0")

        super().__init__()
        self.maps = maps
        self.map = maps[0] # Placeholder until the first reset selects one
        self._random_map = random_map
        self._map_idx: Optional[int] = None
        self.curriculum = curriculum
        self.robot = robot
        self.reward_fn = reward_fn
        self.slip_model = slip_model
        self.terrain_obs = terrain_obs
        self.terrain_obs_noise = float(terrain_obs_noise)
        self.lidar = GridLidar(**lidar_config) if lidar_config else self._default_lidar()
        self.max_steps = max_iteration_steps
        self.render_mode = render_mode
        self.cell_size = cell_size
        self.renderer = None

        # Largest diagonal over all maps, so the observation bounds and reward normalisation hold for every map
        self._map_diagonal = max(np.sqrt(np.sum(np.array(m.grid.shape) **2)) for m in self.maps)

        self.action_space = spaces.Discrete(robot.NUM_ACTIONS, start=0, dtype=np.int8)
        self.observation_space = self._extend_observation_space(self._build_observation_space())

        self.steps = 0
        self.goal_pos = np.array([0, 0])
        self._previous_action = -1 # Do nothing
        self._previous_obs = None
        self._base_obs = None # Lidar + goal part of the last observation (without the terrain values)
        self._previous_dist_to_goal = 0.0
        self._previous_geodesic_dist_to_goal = float("inf")
        self._goal_field = np.full(self.map.grid.shape, np.inf)

        # Dynamics state: last displacement (dc, dr) of the robot, (0, 0) when stopped, and whether it slipped
        self._momentum = (0, 0)
        self._last_slipped = False
        self._slip_rng: Optional[np.random.Generator] = None
        self._obs_rng: Optional[np.random.Generator] = None

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
    def _extend_observation_space(self, space: spaces.Box) -> spaces.Box:
        """Append the bounds of the terrain values (terrain_obs) to the subclass's observation space."""
        n = TERRAIN_OBS_SIZES[self.terrain_obs]
        if n == 0: return space
        if not self._flat_observation: raise ValueError(f"terrain_obs='{self.terrain_obs}' needs a flat observation; {type(self).__name__}'s is not")
        low = np.array([-1.0, -1.0] + [0.0] * (n - 2), dtype=np.float32)
        high = np.ones(n, dtype=np.float32)
        return spaces.Box(low=np.concatenate([space.low.ravel(), low]).astype(np.float32),
                          high=np.concatenate([space.high.ravel(), high]).astype(np.float32))

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
        """Shortest path length from the robot to the goal (wavefront BFS, turns not counted, friction ignored); inf if unreachable."""
        return self._geodesic_dist_to_goal()

    def _robot_cell(self):
        """(col, row) of the robot as ints."""
        c, r = (int(v) for v in self.robot.position)
        return c, r

    # ------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------
    def _observe(self) -> np.ndarray:
        """Subclass observation (lidar + goal) followed by the terrain values of terrain_obs."""
        base = self._get_observation()
        self._base_obs = np.asarray(base)
        if self.terrain_obs == "none": return base
        return np.concatenate([np.asarray(base, dtype=np.float32).ravel(), self._terrain_features()])

    def _estimated_friction(self, row: int, col: int) -> float:
        """Grip of a cell as a sensor would estimate it: true value + Gaussian noise, clipped to [0, 1]."""
        friction = self.map.friction_at(row, col)
        if self.terrain_obs_noise > 0 and self._obs_rng is not None:
            friction += float(self._obs_rng.normal(0.0, self.terrain_obs_noise))
        return float(np.clip(friction, 0.0, 1.0))

    def _terrain_features(self) -> np.ndarray:
        """
        Egocentric proprioceptive values (terrain_obs):
        [momentum forward, momentum left, slipped on the last step, grip here]
        and, for "neighbors", the grip of the front, left, back and right cells.
        """
        heading = self.robot.heading
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        dc, dr = self._momentum
        x, y = dc, -dr # Lidar convention: x = east, y = north (rows grow downwards)
        features = [x * cos_h + y * sin_h, -x * sin_h + y * cos_h, float(self._last_slipped)]
        c, r = self._robot_cell()
        features.append(self._estimated_friction(r, c))
        if self.terrain_obs == "neighbors":
            for ex, ey in ((1, 0), (0, 1), (-1, 0), (0, -1)): # front, left, back, right in the robot frame
                wx, wy = ex * cos_h - ey * sin_h, ex * sin_h + ey * cos_h
                features.append(self._estimated_friction(r - int(round(wy)), c + int(round(wx))))
        return np.asarray(features, dtype=np.float32)

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
        if seed is not None or self._slip_rng is None:
            # Slips and sensor noise get their own generators: the maps and start/goal positions
            # (self.np_random) stay the same for every agent evaluated with the same seed
            slip_seed, obs_seed = np.random.SeedSequence(seed).spawn(2)
            self._slip_rng, self._obs_rng = np.random.default_rng(slip_seed), np.random.default_rng(obs_seed)
        previous_map = self.map
        self.map = self._select_map()
        if self.map is not previous_map: self.renderer = None # The renderer draws a single map

        self.robot.random_reset(self.map, self.np_random)
        self.goal_pos = self._sample_goal()
        self._compute_goal_field()
        self.steps = 0
        self._momentum = (0, 0) # The robot starts stopped
        self._last_slipped = False

        obs = self._observe()
        self._start_episode(obs)
        info = {'map': self.map.name, 'optimal_steps': self._optimal_steps(), 'dist2goal': self._dist_to_goal(),
                'action_mask': self.robot.valid_actions(self.map)}
        if self.curriculum is not None: info['curriculum_stage'] = self.curriculum.stage_idx

        return obs, info

    def step(self, action: int):
        self.steps += 1
        start_c, start_r = self._robot_cell()
        friction = self.map.friction_at(start_r, start_c)

        if self.slip_model is None:
            moved = self.robot.move(self.map, action)
            hit_obstacle, slip, slip_collision = not moved, _NO_SLIP, False
            end_c, end_r = self._robot_cell()
            self._momentum = (end_c - start_c, end_r - start_r) # (0, 0) after a collision or a turn in place
        else:
            hit_obstacle, slip, slip_collision = self._move_with_slip(action, friction)
        self._last_slipped = slip.kind != "none"

        obs = self._observe()
        reward, terminated, truncated, info = self._finish_step(action, obs, hit_obstacle=hit_obstacle,
                                                                slip=slip, slip_collision=slip_collision, friction=friction)
        return obs, reward, terminated, truncated, info

    def _move_with_slip(self, action: int, friction: float):
        """
        Move the robot with the slip model; updates the momentum.

        A move into a wall is a collision, as without slipping, and turning in place
        never slips. Otherwise the slip model picks what happens (core/slip.py) and
        each displacement of its path must land on a free cell: if one does not, the
        slide hit something, which is a collision when into_obstacle is "collide"
        (the robot stays on the last free cell either way).

        Returns
        -------
        tuple
            (hit_obstacle, SlipEvent, slip_collision)
        """
        assert self.slip_model is not None
        intended = self.robot.displacement(action)
        c, r = self._robot_cell()
        if intended == (0, 0) or not self.map.is_free(r + intended[1], c + intended[0]):
            moved = self.robot.move(self.map, action) # Turn in place, or plain collision
            self._momentum = (0, 0)
            return not moved, _NO_SLIP, False

        event = self.slip_model.sample(friction, self._momentum, intended, self._slip_rng) # type: ignore
        for dc, dr in event.path:
            if not self.robot.translate(self.map, dc, dr):
                self._momentum = (0, 0) # Stopped against the obstacle
                collided = self.slip_model.into_obstacle == "collide"
                return collided, event, collided
        self._momentum = event.momentum
        return False, event, False

    def _start_episode(self, obs: np.ndarray):
        """Reset the per-episode memory used to build the next StepContext."""
        self._previous_action = -1 # Do nothing
        self._previous_obs = obs.copy()
        self._previous_dist_to_goal = self._dist_to_goal()
        self._previous_geodesic_dist_to_goal = self._geodesic_dist_to_goal()

    def _finish_step(self, action: int, obs: np.ndarray, hit_obstacle: bool, slip: SlipEvent = _NO_SLIP,
                     slip_collision: bool = False, friction: float = GridMap.DEFAULT_FRICTION):
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
            True if the robot hit a wall or an obstacle (also when a slide did, with into_obstacle="collide").
        slip : SlipEvent, optional
            What the slip model did this step.
        slip_collision : bool, optional
            The collision was caused by a slide.
        friction : float, optional
            Grip of the cell the robot stood on when the step started.

        Returns
        -------
        tuple
            (reward, terminated, truncated, info) as expected by gymnasium's step().
        """
        dist_to_goal = self._dist_to_goal()
        geodesic_dist_to_goal = self._geodesic_dist_to_goal()
        base_obs = self._base_obs if self._base_obs is not None else obs # Lidar + goal, without the terrain values
        rays = np.asarray(base_obs).reshape((-1, 2))[:-1, 0]
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
                          previous_geodesic_dist_to_goal=self._previous_geodesic_dist_to_goal,
                          slipped=slip.kind != "none", slip_kind=slip.kind, friction=float(friction))
        reward, components = evaluate_reward(self.reward_fn, ctx)

        info = {'action_mask': self.robot.valid_actions(self.map), # Valid actions in the new state
                'reached_goal': reached_goal, 'hit_obstacle': hit_obstacle, 'max_steps_reached': max_steps_reached,
                'dist2goal': dist_to_goal, 'min_dist': min_obstacle_dist, 'steps': self.steps,
                'goal_pos': self.goal_pos, 'robot_pos': self.robot.position, 'map': self.map.name,
                'reward': reward, 'reward_components': components, **summarize_components(components)}
        if self.slip_model is not None:
            info.update({'slipped': slip.kind != "none", 'slip_kind': slip.kind, 'slip_maneuver': slip.maneuver,
                         'slip_probability': slip.probability, 'slip_collision': slip_collision, 'friction': float(friction)})
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
        if self.render_mode != 'human' or self._base_obs is None: return
        if self.renderer is None: self.renderer = PygameRenderer(self.map, cell_size=self.cell_size)
        rays = np.asarray(self._base_obs, dtype=np.float64).reshape((-1, 2))[:-1].copy()
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
                curriculum: Optional[SuccessCurriculum] = None,
                slip_model: Optional[SlipModel] = None,
                terrain_obs: str = "none",
                terrain_obs_noise: float = 0.05
                ) -> None:
        if not isinstance(map, GridMap): raise TypeError("The map must be of type GridMap")
        super().__init__(maps=[map], robot=robot, reward_fn=reward_fn, lidar_config=lidar_config,
                         max_iteration_steps=max_iteration_steps, render_mode=render_mode, cell_size=cell_size,
                         curriculum=curriculum, slip_model=slip_model, terrain_obs=terrain_obs,
                         terrain_obs_noise=terrain_obs_noise)

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
    _flat_observation = True
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
                curriculum: Optional[SuccessCurriculum] = None,
                slip_model: Optional[SlipModel] = None,
                terrain_obs: str = "none",
                terrain_obs_noise: float = 0.05
                ) -> None:
        if not isinstance(map, GridMap): raise TypeError("The map must be of type GridMap")
        self._setup(maps=[map], robot=robot, reward_fn=reward_fn, lidar_config=lidar_config,
                    max_iteration_steps=max_iteration_steps, render_mode=render_mode, cell_size=cell_size,
                    num_sections=num_sections, random_map=True, curriculum=curriculum, slip_model=slip_model,
                    terrain_obs=terrain_obs, terrain_obs_noise=terrain_obs_noise)

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
                curriculum: Optional[SuccessCurriculum] = None,
                slip_model: Optional[SlipModel] = None,
                terrain_obs: str = "none",
                terrain_obs_noise: float = 0.05
                ) -> None:
        self._setup(maps=maps, robot=robot, reward_fn=reward_fn, lidar_config=lidar_config,
                    max_iteration_steps=max_iteration_steps, render_mode=render_mode, cell_size=cell_size,
                    num_sections=num_sections, random_map=random_map, curriculum=curriculum, slip_model=slip_model,
                    terrain_obs=terrain_obs, terrain_obs_noise=terrain_obs_noise)

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
