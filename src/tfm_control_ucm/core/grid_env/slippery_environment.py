"""
slippery_environment.py

Grid_Robot_Sections_Env_MultiMap for slippery floors (SlipperyGridMap +
a robot of slippery_robot.py + core/slip.py).

BaseGridEnv is not changed: the robot does the slipping inside move() and
returns False only on a collision, so the parent's step() and end-of-episode
rules apply as they are. This class only adds:

- terrain_obs: proprioceptive values appended to the observation;
- reset(seed): seeds the robot's slip generator and the sensor-noise generator
  from the seed, apart from self.np_random, so a seeded evaluation sees the same
  maps and start/goal positions with or without slips;
- info: slipped, slip_kind, slip_maneuver, slip_probability, slip_collision and
  friction (when the robot has a slip model);
- rendering of the floor grip.

The reward receives the lidar + goal part of the observation in ctx.obs and
ctx.previous_obs (as in the other environments); the slip is in info.

Author: Martin
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Union

import numpy as np
from gymnasium import spaces

from tfm_control_ucm.core.curriculum import SuccessCurriculum
from tfm_control_ucm.core.rewards import RewardCallable
from tfm_control_ucm.renderers.slippery_renderer import SlipperyPygameRenderer
from .environment import Grid_Robot_Sections_Env, Grid_Robot_Sections_Env_MultiMap
from .map import GridMap
from .slippery_map import friction_at
from .slippery_robot import SlipperyRobotMixin

# Values appended to the observation for each terrain_obs mode
TERRAIN_OBS_SIZES = {"none": 0, "proprio": 4, "neighbors": 8}


class Slippery_Grid_Robot_Sections_Env_MultiMap(Grid_Robot_Sections_Env_MultiMap):
    """
    Sections environment over several maps where the robot can slip.

    Parameters
    ----------
    maps, reward_fn, lidar_config, max_iteration_steps, render_mode, cell_size, num_sections, random_map, curriculum
        As in Grid_Robot_Sections_Env_MultiMap. Maps are usually SlipperyGridMap;
        a plain GridMap has the default grip (never slips).
    robot : SlipperyRobotMixin
        SlipperyGridRobotNotTurning or SlipperyGridRobot; its slip_model decides
        the slips (None: no slips, momentum still tracked).
    terrain_obs : str, optional
        "none" (default): the observation of the parent;
        "proprio": + [momentum forward, momentum left, slipped on the last step,
        estimated grip of the current cell] (what odometry + IMU give a real robot);
        "neighbors": proprio + estimated grip of the front, left, back and right
        cells (like a camera-based terrain classifier). Egocentric.
    terrain_obs_noise : float, optional
        Standard deviation of the Gaussian noise of the grip estimates (clipped to [0, 1]).
    """
    def __init__(self, *,
                maps: Iterable[GridMap],
                robot: SlipperyRobotMixin,
                reward_fn: RewardCallable,
                lidar_config: Optional[Dict] = None,
                max_iteration_steps: Union[int, np.integer] = 10_000,
                render_mode: Optional[str] = None,
                cell_size: int = 32,
                num_sections: int = Grid_Robot_Sections_Env._min_num_sections,
                random_map: bool = True,
                curriculum: Optional[SuccessCurriculum] = None,
                terrain_obs: str = "none",
                terrain_obs_noise: float = 0.05
                ) -> None:
        if not isinstance(robot, SlipperyRobotMixin): raise TypeError("The robot must be a slippery robot (slippery_robot.py)")
        if terrain_obs not in TERRAIN_OBS_SIZES: raise ValueError(f"terrain_obs must be one of {list(TERRAIN_OBS_SIZES)}")
        if terrain_obs_noise < 0: raise ValueError("terrain_obs_noise must be >= 0")
        # Needed by _build_observation_space, which the parent's __init__ calls
        self.terrain_obs = terrain_obs
        self.terrain_obs_noise = float(terrain_obs_noise)
        self._base_obs: Optional[np.ndarray] = None # Lidar + goal part of the last observation
        self._obs_rng: Optional[np.random.Generator] = None
        super().__init__(maps=maps, robot=robot, reward_fn=reward_fn, lidar_config=lidar_config,  # type: ignore
                         max_iteration_steps=max_iteration_steps, render_mode=render_mode, cell_size=cell_size,
                         num_sections=num_sections, random_map=random_map, curriculum=curriculum)

    @property
    def slip_model(self):
        return self.robot.slip_model # type: ignore

    # ------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------
    def _build_observation_space(self) -> spaces.Box:
        space = super()._build_observation_space()
        n = TERRAIN_OBS_SIZES[self.terrain_obs]
        if n == 0: return space
        low = np.array([-1.0, -1.0] + [0.0] * (n - 2), dtype=np.float32)
        high = np.ones(n, dtype=np.float32)
        return spaces.Box(low=np.concatenate([space.low.ravel(), low]).astype(np.float32),
                          high=np.concatenate([space.high.ravel(), high]).astype(np.float32))

    def _get_observation(self):
        base = super()._get_observation()
        self._base_obs = base
        if self.terrain_obs == "none": return base
        return np.concatenate([np.asarray(base, dtype=np.float32), self._terrain_features()])

    def _estimated_friction(self, row: int, col: int) -> float:
        """Grip of a cell as a sensor would estimate it: true value + Gaussian noise, clipped to [0, 1]."""
        friction = friction_at(self.map, row, col)
        if self.terrain_obs_noise > 0 and self._obs_rng is not None:
            friction += float(self._obs_rng.normal(0.0, self.terrain_obs_noise))
        return float(np.clip(friction, 0.0, 1.0))

    def _terrain_features(self) -> np.ndarray:
        """
        Egocentric [momentum forward, momentum left, slipped on the last step, grip here]
        and, for "neighbors", the grip of the front, left, back and right cells.
        """
        robot: SlipperyRobotMixin = self.robot # type: ignore
        heading = self.robot.heading
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        dc, dr = robot.momentum
        x, y = dc, -dr # Lidar convention: x = east, y = north (rows grow downwards)
        features = [x * cos_h + y * sin_h, -x * sin_h + y * cos_h, float(robot.last_slip.kind != "none")]
        c, r = (int(v) for v in self.robot.position)
        features.append(self._estimated_friction(r, c))
        if self.terrain_obs == "neighbors":
            for ex, ey in ((1, 0), (0, 1), (-1, 0), (0, -1)): # front, left, back, right in the robot frame
                wx, wy = ex * cos_h - ey * sin_h, ex * sin_h + ey * cos_h
                features.append(self._estimated_friction(r - int(round(wy)), c + int(round(wx))))
        return np.asarray(features, dtype=np.float32)

    # ------------------------------------------------------------
    # Episode
    # ------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None or self._obs_rng is None:
            # Own generators for the slips and the sensor noise: self.np_random (maps, positions) is not touched
            slip_seed, obs_seed = np.random.SeedSequence(seed).spawn(2)
            self.robot.set_rng(np.random.default_rng(slip_seed)) # type: ignore
            self._obs_rng = np.random.default_rng(obs_seed)
        return super().reset(seed=seed, options=options)

    def _start_episode(self, obs: np.ndarray):
        super()._start_episode(self._base_obs) # type: ignore # The parent reads the lidar from it

    def _finish_step(self, action: int, obs: np.ndarray, hit_obstacle: bool):
        reward, terminated, truncated, info = super()._finish_step(action, self._base_obs, hit_obstacle) # type: ignore
        robot: SlipperyRobotMixin = self.robot # type: ignore
        if robot.slip_model is not None:
            slip = robot.last_slip
            info.update({'slipped': slip.kind != "none", 'slip_kind': slip.kind, 'slip_maneuver': slip.maneuver,
                         'slip_probability': slip.probability, 'slip_collision': robot.last_slip_collision,
                         'friction': robot.last_friction})
        return reward, terminated, truncated, info

    # ------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------
    def render(self):
        if self.render_mode == 'human' and self.renderer is None and self._previous_obs is not None:
            self.renderer = SlipperyPygameRenderer(self.map, cell_size=self.cell_size)
        super().render()
