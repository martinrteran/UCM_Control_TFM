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

from .maps.gridmap import GridMap
from .robots.grid_robot import Grid_Robot
from .sensors.grid_lidar import Grid_Lidar

from src.tfm_control_ucm.renderers.grid_renderer import PygameRenderer

class RobotEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        *,
        gridmap: GridMap,
        start_pos=(0, 0),
        start_orientation="N",
        goal_pos=None,
        lidar_config=None,
        max_steps=200,
        render_mode=None,
    ):
        super().__init__()

        self.gridmap = gridmap
        self.start_pos = start_pos
        self.start_orientation = start_orientation
        self.goal_pos = goal_pos
        self.max_steps = max_steps
        self.render_mode = render_mode

        # LIDAR configuration
        if lidar_config is None:
            lidar_config = dict(num_rays=9, max_range=10, fov_deg=180)

        self.lidar = Grid_Lidar(**lidar_config)

        # Action space: forward, turn left, turn right
        self.action_space = spaces.Discrete(3)

        # Observation space: LIDAR distances
        num_rays = lidar_config["num_rays"]
        self.observation_space = spaces.Box(
            low=0.0,
            high=float(lidar_config["max_range"]),
            shape=(num_rays,),
            dtype=np.float32,
        )

        self.robot = None
        self.steps = 0

    # ------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.robot = Grid_Robot(
            gridmap=self.gridmap,
            position=self.start_pos,
            orientation=self.start_orientation,
        )

        self.steps = 0

        obs = self._get_observation()
        info = {}

        return obs, info

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