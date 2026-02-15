"""
pygame_renderer.py

Pygame renderer for GridMap + Robot + Lidar rays.
Author: Martin
"""

import pygame
import numpy as np
from tfm_control_ucm.core.grid_env.map import GridMap
from tfm_control_ucm.core.grid_env.robot import GridRobot

class PygameRenderer:
    def __init__(
        self,
        gridmap: GridMap,
        cell_size=32,
        robot_color=(255, 0, 0),
        obstacle_color=(50, 50, 50),
        free_color=(230, 230, 230),
        ray_color=(0, 120, 255),
    ):
        pygame.init()

        self.gridmap = gridmap
        self.cell_size = cell_size

        self.robot_color = robot_color
        self.obstacle_color = obstacle_color
        self.free_color = free_color
        self.ray_color = ray_color

        width_px, height_px = gridmap.get_size() * cell_size

        self.screen = pygame.display.set_mode((width_px, height_px))
        pygame.display.set_caption("Robot Environment")

    # ------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------
    def draw_grid(self):
        for r in range(self.gridmap.get_height()):
            for c in range(self.gridmap.get_width()):
                rect = pygame.Rect(
                    c * self.cell_size,
                    r * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )

                color = (
                    self.obstacle_color
                    if self.gridmap.is_obstacle(r, c)
                    else self.free_color
                )

                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (180, 180, 180), rect, 1)

    def draw_robot(self, robot):
        r, c = robot.position
        rect = pygame.Rect(
            c * self.cell_size,
            r * self.cell_size,
            self.cell_size,
            self.cell_size,
        )
        pygame.draw.rect(self.screen, self.robot_color, rect)

    def draw_lidar(self, robot: GridRobot, distances):
        c, r = robot.position
        cx = c * self.cell_size + self.cell_size // 2
        cy = r * self.cell_size + self.cell_size // 2


        for dist, angle_rad in distances:
            dx = np.cos(angle_rad) * dist * self.cell_size
            dy = -np.sin(angle_rad) * dist * self.cell_size

            pygame.draw.line(
                self.screen,
                self.ray_color,
                (cx, cy),
                (cx + dx, cy + dy),
                2,
            )

    # ------------------------------------------------------------
    # Main render function
    # ------------------------------------------------------------
    def render(self, robot, lidar, distances):
        self.screen.fill((0, 0, 0))
        self.draw_grid()
        self.draw_robot(robot)
        self.draw_lidar(lidar, distances)
        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True