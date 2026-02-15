"""
pygame_renderer.py

Pygame renderer for GridMap + Robot + Lidar rays.
Author: Martin
"""

import pygame
import numpy as np
from tfm_control_ucm.core.grid_env.map import GridMap
from tfm_control_ucm.core.grid_env.robot import GridRobot
from tfm_control_ucm.core.grid_env.sensor import  GridLidar

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
        self.max_ray_color = (100,120,255)
        self.goal_color = (0, 255, 0)

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

    def draw_robot(self, robot: GridRobot):
        c, r = robot.position
        rect = pygame.Rect(
            c * self.cell_size,
            r * self.cell_size,
            self.cell_size,
            self.cell_size,
        )
        pygame.draw.rect(self.screen, self.robot_color, rect)
        
        c = c * self.cell_size + self.cell_size//2
        r = r * self.cell_size + self.cell_size//2
        ce = c + np.cos(robot.ORIENTATIONS[robot.orientation]['angle']) * self.cell_size*2
        re = r - np.sin(robot.ORIENTATIONS[robot.orientation]['angle']) * self.cell_size*2

        pygame.draw.line(self.screen,(0,150,0),(c,r), (ce,re), 3)
    
    def draw_goal(self, goal_position):
        c, r = goal_position
        rect = pygame.Rect(
            c * self.cell_size,
            r * self.cell_size,
            self.cell_size,
            self.cell_size,
        )
        pygame.draw.rect(self.screen, self.goal_color, rect)

    def draw_vector(self, start, end):
        start = np.asarray(start)
        end = np.asarray(end)
        sc, sr = start * self.cell_size + self.cell_size//2
        ec, er = end * self.cell_size + self.cell_size//2
        pygame.draw.line(self.screen, (255,0,0), (sc, sr), (ec, er), 2)

    def draw_lidar(self, robot, lidar, distances):
        c, r = robot.position
        cx = c * self.cell_size + self.cell_size // 2
        cy = r * self.cell_size + self.cell_size // 2


        for dist, angle_rad in distances:
            if dist < 0: continue
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
    def render(self, robot, lidar, distances, goal_position):
        self.screen.fill((0, 0, 0))
        self.draw_grid()
        
        self.draw_robot(robot)
        self.draw_lidar(robot, lidar, distances)
        self.draw_goal(goal_position)
        self.draw_vector(robot.position, goal_position)

        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True