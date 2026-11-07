"""
slippery_renderer.py

PygameRenderer that also draws the floor grip of a SlipperyGridMap: free cells go
from the normal free colour (friction 1) towards blue as the friction drops.

Author: Martin
"""
import pygame

from tfm_control_ucm.core.grid_env.slippery_map import friction_at
from tfm_control_ucm.renderers.grid_renderer import PygameRenderer


class SlipperyPygameRenderer(PygameRenderer):
    slippery_color = (90, 160, 255) # Colour of a free cell with friction 0

    def _floor_color(self, r, c):
        t = 1.0 - min(max(friction_at(self.gridmap, r, c), 0.0), 1.0)
        return tuple(int(f + t * (s - f)) for f, s in zip(self.free_color, self.slippery_color))

    def draw_grid(self):
        for r in range(self.gridmap.get_height()):
            for c in range(self.gridmap.get_width()):
                rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                color = self.obstacle_color if self.gridmap.is_obstacle(r, c) else self._floor_color(r, c)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (180, 180, 180), rect, 1)
