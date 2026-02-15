from tfm_control_ucm.core.grid_env.environment import *
import pygame


map = GridMap.load("./src/tfm_control_ucm/maps/map_simple.json")
env = Grid_Robot_Env(map=map, robot=GridRobot(),render_mode='human')
env.reset()
env.render()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    

print(map)