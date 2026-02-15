from tfm_control_ucm.core.grid_env.environment import *
import pygame


map = GridMap.load("./src/tfm_control_ucm/maps/map_simple.json")
env = Grid_Robot_Env(map=map, robot=GridRobot(),render_mode='human',lidar_config={})
env.reset()
env.render()
env.lidar.last_results[:,1] = np.rad2deg(env.lidar.last_results[:,1])
print(env.lidar.last_results)
print(env.robot.position)
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    

print(map)