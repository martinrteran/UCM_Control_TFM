from tfm_control_ucm.core.grid_env.environment import *
import pygame


map = GridMap.load("./src/tfm_control_ucm/maps/map_simple.json")
env = Grid_Robot_Env(map=map, robot=GridRobot(),render_mode='human',lidar_config={}, cell_size=10)
env.reset()
env.render()

ACTION_MAP = {pygame.K_SPACE: 0, pygame.K_a:1, pygame.K_d:2}

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break
        if event.type == pygame.KEYDOWN:
            if event.key in ACTION_MAP:
                action = ACTION_MAP[event.key]
                obs, reward, done, truncated, info = env.step(action)
                env.render()
                print("Robot position:", env.robot.position, " with orientation:", env.robot.orientation)
                print("Reward:", reward)

                if done:
                    print("Finished")
                    env.reset()
                    env.render()

pygame.quit()
    

print("End of program")