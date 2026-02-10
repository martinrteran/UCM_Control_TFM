"""
test_render.py

Simple test to verify that the pygame renderer works with RobotEnv.
"""

import time
from tfm_control_ucm.environment.maps.gridmap import GridMap
from tfm_control_ucm.environment.environment import RobotEnv


def test_render():
    gm = GridMap(size=(10, 10), obstacles=[(3,3), (4,3), (5,3)])
    env = RobotEnv(gridmap=gm, start_pos=(5, 5), render_mode="human")

    obs, info = env.reset()

    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.05)

        if terminated or truncated:
            break

    assert True  # If no crash, test passes