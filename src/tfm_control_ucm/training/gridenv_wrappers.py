from torchrl.envs import GymEnv
from tfm_control_ucm.environment.environment import RobotEnv
from tfm_control_ucm.environment.maps.gridmap import GridMap

def make_robot_env(render=False):
    return GymEnv(
        RobotEnv(
            gridmap=GridMap.load(),  # you can pass a map here
            start_pos=(5, 5),
            render_mode="human" if render else None
        ),
        from_pixels=False,
        allow_done=True
    )