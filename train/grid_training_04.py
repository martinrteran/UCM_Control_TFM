from tfm_control_ucm.core.agents.agent import AgentFactory, GridAgent_2_FIXED,GridAgent
from tfm_control_ucm.core.agents.networks import GridAgentNet
from tfm_control_ucm.core.agents.utils import RLConfig, RLAlgorithm
import tfm_control_ucm.core.training.TrainingManager as training_manager
import tfm_control_ucm.core.grid_env.environment as grid_env
import tfm_control_ucm.core.grid_env.map as maps
import tfm_control_ucm.core.grid_env.robot as robot
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
mapa_simple = maps.GridMap.load("./src/tfm_control_ucm/maps/map_simple.json")
robot_simple = robot.GridRobot()
lidar_config = {"num_rays": 30, "max_range": 10, "fov": 2*np.pi, "noise_std": 1e-3, "with_cache": True}
env = grid_env.Grid_Robot_Sections_Env(map=mapa_simple, robot=robot_simple, cell_size=10)

action_dim = env.get_action_dim()
obv_dim = env.get_observation_dim()
rl_config = RLConfig(obs_dim=obv_dim, action_dim=action_dim, algorithm=RLAlgorithm.GRID_AGENT,
                     device=device,name="Grid_Agent", double_dqn = False, batch_size=64, lr=1e-3, gamma=0.90,
                     eps_start=5.0, eps_end=0.005, eps_decay=100_000_000, target_update=1000, buffer_size=1_000_000_000)

policy_net = GridAgentNet(obs_dim=obv_dim, action_dim=action_dim).to(device)
target_net = GridAgentNet(obs_dim=obv_dim, action_dim=action_dim).to(device)
agent = GridAgent(rl_config, policy_net = policy_net, target_net = target_net)

t_config = training_manager.TrainerConfig(
    eps_step=10,
    num_episodes=100_000,
    useSoftReset=False,
)
singleTrainer = training_manager.SingleTrainer("./runs/Test/04/100k","./checkpoints/Test/04/100k",t_config)
singleTrainer.add_agents(agent)
singleTrainer.add_environments(env)

singleTrainer.train()