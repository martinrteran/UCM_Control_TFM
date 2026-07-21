from tfm_control_ucm.core.agents.agent import  PPOAgent
from tfm_control_ucm.core.agents.networks import SimpleQNetwork,ValueNetwork
from tfm_control_ucm.core.agents.utils import RLConfig, RLAlgorithm, PPOConfig
import tfm_control_ucm.core.training.TrainingManager as training_manager
import tfm_control_ucm.core.grid_env.environment as grid_env
import tfm_control_ucm.core.grid_env.map as maps
import tfm_control_ucm.core.grid_env.robot as robot
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
mapa_simple = maps.GridMap.load("./maps/map_4_square_obstacle.json")
robot_simple = robot.GridRobotNotTurning()
n = 2 # 3
n_sections = 4 * n
lidar_config = {"num_rays": n_sections*15, "max_range": 10, "fov": 2*np.pi, "noise_std": 1e-3, "with_cache": True}
env = grid_env.Grid_Robot_Sections_Env(map=mapa_simple, robot=robot_simple, cell_size=10, 
                max_iteration_steps=400, num_sections=n_sections, lidar_config=lidar_config, render_mode="human")
env._max_proximity_penalty = 1e-2# 1.5
env._safety_margin = env._max_range*2/3 #2/3

action_dim = env.get_action_dim()
obv_dim = env.get_observation_dim()

rl_config = PPOConfig(obs_dim=obv_dim, action_dim=action_dim, algorithm=RLAlgorithm.PPO,
                     device=device,name="PPO Agent, 0.01 penalty, 2/3 safety margin and 8 sections", batch_size=64, lr=1e-3, gamma=0.99,
                     eps_start=1.0, eps_end=0.05, eps_decay=1_000_000, buffer_size=100_000,
                     max_grad_norm=5.0)

policy_net = SimpleQNetwork(obs_dim=obv_dim, action_dim=action_dim, num_hidden_layers=2, hidden_dim=64).to(device)
target_net = ValueNetwork(obs_dim=obv_dim, hidden_dim=64).to(device)
agent = PPOAgent(rl_config, policy_network = policy_net, value_network = target_net)
agent.load(rf"checkpoints\Test\19\PPO Agent, 0.01 penalty, 2-3 safety margin and 8 sections 40k\final.pth")

t_config = training_manager.TrainerConfig(
    eps_step=4,
    num_episodes=20_000,
    render=True
)
singleTrainer = training_manager.PPOTrainer("./runs/Test/19","./checkpoints/Test/19",t_config)

singleTrainer.simulate(agent,env,50,True)