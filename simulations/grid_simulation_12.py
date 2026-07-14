from tfm_control_ucm.core.agents.agent import AgentFactory, GridAgent_2_FIXED,GridAgent
from tfm_control_ucm.core.agents.networks import SimpleQNetwork
from tfm_control_ucm.core.agents.utils import RLConfig, RLAlgorithm
import tfm_control_ucm.core.training.TrainingManager as training_manager
import tfm_control_ucm.core.grid_env.environment as grid_env
import tfm_control_ucm.core.grid_env.map as maps
import tfm_control_ucm.core.grid_env.robot as robot
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
mapa_simple = maps.GridMap.load("./src/tfm_control_ucm/maps/map_obstacle_free.json")
robot_simple = robot.GridRobot()
lidar_config = {"num_rays": 80, "max_range": 10, "fov": 2*np.pi, "noise_std": 1e-3, "with_cache": True}
env = grid_env.Grid_Robot_Sections_Env(map=mapa_simple, robot=robot_simple, cell_size=10, max_iteration_steps=300, num_sections=8, lidar_config=lidar_config, render_mode="human")

action_dim = env.get_action_dim()
obv_dim = env.get_observation_dim()
rl_config = RLConfig(obs_dim=obv_dim, action_dim=action_dim, algorithm=RLAlgorithm.SIMPLE_Q_NETWORK,
                     device=device,name="Simple Q Network", double_dqn = False, batch_size=128, lr=1e-3, gamma=0.99,
                     eps_start=1.0, eps_end=0.005, eps_decay=1_000_000, target_update=1000, buffer_size=100_000)

policy_net = SimpleQNetwork(obs_dim=obv_dim, action_dim=action_dim, num_hidden_layers=2, hidden_dim=128).to(device)
target_net = SimpleQNetwork(obs_dim=obv_dim, action_dim=action_dim, num_hidden_layers=2, hidden_dim=128).to(device)
agent = GridAgent(rl_config, policy_net = policy_net, target_net = target_net)
agent.load(rf"./checkpoints/Test/12/Simple Q Network/final.pth")

t_config = training_manager.TrainerConfig(
    eps_step=0,
    num_episodes=50_000,
    render=True
)
singleTrainer = training_manager.SingleTrainer("./runs/Test/12","./checkpoints/Test/12",t_config)
singleTrainer.add_agents(agent)
singleTrainer.add_environments(env)

singleTrainer.simulate(agent, env, num_episodes=100, render=True)
# singleTrainer.train