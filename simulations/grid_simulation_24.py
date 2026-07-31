import argparse
from tfm_control_ucm.core.agents.agent import PPOAgent
from tfm_control_ucm.core.agents.networks import SimpleQNetwork, ValueNetwork
from tfm_control_ucm.core.agents.utils import PPOConfig, RLAlgorithm
import tfm_control_ucm.core.training.TrainingManager as training_manager
import tfm_control_ucm.core.grid_env.environment as grid_env
import tfm_control_ucm.core.grid_env.map as maps
import tfm_control_ucm.core.grid_env.robot as robot
import torch
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for grid environment.")
    
    # Environment args
    parser.add_argument('--max_proximity_penalty', type=float, default=1e-2, help='Max proximity penalty for the environment.')
    parser.add_argument('--safety_margin_factor', type=float, default=2/3, help='Factor of max_range for safety margin.')
    parser.add_argument('--step_penalty', type=float, default=0.05, help='Factor of base penalty.')
    parser.add_argument('--n', type=int, default=2, help='Factor to determine number of sections (4*n).')
    parser.add_argument('--max_steps', type=int, default=400, help='Maximum number of steps per episode.')
    
    # Agent config args
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--buffer_size', type=int, default=100_000, help='Replay buffer size.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Max gradient norm.')
    parser.add_argument('--clip_eps', type=float, default=0.2, help='PPO clip epsilon.')
    parser.add_argument('--entropy_coeff', type=float, default=0.01, help='Entropy coefficient.')
    parser.add_argument('--value_coeff', type=float, default=0.5, help='Value function coefficient.')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor.')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda.')

    # Trainer config args
    parser.add_argument('--num_episodes', type=int, default=40_000, help='Total number of episodes to train.')
    parser.add_argument('--expanded_logs', type=bool, default=False, help='Whether to use expanded logs while training.')
    parser.add_argument('--agent_name', type=str, default="PPO Agent", help='Name of the agent.')
    parser.add_argument('--render', type=bool, default=False, help='Whether to render the environment during training.')
    parser.add_argument('--test_number', type=int,required=False,default=24, help='Test number')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    mapa_4 = maps.GridMap.load("./maps/map_4_square_obstacle.json")
    mapa_1 = maps.GridMap.load("./maps/map_1_obstacle.json")
    mapa_0 = maps.GridMap.load("./maps/map_obstacle_free.json")
    mapa_simple = maps.GridMap.load("./maps/map_simple.json")

    robot_simple = robot.GridRobotNotTurning()
    
    n_sections = 4 * args.n
    lidar_config = {"num_rays": n_sections * 15, "max_range": 10, "fov": 2 * np.pi, "noise_std": 1e-3, "with_cache": True}
    
    env = grid_env.Grid_Robot_Sections_Env_MultiMap(maps=[mapa_0, mapa_1, mapa_4, mapa_simple], robot=robot_simple, cell_size=10, 
                    max_iteration_steps=args.max_steps, num_sections=n_sections, lidar_config=lidar_config, render_mode="human")
    env._max_proximity_penalty = args.max_proximity_penalty
    env._safety_margin = env.lidar.max_range * args.safety_margin_factor
    env._step_penalty = args.step_penalty

    action_dim = env.get_action_dim()
    obv_dim = env.get_observation_dim()

    rl_config = PPOConfig(obs_dim=obv_dim, action_dim=action_dim, algorithm=RLAlgorithm.PPO,
                         device=device, name=args.agent_name,
                         lr=args.lr,
                         buffer_size=args.buffer_size,
                         batch_size=args.batch_size,
                         max_grad_norm=args.max_grad_norm,
                         clip_eps=args.clip_eps,
                         entropy_coeff=args.entropy_coeff,
                         value_coeff=args.value_coeff,
                         gamma=args.gamma,
                         gae_lambda=args.gae_lambda)

    policy_net = SimpleQNetwork(obs_dim=obv_dim, action_dim=action_dim, num_hidden_layers=3, hidden_dim=64).to(device)
    target_net = ValueNetwork(obs_dim=obv_dim, hidden_dim=64).to(device)
    agent = PPOAgent(rl_config, policy_network=policy_net, value_network=target_net)
    agent.load(rf"./checkpoints/Test/{args.test_number}/{args.agent_name}/final.pth")

    t_config = training_manager.TrainerConfig(
        eps_step=4,
        num_episodes=args.num_episodes,
        render=False
    )
    singleTrainer = training_manager.PPOTrainer(f"./runs/Test/{args.test_number}", f"./checkpoints/Test/{args.test_number}", t_config)
    singleTrainer.add_agents(agent)
    singleTrainer.add_environments(env)

    singleTrainer.simulate(agent,env,50,True)

if __name__ == '__main__':
    main()