"""
Test 37 — the plain DQNAgent (no action masking) on the setup of test 36.

Test 36 trains MaskedDQNAgent (Double DQN + masking). This one trains the
original DQNAgent (core/agents/agent.py, with the 2026-11-07 fixes of its
target-network schedule and evaluation epsilon), which does not mask actions:
the robot can move into walls and those collisions end the episode, as in the
PPO tests before 31 (test 25).

--no-double_dqn (default here): classic DQN target r + gamma * max_a Q_target(s', a).
--double_dqn: Double DQN target (online network picks, target network evaluates).
Run the Double DQN variant as test 38:
    python train/grid_training_37.py --seed 0 --double_dqn --test_number 38

Same maps, reward, lidar, step limit, seed, network body, episode budget and DQN
hyperparameters as test 36, except what DQNAgent does differently from
MaskedDQNAgent: no masking, epsilon decays exponentially
(eps = eps_end + (eps_start - eps_end) * exp(-step / eps_decay); eps_decay 150k
gives eps 0.084 at 500k steps, close to test 36's linear schedule reaching 0.05
there) and training starts as soon as the buffer holds one batch.
"""
import argparse
import json
import os

from tfm_control_ucm.core.agents.agent import DQNAgent
from tfm_control_ucm.core.agents.networks import SimpleQNetwork
from tfm_control_ucm.core.agents.utils import DQNConfig, RLAlgorithm
from tfm_control_ucm.core.rewards import ProgressReward
import tfm_control_ucm.core.training.TrainingManager as training_manager
import tfm_control_ucm.core.grid_env.environment as grid_env
import tfm_control_ucm.core.grid_env.map as maps
import tfm_control_ucm.core.grid_env.robot as robot
import torch
import numpy as np

MAP_PATHS = [
    "./maps/map_obstacle_free.json",
    "./maps/map_1_obstacle.json",
    "./maps/map_4_square_obstacle.json",
    "./maps/map_simple.json",
]

def main():
    parser = argparse.ArgumentParser(description="Train the plain DQNAgent (no masking) for the grid environment.")

    # Reward args (ProgressReward), as in test 31
    parser.add_argument('--progress_scale', type=float, default=10.0, help='Weight of the normalised progress term.')
    parser.add_argument('--clip_negative_progress', action=argparse.BooleanOptionalAction, default=True, help='Give 0 instead of a penalty when moving away from the goal.')
    parser.add_argument('--step_penalty', type=float, default=0.05, help='Constant cost per step.')
    parser.add_argument('--max_proximity_penalty', type=float, default=1e-2, help='Proximity penalty when touching an obstacle.')
    parser.add_argument('--safety_margin_factor', type=float, default=2/3, help='Safety margin as a fraction of the lidar max range.')
    parser.add_argument('--goal_reward', type=float, default=20.0, help='Bonus for reaching the goal.')
    parser.add_argument('--collision_penalty', type=float, default=5.0, help='Penalty for hitting a wall or an obstacle.')

    # Environment args
    parser.add_argument('--n', type=int, default=2, help='Factor to determine number of sections (4*n).')
    parser.add_argument('--max_steps', type=int, default=400, help='Maximum number of steps per episode.')

    # DQN args
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate.')
    parser.add_argument('--buffer_size', type=int, default=100_000, help='Replay buffer size (transitions).')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor.')
    parser.add_argument('--max_grad_norm', type=float, default=10.0, help='Max gradient norm.')
    parser.add_argument('--eps_start', type=float, default=1.0, help='Initial epsilon.')
    parser.add_argument('--eps_end', type=float, default=0.05, help='Final epsilon.')
    parser.add_argument('--eps_decay', type=int, default=150_000, help='Time constant (training steps) of the exponential epsilon decay.')
    parser.add_argument('--target_update', type=int, default=1_000, help='Gradient updates between target-network copies.')
    parser.add_argument('--double_dqn', action=argparse.BooleanOptionalAction, default=False, help='Double DQN target (test 38).')
    parser.add_argument('--train_every', type=int, default=4, help='Environment steps between gradient updates.')

    # Trainer config args
    parser.add_argument('--num_episodes', type=int, default=40_000, help='Total number of episodes to train.')
    parser.add_argument('--agent_name', type=str, default="DQN Agent", help='Name of the agent.')
    parser.add_argument('--render', action=argparse.BooleanOptionalAction, default=False, help='Render the environment during training.')
    parser.add_argument('--seed', type=int, default=None, help='Seed for maps, start/goal positions, exploration and network initialisation.')
    parser.add_argument('--test_number', type=int, required=False, default=37, help='Test number')

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    mapas = [maps.GridMap.load(path) for path in MAP_PATHS]

    robot_simple = robot.GridRobotNotTurning()

    n_sections = 4 * args.n
    lidar_config = {"num_rays": n_sections * 15, "max_range": 10, "fov": 2 * np.pi, "noise_std": 1e-3, "with_cache": True}

    reward = ProgressReward(progress_scale=args.progress_scale,
                            clip_negative_progress=args.clip_negative_progress,
                            step_penalty=args.step_penalty,
                            safety_margin=lidar_config["max_range"] * args.safety_margin_factor,
                            max_proximity_penalty=args.max_proximity_penalty,
                            goal_reward=args.goal_reward,
                            collision_penalty=args.collision_penalty)

    env = grid_env.Grid_Robot_Sections_Env_MultiMap(maps=mapas, robot=robot_simple, reward_fn=reward, cell_size=10,
                    max_iteration_steps=args.max_steps, num_sections=n_sections, lidar_config=lidar_config,
                    render_mode="human" if args.render else None)
    env.reset(seed=args.seed) # Seeds the map selection; later resets keep the same generator

    action_dim = env.get_action_dim()
    obv_dim = env.get_observation_dim()

    rl_config = DQNConfig(obs_dim=obv_dim, action_dim=action_dim,
                          algorithm=RLAlgorithm.DDQN if args.double_dqn else RLAlgorithm.DQN,
                          device=device, name=args.agent_name,
                          lr=args.lr,
                          buffer_size=args.buffer_size,
                          batch_size=args.batch_size,
                          gamma=args.gamma,
                          max_grad_norm=args.max_grad_norm,
                          eps_start=args.eps_start,
                          eps_end=args.eps_end,
                          eps_decay=args.eps_decay,
                          target_update=args.target_update,
                          double_dqn=args.double_dqn)

    q_net = SimpleQNetwork(obs_dim=obv_dim, action_dim=action_dim, num_hidden_layers=3, hidden_dim=64).to(device)
    agent = DQNAgent(rl_config, q_network=q_net)

    logs_dir = f"./runs/Test/{args.test_number}"
    os.makedirs(logs_dir, exist_ok=True)
    with open(os.path.join(logs_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"script": os.path.basename(__file__),
                   "algorithm": "ddqn" if args.double_dqn else "dqn",
                   "agent": "DQNAgent",
                   "args": vars(args),
                   "reward": reward.get_config(),
                   "maps": MAP_PATHS,
                   "random_map": True,
                   "action_masking": False,
                   "robot": type(robot_simple).__name__,
                   "lidar": lidar_config,
                   "num_sections": n_sections}, f, indent=2)

    t_config = training_manager.TrainerConfig(
        eps_step=args.train_every,
        num_episodes=args.num_episodes,
        render=args.render,
        log_steps=False,
        action_masking=False # DQNAgent does not mask actions
    )
    singleTrainer = training_manager.SingleTrainer(logs_dir, f"./checkpoints/Test/{args.test_number}", t_config)
    singleTrainer.add_agents(agent)
    singleTrainer.add_environments(env)

    singleTrainer.train()

if __name__ == '__main__':
    main()
