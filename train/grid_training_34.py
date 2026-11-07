"""
Test 34 — test 33 with proprioception in the observation.

Same slippery maps and slip model as test 33 (core/slip.py; a slide into an
obstacle is a collision). The observation adds what a real robot gets from its
odometry and IMU (terrain_obs="proprio", egocentric): its momentum (last
displacement: forward, left), whether the last move slipped, and a noisy
estimate (std 0.05) of the grip of the cell it stands on. It does not see the
grip of the cells around it ("neighbors" would, like a camera terrain classifier).
With the momentum the slip kind becomes predictable from the observation (the
maneuver depends on it), so the slip dynamics are Markov in the observation.

--no-slippery runs the original maps of test 31 without slips but with the same
observation: test 31 + momentum only, to tell whether the momentum alone changes
the greedy loops (run it as test 35: --no-slippery --test_number 35).

Everything else is as in test 33.
"""
import argparse
import json
import os

from tfm_control_ucm.core.agents.agent import PPOAgent
from tfm_control_ucm.core.agents.networks import SimpleQNetwork, ValueNetwork
from tfm_control_ucm.core.agents.utils import PPOConfig, RLAlgorithm
from tfm_control_ucm.core.rewards import ProgressReward
from tfm_control_ucm.core.slip import SLIP_KINDS, SlipModel
import tfm_control_ucm.core.training.TrainingManager as training_manager
from tfm_control_ucm.core.training.slip_trainer import SlipSingleTrainer
from tfm_control_ucm.core.grid_env.slippery_environment import Slippery_Grid_Robot_Sections_Env_MultiMap
from tfm_control_ucm.core.grid_env.slippery_map import SlipperyGridMap
from tfm_control_ucm.core.grid_env.slippery_robot import SlipperyGridRobotNotTurning
import torch
import numpy as np

MAP_PATHS = [
    "./maps/map_obstacle_free_slippery.json",
    "./maps/map_1_obstacle_slippery.json",
    "./maps/map_4_square_obstacle_slippery.json",
    "./maps/map_simple_slippery.json",
]
PLAIN_MAP_PATHS = [ # Test 31's maps, for --no-slippery
    "./maps/map_obstacle_free.json",
    "./maps/map_1_obstacle.json",
    "./maps/map_4_square_obstacle.json",
    "./maps/map_simple.json",
]

def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for grid environment.")

    # Reward args (ProgressReward)
    parser.add_argument('--progress_scale', type=float, default=10.0, help='Weight of the normalised progress term.')
    parser.add_argument('--clip_negative_progress', action=argparse.BooleanOptionalAction, default=True, help='Give 0 instead of a penalty when moving away from the goal.')
    parser.add_argument('--step_penalty', type=float, default=0.05, help='Constant cost per step.')
    parser.add_argument('--max_proximity_penalty', type=float, default=1e-2, help='Proximity penalty when touching an obstacle.')
    parser.add_argument('--safety_margin_factor', type=float, default=2/3, help='Safety margin as a fraction of the lidar max range.')
    parser.add_argument('--goal_reward', type=float, default=20.0, help='Bonus for reaching the goal.')
    parser.add_argument('--collision_penalty', type=float, default=5.0, help='Penalty for hitting a wall or an obstacle.')

    # Slip args (SlipModel)
    parser.add_argument('--demand_straight', type=float, default=0.3, help='Grip needed to keep going straight (slip: overshoot).')
    parser.add_argument('--demand_start', type=float, default=0.4, help='Grip needed to start from rest (slip: wheels spin).')
    parser.add_argument('--demand_turn', type=float, default=0.6, help='Grip needed to turn 90 degrees (slip: diagonal).')
    parser.add_argument('--demand_reverse', type=float, default=0.8, help='Grip needed to reverse (slip: drag).')
    parser.add_argument('--slip_gain', type=float, default=2.0, help='Slope of p = clip(gain * (demand - mu), 0, max_prob).')
    parser.add_argument('--slip_max_prob', type=float, default=0.9, help='Upper bound of the slip probability.')
    parser.add_argument('--slip_kinds', nargs='+', choices=SLIP_KINDS, default=["overshoot", "stay", "diagonal", "drag"], help='Slip kinds that can happen.')
    parser.add_argument('--lateral_demand', type=float, default=0.3, help='Demand of the sideways drift (only with "lateral" in --slip_kinds).')
    parser.add_argument('--into_obstacle', choices=["collide", "stop"], default="collide", help='What a slide into an obstacle is.')
    parser.add_argument('--terrain_obs', choices=["none", "proprio", "neighbors"], default="proprio", help='Proprioceptive values added to the observation.')
    parser.add_argument('--slippery', action=argparse.BooleanOptionalAction, default=True, help='Slippery maps and slips (--no-slippery: test 31 maps, no slips, same observation).')
    parser.add_argument('--terrain_obs_noise', type=float, default=0.05, help='Std of the noise of the grip estimates.')

    # Environment args
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
    parser.add_argument('--agent_name', type=str, default="PPO Agent", help='Name of the agent.')
    parser.add_argument('--render', action=argparse.BooleanOptionalAction, default=False, help='Render the environment during training.')
    parser.add_argument('--seed', type=int, default=None, help='Seed for maps, start/goal positions, slips and network initialisation.')
    parser.add_argument('--test_number', type=int, required=False, default=34, help='Test number')

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    map_paths = MAP_PATHS if args.slippery else PLAIN_MAP_PATHS
    mapas = [SlipperyGridMap.load(path) for path in map_paths] # Plain map files load with the default grip

    n_sections = 4 * args.n
    lidar_config = {"num_rays": n_sections * 15, "max_range": 10, "fov": 2 * np.pi, "noise_std": 1e-3, "with_cache": True}

    reward = ProgressReward(progress_scale=args.progress_scale,
                            clip_negative_progress=args.clip_negative_progress,
                            step_penalty=args.step_penalty,
                            safety_margin=lidar_config["max_range"] * args.safety_margin_factor,
                            max_proximity_penalty=args.max_proximity_penalty,
                            goal_reward=args.goal_reward,
                            collision_penalty=args.collision_penalty)

    slip_model = SlipModel(demand_straight=args.demand_straight, demand_start=args.demand_start,
                           demand_turn=args.demand_turn, demand_reverse=args.demand_reverse,
                           gain=args.slip_gain, max_prob=args.slip_max_prob, kinds=tuple(args.slip_kinds),
                           lateral_demand=args.lateral_demand, into_obstacle=args.into_obstacle) if args.slippery else None
    robot_simple = SlipperyGridRobotNotTurning(slip_model=slip_model) # No slip model: deterministic, momentum still observed

    env = Slippery_Grid_Robot_Sections_Env_MultiMap(maps=mapas, robot=robot_simple, reward_fn=reward, cell_size=10,
                    max_iteration_steps=args.max_steps, num_sections=n_sections, lidar_config=lidar_config,
                    render_mode="human" if args.render else None,
                    terrain_obs=args.terrain_obs, terrain_obs_noise=args.terrain_obs_noise)
    env.reset(seed=args.seed) # Seeds the map selection and the slips; later resets keep the same generators

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

    logs_dir = f"./runs/Test/{args.test_number}"
    os.makedirs(logs_dir, exist_ok=True)
    with open(os.path.join(logs_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"script": os.path.basename(__file__),
                   "args": vars(args),
                   "reward": reward.get_config(),
                   "maps": map_paths,
                   "random_map": True,
                   "action_masking": True,
                   "robot": type(robot_simple).__name__,
                   "lidar": lidar_config,
                   "num_sections": n_sections,
                   "slip": slip_model.get_config() if slip_model else None,
                   "terrain_obs": args.terrain_obs,
                   "terrain_obs_noise": args.terrain_obs_noise}, f, indent=2)

    t_config = training_manager.TrainerConfig(
        eps_step=4,
        num_episodes=args.num_episodes,
        render=args.render,
        log_steps=False,
        action_masking=True
    )
    singleTrainer = SlipSingleTrainer(logs_dir, f"./checkpoints/Test/{args.test_number}", t_config) # Also logs Slip/* (none with --no-slippery)
    singleTrainer.add_agents(agent)
    singleTrainer.add_environments(env)

    singleTrainer.train()

if __name__ == '__main__':
    main()
