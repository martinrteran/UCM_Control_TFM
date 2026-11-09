"""
Test 45 — test 42 with dual-clip PPO (c = 3).

Dual-clip PPO (Ye et al. 2020, "Mastering Complex Control in MOBA Games with Deep RL",
AAAI): for negative advantages the clipped objective is also bounded from below,
L = max(min(r·A, clip(r)·A), c·A), c = 3 as in the paper. It limits the variance of very large
ratios on bad actions; it was designed for off-policy batches, so with on-policy rollouts and
target_kl it may change little.

Test 42 = textbook PPO (StandardPPOAgent, rollout 2048, 10 epochs, minibatch 64, lr 3e-4,
GAE 0.95, clip 0.2, entropy 0.01, value 0.5, grad clip 0.5) + target_kl 0.015, on the setup of
test 35 (masking + momentum observation, plain maps, no slips): greedy 97.0 ± 0.8 %, sampled
98.3 ± 0.6 % over 5 seeds; the remaining gap to PPOAgent (99.4 %) is simple_map (90 vs 98 %).
Everything not mentioned is as in test 42. Launch with seeds 0-4:
    powershell -File scripts/launch_seeds.ps1 -Script train/grid_training_45.py -Test 45

Original description of test 42:

Test 42 — test 41 with target_kl = 0.015.

Textbook PPO (test 41, 5 seeds) learned much faster than PPOAgent but converged
prematurely: entropy fell to 0.30 within 8k episodes and stayed there, training
success stuck at ~98 % and the approximate KL per update was 0.03-0.07 (usual
target <= 0.015-0.02). Greedy evaluation 93.1 ± 1.1 % vs 99.4 ± 0.4 % for PPOAgent
(test 35). With target_kl = 0.015 the epochs of an update stop as soon as the
approximate KL of a minibatch exceeds 1.5 x 0.015, so each update moves the policy
less. Everything else is as in test 41.

Original description of test 41:

Test 41 — test 35 with textbook PPO (StandardPPOAgent) instead of PPOAgent.

Test 35 (PPOAgent + invalid-action masking + momentum observation, plain maps, no
slips) is the best agent: 99.4 ± 0.4 % greedy over 5 seeds. But PPOAgent is not the
PPO of the papers: one gradient step of 64 transitions per episode, sampled from a
100k buffer of past episodes with stale log-probabilities and advantages (§2i.2).
For a thesis that names the algorithm "PPO", this test trains the textbook version
on exactly the setup of test 35:

- StandardPPOAgent (core/agents/standard_ppo.py): rollouts of 2048 steps, GAE(0.95),
  10 epochs of minibatches of 64, clipped surrogate (0.2), per-minibatch advantage
  normalisation, value loss 0.5, entropy 0.01, gradient clip 0.5, lr 3e-4
  (the usual defaults of Stable-Baselines3 except the entropy coefficient, kept as
  in test 35); the rollout is discarded after each update.
- StepInfoSingleTrainer (core/training/step_info_trainer.py): relays the step info so
  a collision is terminal and the step limit bootstraps in GAE (both are `truncated`
  in the environment). With masking and no slips collisions cannot happen, so here
  every truncation is the step limit.

Same environment, observation, masking, reward, network bodies (SimpleQNetwork 3x64
policy, ValueNetwork 64 critic), seed and 40k episodes as test 35.
"""
import argparse
import json
import os

from tfm_control_ucm.core.agents.networks import SimpleQNetwork, ValueNetwork
from tfm_control_ucm.core.agents.standard_ppo import StandardPPOAgent, StandardPPOConfig
from tfm_control_ucm.core.agents.utils import RLAlgorithm
from tfm_control_ucm.core.rewards import ProgressReward
import tfm_control_ucm.core.training.TrainingManager as training_manager
from tfm_control_ucm.core.training.step_info_trainer import StepInfoSingleTrainer
from tfm_control_ucm.core.grid_env.slippery_environment import Slippery_Grid_Robot_Sections_Env_MultiMap
from tfm_control_ucm.core.grid_env.slippery_map import SlipperyGridMap
from tfm_control_ucm.core.grid_env.slippery_robot import SlipperyGridRobotNotTurning
import torch
import numpy as np

MAP_PATHS = [
    "./maps/map_obstacle_free.json",
    "./maps/map_1_obstacle.json",
    "./maps/map_4_square_obstacle.json",
    "./maps/map_simple.json",
]

def main():
    parser = argparse.ArgumentParser(description="Train textbook PPO for the grid environment.")

    # Reward args (ProgressReward), as in tests 31 and 35
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
    parser.add_argument('--terrain_obs', choices=["none", "proprio", "neighbors"], default="proprio", help='Proprioceptive values added to the observation (test 35: proprio).')
    parser.add_argument('--terrain_obs_noise', type=float, default=0.05, help='Std of the noise of the grip estimate.')

    # PPO args
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate.')
    parser.add_argument('--rollout_steps', type=int, default=2048, help='Environment steps per rollout (per update).')
    parser.add_argument('--n_epochs', type=int, default=10, help='Passes over each rollout.')
    parser.add_argument('--batch_size', type=int, default=64, help='Minibatch size.')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor.')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda.')
    parser.add_argument('--clip_eps', type=float, default=0.2, help='PPO clip epsilon.')
    parser.add_argument('--entropy_coeff', type=float, default=0.01, help='Entropy coefficient.')
    parser.add_argument('--value_coeff', type=float, default=0.5, help='Value loss coefficient.')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Max gradient norm.')
    parser.add_argument('--clip_value', type=float, default=None, help='Clip of the value update (None: no clipping).')
    parser.add_argument('--target_kl', type=float, default=0.015, help='Stop the epochs when the approximate KL exceeds 1.5x it (test 41: None).')
    parser.add_argument('--clip_mode', choices=["clip", "dual_clip", "rollback", "none"], default="dual_clip", help='Policy objective: clip (PPO), dual_clip (Ye et al. 2020), rollback (PPO-RB, Wang et al. 2019), none (use a KL penalty).')
    parser.add_argument('--dual_clip_c', type=float, default=3.0, help='Lower bound c of dual-clip PPO (c > 1).')
    parser.add_argument('--rollback_alpha', type=float, default=0.3, help='Rollback slope alpha of PPO-RB.')
    parser.add_argument('--kl_coef', type=float, default=0.0, help='Initial weight beta of the KL penalty (0: no penalty).')
    parser.add_argument('--kl_target', type=float, default=None, help='Target KL of the adaptive beta (None: fixed beta).')
    parser.add_argument('--anneal_lr', action=argparse.BooleanOptionalAction, default=False, help='Decay the learning rate linearly to 0 over num_episodes.')
    parser.add_argument('--policy_init_scale', type=float, default=1.0, help='Multiply the weights of the last policy layer (and zero its bias) by this; 1 = default init.')

    # Trainer config args
    parser.add_argument('--num_episodes', type=int, default=40_000, help='Total number of episodes to train.')
    parser.add_argument('--agent_name', type=str, default="PPO Agent", help='Name of the agent.')
    parser.add_argument('--render', action=argparse.BooleanOptionalAction, default=False, help='Render the environment during training.')
    parser.add_argument('--seed', type=int, default=None, help='Seed for maps, start/goal positions, sampling and network initialisation.')
    parser.add_argument('--test_number', type=int, required=False, default=45, help='Test number')

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    mapas = [SlipperyGridMap.load(path) for path in MAP_PATHS] # Plain map files: grip 1 everywhere

    robot_simple = SlipperyGridRobotNotTurning(slip_model=None) # Deterministic moves (as test 31), momentum tracked

    n_sections = 4 * args.n
    lidar_config = {"num_rays": n_sections * 15, "max_range": 10, "fov": 2 * np.pi, "noise_std": 1e-3, "with_cache": True}

    reward = ProgressReward(progress_scale=args.progress_scale,
                            clip_negative_progress=args.clip_negative_progress,
                            step_penalty=args.step_penalty,
                            safety_margin=lidar_config["max_range"] * args.safety_margin_factor,
                            max_proximity_penalty=args.max_proximity_penalty,
                            goal_reward=args.goal_reward,
                            collision_penalty=args.collision_penalty)

    env = Slippery_Grid_Robot_Sections_Env_MultiMap(maps=mapas, robot=robot_simple, reward_fn=reward, cell_size=10,
                    max_iteration_steps=args.max_steps, num_sections=n_sections, lidar_config=lidar_config,
                    render_mode="human" if args.render else None,
                    terrain_obs=args.terrain_obs, terrain_obs_noise=args.terrain_obs_noise)
    env.reset(seed=args.seed) # Seeds the map selection; later resets keep the same generator

    action_dim = env.get_action_dim()
    obv_dim = env.get_observation_dim()

    rl_config = StandardPPOConfig(obs_dim=obv_dim, action_dim=action_dim, algorithm=RLAlgorithm.PPO,
                                  device=device, name=args.agent_name,
                                  lr=args.lr,
                                  rollout_steps=args.rollout_steps,
                                  n_epochs=args.n_epochs,
                                  batch_size=args.batch_size,
                                  gamma=args.gamma,
                                  gae_lambda=args.gae_lambda,
                                  clip_eps=args.clip_eps,
                                  entropy_coeff=args.entropy_coeff,
                                  value_coeff=args.value_coeff,
                                  max_grad_norm=args.max_grad_norm,
                                  clip_value=args.clip_value,
                                  target_kl=args.target_kl,
                                  clip_mode=args.clip_mode,
                                  dual_clip_c=args.dual_clip_c,
                                  rollback_alpha=args.rollback_alpha,
                                  kl_coef=args.kl_coef,
                                  kl_target=args.kl_target,
                                  anneal_lr_episodes=args.num_episodes if args.anneal_lr else None)

    policy_net = SimpleQNetwork(obs_dim=obv_dim, action_dim=action_dim, num_hidden_layers=3, hidden_dim=64).to(device)
    value_net = ValueNetwork(obs_dim=obv_dim, hidden_dim=64).to(device)
    if args.policy_init_scale != 1.0: # Andrychowicz et al. 2021: near-uniform initial policy
        with torch.no_grad():
            policy_net.net[-1].weight.mul_(args.policy_init_scale)
            policy_net.net[-1].bias.zero_()
    agent = StandardPPOAgent(rl_config, policy_network=policy_net, value_network=value_net)

    logs_dir = f"./runs/Test/{args.test_number}"
    os.makedirs(logs_dir, exist_ok=True)
    with open(os.path.join(logs_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"script": os.path.basename(__file__),
                   "algorithm": "ppo",
                   "agent": "StandardPPOAgent",
                   "args": vars(args),
                   "reward": reward.get_config(),
                   "maps": MAP_PATHS,
                   "random_map": True,
                   "action_masking": True,
                   "robot": type(robot_simple).__name__,
                   "lidar": lidar_config,
                   "num_sections": n_sections,
                   "slip": None, # No slips: eval_agent rebuilds the slippery env with a robot without slip model
                   "terrain_obs": args.terrain_obs,
                   "terrain_obs_noise": args.terrain_obs_noise}, f, indent=2)

    t_config = training_manager.TrainerConfig(
        num_episodes=args.num_episodes,
        render=args.render,
        log_steps=False,
        action_masking=True
    )
    trainer = StepInfoSingleTrainer(logs_dir, f"./checkpoints/Test/{args.test_number}", t_config)
    trainer.add_agents(agent)
    trainer.add_environments(env)

    trainer.train()

if __name__ == '__main__':
    main()
