"""
eval_agent.py
-------------
Load a saved checkpoint and run greedy evaluation episodes.
Prints per-episode stats and a summary, overall and per map.

The environment is rebuilt to match training:
- if runs/Test/<N>/config.json exists (written by grid_training_25.py and later),
  its maps, lidar, sections, step limit and reward are used;
- otherwise the robot is inferred from action_dim (3 = GridRobot, 4 = GridRobotNotTurning)
  and the number of sections from obs_dim, with the lidar used by tests 22-25.

Networks are rebuilt from the checkpoint's state_dict shapes, so any MLP policy
(GridAgentNet, SimpleQNetwork, PolicyNetwork...) loads. Recurrent agents are not supported.

Usage:
    python scripts/eval_agent.py <checkpoint.pth> [options]

Examples:
    python scripts/eval_agent.py "checkpoints/Test/25/PPO Agent/final.pth"
    python scripts/eval_agent.py "checkpoints/Test/22/PPO Agent/final.pth" --maps map_4_square_obstacle.json
    python scripts/eval_agent.py "checkpoints/Test/25/PPO Agent/final.pth" --episodes 50 --render
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tfm_control_ucm.core.grid_env.environment import Grid_Robot_Sections_Env_MultiMap
from tfm_control_ucm.core.grid_env.map import GridMap
from tfm_control_ucm.core.grid_env.robot import GridRobot, GridRobotNotTurning
from tfm_control_ucm.core.agents.agent import GridAgent, PPOAgent
from tfm_control_ucm.core.agents.utils import RLConfig, RLAlgorithm, PPOConfig
from tfm_control_ucm.core.rewards import ProgressReward


MAPS_DIR = Path("./maps")
DEFAULT_MAPS = ["map_obstacle_free.json", "map_1_obstacle.json", "map_4_square_obstacle.json", "map_simple.json"]
DEFAULT_MAX_STEPS = 400


class StateDictMLP(nn.Module):
    """Linear/ReLU stack rebuilt from a `net.<i>.weight` state_dict, matching the project's MLPs."""

    def __init__(self, state_dict: dict):
        super().__init__()
        indices = sorted(int(k.split(".")[1]) for k in state_dict if k.startswith("net.") and k.endswith(".weight"))
        if not indices:
            raise ValueError("Unsupported network: expected 'net.<i>.weight' keys (recurrent agents are not supported)")
        layers = []
        for idx in indices:
            out_dim, in_dim = state_dict[f"net.{idx}.weight"].shape
            while len(layers) < idx:
                layers.append(nn.ReLU())
            layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def find_run_config(checkpoint: Path) -> Optional[Path]:
    """checkpoints/Test/<N>/<agent>/<file>.pth -> runs/Test/<N>/config.json, if it exists."""
    for parent in checkpoint.resolve().parents:
        if parent.parent.name == "Test" and parent.parent.parent.name == "checkpoints":
            candidate = parent.parent.parent.parent / "runs" / "Test" / parent.name / "config.json"
            return candidate if candidate.exists() else None
    return None


def load_agent_from_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)

    alg_str = ckpt.get("algorithm", "grid_agent")
    cfg_dict = ckpt.get("config", {})
    obs_dim    = int(cfg_dict.get("obs_dim", 32))
    action_dim = int(cfg_dict.get("action_dim", 3))
    lr         = cfg_dict.get("lr", 1e-3)
    gamma      = cfg_dict.get("gamma", 0.99)

    print(f"[checkpoint] algorithm={alg_str}  obs_dim={obs_dim}  action_dim={action_dim}  "
          f"global_step={ckpt.get('global_step', '?')}  episodes={ckpt.get('episode_count', '?')}")

    policy_net = StateDictMLP(ckpt["policy_state_dict"])
    if alg_str in ("grid_agent", "dqn", "ddqn"):
        config = RLConfig(name="eval", obs_dim=obs_dim, action_dim=action_dim,
                          algorithm=RLAlgorithm.GRID_AGENT, device=device, gamma=gamma, lr=lr)
        agent = GridAgent(config, policy_net=policy_net, target_net=StateDictMLP(ckpt["policy_state_dict"]))
    elif alg_str == "ppo":
        config = PPOConfig(name="eval", obs_dim=obs_dim, action_dim=action_dim,  # type: ignore
                           algorithm=RLAlgorithm.PPO, device=device, gamma=gamma, lr=lr)
        agent = PPOAgent(config, policy_network=policy_net, value_network=StateDictMLP(ckpt["value_state_dict"]))  # type: ignore
    else:
        raise ValueError(f"Unsupported algorithm '{alg_str}' (only grid_agent/dqn/ddqn and ppo MLP agents)")

    agent.load(path)
    return agent, obs_dim, action_dim


def build_env(args, obs_dim: int, action_dim: int, run_config: Optional[dict]):
    if run_config:
        map_paths = run_config["maps"]
        lidar_config = run_config["lidar"]
        num_sections = run_config["num_sections"]
        max_steps = run_config["args"].get("max_steps", DEFAULT_MAX_STEPS)
        reward_params = {k: v for k, v in run_config["reward"].items() if k != "name"}
        reward = ProgressReward(**reward_params) if run_config["reward"]["name"] == "ProgressReward" else ProgressReward()
    else:
        map_paths = [str(MAPS_DIR / name) for name in DEFAULT_MAPS]
        num_sections = (obs_dim - 2) // 2
        lidar_config = {"num_rays": num_sections * 15, "max_range": 10, "fov": 2 * np.pi, "noise_std": 1e-3, "with_cache": True}
        max_steps = DEFAULT_MAX_STEPS
        reward = ProgressReward()

    if args.maps:
        map_paths = [str(MAPS_DIR / name) for name in args.maps]
    if args.max_steps:
        max_steps = args.max_steps

    robot = GridRobot() if action_dim == GridRobot.NUM_ACTIONS else GridRobotNotTurning()
    env = Grid_Robot_Sections_Env_MultiMap(maps=[GridMap.load(p) for p in map_paths], robot=robot, reward_fn=reward,
                                           lidar_config=lidar_config, num_sections=num_sections,
                                           max_iteration_steps=max_steps, cell_size=args.cell_size,
                                           render_mode="human" if args.render else None)
    # Runs without config.json (tests <= 24) were trained with the mirrored goal angle
    env._legacy_goal_angle = run_config is None if args.legacy_goal_angle is None else args.legacy_goal_angle
    if env.get_observation_dim() != obs_dim:
        raise ValueError(f"The rebuilt environment gives obs_dim={env.get_observation_dim()} but the checkpoint "
                         f"expects {obs_dim}. Pass the training settings explicitly or add a config.json to the run.")
    print(f"[env] robot={type(robot).__name__}  sections={num_sections}  max_steps={max_steps}  "
          f"maps={[Path(p).stem for p in map_paths]}  legacy_goal_angle={env._legacy_goal_angle}")
    return env


def run_eval(agent, env, num_episodes: int, render: bool, seed: Optional[int]):
    records = []
    for ep in range(num_episodes):
        state, _ = env.reset(seed=seed if ep == 0 else None)
        map_name = env.map.name
        done = False
        ep_reward = 0.0
        ep_steps  = 0
        info = {}

        while not done:
            if render:
                env.render()
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_steps  += 1

        outcome = "success" if info.get("reached_goal") else "collision" if info.get("hit_obstacle") else "timeout"
        records.append({"ep": ep + 1, "map": map_name, "reward": ep_reward, "steps": ep_steps, "outcome": outcome})
        print(f"  ep {ep+1:4d}  {map_name:<18s} reward={ep_reward:7.2f}  steps={ep_steps:5d}  {outcome}")

    rewards = [r["reward"] for r in records]
    steps   = [r["steps"]  for r in records]
    outcomes = defaultdict(int)
    for r in records:
        outcomes[r["outcome"]] += 1

    print("\n── Summary ─────────────────────────────")
    print(f"  Episodes : {num_episodes}")
    for outcome in ("success", "collision", "timeout"):
        print(f"  {outcome.capitalize():<9s}: {outcomes[outcome]}/{num_episodes}  ({100*outcomes[outcome]/num_episodes:.1f}%)")
    print(f"  Reward   : mean={np.mean(rewards):.2f}  std={np.std(rewards):.2f}  "
          f"min={np.min(rewards):.2f}  max={np.max(rewards):.2f}")
    print(f"  Steps    : mean={np.mean(steps):.1f}  std={np.std(steps):.1f}  "
          f"min={np.min(steps)}  max={np.max(steps)}")

    by_map = defaultdict(list)
    for r in records:
        by_map[r["map"]].append(r)
    if len(by_map) > 1:
        print("\n── Per map ─────────────────────────────")
        for name, rs in sorted(by_map.items()):
            wins = sum(r["outcome"] == "success" for r in rs)
            print(f"  {name:<18s} success={wins}/{len(rs)} ({100*wins/len(rs):.1f}%)  "
                  f"mean steps={np.mean([r['steps'] for r in rs]):.1f}")
    return records


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved agent checkpoint")
    parser.add_argument("checkpoint", help="Path to .pth checkpoint file")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of evaluation episodes (default: 20)")
    parser.add_argument("--maps", nargs="+", choices=sorted(p.name for p in MAPS_DIR.glob("*.json")),
                        help="Maps from ./maps to evaluate on, one picked at random per episode "
                             "(default: the run's maps, or all four)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help=f"Step limit per episode (default: the run's, or {DEFAULT_MAX_STEPS})")
    parser.add_argument("--config", type=Path, default=None,
                        help="Run config.json to rebuild the environment (default: runs/Test/<N>/config.json)")
    parser.add_argument("--seed", type=int, default=None, help="Seed for map choice and start/goal positions")
    parser.add_argument("--legacy-goal-angle", action=argparse.BooleanOptionalAction, default=None,
                        help="Use the mirrored goal angle of tests <= 24 (default: on when the run has no config.json)")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment (requires pygame)")
    parser.add_argument("--cell-size", type=int, default=10,
                        help="Cell size in pixels for rendering (default: 10)")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    config_path = args.config or find_run_config(Path(args.checkpoint))
    run_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path else None
    print(f"[config] {config_path if config_path else 'none found, inferring from the checkpoint'}")

    agent, obs_dim, action_dim = load_agent_from_checkpoint(args.checkpoint, device)
    env = build_env(args, obs_dim, action_dim, run_config)

    print(f"\nRunning {args.episodes} evaluation episode(s)...\n")
    run_eval(agent, env, args.episodes, args.render, args.seed)
    env.close()


if __name__ == "__main__":
    main()
