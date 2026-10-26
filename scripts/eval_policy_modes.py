"""
eval_policy_modes.py
--------------------
Evaluate a trained agent greedily (most likely action) and sampling its policy,
with the same seed, on the run's own environment (rebuilt from
runs/Test/<N>/config.json, final distribution: all maps, any distance).

Per mode and map it counts success / collision / timeout, the timeouts that are
loops (the robot visits one cell >= 20 times) and the median steps / optimal
steps of successful episodes. Action masking is applied when the run used it.

Usage:
    python scripts/eval_policy_modes.py 31 --episodes 1000 --seed 0
    python scripts/eval_policy_modes.py 31 --out plots/evals/eval_31_e1000_s0.json
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_agent import build_env, load_agent_from_checkpoint  # noqa: E402

LOOP_VISITS = 20 # A timeout counts as a loop if the robot visited one cell at least this many times


def checkpoint_of(test: int) -> Path:
    candidates = sorted(Path(f"checkpoints/Test/{test}").glob("*/final.pth"))
    if not candidates: raise FileNotFoundError(f"No checkpoints/Test/{test}/*/final.pth")
    return candidates[0]


def evaluate_mode(test: int, mode: str, episodes: int, seed: int) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = json.loads(Path(f"runs/Test/{test}/config.json").read_text(encoding="utf-8"))
    agent, obs_dim, action_dim = load_agent_from_checkpoint(str(checkpoint_of(test)), torch.device("cpu"))
    args = argparse.Namespace(maps=None, max_steps=None, cell_size=10, render=False, legacy_goal_angle=None)
    env = build_env(args, obs_dim, action_dim, config)
    masking = bool(config.get("action_masking", False))

    per_map, ratios = defaultdict(Counter), defaultdict(list)
    loops = 0
    for ep in range(episodes):
        state, info = env.reset(seed=seed if ep == 0 else None)
        optimal = info["optimal_steps"]
        agent.on_episode_start()
        visited, steps, done = Counter(), 0, False
        while not done:
            mask = info.get("action_mask") if masking else None
            action = agent.act(state, training=(mode == "sampled"), action_mask=mask)
            state, _, terminated, truncated, info = env.step(action)
            visited[tuple(int(v) for v in env.robot.position)] += 1
            steps += 1
            done = terminated or truncated
        outcome = "success" if info.get("reached_goal") else "collision" if info.get("hit_obstacle") else "timeout"
        per_map[env.map.name][outcome] += 1
        if outcome == "success" and 0 < optimal < float("inf"):
            ratios[env.map.name].append(steps / optimal)
        if outcome == "timeout" and visited.most_common(1)[0][1] >= LOOP_VISITS:
            loops += 1
    return {"per_map": {m: dict(c) for m, c in per_map.items()},
            "steps_ratio_median": {m: float(np.median(r)) for m, r in ratios.items()},
            "timeout_loops": loops, "action_masking": masking}


def evaluate(test: int, episodes: int = 1000, seed: int = 0) -> dict:
    return {mode: evaluate_mode(test, mode, episodes, seed) for mode in ("greedy", "sampled")}


def totals(result_mode: dict) -> Counter:
    total = Counter()
    for counts in result_mode["per_map"].values():
        total.update(counts)
    return total


def main():
    parser = argparse.ArgumentParser(description="Greedy vs sampled evaluation of a trained test")
    parser.add_argument("test", type=int)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="JSON output (default: plots/evals/eval_<N>_e<episodes>_s<seed>.json)")
    args = parser.parse_args()
    out = args.out or Path(f"plots/evals/eval_{args.test}_e{args.episodes}_s{args.seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    result = evaluate(args.test, args.episodes, args.seed)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for mode, r in result.items():
        t = totals(r)
        print(f"test {args.test} {mode}: success {t['success']}/{args.episodes} collision {t['collision']} "
              f"timeout {t['timeout']} (loops {r['timeout_loops']}) -> {out}")


if __name__ == "__main__":
    main()
