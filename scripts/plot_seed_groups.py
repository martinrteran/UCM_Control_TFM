"""
plot_seed_groups.py
-------------------
Compare configurations trained with several seeds (e.g. DQN test 40 vs PPO test 35, 5 seeds each).

Figure 1 (seed_groups_training.png): training success and steps ratio (rolling) per group, as the
mean over the seeds with a band from the worst to the best seed.
Figure 2 (seed_groups_eval.png): final evaluation (plots/evals/eval_<N>_e<episodes>_s<seed>.json), greedy
and sampled success per map and loops per 1000 episodes, mean ± std over the seeds with each seed as a dot.

Needs runs/Test/<N>/episodes.csv (scripts/analyze_tests.py or export_episode_metrics.py write it) and the
cached evaluations (analyze_tests.py or eval_policy_modes.py).

Usage:
    python scripts/plot_seed_groups.py --group "PPO (35)" 35 351 352 353 354 --group "DQN (40)" 40 401 402 403 404 \
        --out plots/seeds_35_vs_40
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_episode_csv import load  # noqa: E402

COLORS = ["#264653", "#e76f51", "#2a9d8f", "#e9c46a", "#8d6cab", "#7f8c8d"]
MAPS = ["obstacle_free_map", "1_obstacle", "4_obstacles", "simple_map"]


def _base_map(name: str) -> str:
    return name[:-len("_slippery")] if name.endswith("_slippery") else name


def rolling_band(dfs, column, window, by_success=False):
    """Mean, min and max over the seeds of a rolling statistic, on the common episode index."""
    series = []
    for df in dfs:
        s = df[column]
        if by_success: # Median over the successful episodes only (steps ratio)
            s = s.where(df["Success/Episode"] == 1).rolling(window, min_periods=50).median()
        else:
            s = s.rolling(window, min_periods=window // 5).mean()
        series.append(s)
    table = pd.concat(series, axis=1).dropna()
    return table.index, table.mean(axis=1), table.min(axis=1), table.max(axis=1)


def plot_training(groups, window, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for color, (label, tests) in zip(COLORS, groups.items()):
        dfs = [load(Path(f"runs/Test/{t}/episodes.csv")) for t in tests]
        x, mean, low, high = rolling_band(dfs, "Success/Episode", window)
        axes[0].plot(x, mean, color=color, lw=1.8, label=f"{label}, mean of {len(tests)} seeds")
        axes[0].fill_between(x, low, high, color=color, alpha=0.2, lw=0)
        x, mean, low, high = rolling_band(dfs, "Steps/Ratio/Episode", window, by_success=True)
        axes[1].plot(x, mean, color=color, lw=1.8, label=label)
        axes[1].fill_between(x, low, high, color=color, alpha=0.2, lw=0)
    axes[0].set(title="Training success (band: worst to best seed)", xlabel="Episode",
                ylabel=f"Success rate (rolling {window})", ylim=(0, 1.02))
    axes[1].set(title="Path efficiency of successful episodes", xlabel="Episode",
                ylabel="Steps / optimal (rolling median)", ylim=(0.95, 2.0))
    axes[1].axhline(1.0, color="grey", ls="--", lw=1)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "seed_groups_training.png", dpi=150)
    plt.close(fig)


def eval_rates(result_mode):
    """Success rate per map (base names), total success and loops of one evaluation mode."""
    per_map = {}
    for name, counts in result_mode["per_map"].items():
        base = _base_map(name)
        n = sum(counts.values())
        per_map[base] = counts.get("success", 0) / n if n else np.nan
    total = sum(c.get("success", 0) for c in result_mode["per_map"].values()) / sum(sum(c.values()) for c in result_mode["per_map"].values())
    return per_map, total, result_mode["timeout_loops"]


def plot_eval(groups, episodes, seed, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), gridspec_kw={"width_ratios": [3, 3, 1.4]})
    categories = MAPS + ["total"]
    width = 0.8 / len(groups)
    for gi, (color, (label, tests)) in enumerate(zip(COLORS, groups.items())):
        evals = [json.loads(Path(f"plots/evals/eval_{t}_e{episodes}_s{seed}.json").read_text(encoding="utf-8")) for t in tests]
        for ax, mode in zip(axes[:2], ("greedy", "sampled")):
            rates = [eval_rates(e[mode]) for e in evals]
            values = np.array([[r[0].get(m, np.nan) for m in MAPS] + [r[1]] for r in rates]) * 100
            x = np.arange(len(categories)) + (gi - (len(groups) - 1) / 2) * width
            ax.bar(x, values.mean(axis=0), width, yerr=values.std(axis=0, ddof=1), color=color, alpha=0.85,
                   capsize=3, label=label)
            for j in range(len(categories)):
                ax.scatter(np.full(len(values), x[j]), values[:, j], color="black", s=6, zorder=3)
        loops = np.array([[eval_rates(e[m])[2] for m in ("greedy", "sampled")] for e in evals])
        x = np.arange(2) + (gi - (len(groups) - 1) / 2) * width
        axes[2].bar(x, loops.mean(axis=0), width, yerr=loops.std(axis=0, ddof=1), color=color, alpha=0.85, capsize=3, label=label)
        for j in range(2):
            axes[2].scatter(np.full(len(loops), x[j]), loops[:, j], color="black", s=6, zorder=3)
    for ax, mode in zip(axes[:2], ("Greedy", "Sampled")):
        ax.set(title=f"{mode} evaluation ({episodes} episodes per seed)", ylabel="Success %", ylim=(50, 101))
        ax.set_xticks(np.arange(len(categories)), [c.replace("_map", "") for c in categories])
    axes[2].set(title="Loops per 1000 episodes", ylabel="Timeouts that are loops")
    axes[2].set_xticks(np.arange(2), ["greedy", "sampled"])
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(loc="lower left")
    fig.suptitle("Bars: mean over seeds ± std; dots: individual seeds", fontsize=9, y=0.02)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "seed_groups_eval.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare configurations trained with several seeds")
    parser.add_argument("--group", nargs="+", action="append", required=True, metavar=("LABEL", "TEST"),
                        help="Label followed by the test numbers of its seeds; repeat for each configuration")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    groups = {g[0]: [int(t) for t in g[1:]] for g in args.group}
    args.out.mkdir(parents=True, exist_ok=True)
    plot_training(groups, args.window, args.out)
    plot_eval(groups, args.episodes, args.seed, args.out)
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
