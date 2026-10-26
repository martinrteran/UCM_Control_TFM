"""
compare_episode_csv.py
----------------------
Compare runs from the CSVs written by export_episode_metrics.py.

Figures (in --out):
  compare_success.png         success rate vs episode and vs cumulative environment steps
  compare_success_per_map.png success rate per map
  compare_learning.png        path efficiency (successful episodes) and policy entropy
A table of means per phase is printed and saved as compare_by_phase.csv.

Runs with a curriculum (Curriculum/Stage tag) get their stage changes marked.

Usage:
    python scripts/compare_episode_csv.py runs/Test/25/episodes.csv runs/Test/26/episodes.csv \
        --labels "25 (no curriculum)" "26 (curriculum)" --out plots/compare_25_26
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_episode_csv import load

COLORS = ["#264653", "#e76f51", "#2a9d8f", "#e9c46a"]


def stage_starts(df: pd.DataFrame) -> list:
    if "Curriculum/Stage" not in df:
        return []
    stage = df["Curriculum/Stage"].dropna()
    return list(stage.index[stage.diff().fillna(0) != 0])


def mark_stages(ax, starts, color):
    for i, episode in enumerate(starts):
        ax.axvline(episode, color=color, ls=":", lw=1, alpha=0.8, label="curriculum stage change" if i == 0 else None)


def plot_success(runs, window, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for color, (label, df) in zip(COLORS, runs.items()):
        success = df["Success/Episode"].rolling(window, min_periods=window // 5).mean()
        axes[0].plot(df.index, success, color=color, lw=1.8, label=label)
        mark_stages(axes[0], stage_starts(df), color)
        cumulative_steps = df["Steps/Episode"].cumsum() / 1e6
        axes[1].plot(cumulative_steps, success, color=color, lw=1.8, label=label)
    axes[0].set(xlabel="Episode", ylabel=f"Success rate (rolling {window})", ylim=(0, 1), title="Success vs episodes")
    axes[1].set(xlabel="Environment steps (millions)", ylabel=f"Success rate (rolling {window})", ylim=(0, 1),
                title="Success vs experience")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "compare_success.png", dpi=150)
    plt.close(fig)


def plot_success_per_map(runs, window, out):
    maps = sorted({c for df in runs.values() for c in df.columns if c.startswith("Success/Map/")})
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True, sharey=True)
    for ax, col in zip(axes.flat, maps):
        for color, (label, df) in zip(COLORS, runs.items()):
            series = df[col].dropna()
            ax.plot(series.index, series.rolling(window // 4, min_periods=20).mean(), color=color, lw=1.6, label=label)
        ax.set(title=col.split("/")[-1], ylim=(0, 1.02))
        ax.grid(alpha=0.3)
    for ax in axes[1]: ax.set_xlabel("Episode")
    for ax in axes[:, 0]: ax.set_ylabel(f"Success (rolling {window // 4} episodes of the map)")
    axes[0, 0].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "compare_success_per_map.png", dpi=150)
    plt.close(fig)


def plot_learning(runs, window, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for color, (label, df) in zip(COLORS, runs.items()):
        won = df[df["Success/Episode"] == 1]["Steps/Ratio/Episode"]
        axes[0].plot(won.index, won.rolling(window, min_periods=50).median(), color=color, lw=1.8, label=label)
        if "Agent/Entropy" in df:
            axes[1].plot(df.index, df["Agent/Entropy"].rolling(window, min_periods=50).mean(), color=color, lw=1.8, label=label)
    axes[0].axhline(1.0, color="grey", ls="--", lw=1)
    axes[0].set(xlabel="Episode", ylabel="Steps / optimal (median, successful)", title="Path efficiency", ylim=(0.9, 3))
    axes[1].set(xlabel="Episode", ylabel="Entropy (nats)", title="Policy entropy")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "compare_learning.png", dpi=150)
    plt.close(fig)


def phase_table(runs, from_episode: int, blocks: int = 4) -> pd.DataFrame:
    """Means per block of episodes, from `from_episode` on (e.g. when every run plays the same distribution)."""
    rows = []
    for label, df in runs.items():
        df = df[df.index >= from_episode]
        for part in np.array_split(df.index.to_numpy(), blocks):
            d = df.loc[part]
            won = d["Success/Episode"] == 1
            rows.append({"run": label, "episodes": f"{part[0]}-{part[-1]}",
                         "success": d["Success/Episode"].mean(),
                         "collision": d["Done/Hit Obstacle"].mean(),
                         "timeout": d["Done/Max Steps Reached"].mean(),
                         "steps_ratio_median": d.loc[won, "Steps/Ratio/Episode"].median(),
                         **{c.split("/")[-1]: d[c].mean() for c in sorted(d.columns) if c.startswith("Success/Map/")}})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Compare runs from their episodes.csv files")
    parser.add_argument("csvs", type=Path, nargs="+")
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--from-episode", type=int, default=None,
                        help="First episode of the phase table (default: last curriculum stage change of any run)")
    args = parser.parse_args()
    if len(args.labels) != len(args.csvs): parser.error("one label per CSV")
    args.out.mkdir(parents=True, exist_ok=True)

    runs = {label: load(csv) for label, csv in zip(args.labels, args.csvs)}
    plot_success(runs, args.window, args.out)
    plot_success_per_map(runs, args.window, args.out)
    plot_learning(runs, args.window, args.out)

    starts = [s for df in runs.values() for s in stage_starts(df)]
    from_episode = args.from_episode if args.from_episode is not None else (max(starts) if starts else 0)
    table = phase_table(runs, from_episode)
    table.to_csv(args.out / "compare_by_phase.csv", index=False)
    print(f"Phase table from episode {from_episode}; curriculum stage changes: {starts}")
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.precision", 3):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
