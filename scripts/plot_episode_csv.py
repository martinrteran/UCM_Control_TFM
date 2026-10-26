"""
plot_episode_csv.py
-------------------
Plots of one run from the CSV written by export_episode_metrics.py, and a
summary of training by phase.

Figures (in --out):
  outcomes.png        success / collision / timeout rates over training
  success_per_map.png success rate per map
  efficiency.png      steps and steps ratio (successful episodes only)
  learning.png        episode reward, critic loss, policy loss, entropy

Usage:
    python scripts/plot_episode_csv.py runs/Test/25/episodes.csv --out plots/test25 --window 1000
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTCOME_COLORS = {"Success": "#2a9d8f", "Collision": "#e76f51", "Timeout": "#e9c46a"}
MAP_COLORS = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"]


def load(csv_path: Path) -> pd.DataFrame:
    """Wide table: one row per episode, one column per tag."""
    long = pd.read_csv(csv_path)
    return long.pivot_table(index="step", columns="tag", values="value", aggfunc="last").sort_index()


def rolling(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=max(1, window // 5)).mean()


def plot_outcomes(df, window, out):
    success = df["Success/Episode"]
    collision = df["Done/Hit Obstacle"]
    timeout = df["Done/Max Steps Reached"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.stackplot(df.index, rolling(success, window), rolling(collision, window), rolling(timeout, window),
                 labels=list(OUTCOME_COLORS), colors=list(OUTCOME_COLORS.values()), alpha=0.9)
    ax.set(xlabel="Episode", ylabel=f"Share of episodes (rolling {window})", ylim=(0, 1), xlim=(df.index[0], df.index[-1]),
           title="How episodes end: success rises above 90 %")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "outcomes.png", dpi=150)
    plt.close(fig)


def plot_success_per_map(df, window, out):
    map_cols = sorted(c for c in df.columns if c.startswith("Success/Map/"))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for color, col in zip(MAP_COLORS, map_cols):
        series = df[col].dropna()
        ax.plot(series.index, series.rolling(window // len(map_cols), min_periods=20).mean(), label=col.split("/")[-1], color=color, lw=1.8)
    ax.set(xlabel="Episode", ylabel=f"Success rate (rolling {window // len(map_cols)} episodes of the map)", ylim=(0, 1.02),
           title="Success per map: map_simple stays behind")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "success_per_map.png", dpi=150)
    plt.close(fig)


def plot_efficiency(df, window, out):
    won = df[df["Success/Episode"] == 1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ratio = won["Steps/Ratio/Episode"]
    axes[0].plot(ratio.index, ratio.rolling(window, min_periods=50).median(), color="#264653", lw=1.8, label="median")
    axes[0].fill_between(ratio.index, ratio.rolling(window, min_periods=50).quantile(0.25),
                         ratio.rolling(window, min_periods=50).quantile(0.75), color="#264653", alpha=0.2, label="25–75 %")
    axes[0].axhline(1.0, color="grey", ls="--", lw=1, label="optimal (BFS)")
    axes[0].set(xlabel="Episode", ylabel="Steps / optimal steps", title="Path efficiency (successful episodes)", ylim=(0.9, None))
    axes[0].legend()
    steps = df["Steps/Episode"]
    axes[1].plot(steps.index, rolling(steps, window), color="#e76f51", lw=1.8, label="all episodes")
    axes[1].plot(won.index, rolling(won["Steps/Episode"], window), color="#2a9d8f", lw=1.8, label="successful")
    axes[1].set(xlabel="Episode", ylabel=f"Steps (rolling {window})", title="Episode length")
    axes[1].legend()
    for ax in axes: ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "efficiency.png", dpi=150)
    plt.close(fig)


def plot_learning(df, window, out):
    panels = [("Reward/Total", "Episode reward"), ("Agent/Value/Loss", "Critic loss"),
              ("Agent/Policy/Loss", "Policy loss"), ("Agent/Entropy", "Policy entropy (nats)")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, (tag, title) in zip(axes.flat, panels):
        if tag not in df:
            ax.set_visible(False)
            continue
        ax.plot(df.index, df[tag], color="#bbbbbb", lw=0.3)
        ax.plot(df.index, rolling(df[tag], window), color="#264653", lw=1.8)
        ax.set(title=title, xlabel="Episode")
        ax.grid(alpha=0.3)
    axes[0, 1].set_yscale("log")
    fig.tight_layout()
    fig.savefig(out / "learning.png", dpi=150)
    plt.close(fig)


def summary(df, blocks=5) -> pd.DataFrame:
    """Means per training phase (equal blocks of episodes)."""
    won = df["Success/Episode"] == 1
    rows = []
    for i, part in enumerate(np.array_split(df.index.to_numpy(), blocks)):
        d = df.loc[part]
        w = won.loc[part]
        rows.append({
            "episodes": f"{part[0]}-{part[-1]}",
            "success": d["Success/Episode"].mean(),
            "collision": d["Done/Hit Obstacle"].mean(),
            "timeout": d["Done/Max Steps Reached"].mean(),
            "steps_ratio_median_success": d.loc[w, "Steps/Ratio/Episode"].median(),
            "steps_mean": d["Steps/Episode"].mean(),
            "reward_mean": d["Reward/Total"].mean(),
            **{f"success_{c.split('/')[-1]}": d[c].mean() for c in sorted(df.columns) if c.startswith("Success/Map/")},
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Plot one run from its episodes.csv")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window", type=int, default=1000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = load(args.csv)
    plot_outcomes(df, args.window, args.out)
    plot_success_per_map(df, args.window, args.out)
    plot_efficiency(df, args.window, args.out)
    plot_learning(df, args.window, args.out)

    table = summary(df)
    table.to_csv(args.out / "summary_by_phase.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 3):
        print(table.T)


if __name__ == "__main__":
    main()
