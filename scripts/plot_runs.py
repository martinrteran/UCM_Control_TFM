"""
plot_runs.py
------------
Read TensorBoard event files from one or more runs and produce comparison plots.

Usage:
    python scripts/plot_runs.py               # all runs under runs/Test/
    python scripts/plot_runs.py 14 21 24      # specific test numbers
    python scripts/plot_runs.py --smooth 20   # window size for rolling average
    python scripts/plot_runs.py --save        # save PNGs instead of showing
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    sys.exit("tensorboard package not found. Run: pip install tensorboard")


# ── Metrics we care about ────────────────────────────────────────────────────
METRICS = {
    "Reward/Total":         "Total reward / episode",
    "Success/Episode":      "Success rate (1 = reached goal)",
    "Steps/Ratio/Episode":  "Steps / optimal steps (lower = better)",
    "Loss/Episode":         "Training loss",
    "Policy/Epsilon/Episode": "Epsilon (exploration)",
}


def smooth(values, window):
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


# Tags written by the per-algorithm PPO trainers used before test 25
TAG_ALIASES = {
    "Reward/Total":        ["Done/Reward/Episode", "Reward/Total/Episode"],
    "Success/Episode":     ["Done/Success"],
    "Steps/Ratio/Episode": ["Steps/Ratio"],
    "Loss/Episode":        ["Agent/Loss", "Agent/Loss/Total", "Policy/Loss"],
}


def load_run(run_dir: Path) -> dict[str, tuple[list, list]]:
    """Return {tag: (steps, values)} for each available metric (old PPO tags are read as their aliases)."""
    acc = EventAccumulator(str(run_dir))
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    result = {}
    for tag in METRICS:
        source = next((t for t in [tag] + TAG_ALIASES.get(tag, []) if t in tags), None)
        if source:
            events = acc.Scalars(source)
            steps = [e.step for e in events]
            vals  = [e.value for e in events]
            result[tag] = (steps, vals)
    return result


def find_runs(test_numbers: list[int] | None) -> dict[str, dict]:
    """
    Return {label: {tag: (steps, values)}} for each agent inside each test.
    label = "Test {N} / {agent_name}"
    """
    base = Path("runs/Test")
    if not base.exists():
        sys.exit(f"Directory {base} not found. Run from the repo root.")

    runs = {}
    test_dirs = sorted(base.iterdir(), key=lambda p: p.name)

    for test_dir in test_dirs:
        if not test_dir.is_dir():
            continue
        # parse number
        try:
            n = int(test_dir.name.split()[0])
        except ValueError:
            n = None
        if test_numbers and n not in test_numbers:
            continue

        for agent_dir in sorted(test_dir.rglob("events.out.tfevents.*")):
            agent_name = agent_dir.parent.name
            label = f"Test {test_dir.name} / {agent_name}"
            data = load_run(agent_dir.parent)
            if data:
                runs[label] = data

    return runs


def plot(runs: dict, smooth_window: int, save: bool):
    available_tags = set()
    for data in runs.values():
        available_tags |= data.keys()
    available_tags = [t for t in METRICS if t in available_tags]

    fig, axes = plt.subplots(len(available_tags), 1,
                             figsize=(12, 4 * len(available_tags)),
                             squeeze=False)
    fig.suptitle("Training run comparison", fontsize=14)

    colors = plt.cm.tab10.colors  # type: ignore

    for ax_row, tag in zip(axes, available_tags):
        ax = ax_row[0]
        for idx, (label, data) in enumerate(runs.items()):
            if tag not in data:
                continue
            steps, vals = data[tag]
            vals_s = smooth(vals, smooth_window)
            steps_s = steps[len(steps) - len(vals_s):]
            color = colors[idx % len(colors)]
            ax.plot(steps_s, vals_s, label=label, color=color, linewidth=1.2)
            # faint raw line
            if smooth_window > 1:
                ax.plot(steps, vals, color=color, alpha=0.15, linewidth=0.5)

        ax.set_title(METRICS[tag])
        ax.set_xlabel("Episode")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        out = Path("plots")
        out.mkdir(exist_ok=True)
        fname = out / "run_comparison.png"
        plt.savefig(fname, dpi=150)
        print(f"Saved to {fname}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot TensorBoard training runs")
    parser.add_argument("tests", nargs="*", type=int,
                        help="Test numbers to include (default: all)")
    parser.add_argument("--smooth", type=int, default=50,
                        help="Rolling average window (default: 50)")
    parser.add_argument("--save", action="store_true",
                        help="Save PNG instead of displaying")
    args = parser.parse_args()

    runs = find_runs(args.tests or None)
    if not runs:
        sys.exit("No event files found for the requested test numbers.")

    print(f"Loaded {len(runs)} run(s):")
    for label in runs:
        tags = list(runs[label].keys())
        print(f"  {label}  →  {tags}")

    plot(runs, args.smooth, args.save)


if __name__ == "__main__":
    main()
