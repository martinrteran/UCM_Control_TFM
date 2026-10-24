"""
compare_runs.py
---------------
Print a side-by-side statistical comparison of multiple training runs.
Reads TensorBoard event files and computes summary statistics for the
final N% of training (the "converged" phase).

Usage:
    python scripts/compare_runs.py 14 21 24
    python scripts/compare_runs.py 14 21 24 --tail 0.2   # last 20% of episodes
    python scripts/compare_runs.py --all                  # compare all runs

Outputs a markdown table ready to paste into the thesis.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    sys.exit("pip install tensorboard")


METRICS = [
    ("Reward/Total",        "Reward (mean)"),
    ("Success/Episode",     "Success rate"),
    ("Steps/Ratio/Episode", "Steps ratio"),
    ("Loss/Episode",        "Loss (mean)"),
]


def load_scalar(run_dir: Path, tag: str) -> list[float] | None:
    acc = EventAccumulator(str(run_dir))
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    if tag not in tags:
        return None
    return [e.value for e in acc.Scalars(tag)]


def tail(values: list[float], fraction: float) -> list[float]:
    n = max(1, int(len(values) * fraction))
    return values[-n:]


def find_agent_dirs(test_numbers: list[int] | None, all_runs: bool) -> dict[str, Path]:
    """Return {label: event_dir}."""
    base = Path("runs/Test")
    if not base.exists():
        sys.exit(f"{base} not found — run from repo root.")
    result = {}
    for test_dir in sorted(base.iterdir()):
        if not test_dir.is_dir():
            continue
        try:
            n = int(test_dir.name.split()[0])
        except ValueError:
            continue
        if not all_runs and test_numbers and n not in test_numbers:
            continue
        for ev in sorted(test_dir.rglob("events.out.tfevents.*")):
            agent_name = ev.parent.name
            label = f"Test {n} / {agent_name}"
            result[label] = ev.parent
    return result


def summarise(dirs: dict[str, Path], tail_frac: float):
    rows = []
    for label, dpath in dirs.items():
        row = {"Run": label}
        for tag, col_name in METRICS:
            vals = load_scalar(dpath, tag)
            if vals is None:
                row[col_name] = "—"
            else:
                t = tail(vals, tail_frac)
                row[col_name] = f"{np.mean(t):.3f} ± {np.std(t):.3f}"
        rows.append(row)
    return rows


def print_table(rows: list[dict]):
    if not rows:
        print("No data found.")
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}

    header = " | ".join(c.ljust(widths[c]) for c in cols)
    sep    = "-+-".join("-" * widths[c] for c in cols)
    print(header)
    print(sep)
    for row in rows:
        print(" | ".join(str(row[c]).ljust(widths[c]) for c in cols))

    print()
    print("<!-- Markdown table -->")
    md_header = "| " + " | ".join(c.ljust(widths[c]) for c in cols) + " |"
    md_sep    = "| " + " | ".join("-" * widths[c] for c in cols) + " |"
    print(md_header)
    print(md_sep)
    for row in rows:
        print("| " + " | ".join(str(row[c]).ljust(widths[c]) for c in cols) + " |")


def main():
    parser = argparse.ArgumentParser(description="Compare training runs statistically")
    parser.add_argument("tests", nargs="*", type=int,
                        help="Test numbers to compare")
    parser.add_argument("--all", action="store_true",
                        help="Compare all available runs")
    parser.add_argument("--tail", type=float, default=0.25,
                        help="Fraction of episodes at the end to use (default: 0.25)")
    args = parser.parse_args()

    if not args.all and not args.tests:
        parser.print_help()
        sys.exit(1)

    dirs = find_agent_dirs(args.tests or None, args.all)
    if not dirs:
        sys.exit("No matching runs found.")

    print(f"Comparing {len(dirs)} run(s) using last {args.tail*100:.0f}% of episodes\n")
    rows = summarise(dirs, args.tail)
    print_table(rows)


if __name__ == "__main__":
    main()
