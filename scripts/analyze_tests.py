"""
analyze_tests.py
----------------
One command to compare trained tests: export, evaluate, plot and tabulate.

For each test number:
  1. runs/Test/<N>/episodes.csv is exported from the TensorBoard event file if missing;
  2. plots/evals/eval_<N>_e<episodes>_s<seed>.json (greedy + sampled evaluation) is
     computed if missing — existing evaluations are reused, so re-running is cheap;
Then:
  3. comparison plots and a phase table are written to --out (compare_episode_csv.py);
  4. summary.md in --out has the evaluation table, with a two-proportion z-test of each
     test against the first one (|z| > 1.96 ~ significant at 5 %).

Usage (from the repo root):
    python scripts/analyze_tests.py 25 31
    python scripts/analyze_tests.py 25 30 31 --labels "25 base" "30 lidar 30" "31 masking" --out plots/compare_25_30_31
    python scripts/analyze_tests.py 31 --episodes 300      # quicker evaluation
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_episode_csv as compare  # noqa: E402
from eval_policy_modes import evaluate, totals  # noqa: E402
from export_episode_metrics import export  # noqa: E402

MAPS = ["obstacle_free_map", "1_obstacle", "4_obstacles", "simple_map"]


def ensure_csv(test: int) -> Path:
    run = Path(f"runs/Test/{test}")
    csv = run / "episodes.csv"
    event_files = sorted(run.glob("*/events.out.tfevents.*"))
    if not csv.exists() or (event_files and event_files[-1].stat().st_mtime > csv.stat().st_mtime):
        if not event_files: raise FileNotFoundError(f"No TensorBoard event file under {run}")
        print(f"[test {test}] exporting {event_files[-1].parent.name} -> {csv}")
        export(event_files[-1].parent, csv)
    return csv


def ensure_eval(test: int, episodes: int, seed: int) -> dict:
    path = Path(f"plots/evals/eval_{test}_e{episodes}_s{seed}.json")
    if not path.exists():
        print(f"[test {test}] evaluating {episodes} episodes x 2 modes (seed {seed}) -> {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(evaluate(test, episodes, seed), indent=2), encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def success_counts(result_mode: dict, map_name=None):
    maps = [map_name] if map_name else list(result_mode["per_map"])
    wins = sum(result_mode["per_map"].get(m, {}).get("success", 0) for m in maps)
    n = sum(sum(result_mode["per_map"].get(m, {}).values()) for m in maps)
    return wins, n


def z_test(a, b) -> float:
    (x1, n1), (x2, n2) = a, b
    p = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) if 0 < p < 1 else 0.0
    return (x2 / n2 - x1 / n1) / se if se > 0 else 0.0


def summary_table(labels, evals) -> str:
    ref_label, ref = labels[0], evals[0]
    lines = []
    for mode in ("greedy", "sampled"):
        lines += [f"### {mode.capitalize()} ({sum(totals(ref[mode]).values())} episodes, z vs {ref_label})", "",
                  "| Test | Success | Collisions | Timeouts (loops) | " + " | ".join(MAPS) + " | z total | z simple_map |",
                  "|---|---|---|---|" + "---|" * len(MAPS) + "---|---|"]
        for label, result in zip(labels, evals):
            r = result[mode]
            t = totals(r)
            n = sum(t.values())
            per_map = " | ".join(f"{100 * s / k:.1f} %" if k else "–" for s, k in (success_counts(r, m) for m in MAPS))
            if result is ref:
                z_total = z_simple = "–"
            else:
                z_total = f"{z_test(success_counts(ref[mode]), success_counts(r)):+.2f}"
                z_simple = f"{z_test(success_counts(ref[mode], 'simple_map'), success_counts(r, 'simple_map')):+.2f}"
            lines.append(f"| {label} | {100 * t['success'] / n:.1f} % | {t['collision']} | {t['timeout']} ({r['timeout_loops']}) | "
                         f"{per_map} | {z_total} | {z_simple} |")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Export, evaluate, plot and compare trained tests")
    parser.add_argument("tests", type=int, nargs="+", help="Test numbers; the first one is the reference for the z-tests")
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--out", type=Path, default=None, help="Output folder (default: plots/compare_<tests>)")
    parser.add_argument("--episodes", type=int, default=1000, help="Evaluation episodes per mode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window", type=int, default=1000, help="Rolling window of the training plots")
    args = parser.parse_args()
    labels = args.labels or [f"Test {t}" for t in args.tests]
    if len(labels) != len(args.tests): parser.error("one label per test")
    out = args.out or Path("plots/compare_" + "_".join(str(t) for t in args.tests))
    out.mkdir(parents=True, exist_ok=True)

    csvs = [ensure_csv(t) for t in args.tests]
    evals = [ensure_eval(t, args.episodes, args.seed) for t in args.tests]

    runs = {label: compare.load(csv) for label, csv in zip(labels, csvs)}
    compare.plot_success(runs, args.window, out)
    compare.plot_success_per_map(runs, args.window, out)
    compare.plot_learning(runs, args.window, out)
    starts = [s for df in runs.values() for s in compare.stage_starts(df)]
    phases = compare.phase_table(runs, max(starts) if starts else 0)
    phases.to_csv(out / "compare_by_phase.csv", index=False)

    table = summary_table(labels, evals)
    (out / "summary.md").write_text(f"# Comparison of tests {', '.join(map(str, args.tests))}\n\n"
                                    f"Evaluation: {args.episodes} episodes per mode, seed {args.seed}.\n\n{table}\n"
                                    "Training phases: `compare_by_phase.csv`. Plots: `compare_success.png`, "
                                    "`compare_success_per_map.png`, `compare_learning.png`.\n", encoding="utf-8")
    print(table)
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
