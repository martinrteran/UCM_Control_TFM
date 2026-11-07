"""
seed_stats.py
-------------
Aggregate the evaluation of one configuration trained with several seeds, and
optionally compare it with another configuration (one run or several seeds).

The inputs are the cached evaluations plots/evals/eval_<N>_e<episodes>_s<seed>.json
written by eval_policy_modes.py / analyze_tests.py (greedy + sampled, per map);
missing ones are computed first (needs the run's config.json and checkpoint, like
analyze_tests.py) unless --no-eval is given. Every run is evaluated with the same
evaluation seed, so all of them play the same episodes.

For each mode and metric (total success, success per map, timeouts that are loops)
the table gives the value of every seed, mean +- sample std across seeds and a 95 %
t interval (mean +- t(0.975, n-1) * std / sqrt(n), percentages capped at [0, 100]).
These describe training-seed variability, not evaluation noise.

Comparison (--vs):
  - one run: its value and its distance to the group, (value - mean) / std;
  - two or more runs: their mean +- std, the difference of means and Welch's t
    (p-value when scipy is installed).

Usage (from the repo root):
    python scripts/seed_stats.py 31 311 312 313 314 --vs 32
    python scripts/seed_stats.py 31 311 312 313 314 --vs 32 321 322 323 324 --name "31 masking" --vs-name "32 signed progress"
    python scripts/seed_stats.py 31 311 312 313 314 --out plots/seeds_31/seeds_stats.md --no-eval
"""
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

MAPS = ["obstacle_free_map", "1_obstacle", "4_obstacles", "simple_map"]
MODES = ("greedy", "sampled")

# t(0.975, df) for df = 1..30; above that the normal quantile is close enough
_T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
         2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
         2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def t975(df: int) -> float:
    """Two-sided 95 % Student-t quantile for df degrees of freedom."""
    if df < 1: raise ValueError("df must be >= 1")
    return _T975[df - 1] if df <= len(_T975) else 1.960


def eval_path(test: int, episodes: int, seed: int) -> Path:
    return Path(f"plots/evals/eval_{test}_e{episodes}_s{seed}.json")


def load_eval(test: int, episodes: int, seed: int, allow_eval: bool) -> dict:
    """Cached evaluation of a run; computed with analyze_tests.ensure_eval when missing and allowed."""
    path = eval_path(test, episodes, seed)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if not allow_eval:
        raise FileNotFoundError(f"{path} is missing (run without --no-eval to compute it)")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analyze_tests import ensure_eval  # noqa: E402  (imports torch; only needed here)
    return ensure_eval(test, episodes, seed)


def success_rate(result_mode: dict, map_name: Optional[str] = None) -> float:
    """Success in % over all maps, or over one map; nan if the map was never played."""
    per_map = result_mode["per_map"]
    maps = [map_name] if map_name else list(per_map)
    wins = sum(per_map.get(m, {}).get("success", 0) for m in maps)
    n = sum(sum(per_map.get(m, {}).values()) for m in maps)
    return 100.0 * wins / n if n else float("nan")


Metric = Tuple[str, Callable[[dict], float], bool]  # (name, value from one mode's result, is a percentage)


def metrics() -> List[Metric]:
    rows: List[Metric] = [("Success (total) %", lambda r: success_rate(r), True)]
    rows += [(f"{m} %", (lambda r, m=m: success_rate(r, m)), True) for m in MAPS]
    rows.append(("Timeouts that are loops", lambda r: float(r["timeout_loops"]), False))
    return rows


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    """Mean and sample standard deviation (0 for a single value)."""
    n = len(values)
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    return mean, std


def t_interval(values: Sequence[float], percent: bool) -> Tuple[float, float]:
    """95 % t interval of the mean (nan for a single value); percentages clipped to [0, 100]."""
    n = len(values)
    mean, std = mean_std(values)
    if n < 2: return float("nan"), float("nan")
    half = t975(n - 1) * std / math.sqrt(n)
    low, high = mean - half, mean + half
    if percent: low, high = max(low, 0.0), min(high, 100.0)
    return low, high


def welch(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, Optional[float]]:
    """Welch's t statistic of mean(b) - mean(a), its degrees of freedom and the two-sided p (None without scipy)."""
    (ma, sa), (mb, sb) = mean_std(a), mean_std(b)
    va, vb = sa ** 2 / len(a), sb ** 2 / len(b)
    if va + vb == 0: return float("nan"), float("nan"), None
    t = (mb - ma) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / ((va ** 2 / (len(a) - 1)) + (vb ** 2 / (len(b) - 1)))
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), df))
    except ImportError:
        p = None
    return t, df, p


def _fmt(value: float, percent: bool) -> str:
    if math.isnan(value): return "–"
    return f"{value:.1f}" if percent else f"{value:.0f}" if float(value).is_integer() else f"{value:.1f}"


def build_table(group: Dict[str, dict], vs: Optional[Dict[str, dict]], mode: str) -> List[str]:
    """Markdown table of one mode. group / vs map a label to that run's evaluation."""
    labels = list(group)
    head = ["Metric", *labels, "Mean ± std", "95 % interval"]
    if vs and len(vs) == 1:
        head += [next(iter(vs)), "Distance (std)"]
    elif vs:
        head += ["Other mean ± std", "Difference", "Welch t (df)", "p"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, value_of, percent in metrics():
        values = [value_of(group[l][mode]) for l in labels]
        if any(math.isnan(v) for v in values):
            continue  # A map not played by every run (e.g. a different map set)
        mean, std = mean_std(values)
        low, high = t_interval(values, percent)
        cells = [name, *(_fmt(v, percent) for v in values), f"{mean:.1f} ± {std:.1f}",
                 f"[{low:.1f}, {high:.1f}]" if not math.isnan(low) else "–"]
        if vs and len(vs) == 1:
            other = value_of(next(iter(vs.values()))[mode])
            cells += [_fmt(other, percent), f"{(other - mean) / std:+.1f}" if std > 0 else "–"]
        elif vs:
            others = [value_of(r[mode]) for r in vs.values()]
            om, osd = mean_std(others)
            t, df, p = welch(values, others)
            cells += [f"{om:.1f} ± {osd:.1f}", f"{om - mean:+.1f}",
                      f"{t:+.2f} ({df:.1f})" if not math.isnan(t) else "–",
                      f"{p:.3f}" if p is not None else "–"]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def report(group: Dict[str, dict], vs: Optional[Dict[str, dict]], name: str, vs_name: Optional[str],
           tests: Sequence[int], vs_tests: Sequence[int], episodes: int, seed: int) -> str:
    n = len(group)
    t_used = f"t = {t975(n - 1):.3f}" if n > 1 else "no interval with one run"
    collisions = {mode: sum(sum(r[mode]["per_map"].get(m, {}).get("collision", 0) for m in r[mode]["per_map"])
                            for r in group.values()) for mode in MODES}
    lines = [f"# {name} across {n} training seeds", "",
             f"Runs: {', '.join(f'{l} = test {t}' for l, t in zip(group, tests))}.",
             f"Evaluation: {episodes} episodes per mode, evaluation seed {seed}, so every run plays the same episodes.",
             f"Collisions summed over the runs: greedy {collisions['greedy']}, sampled {collisions['sampled']}.", "",
             f"Mean ± sample std over the runs; 95 % interval = mean ± t(0.975, n−1) · std / √n ({t_used}),",
             "percentages capped at [0, 100]. It describes training-seed variability, not evaluation noise; with few",
             "seeds it is approximate."]
    if vs and len(vs) == 1:
        lines.append(f"Comparison: {vs_name} (test {vs_tests[0]}, one run); distance = (value − mean) / std of the group.")
    elif vs:
        lines.append(f"Comparison: {vs_name} (tests {', '.join(map(str, vs_tests))}); difference = other − this group, "
                     "Welch's t test (p needs scipy).")
    lines.append("")
    for mode in MODES:
        lines += [f"## {mode.capitalize()}", "", *build_table(group, vs, mode), ""]
    sources = ", ".join(str(eval_path(t, episodes, seed)) for t in [*tests, *vs_tests])
    lines.append(f"Generated by `scripts/seed_stats.py` from {sources}.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Mean ± std and 95 % interval of a configuration across training seeds")
    parser.add_argument("tests", type=int, nargs="+", help="Test numbers of the same configuration, one per training seed")
    parser.add_argument("--labels", nargs="+", default=None, help="One label per test (default: s0, s1, ...)")
    parser.add_argument("--name", default=None, help="Name of the configuration (default: Test <first>)")
    parser.add_argument("--vs", type=int, nargs="+", default=None, help="Test number(s) of another configuration to compare with")
    parser.add_argument("--vs-name", default=None, help="Name of the other configuration (default: Test <first --vs>)")
    parser.add_argument("--episodes", type=int, default=1000, help="Evaluation episodes per mode (selects the cached file)")
    parser.add_argument("--seed", type=int, default=0, help="Evaluation seed (selects the cached file)")
    parser.add_argument("--out", type=Path, default=None, help="Markdown output (default: plots/seeds_<first>/seeds_stats.md)")
    parser.add_argument("--no-eval", action="store_true", help="Fail if an evaluation is missing instead of computing it")
    args = parser.parse_args()

    labels = args.labels or [f"s{i}" for i in range(len(args.tests))]
    if len(labels) != len(args.tests): parser.error("one label per test")
    if len(args.tests) < 2: parser.error("give at least two runs of the configuration")
    vs_tests = args.vs or []
    name = args.name or f"Test {args.tests[0]}"
    vs_name = args.vs_name or (f"Test {vs_tests[0]}" if vs_tests else None)
    out = args.out or Path(f"plots/seeds_{args.tests[0]}/seeds_stats.md")

    group = {l: load_eval(t, args.episodes, args.seed, not args.no_eval) for l, t in zip(labels, args.tests)}
    vs = {str(t): load_eval(t, args.episodes, args.seed, not args.no_eval) for t in vs_tests} or None

    text = report(group, vs, name, vs_name, args.tests, vs_tests, args.episodes, args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
