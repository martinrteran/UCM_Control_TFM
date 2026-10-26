"""
export_episode_metrics.py
-------------------------
Stream a TensorBoard event file once and save its per-episode scalars as CSV
(long format: tag, step, value). Per-step tags (Info/*, Loss/Global step) are
skipped, so huge event files only need to be parsed one time.

Usage:
    python scripts/export_episode_metrics.py "runs/Test/25/PPO Agent"
    python scripts/export_episode_metrics.py "runs/Test/25/PPO Agent" --out runs/Test/25/episodes.csv
"""
import argparse
import csv
from pathlib import Path

from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

SKIP_PREFIXES = ("Info/", "Loss/Global step")


def export(run_dir: Path, out: Path) -> int:
    rows = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tag", "step", "value"])
        for event_file in sorted(run_dir.glob("events.out.tfevents.*")):
            for event in EventFileLoader(str(event_file)).Load():
                for value in event.summary.value:
                    if value.tag.startswith(SKIP_PREFIXES):
                        continue
                    if value.HasField("simple_value"):
                        number = value.simple_value
                    elif value.HasField("tensor") and value.tensor.float_val:
                        number = value.tensor.float_val[0]
                    else:
                        continue  # text summaries (Config/*)
                    writer.writerow([value.tag, event.step, number])
                    rows += 1
    return rows


def main():
    parser = argparse.ArgumentParser(description="Export per-episode TensorBoard scalars to CSV")
    parser.add_argument("run_dir", type=Path, help="Directory with the events.out.tfevents.* file(s)")
    parser.add_argument("--out", type=Path, default=None, help="Output CSV (default: <run_dir>/../episodes.csv)")
    args = parser.parse_args()
    out = args.out or args.run_dir.parent / "episodes.csv"
    rows = export(args.run_dir, out)
    print(f"{rows} rows -> {out}")


if __name__ == "__main__":
    main()
