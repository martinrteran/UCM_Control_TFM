"""
logger.py

Training logger: TensorBoard + CSV.

Usage:
    tensorboard --logdir logs/tensorboard

Author: RL Project
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import csv
import time

from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    """
    Logs scalar metrics to TensorBoard and a CSV file.

    Outputs:
        logs/tensorboard/<run_name>_<timestamp>/   ← TensorBoard events
        logs/csv/<run_name>_<timestamp>.csv        ← flat CSV
    """

    def __init__(self, log_dir: str = "logs", run_name: str = "run"):
        ts = time.strftime("%Y%m%d_%H%M%S")

        # ── TensorBoard ───────────────────────────────────────────────────────
        tb_path = Path(log_dir) / "tensorboard" / f"{run_name}_{ts}"
        tb_path.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(tb_path))

        # ── CSV ───────────────────────────────────────────────────────────────
        csv_dir = Path(log_dir) / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path   = csv_dir / f"{run_name}_{ts}.csv"
        self._csv_file   = open(self._csv_path, "w", newline="")
        self._csv_writer = None   # created lazily once we know the keys

        print(f"[TensorBoard] {tb_path}")
        print(f"[TensorBoard] Run:  tensorboard --logdir {Path(log_dir) / 'tensorboard'}")
        print(f"[CSV]         {self._csv_path}")

    def log(self, metrics: Dict[str, Any], step: int):
        row = {"step": step, **metrics}

        # TensorBoard
        for tag, value in metrics.items():
            try:
                self.writer.add_scalar(tag, float(value), global_step=step)
            except (TypeError, ValueError):
                pass

        # CSV (header written on first call)
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=list(row.keys())
            )
            self._csv_writer.writeheader()
        try:
            self._csv_writer.writerow(row)
            self._csv_file.flush()
        except ValueError:
            pass  # new keys appeared mid-run — skip row safely

    def close(self):
        self.writer.flush()
        self.writer.close()
        self._csv_file.close()
        print("[Logger] Closed TensorBoard writer and CSV file.")