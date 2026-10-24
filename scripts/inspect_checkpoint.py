"""
inspect_checkpoint.py
---------------------
Print metadata stored in a .pth checkpoint file without loading the full model.

Usage:
    python scripts/inspect_checkpoint.py <checkpoint.pth>
    python scripts/inspect_checkpoint.py checkpoints/Test/24/GridAgent/final.pth
    python scripts/inspect_checkpoint.py checkpoints/   # inspect all .pth files recursively
"""
import argparse
import sys
from pathlib import Path

import torch


def inspect(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    print(f"\n{'─'*60}")
    print(f"  {path}")
    print(f"{'─'*60}")

    # Top-level metadata
    for key in ("algorithm", "epsilon", "global_step", "episode_count"):
        if key in ckpt:
            print(f"  {key:20s} = {ckpt[key]}")

    # Config dict
    if "config" in ckpt and isinstance(ckpt["config"], dict):
        print("  config:")
        cfg = ckpt["config"]
        interesting = ("obs_dim", "action_dim", "lr", "gamma", "batch_size",
                       "buffer_size", "eps_start", "eps_end", "eps_decay",
                       "target_update", "double_dqn", "device", "name")
        for k in interesting:
            if k in cfg:
                print(f"    {k:20s} = {cfg[k]}")

    # Network layer shapes
    for net_key in ("policy_state_dict", "target_state_dict"):
        if net_key in ckpt:
            sd = ckpt[net_key]
            layers = [f"{k}: {tuple(v.shape)}"
                      for k, v in sd.items() if "weight" in k]
            if layers:
                print(f"  {net_key} layers:")
                for l in layers:
                    print(f"    {l}")

    # Training history summary
    if "train_history" in ckpt and isinstance(ckpt["train_history"], dict):
        hist = ckpt["train_history"]
        print(f"  train_history keys: {list(hist.keys())}")

    # File size
    size_mb = path.stat().st_size / 1e6
    print(f"  file size            = {size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Inspect .pth checkpoint metadata")
    parser.add_argument("path", help="Path to a .pth file or directory")
    args = parser.parse_args()

    p = Path(args.path)
    if p.is_dir():
        files = sorted(p.rglob("*.pth"))
        if not files:
            sys.exit(f"No .pth files found under {p}")
        for f in files:
            try:
                inspect(f)
            except Exception as e:
                print(f"  [error reading {f}]: {e}")
    elif p.is_file():
        inspect(p)
    else:
        sys.exit(f"Path not found: {p}")


if __name__ == "__main__":
    main()
