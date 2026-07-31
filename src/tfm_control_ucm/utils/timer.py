import time
import numpy as np
import torch
from collections import defaultdict
 
 
class Timer:
    """Context manager acumulativo. Usa torch.cuda.synchronize()
    para que el tiempo de GPU no se mida de forma engañosa
    (CUDA es asíncrono: sin sync, el 'tiempo' de una op en GPU
    puede parecer ~0 aunque tarde varios ms en ejecutarse)."""
 
    totals = defaultdict(float)
    counts = defaultdict(int)
 
    def __init__(self, name, sync_cuda=False):
        self.name = name
        self.sync_cuda = sync_cuda and torch.cuda.is_available()
 
    def __enter__(self):
        if self.sync_cuda:
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self
 
    def __exit__(self, *args):
        if self.sync_cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - self.t0
        Timer.totals[self.name] += dt
        Timer.counts[self.name] += 1
 
    @staticmethod
    def report():
        total = sum(Timer.totals.values())
        print(f"\n{'fase':<25}{'tiempo (s)':>12}{'%':>8}{'ms/it':>10}")
        print("-" * 55)
        for name, t in sorted(Timer.totals.items(), key=lambda x: -x[1]):
            n = Timer.counts[name]
            pct = 100 * t / total if total else 0
            print(f"{name:<25}{t:>12.3f}{pct:>7.1f}%{1000*t/n:>9.2f}")
        print("-" * 55)
        print(f"{'TOTAL':<25}{total:>12.3f}\n")