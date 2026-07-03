
from abc import ABC, abstractmethod
from ast import Call
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from msilib.schema import SelfReg
import time
import datetime
from typing import Callable, Optional
import gymnasium as gym
import numpy as np
import signal
import sys
import tqdm

class InterruptHandler:
    """
    Context manager for handling keyboard interrupts with auto-save.
    - Ctrl+C → Save checkpoint and exit
    - Ctrl+\\ (SIGQUIT) → Soft reset agent (Linux/Mac only)
    """
    
    def __init__(self, agent, writer=None, save_dir="checkpoints", print_fn: Callable=print):
        self.agent = agent
        self.writer = writer
        self.save_dir = save_dir
        self.interrupted = False
        self.soft_reset_requested = False
        self.print_fn = print_fn
        self.original_sigint = None
        self.original_sigquit = None
        
    def __enter__(self):
        self.original_sigint = signal.signal(signal.SIGINT, self._handle_exit)

        # Ctrl+\ triggers soft reset (Linux/Mac only, not available on Windows)
        if hasattr(signal, 'SIGQUIT'):
            self.original_sigquit = signal.signal(signal.SIGQUIT, self._handle_soft_reset) # type: ignore
            
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original handlers
        signal.signal(signal.SIGINT, self.original_sigint)
        if hasattr(signal, 'SIGQUIT') and self.original_sigquit:
            signal.signal(signal.SIGQUIT, self.original_sigquit) # type: ignore
        
        if self.interrupted:
            return True  # Suppress KeyboardInterrupt
        
        if exc_type is not None and exc_type is not KeyboardInterrupt:
            self._save_emergency(exc_type, exc_val)
            return False
            
        return False
    
    def _handle_exit(self, sig, frame):
        """Ctrl+C → save and exit."""
        self.interrupted = True
        self.print_fn("\n" + "="*60)
        self.print_fn("🛑 Training interrupted by user (Ctrl+C)")
        self.print_fn("="*60)
        
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        checkpoint_path = f"{self.save_dir}/agent_interrupted_{timestamp}.pth"
        
        self.print_fn("Saving checkpoint...")
        try:
            self.agent.save(checkpoint_path)
            self.print_fn(f"✅ Checkpoint saved to: {checkpoint_path}")
            self.print_fn(f"   - Global step: {self.agent.global_step}")
            self.print_fn(f"   - Epsilon: {self.agent.epsilon:.4f}")
            self.print_fn(f"   - Buffer size: {len(self.agent.buffer)}")
        except Exception as e:
            self.print_fn(f"❌ Error saving checkpoint: {e}")
        
        if self.writer is not None:
            self.writer.close()
        
        self.print_fn("✅ Cleanup complete. Exiting...")
        self.print_fn("="*60)
        sys.exit(0)

    def _save_emergency(self, exc_type, exc_val):
        """Save emergency checkpoint on unexpected exception."""
        self.print_fn("\n" + "="*60)
        self.print_fn(f"❌ Unexpected error: {exc_type.__name__}: {exc_val}")
        self.print_fn("="*60)
        
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        checkpoint_path = f"{self.save_dir}/agent_error_{timestamp}.pth"
        
        try:
            self.agent.save(checkpoint_path)
            self.print_fn(f"💾 Emergency checkpoint saved to: {checkpoint_path}")
        except Exception as e:
            self.print_fn(f"❌ Could not save emergency checkpoint: {e}")


class EarlyStopping:
    """
    Detiene el entrenamiento si la métrica de recompensa no mejora tras un número 
    determinado de episodios (patience).
    """
    def __init__(self, save_path: str, patience: int = 10000, min_delta: float = 0.0, mode: str = 'max'):
        if not isinstance(patience, (int, np.integer)): 
            raise TypeError("The patience must be an integer")
        if patience <= 0: 
            raise ValueError("The patience must be greater than 0")
            
        if not isinstance(min_delta, (int, float, np.number)): 
            raise TypeError("The min_delta must be a numeric value")
            
        if not isinstance(mode, str): 
            raise TypeError("The mode must be a string")
        if mode not in ['max', 'min']: 
            raise ValueError("The mode must be 'max' or 'min'")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.best_checkpoint_saved = False
        self.save_path = save_path

    def __call__(self, current_score: float, agent, print_fn: Callable) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            self._save_checkpoint(agent, print_fn)
            return False

        if self.mode == 'max':
            improvement = current_score - self.best_score > self.min_delta
        else:
            improvement = self.best_score - current_score > self.min_delta

        if improvement:
            self.best_score = current_score
            self.counter = 0
            self._save_checkpoint(agent, print_fn)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print_fn(f"\n[Early Stopping] Límite de paciencia ({self.patience}) alcanzado. "
                         f"Mejor puntuación: {self.best_score:.2f}")
        
        return self.early_stop

    def _save_checkpoint(self, agent, print_fn: Callable):
        try:
            agent.save(self.save_path)
            self.best_checkpoint_saved = True
        except Exception as e:
            print_fn(f"Error guardando checkpoint de early stopping: {e}")