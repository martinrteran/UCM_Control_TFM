"""
slip_trainer.py

Trainers that also log the slips of the slippery environment
(grid_env/slippery_environment.py). TrainingManager is not changed: the
training loop is the parent's; each episode's environment is wrapped in a
counter that reads info['slipped'] / info['slip_kind'] / info['slip_collision']
from every step, and the counts are logged after the parent logs the episode.

TensorBoard tags, per episode (only when the environment reports slips, i.e. the
robot has a slip model):
    Slip/Count        slips in the episode
    Slip/Rate         slips / steps
    Slip/Collision    1 if the episode ended because a slide hit an obstacle
    Slip/Kind/<kind>  slips of each kind the slip model can produce
plus the text Config/Slip with the slip model's parameters.

Author: Martin
"""
from __future__ import annotations

import json
from collections import Counter

from torch.utils.tensorboard import SummaryWriter

from ..agents.base_rl_agent import BaseRLAgent
from .TrainingManager import MultiEnvironmentTrainer, SingleTrainer


class SlipCounter:
    """
    Pass-through wrapper of an environment that counts the slips of one episode.

    Only reset, step and render are used by the training loop; any other attribute
    is read from the wrapped environment.
    """
    def __init__(self, env):
        self.env = env
        self.reports_slips = False # The env put slip keys in info (its robot has a slip model)
        self.steps = 0
        self.slips = 0
        self.kinds: Counter = Counter()
        self.slip_collision = False

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        result = self.env.step(action)
        info = result[-1]
        self.steps += 1
        if 'slipped' in info:
            self.reports_slips = True
            if info['slipped']:
                self.slips += 1
                self.kinds[info['slip_kind']] += 1
            self.slip_collision = bool(info.get('slip_collision', False))
        return result

    def render(self):
        return self.env.render()

    def __getattr__(self, name):
        if name == "env": raise AttributeError(name) # Not set yet (e.g. while copying): avoid infinite recursion
        return getattr(self.env, name)


class SlipLoggingMixin:
    """Adds the slip tags to a trainer (put it before the trainer class in the bases)."""

    def _run_episode(self, agent: BaseRLAgent, env, episode: int, global_step: int, eps_step: int,
                     writer: SummaryWriter) -> int:
        counter = SlipCounter(env)
        global_step = super()._run_episode(agent, counter, episode, global_step, eps_step, writer) # type: ignore
        if counter.reports_slips:
            writer.add_scalar("Slip/Count", counter.slips, episode)
            writer.add_scalar("Slip/Rate", counter.slips / counter.steps if counter.steps else 0.0, episode)
            writer.add_scalar("Slip/Collision", int(counter.slip_collision), episode)
            slip_model = getattr(env, "slip_model", None)
            for kind in (slip_model.kinds if slip_model is not None else counter.kinds):
                writer.add_scalar(f"Slip/Kind/{kind}", counter.kinds[kind], episode)
        return global_step

    def _log_config(self, writer: SummaryWriter, agent: BaseRLAgent, env):
        super()._log_config(writer, agent, env) # type: ignore
        slip_model = getattr(env, "slip_model", None)
        if slip_model is not None and hasattr(slip_model, "get_config"):
            writer.add_text("Config/Slip", json.dumps(slip_model.get_config(), indent=2, default=str))


class SlipSingleTrainer(SlipLoggingMixin, SingleTrainer):
    """SingleTrainer (one agent, one environment) that also logs the slips."""


class SlipMultiEnvironmentTrainer(SlipLoggingMixin, MultiEnvironmentTrainer):
    """MultiEnvironmentTrainer (one agent, several environments) that also logs the slips."""
