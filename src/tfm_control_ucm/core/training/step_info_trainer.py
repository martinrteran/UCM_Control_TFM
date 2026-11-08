"""
step_info_trainer.py

Trainers that relay each step's info dict to the agent. TrainingManager is not
changed: its loop calls agent.observe(state, action, reward, next_state,
terminated, truncated) without the info, so an agent cannot tell why a
truncated episode ended. In these environments a collision and the step limit
are both `truncated`, but only the step limit should bootstrap.

These trainers wrap each episode's environment in StepInfoRelay, which calls
agent.on_step_info(info) right after env.step() and before the loop's
agent.observe(). Agents without on_step_info are unaffected. The loop, the
logging and the TensorBoard tags are the parent's.

Combine with the slip logging: class X(SlipLoggingMixin, StepInfoMixin, SingleTrainer).

Author: Martin
"""
from __future__ import annotations

from torch.utils.tensorboard import SummaryWriter

from ..agents.base_rl_agent import BaseRLAgent
from .slip_trainer import SlipLoggingMixin
from .TrainingManager import MultiEnvironmentTrainer, SingleTrainer


class StepInfoRelay:
    """Pass-through environment wrapper that hands each step's info to agent.on_step_info."""

    def __init__(self, env, agent):
        self.env = env
        self._on_step_info = getattr(agent, "on_step_info", None)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        result = self.env.step(action)
        if self._on_step_info is not None:
            self._on_step_info(result[-1])
        return result

    def render(self):
        return self.env.render()

    def __getattr__(self, name):
        if name == "env": raise AttributeError(name) # Not set yet (e.g. while copying): avoid infinite recursion
        return getattr(self.env, name)


class StepInfoMixin:
    """Relays the step info to the agent (put it before the trainer class in the bases)."""

    def _run_episode(self, agent: BaseRLAgent, env, episode: int, global_step: int, eps_step: int,
                     writer: SummaryWriter) -> int:
        return super()._run_episode(agent, StepInfoRelay(env, agent), episode, global_step, eps_step, writer) # type: ignore


class StepInfoSingleTrainer(StepInfoMixin, SingleTrainer):
    """SingleTrainer that relays the step info to the agent."""


class StepInfoMultiEnvironmentTrainer(StepInfoMixin, MultiEnvironmentTrainer):
    """MultiEnvironmentTrainer that relays the step info to the agent."""


class SlipStepInfoSingleTrainer(SlipLoggingMixin, StepInfoMixin, SingleTrainer):
    """SingleTrainer that relays the step info and logs the slips (slippery environments)."""
