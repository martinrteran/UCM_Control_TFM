"""
TrainingManager.py

One training loop for every agent. The trainer talks to the agent only through
the episode hooks of BaseRLAgent (on_episode_start, act, observe,
on_episode_end, episode_metrics) and to the environment only through the
gymnasium API (reset/step and their info dicts), so DQN, PPO and the recurrent
PPO variants are trained and logged in exactly the same way.

Trainers differ only in which environment each episode uses:
- SingleTrainer: one agent, one environment.
- MultiEnvironmentTrainer: one agent, several environments.

TensorBoard tags (per episode unless noted):
    Reward/Total, Reward/Avg, Success/Episode, Success/Map/<map>,
    Done/Hit Obstacle, Done/Max Steps Reached,
    Steps/Episode, Steps/Min/Episode, Steps/Ratio/Episode,
    Distances/Start, Distances/End, Distances/Change, Loss/Episode,
    the agent's episode_metrics() (e.g. Policy/Epsilon/Episode, Agent/Entropy),
    per step: Loss/Global step, Info/Reward/Total/Steps, Info/Reward/Components/<name>,
    Info/Reward/{min,max,diff}/Steps.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import math
import time
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import tqdm

from torch.utils.tensorboard import SummaryWriter

from tfm_control_ucm.core.training.utils import InterruptHandler, EarlyStopping

from ..agents.base_rl_agent import BaseRLAgent


@dataclass
class TrainerConfig:
    """Configuration dataclass for training hyperparameters."""

    # General
    parallel: bool = True # Kept for compatibility; agents are trained one after another
    useTQDM: bool = True # Whether to use tqdm progress bars
    useEarlyStop: bool = True # Whether to use early stopping
    num_episodes: int = 100_000 # Total number of episodes to train
    render: bool = False # Whether to render the environment during training (may slow down training)
    log_steps: bool = True # Whether to log the reward components of every step (large event files)


    # For multi-environment
    onePerEnv: bool = True # True: train on each environment in turn (one block of episodes each); False: change the environment every episode
    cyclesPerEnv: int = 1 # If onePerEnv is True, how many times the blocks go through all the environments

    # For multi environment-multi agent
    style: str = "single" # "single" (1 to 1), "mixed" (all agents all environments) or "multi-env" (each agent trains on all environments at once)

    eps_step: int = 1 # Off-policy agents: train every N steps (0 for end of episode). On-policy agents train at the end of each episode

    def __post_init__(self):
        assert isinstance(self.parallel, bool), "parallel must be a boolean"
        assert isinstance(self.useTQDM, bool), "useTQDM must be a boolean"
        assert isinstance(self.num_episodes, int) and self.num_episodes > 1_000, "num_episodes must be a positive integer and greater than 1000 for meaningful training"
        assert isinstance(self.log_steps, bool), "log_steps must be a boolean"
        assert isinstance(self.onePerEnv, bool), "onePerEnv must be a boolean"
        assert isinstance(self.cyclesPerEnv, int) and self.cyclesPerEnv > 0, "cyclesPerEnv must be a positive integer"
        assert isinstance(self.eps_step, int) and self.eps_step >= 0, "eps_step must be a non-negative integer"
        assert isinstance(self.style, str) and self.style in {"single", "mixed", "multi-env"}, "style must be a string and must be one of 'single', 'mixed' or 'multi-env'"
        assert isinstance(self.useEarlyStop, bool), "useEarlyStop must be a boolean"

class BaseTrainer(ABC):
    """
    Abstract base class for trainers: holds the agents and environments and the
    single training loop. Subclasses decide which environment each episode uses.
    """
    envs: list[gym.Env]
    agents: list[BaseRLAgent]
    config: TrainerConfig

    def __init__(self, logs_save_dir: str, models_save_dir: str, config: Optional[TrainerConfig] = None) -> None:
        super().__init__()
        self.logs_save_dir = logs_save_dir
        self.models_save_dir = models_save_dir
        self.config = config or TrainerConfig()
        self.exec_date = time.strftime("%Y%m%d-%H%M%S")
        self.print = lambda msg: tqdm.tqdm.write(msg) if self.config.useTQDM else print(msg) # type: ignore

        self.envs = []
        self.agents = []

    def add_environments(self, *envs: gym.Env):
        if len(envs) == 0:
            raise ValueError("No environments provided")

        assert all(env.get_observation_dim() == envs[0].get_observation_dim() for env in envs),"All environments must have the same observation space" # type: ignore
        assert all(env.get_action_dim() == envs[0].get_action_dim() for env in envs), "All environments must have the same action space" # type: ignore

        self.obs_dim = envs[0].get_observation_dim()# type: ignore
        self.act_dim = envs[0].get_action_dim()# type: ignore

        self.envs.extend(envs)
        return self

    def add_agents(self, *agents: BaseRLAgent):
        if len(agents) == 0:
            raise ValueError("No agents provided")

        self.agents.extend(agents)
        return self

    @abstractmethod
    def train(self, num_episodes: Optional[int] = None, eps_step: Optional[int] = None):
        raise NotImplementedError("train() must be implemented by subclasses")

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    def _train_agent(self, agent: BaseRLAgent, env_for_episode: Callable[[int], gym.Env], num_episodes: int, eps_step: int):
        """
        Train one agent for num_episodes; env_for_episode(i) gives the environment of episode i.
        Saves <models_save_dir>/<agent name>/final.pth at the end.
        """
        global_step = 0
        episodes = tqdm.tqdm(range(num_episodes), desc="Training") if self.config.useTQDM else range(num_episodes)

        with SummaryWriter(log_dir=f"{self.logs_save_dir}/{agent.config.name}") as writer:
            self._log_config(writer, agent, env_for_episode(0))
            with InterruptHandler(agent, writer, f"{self.models_save_dir}/{agent.config.name}") as handler:
                for episode in episodes:
                    global_step = self._run_episode(agent, env_for_episode(episode), episode, global_step, eps_step, writer)

        checkpoint_path = f"{self.models_save_dir}/{agent.config.name}/final.pth"
        agent.save(checkpoint_path)

    def _inner_training(self, agent: BaseRLAgent, env: gym.Env, num_episodes: int, eps_step: int):
        """Train one agent on one environment (kept for scripts that call it directly)."""
        self._train_agent(agent, lambda episode: env, num_episodes, eps_step)

    def _run_episode(self, agent: BaseRLAgent, env: gym.Env, episode: int, global_step: int, eps_step: int,
                     writer: SummaryWriter) -> int:
        """Play and learn from one episode; returns the updated global step."""
        state, reset_info = env.reset()
        agent.on_episode_start()

        done = False
        terminated = False
        ep_steps = 0
        ep_reward = 0.0
        losses = []
        info = {}

        if self.config.render:
            env.render()

        while not done:
            action = agent.act(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.observe(state, action, float(reward), next_state, terminated, truncated)

            ep_steps += 1
            ep_reward += float(reward)
            global_step += 1

            if agent.train_on_step and eps_step > 0 and ep_steps % eps_step == 0:
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)
                    writer.add_scalar("Loss/Global step", loss, global_step)

            if self.config.log_steps:
                self._log_step(writer, info, float(reward), global_step)
            if self.config.render:
                env.render()
            state = next_state

        if agent.train_on_step and eps_step == 0:
            loss = agent.train_step()
            if loss is not None: losses.append(loss)
        loss = agent.on_episode_end(state)
        if loss is not None: losses.append(loss)

        self._log_episode(writer, agent, episode, ep_reward, ep_steps, terminated, losses, reset_info, info)
        return global_step

    # ------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------
    def _log_config(self, writer: SummaryWriter, agent: BaseRLAgent, env: gym.Env):
        """Record the agent and reward configuration with the run."""
        writer.add_text("Config/Agent", json.dumps(agent.config.to_dict(), indent=2, default=str))
        reward_fn = getattr(env, "reward_fn", None)
        if reward_fn is not None and hasattr(reward_fn, "get_config"):
            writer.add_text("Config/Reward", json.dumps(reward_fn.get_config(), indent=2, default=str))

    def _log_step(self, writer: SummaryWriter, info: dict, reward: float, global_step: int):
        writer.add_scalar("Info/Reward/Total/Steps", reward, global_step)
        for component, value in info.get('reward_components', {}).items():
            writer.add_scalar(f"Info/Reward/Components/{component}", value, global_step)
        for stat in ("min", "max", "diff"):
            if f"componente_reward_{stat}" in info:
                writer.add_scalar(f"Info/Reward/{stat}/Steps", info[f"componente_reward_{stat}"], global_step)

    def _log_episode(self, writer: SummaryWriter, agent: BaseRLAgent, episode: int, ep_reward: float, ep_steps: int,
                     terminated: bool, losses: list, reset_info: dict, info: dict):
        writer.add_scalar("Reward/Total", ep_reward, episode)
        writer.add_scalar("Reward/Avg", ep_reward / ep_steps if ep_steps > 0 else 0, episode)

        writer.add_scalar("Success/Episode", int(terminated), episode)
        writer.add_scalar("Done/Hit Obstacle", int(info.get('hit_obstacle', False)), episode)
        writer.add_scalar("Done/Max Steps Reached", int(info.get('max_steps_reached', False)), episode)
        if 'map' in info:
            writer.add_scalar(f"Success/Map/{info['map']}", int(terminated), episode)

        writer.add_scalar("Steps/Episode", ep_steps, episode)
        optimal_steps = reset_info.get('optimal_steps')
        if optimal_steps is not None and 0 < optimal_steps < math.inf:
            writer.add_scalar("Steps/Min/Episode", optimal_steps, episode)
            writer.add_scalar("Steps/Ratio/Episode", ep_steps / optimal_steps, episode)

        if 'dist2goal' in reset_info and 'dist2goal' in info:
            writer.add_scalar("Distances/Start", reset_info['dist2goal'], episode)
            writer.add_scalar("Distances/End", info['dist2goal'], episode)
            writer.add_scalar("Distances/Change", reset_info['dist2goal'] - info['dist2goal'], episode)

        if losses:
            writer.add_scalar("Loss/Episode", losses[-1], episode)
        for tag, value in agent.episode_metrics().items():
            writer.add_scalar(tag, value, episode)

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------
    def simulate(self, agent: BaseRLAgent, env: gym.Env, num_episodes: int = 10, render: bool = False):
        """
        Run the agent greedily (no learning) for a number of episodes and print each result.
        """
        for episode in range(num_episodes):
            state, _ = env.reset()
            agent.on_episode_start()
            done = False
            terminated = False
            ep_steps = 0
            ep_reward = 0

            while not done:
                action = agent.act(state, training=False)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                ep_steps += 1
                ep_reward += float(reward)
                state = next_state
                if render:
                    env.render()

            print(f"Episode {episode + 1}/{num_episodes} - Reward: {ep_reward:.2f} - Steps: {ep_steps} - Success: {bool(terminated)}")


class SingleTrainer(BaseTrainer):
    """
    Trainer class for training a single agent in a single environment.
    """

    def __init__(self, logs_save_dir: str, models_save_dir: str, config: Optional[TrainerConfig] = None):
        super().__init__(logs_save_dir, models_save_dir, config)


    def add_environments(self, *envs: gym.Env):
        if len(self.envs) == 1 or len(envs) > 1:
            raise ValueError("SingleAgentSingleEnvironmentTrainer requires exactly one environment")
        return super().add_environments(*envs)


    def add_agents(self, *agents: BaseRLAgent):
        if len(self.agents) == 1 or  len(agents) > 1:
            raise ValueError("SingleAgentSingleEnvironmentTrainer requires exactly one agent")
        return super().add_agents(*agents)


    def train(self, num_episodes: Optional[int] = None, eps_step: Optional[int] = None):
        num_episodes = num_episodes or self.config.num_episodes
        eps_step = self.config.eps_step if eps_step is None else eps_step
        if len(self.envs) == 0 or len(self.agents) == 0:
            raise ValueError("One environment and one agent must be added before training")
        env = self.envs[0]
        self._train_agent(self.agents[0], lambda episode: env, num_episodes, eps_step)


class MultiEnvironmentTrainer(BaseTrainer):
    """
    Trainer class for training one agent on several environments.

    With TrainerConfig.onePerEnv the episodes are split into blocks, one
    environment per block, repeated cyclesPerEnv times; otherwise the
    environment changes every episode. num_episodes is the total.
    """

    def __init__(self, logs_save_dir: str, models_save_dir: str, config: Optional[TrainerConfig] = None):
        super().__init__(logs_save_dir, models_save_dir, config)

    def add_agents(self, *agents: BaseRLAgent):
        if len(self.agents) == 1 or len(agents) > 1:
            raise ValueError("MultiEnvironmentTrainer allows only one agent")
        return super().add_agents(*agents)

    def _env_schedule(self, num_episodes: int) -> Callable[[int], gym.Env]:
        envs = list(self.envs)
        if not self.config.onePerEnv:
            return lambda episode: envs[episode % len(envs)]
        blocks = len(envs) * self.config.cyclesPerEnv
        block_size = max(1, math.ceil(num_episodes / blocks))
        return lambda episode: envs[(episode // block_size) % len(envs)]

    def train(self, num_episodes: Optional[int] = None, eps_step: Optional[int] = None):
        num_episodes = num_episodes or self.config.num_episodes
        eps_step = self.config.eps_step if eps_step is None else eps_step
        if len(self.envs) == 0 or len(self.agents) == 0:
            raise ValueError("At least one environment and one agent must be added before training")
        self._train_agent(self.agents[0], self._env_schedule(num_episodes), num_episodes, eps_step)


class PPOTrainer(SingleTrainer):
    """Deprecated: every agent uses the same loop now. Alias of SingleTrainer kept for old scripts."""

class PPOGRUTrainer(SingleTrainer):
    """Deprecated: every agent uses the same loop now. Alias of SingleTrainer kept for old scripts."""

class RecurrentPPOTrainer(SingleTrainer):
    """Deprecated: every agent uses the same loop now. Alias of SingleTrainer kept for old scripts."""
