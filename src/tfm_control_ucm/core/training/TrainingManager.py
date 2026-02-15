
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
from typing import Callable, Optional, override
import gymnasium as gym
import numpy as np
import signal
import sys
import tqdm

from torch.utils.tensorboard import SummaryWriter

from tfm_control_ucm.core.training.utils import InterruptHandler, EarlyStopping

from ..agents.base_rl_agent import BaseRLAgent


@dataclass
class TrainerConfig:
    """Configuration dataclass for training hyperparameters."""
    
    # General
    parallel: bool = True # Whether to train agents in parallel threads
    useTQDM: bool = True # Whether to use tqdm progress bars
    useEarlyStop: bool = True # Whether to use early stopping
    num_episodes: int = 100_000 # Total number of episodes to train
    useSoftReset: bool = True # Whether to enable soft reset on Ctrl+\

    # For multi-environment
    onePerEnv: bool = True # Whether to train one agent one environment at a time vs change the environment between episodes
    cyclesPerEnv: int = 1 # If onePerEnv is False, how many times we iterate through all the environments for all the episodes

    # For multi environment-multi agent
    style: str = "single" # "single" (1 to 1), "mixed" (all agents all environments) or "multi-env" (each agent trains on all environments at once)

    eps_step: int = 1 # Train every N steps (0 for end of episode)
    
    def __post_init__(self):
        assert isinstance(self.parallel, bool), "parallel must be a boolean"
        assert isinstance(self.useTQDM, bool), "useTQDM must be a boolean"  
        assert isinstance(self.num_episodes, int) and self.num_episodes > 1_000, "num_episodes must be a positive integer and greater than 1000 for meaningful training"
        assert isinstance(self.onePerEnv, bool), "onePerEnv must be a boolean"
        assert isinstance(self.cyclesPerEnv, int) and self.cyclesPerEnv > 0, "cyclesPerEnv must be a positive integer"
        assert isinstance(self.eps_step, int) and self.eps_step >= 0, "eps_step must be a non-negative integer"
        assert isinstance(self.style, str) and self.style in {"single", "mixed", "multi-env"}, "style must be a string and must be one of 'single', 'mixed' or 'multi-env'"
        assert isinstance(self.useEarlyStop, bool), "useEarlyStop must be a boolean"
        assert isinstance(self.useSoftReset, bool), "useSoftReset must be a boolean"

class BaseTrainer(ABC):
    envs: list[gym.Env] = []
    agents: list[BaseRLAgent] = []
    config: TrainerConfig

    """Abstract base class for trainers."""
    def __init__(self, logs_save_dir: str, models_save_dir: str, config: Optional[TrainerConfig] = None) -> None:
        super().__init__()
        self.logs_save_dir = logs_save_dir
        self.models_save_dir = models_save_dir
        self.config = config or TrainerConfig()
        self.exec_date = time.strftime("%Y%m%d-%H%M%S")
        self.print = lambda msg: tqdm.tqdm.write(msg) if self.config.useTQDM else print(msg) # type: ignore
    
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
        
    def is_stuck(self, recent_actions: deque, recent_rewards: deque, recent_successes: deque, episode: int) -> tuple[bool, str]:
        """
        Heuristics to detect whether the agent is stuck.

        Parameters
        ----------
        recent_actions : deque
            Last N episode-level dominant actions
        recent_rewards : deque
            Last N episode rewards
        recent_successes : deque
            Last N episode success indicators
        episode : int
            Current episode number

        Returns
        -------
        (stuck: bool, reason: str)
        """
        # Don't check too early (let buffer fill first)
        if episode < 5000:
            return False, ""

        # Check 1: Agent only ever picks the same action
        if len(recent_actions) >= 50 and len(set(recent_actions)) == 1:
            return True, f"Only using action {recent_actions[0]} for 50+ episodes"
        
        # Check 3: No successes in recent episodes
        if len(recent_successes) >= 1000 and sum(recent_successes) == 0:
            return True, f"No successes in 1000+ episodes"

        # Check 2: Rewards are not improving and all negative
        if len(recent_rewards) >= 500:
            mean_reward = sum(recent_rewards) / len(recent_rewards)
            if mean_reward < -200.0:
                return True, f"Mean reward stuck at {mean_reward:.2f} for 500+ episodes"

        
        return False, ""

    def do_soft_reset(self, agent, episode, writer):
        """
        Perform a soft reset: save current state, reset exploration and buffer.

        Parameters
        ----------
        agent : GridAgent_2
            The DQN agent
        episode : int
            Current episode (for logging)
        """
        self.print("\n" + "="*60)
        self.print("🔄 Performing soft reset...")
        self.print("="*60)

        # Save pre-reset checkpoint so you can always go back
        pre_reset_path = f"{self.models_save_dir}/{self.exec_date}/agent_pre_reset_ep{episode}.pth"
        agent.save(pre_reset_path)
        self.print(f"💾 Pre-reset checkpoint saved: {pre_reset_path}")

        agent.soft_reset()

        # Log the reset event in TensorBoard
        writer.add_scalar("Events/SoftReset", 1.0, episode)

        self.print(f"✅ Soft reset complete at episode {episode}")
        self.print(f"   - Epsilon: {agent.epsilon:.4f}")
        self.print(f"   - Buffer: cleared")
        self.print("="*60 + "\n")   

    def _inner_training(self, agent: BaseRLAgent, env: gym.Env, num_episodes: int, eps_step: int):
        global_step = 0
        soft_reset_count = 0
        episodes = tqdm.tqdm(range(num_episodes),desc="Training") if self.config.useTQDM else range(num_episodes)
        recent_successes = deque(maxlen=1000)
        recent_rewards = deque(maxlen=600)
        recent_actions = deque(maxlen=80)

        earlyStopping = EarlyStopping(save_path=f"{self.models_save_dir}/{agent.config.name}/early_stopping.pth") if self.config.useEarlyStop else None

        with SummaryWriter(log_dir=f"{self.logs_save_dir}/{agent.config.name}") as writer:
            with InterruptHandler(agent, writer,f"{self.models_save_dir}/{agent.config.name}") as handler:
                for episode in episodes:
                    state, _ = env.reset()
                    done = False
                    terminated= False

                    ep_steps = 0
                    ep_reward = 0
                    ep_action_counts = [0]*self.act_dim
                    ep_terminated_count= [0]*2
                    start_dist = state[-1][0]

                    while not done:
                        action = agent.select_action(state)
                        next_state, reward, terminated, truncated, info = env.step(action)
                        done = terminated or truncated
                        agent.store(state, action, reward, next_state, done)

                        if eps_step>0 and ep_steps % eps_step == 0:
                            loss = agent.train_step()
                        
                        ep_steps += 1
                        ep_reward += float(reward)
                        ep_action_counts[action] += 1
                        ep_terminated_count[int(terminated)] += 1
                        global_step += 1

                        state = next_state
                 
                    dominant_action = int(np.argmax(ep_action_counts))
                    dominant_success = int(np.argmax(ep_terminated_count))
                    recent_actions.append(dominant_action)
                    recent_rewards.append(ep_reward)
                    recent_successes.append(dominant_success)

                    if eps_step==0:
                        loss = agent.train_step()
                        if loss is not None:
                            writer.add_scalar(f"Loss/Episode", loss, episode)

                    if self.config.useSoftReset:
                        stuck, reason = self.is_stuck(recent_actions, recent_rewards, recent_successes, episode)
                        if stuck:
                            soft_reset_count += 1
                            self.print(f"\n⚠️  Agent stuck: {reason}")
                            self.print(f"   Triggering automatic soft reset #{soft_reset_count}")
                            self.do_soft_reset(agent, episode, writer)
                            recent_actions.clear()
                            recent_rewards.clear()
                            recent_successes.clear()
                    
                    if self.config.useEarlyStop and earlyStopping is not None:
                        if earlyStopping(ep_reward, agent, self.print):
                            break
                    
                    if episode % 500 == 0 and episode > 0:
                        checkpoint_path = f"{self.models_save_dir}/{agent.config.name}/{self.exec_date}/ep_{episode}.pth"
                        agent.save(checkpoint_path)
                    
                    writer.add_scalar("Policy/Epsilon/Episode",  agent.epsilon,              episode)
                    writer.add_scalar("Action/Dominant/Episode",  dominant_action,            episode)

                    writer.add_scalar("Steps/Episode",           ep_steps,                   episode)
                    
                    writer.add_scalar(f"Reward/Accumulated", ep_reward, episode)
                    writer.add_scalar(f"Reward/Avg", ep_reward/ep_steps if ep_steps > 0 else 0, episode)

                    writer.add_scalar(f"Success/Episode", int(terminated), episode)
                    writer.add_scalar("SoftResets/Total",        soft_reset_count,            episode)

                    writer.add_scalar(f"Distances/Start", start_dist, episode)
                    writer.add_scalar(f"Distances/End", state[-1][0], episode)
                    writer.add_scalar(f"Distances/Change", start_dist - state[-1][0], episode)
                    
                    

class SingleTrainer(BaseTrainer):
    """
    Trainer class for training a single agent in a single environment.
    """

    def __init__(self, logs_save_dir: str, models_save_dir: str, config: Optional[TrainerConfig] = None):
        super().__init__(logs_save_dir, models_save_dir, config)

    @override
    def add_environments(self, *envs: gym.Env):
        if len(self.envs) == 1 or len(envs) > 1:
            raise ValueError("SingleAgentSingleEnvironmentTrainer requires exactly one environment")
        return super().add_environments(*envs)

    @override
    def add_agents(self, *agents: BaseRLAgent):
        if len(self.agents) == 1 or  len(agents) > 1:
            raise ValueError("SingleAgentSingleEnvironmentTrainer requires exactly one agent")
        return super().add_agents(*agents)

    @override
    def train(self, num_episodes: Optional[int] = None, eps_step: Optional[int] = None):
        num_episodes = num_episodes or self.config.num_episodes
        eps_step = eps_step or self.config.eps_step
        if len(self.envs) == 0 or len(self.agents) == 0:
            raise ValueError("One environment and one agent must be added before training")
        super()._inner_training(self.agents[0], self.envs[0], num_episodes, eps_step)

class MultiAgentTrainer(BaseTrainer):
    """
    Trainer class for training multiple agents in a shared environment.
    Each agent learns its own policy while interacting with the same environment.
    """

    def __init__(self, logs_save_dir: str, models_save_dir: str, config: Optional[TrainerConfig] = None):
        super().__init__(logs_save_dir, models_save_dir, config)
    
    @override
    def add_environments(self, *envs: gym.Env):
        if len(self.envs) == 1 or len(envs) > 1:
            raise ValueError("MultiAgentTrainer allows only one environment")
        return super().add_environments(*envs)

    
    @override
    def train(self, num_episodes: Optional[int] = None, eps_step: Optional[int] = None):
        if len(self.envs) == 0 or len(self.agents) == 0:
            raise ValueError("One environment and at least one agent must be added before training")
        # If parallel is True, we will train each agent in a separate thread, sharing the same environment.
        num_episodes = num_episodes or self.config.num_episodes
        eps_step = eps_step or self.config.eps_step
        
        if self.config.parallel:
            with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
                futures = [executor.submit(super()._inner_training, agent, self.envs[0], num_episodes, eps_step) for agent in self.agents]
                for future in futures:
                    future.result()  # Wait for all agents to finish
        else:
            for agent in self.agents:
                super()._inner_training(agent, self.envs[0], num_episodes, eps_step)

class MultiEnvironmentTrainer(BaseTrainer):
    """
    Trainer class for training multiple agents in a shared environment.
    Each agent learns its own policy while interacting with the same environment.
    """

    def __init__(self, logs_save_dir: str, models_save_dir: str, config: Optional[TrainerConfig] = None):
        super().__init__(logs_save_dir, models_save_dir, config)

    @override
    def add_agents(self, *agents: BaseRLAgent):
        if len(self.agents) == 1 or len(agents) > 1:
            raise ValueError("MultiEnvironmentTrainer allows only one agent")
        return super().add_agents(*agents)
    

    @override
    def train(self, num_episodes: Optional[int] = None, eps_step: Optional[int] = None):
        num_episodes = num_episodes or self.config.num_episodes
        eps_step = eps_step or self.config.eps_step
        if len(self.envs) == 0 or len(self.agents) == 0:
            raise ValueError("At least one environment and one agent must be added before training")
        user = super() if self.config.onePerEnv else self
        # If parallel is True, we will train each agent in a separate thread, sharing the same environment.
        if self.config.onePerEnv:
            if self.config.parallel:
                with ThreadPoolExecutor(max_workers=len(self.envs)) as executor:
                    futures = [executor.submit(super()._inner_training, self.agents[0], env, num_episodes, eps_step) for env in self.envs]
                    for future in futures:
                        future.result()  # Wait for all agents to finish
            else:
                for env in self.envs:
                    super()._inner_training(self.agents[0], env, num_episodes, eps_step)
        else:
            self._inner_training(self.agents[0], None, num_episodes, eps_step)
    
    @override
    def _inner_training(self, agent: BaseRLAgent, env: Optional[gym.Env], num_episodes: int, eps_step: int):
        global_step = 0
        soft_reset_count = 0
        agent_id = id(agent)
        episodePerEnv = num_episodes//max(1, self.config.cyclesPerEnv)
        idx_env = 0
        current_env = self.envs[idx_env]

        episodes = tqdm.tqdm(range(num_episodes)) if self.config.useTQDM else range(num_episodes)
        recent_successes = deque(maxlen=1000)
        recent_rewards = deque(maxlen=50)
        recent_actions = deque(maxlen=50)
        earlyStopping = EarlyStopping(save_path=f"{self.models_save_dir}/{agent.config.name}/early_stopping.pth") if self.config.useEarlyStop else None

        with SummaryWriter(log_dir=f"{self.logs_save_dir}/{self.exec_date}/agent_{agent_id}") as writer:
            with InterruptHandler(agent, writer, self.logs_save_dir) as handler:
                for episode in episodes:
                    state, _ = current_env.reset()
                    done = False

                    ep_step = 0
                    ep_reward = 0
                    terminated = False
                    ep_action_counts = [0]*self.act_dim
                    ep_terminated_count= [0]*2
                    start_dist = state[-2]

                    while not done:
                        action = agent.select_action(state)
                        next_state, reward, terminated, truncated, info = current_env.step(action)
                        done = terminated or truncated
                        agent.store(state, action, reward, next_state, done)

                        ep_step += 1
                        ep_reward += float(reward)
                        ep_action_counts[action] += 1
                        ep_terminated_count[int(terminated)] += 1
                        global_step += 1

                        state = next_state

                        if eps_step>0 and ep_step % eps_step == 0:
                            loss = agent.train_step()

                        if done:
                            break
                    if self.config.useTQDM:
                        episodes.set_description(f"Agent {agent_id} - Episode {episode} - Reward: {ep_reward:.2f} - Steps: {ep_step} - Start Dist: {start_dist:.2f} - End Dist: {state[-2]:.2f}") # type: ignore
                    
                    dominant_action = int(np.argmax(ep_action_counts))
                    dominant_success = int(np.argmax(ep_terminated_count))
                    recent_actions.append(dominant_action)
                    recent_rewards.append(ep_reward)
                    recent_successes.append(dominant_success)

                    if eps_step==0:
                        loss = agent.train_step()
                        if loss is not None:
                            writer.add_scalar(f"Loss/Episode", loss, episode)

                    
                    writer.add_scalar("Policy/Epsilon/Episode",  agent.epsilon,              episode)
                    writer.add_scalar("Action/Dominant/Episode",  dominant_action,            episode)

                    writer.add_scalar("Steps/Episode",           ep_step,                   episode)
                    
                    writer.add_scalar(f"Reward/Accumulated", ep_reward, episode)
                    writer.add_scalar(f"Reward/Avg", ep_reward/ep_step if ep_step > 0 else 0, episode)

                    writer.add_scalar(f"Success/Episode", int(terminated), episode)
                    writer.add_scalar("SoftResets/Total",        soft_reset_count,            episode)

                    writer.add_scalar(f"Distances/Start", start_dist, episode)
                    writer.add_scalar(f"Distances/End", state[-2], episode)
                    writer.add_scalar(f"Distances/Change", start_dist - state[-2], episode)
                    
                    if self.config.useSoftReset:
                        stuck, reason = self.is_stuck(recent_actions, recent_rewards, recent_successes, episode)
                        if stuck:
                            soft_reset_count += 1
                            self.print(f"\n⚠️  Agent stuck: {reason}")
                            self.print(f"   Triggering automatic soft reset #{soft_reset_count}")
                            self.do_soft_reset(agent, episode, writer)
                            recent_actions.clear()
                            recent_rewards.clear()
                            recent_successes.clear()
                    
                    if self.config.useEarlyStop and earlyStopping is not None:
                        if earlyStopping(ep_reward, agent, self.print):
                            break

                    if episode % 500 == 0 and episode > 0:
                        checkpoint_path = f"{self.models_save_dir}/{self.exec_date}/{agent_id}_ep_{episode}.pth"
                        agent.save(checkpoint_path)
                    
                    if episode % episodePerEnv == 0 and episode > 0:
                        idx_env = (idx_env + 1) % len(self.envs)
                        current_env = self.envs[idx_env]

class MultiTrainer(BaseTrainer):
    """
    Trainer class for training multiple agents in multiple environments.
    Each agent learns its own policy while interacting with its own environment.
    """

    def __init__(self, logs_save_dir: str, models_save_dir: str, config: Optional[TrainerConfig] = None):
        super().__init__(logs_save_dir, models_save_dir, config)

    @override
    def train(self, num_episodes: int | None = None, eps_step: int | None = None):
        num_episodes = num_episodes or self.config.num_episodes
        eps_step = eps_step or self.config.eps_step
        if len(self.envs) == 0 or len(self.agents) == 0:
            raise ValueError("At least one environment and one agent must be added before training")
        
        iterator = list(zip(self.agents, self.envs)) if self.config.style == "single" else [(agent, env) for agent in self.agents for env in self.envs] if self.config.style == "mixed" else [(agent, self.envs) for agent in self.agents]
        
        if self.config.style == "multi-env":
            if self.config.parallel:
                with ThreadPoolExecutor(max_workers=len(iterator)) as executor:
                    futures = [executor.submit(self.__multi_env_training, agent, envs, num_episodes, eps_step) for agent, envs in iterator] # type: ignore
                    for future in futures:
                        future.result()  # Wait for all agents to finish
            else:
                for agent, envs in iterator:
                    super().__multi_env_training(agent, envs, num_episodes, eps_step) # type: ignore
        else:
            if self.config.parallel:
                with ThreadPoolExecutor(max_workers=len(iterator)) as executor:
                    futures = [executor.submit(super()._inner_training, agent, env, num_episodes, eps_step) for agent, env in iterator] # type: ignore
                    for future in futures:
                        future.result()  # Wait for all agents to finish
            else:
                for agent, env in iterator:
                    super()._inner_training(agent, env, num_episodes, eps_step) # type: ignore
    
    def __multi_env_training(self, agent: BaseRLAgent, envs: list[gym.Env], num_episodes: int, eps_step: int):
        # This method is used when style is "multi-env". It trains a single agent on multiple environments sequentially.
        # The agent's experience buffer and policy are shared across all environments.
        aux_env = MultiEnvironmentTrainer(logs_save_dir=self.logs_save_dir, models_save_dir=self.models_save_dir, config=self.config)
        aux_env.config.onePerEnv = False
        aux_env.add_environments(*envs) # type: ignore
        aux_env.add_agents(agent)
        aux_env.train(num_episodes, eps_step)

