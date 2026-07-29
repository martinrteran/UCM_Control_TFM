
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
from tfm_control_ucm.utils.timer import Timer
from typing import Callable, Optional
import gymnasium as gym
import numpy as np
import signal
import sys
import torch

import tqdm

from torch.utils.tensorboard import SummaryWriter

from tfm_control_ucm.core.training.utils import InterruptHandler, EarlyStopping
from tfm_control_ucm.utils.path_planner import PathPlanner

from ..agents.base_rl_agent import BaseRLAgent, compute_gae


@dataclass
class TrainerConfig:
    """Configuration dataclass for training hyperparameters."""
    
    # General
    parallel: bool = True # Whether to train agents in parallel threads
    useTQDM: bool = True # Whether to use tqdm progress bars
    useEarlyStop: bool = True # Whether to use early stopping
    num_episodes: int = 100_000 # Total number of episodes to train
    render: bool = False # Whether to render the environment during training (may slow down training)


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

class BaseTrainer(ABC):
    envs: list[gym.Env]
    agents: list[BaseRLAgent]
    config: TrainerConfig

    """Abstract base class for trainers."""
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
   
    def _inner_training(self, agent: BaseRLAgent, env: gym.Env, num_episodes: int, eps_step: int):
        global_step = 0
        episodes = tqdm.tqdm(range(num_episodes),desc="Training") if self.config.useTQDM else range(num_episodes)
        # recent_successes = deque(maxlen=1000)
        # recent_rewards = deque(maxlen=600)
        # recent_actions = deque(maxlen=80)
        path_planner = PathPlanner(env.map.grid) # type: ignore

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
                    start_dist = state[-2] # state[-1][0]
                    start_pos = env.robot.position# type: ignore
                    end_pos = env.goal_pos# type: ignore
                    start_pos = (int(start_pos[1]), int(start_pos[0]))
                    end_pos = (int(end_pos[1]), int(end_pos[0]))
                    min_steps = path_planner.min_steps(start=start_pos, goal=end_pos, method="wavefront") 
                    if self.config.render:
                        env.render()
                    loss = None
                    while not done:
                        if self.config.render:
                            env.render()
                        action = agent.select_action(state)
                        next_state, reward, terminated, truncated, info = env.step(action)
                        done = terminated or truncated

                        agent.store(state, action, reward, next_state, done)

                        if ep_steps > 0 and eps_step>0 and ep_steps % eps_step == 0:
                            loss = agent.train_step()
                            if loss is not None:
                                writer.add_scalar(f"Loss/Global step", loss, global_step)
                        
                        ep_steps += 1
                        ep_reward += float(reward)
                        ep_action_counts[action] += 1
                        ep_terminated_count[int(terminated)] += 1
                        global_step += 1

                        state = next_state
                        

                    if eps_step==0:
                        loss = agent.train_step()
                    
                    if loss is not None:
                        writer.add_scalar(f"Loss/Episode", loss, episode)
                    
                    writer.add_scalar("Policy/Epsilon/Episode",  agent.epsilon,              episode)
                    writer.add_scalar("Policy/Epsilon/Global Step",  agent.epsilon,              global_step)
                    # writer.add_scalar("Action/Dominant/Episode",  dominant_action,            episode)

                    writer.add_scalar("Steps/Episode",           ep_steps,                   episode)
                    writer.add_scalar("Steps/Min/Episode",       min_steps,                  episode)
                    writer.add_scalar("Steps/Ratio/Episode",     ep_steps/min_steps if min_steps>0 else 0, episode)
                    
                    writer.add_scalar(f"Reward/Total", ep_reward, episode)
                    writer.add_scalar(f"Reward/Avg", ep_reward/ep_steps if ep_steps > 0 else 0, episode)

                    writer.add_scalar(f"Success/Episode", int(terminated), episode)

                    writer.add_scalar(f"Distances/Start", start_dist, episode)
                    writer.add_scalar(f"Distances/End", state[-2], episode) # state[-1][0], episode)
                    writer.add_scalar(f"Distances/Change", start_dist - state[-2], episode) # state[-1][0], episode)

        checkpoint_path = f"{self.models_save_dir}/{agent.config.name}/final.pth"# {self.exec_date}
        agent.save(checkpoint_path)
    def simulate(self, agent: BaseRLAgent, env: gym.Env, num_episodes: int = 10, render: bool = False):
        """
        Simulate the agent in the environment for a given number of episodes.
        """
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            terminated = False
            ep_steps = 0
            ep_reward = 0

            while not done:
                action = agent.select_action(state, training=False)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                ep_steps += 1
                ep_reward += float(reward)
                state = next_state
                if render:
                    env.render()

            print(f"Episode {episode + 1}/{num_episodes} - Reward: {ep_reward:.2f} - Steps: {ep_steps}")


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
        eps_step = eps_step or self.config.eps_step
        if len(self.envs) == 0 or len(self.agents) == 0:
            raise ValueError("One environment and one agent must be added before training")
        super()._inner_training(self.agents[0], self.envs[0], num_episodes, eps_step)

class PPOTrainer(BaseTrainer):
    def train(self, num_episodes: Optional[int] = None, eps_step: Optional[int] = None):

        global_step = 0

        agent = self.agents[0]
        env = self.envs[0]


        num_episodes = num_episodes or self.config.num_episodes
        episodes = tqdm.tqdm(range(num_episodes),desc="Training") if self.config.useTQDM else range(num_episodes)
        eps_step = eps_step or self.config.eps_step
        
        path_planner = PathPlanner(env.map.grid) # type: ignore
        with SummaryWriter(log_dir=f"{self.logs_save_dir}/{agent.config.name}") as writer:
            with InterruptHandler(agent, writer,f"{self.models_save_dir}/{agent.config.name}") as handler:
                for episode in episodes:
                    state, _ = env.reset()

                    states = []
                    actions = []
                    rewards = []
                    dones = []
                    values = []
                    log_probs = []

                    done = False
                    terminated= False

                    start_dist = state[-2] # state[-1][0]
                    start_pos = env.robot.position# type: ignore
                    end_pos = env.goal_pos# type: ignore
                    start_pos = (int(start_pos[1]), int(start_pos[0]))
                    end_pos = (int(end_pos[1]), int(end_pos[0]))
                    min_steps = path_planner.min_steps(start=start_pos, goal=end_pos, method="wavefront") 
                    if self.config.render:
                        env.render()
                    
                    loss = None
                    info = {}

                    while not done:
                        if self.config.render:
                            env.render()

                        # 1. Value
                        # with Timer("Value Computation", sync_cuda=True):
                        value = agent.value_net(agent._to_tensor(state)).item()
                        values.append(value)

                        # 2. Action
                        # with Timer("Action Selection", sync_cuda=True):
                        action = agent.select_action(state, training=True)

                        # 3. Log prob
                        # with Timer("Log Prob Computation", sync_cuda=False):
                        #logits = agent.policy_net(agent._to_tensor(state))
                        #dist = torch.distributions.Categorical(logits=logits)
                        #log_prob = dist.log_prob(torch.tensor(action, device=agent.device)).item()
                        log_prob = agent.last_log_prob.item() if hasattr(agent, 'last_log_prob') else 0.0

                        # 4. Step
                        # with Timer("Environment Step"):
                        next_state, reward, terminated, truncated, info = env.step(action)
                        done = terminated or truncated

                        # 5. Store
                        states.append(state)
                        actions.append(action)
                        rewards.append(reward)
                        dones.append(done)
                        log_probs.append(log_prob)

                        if info:
                            writer.add_scalar("Info/Max Steps Reached", int(info.get('max_steps_reached', False)), episode)
                            for component, value in info.get('reward_components', {}).items():
                                writer.add_scalar(f"Reward/Components/{component}", value, global_step)
                            writer.add_scalar(f"Reward/Total/Steps", info.get('reward', 0), global_step)
                            # writer.add_scalar("Info/Hit Obstacle", int(info.get('hit_obstacle', False)), episode)
                            # writer.add_scalar("Info/Min Dist", info.get('min_dist', 0), episode)
                            # writer.add_scalar("Info/Max Dist", info.get('max_dist', 0), episode)
                            # writer.add_scalar("Info/Avg Dist", info.get('avg_dist', 0), episode)
                        if self.config.render:
                            env.render()
                        global_step += 1
                        state = next_state

                    # Final value
                    last_value = agent.value_net(agent._to_tensor(state)).item()
                    values.append(last_value)

                    # GAE
                    advantages, returns = compute_gae(
                        rewards=np.array(rewards),
                        values=np.array(values),
                        dones=np.array(dones),
                        gamma=agent.gamma,
                        lam=agent.lam
                    )

                    # Store PPO transitions
                    for t in range(len(states)):
                        agent.buffer.add(
                            states[t],
                            actions[t],
                            rewards[t],
                            states[t+1] if t < len(states)-1 else state,
                            dones[t],
                            log_probs[t],
                            advantages[t],
                            returns[t]
                        )

                    # Train PPO
                    # with Timer("PPO Training Step", sync_cuda=True):
                    loss = agent.train_step()
                    
                    ep_steps = len(states)
                    ep_reward = sum(rewards)
                    #with Timer("Logging", sync_cuda=False):
                    writer.add_scalar("Agent/Policy/Value", last_value, episode)
                    if loss is not None:
                        writer.add_scalar("Agent/Policy/Loss", loss, episode)
                    writer.add_scalar("Agent/Policy/Entropy", agent.last_entropy, episode)
                    writer.add_scalar("Agent/Policy/PolicyLoss", agent.last_policy_loss, episode)
                    writer.add_scalar("Agent/Policy/ValueLoss", agent.last_value_loss, episode)

                    writer.add_scalar("Steps/Episode",           ep_steps,                   episode)
                    writer.add_scalar("Steps/Min",       min_steps,                  episode)
                    writer.add_scalar("Steps/Ratio",     ep_steps/min_steps if min_steps>0 else 0, episode)
                    
                    
                    writer.add_scalar(f"Reward/Total/Episode", ep_reward, episode)

                    writer.add_scalar(f"Done/Success", int(terminated), episode)
                    writer.add_scalar(f"Done/Max Steps Reached", info.get('max_steps_reached', False), episode)
                    writer.add_scalar(f"Done/Hit Obstacle", info.get('hit_obstacle', False), episode)

                    writer.add_scalar(f"Distances/Start", start_dist, episode)
                    writer.add_scalar(f"Distances/End", state[-2], episode) # state[-1][0], episode)
                    writer.add_scalar(f"Distances/Change", start_dist - state[-2], episode) # state[-1][0], episode)
        
        checkpoint_path = f"{self.models_save_dir}/{agent.config.name}/final.pth"# {self.exec_date}
        agent.save(checkpoint_path)
class PPOGRUTrainer(BaseTrainer):

    def simulate(self, agent: BaseRLAgent, env: gym.Env, num_episodes: int = 10, render: bool = False):
        for episode in range(num_episodes):
            state, _ = env.reset()
            agent.reset_hidden()

            done = False
            ep_reward = 0.0
            steps = 0

            while not done:
                action = agent.select_action(state, training=False)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                state = next_state
                ep_reward += float(reward)
                steps += 1

                if render:
                    env.render()

            print(f"[SIM] Episode {episode+1}/{num_episodes} - Reward: {ep_reward:.2f} - Steps: {steps}")


    def train(self, num_episodes: Optional[int] = None, eps_step: Optional[int] = None):
        global_step = 0

        agent = self.agents[0]      # PPOGRUAgent
        env = self.envs[0]

        num_episodes = num_episodes or self.config.num_episodes
        episodes = tqdm.tqdm(range(num_episodes), desc="Training") if self.config.useTQDM else range(num_episodes)
        eps_step = eps_step or self.config.eps_step

        path_planner = PathPlanner(env.map.grid)  # type: ignore
    
        with SummaryWriter(log_dir=f"{self.logs_save_dir}/{agent.config.name}") as writer:
            with InterruptHandler(agent, writer, f"{self.models_save_dir}/{agent.config.name}") as handler:
                for episode in episodes:
                    state, _ = env.reset()
                    agent.reset_hidden()

                    states   = []
                    actions  = []
                    rewards  = []
                    dones    = []
                    log_probs = []
                    values   = []

                    done = False
                    terminated = False

                    start_dist = state[-2]
                    start_pos = env.robot.position  # type: ignore
                    end_pos   = env.goal_pos        # type: ignore
                    start_pos = (int(start_pos[1]), int(start_pos[0]))
                    end_pos   = (int(end_pos[1]),   int(end_pos[0]))
                    min_steps = path_planner.min_steps(start=start_pos, goal=end_pos, method="wavefront")

                    if self.config.render:
                        env.render()

                    info = {}

                    while not done:
                        # 1. Value recurrente
                        value = agent.evaluate_value(state)
                        values.append(value)

                        # 2. Acción recurrente
                        action = agent.select_action(state, training=True)

                        # 3. Log prob recurrente
                        log_prob = agent.last_log_prob.item()

                        # 4. Step
                        next_state, reward, terminated, truncated, info = env.step(action)
                        done = terminated or truncated

                        # 5. Guardar transición en el agente
                        agent.store_transition(state, action, reward, done, log_prob, value)

                        states.append(state)
                        actions.append(action)
                        rewards.append(reward)
                        dones.append(done)
                        log_probs.append(log_prob)

                        state = next_state
                        if info:
                            writer.add_scalar("Info/Max Steps Reached", int(info.get('max_steps_reached', False)), episode)
                            for component, value in info.get('reward_components', {}).items():
                                writer.add_scalar(f"Reward/Components/{component}", value, global_step)
                            writer.add_scalar(f"Reward/Total/Steps", info.get('reward', 0), global_step)
                        
                        if self.config.render:
                            env.render()
                        global_step += 1

                    # Entrenar PPO (GAE + update dentro del agente)
                    loss = agent.train_step()

                    ep_steps  = len(states)
                    ep_reward = sum(rewards)

                    # métricas del agente
                    last_value   = agent.last_return
                    entropy      = agent.last_entropy
                    value_loss   = agent.last_value_loss
                    policy_loss  = agent.last_policy_loss
                    advantage    = agent.last_advantage

                    # logs principales
                    writer.add_scalar("Agent/Loss/Total", loss, episode)
                    writer.add_scalar("Agent/Loss/Policy", policy_loss, episode)
                    writer.add_scalar("Agent/Loss/Value", value_loss, episode)
                    writer.add_scalar("Agent/Entropy/Episode", entropy, episode)
                    writer.add_scalar("Agent/Value/Last", last_value, episode)
                    writer.add_scalar("Agent/Advantage/Mean", advantage, episode)

                    writer.add_scalar("Steps/Episode", ep_steps, episode)
                    writer.add_scalar("Steps/Min/Episode", min_steps, episode)
                    writer.add_scalar("Steps/Ratio/Episode", ep_steps / min_steps if min_steps > 0 else 0, episode)

                    writer.add_scalar("Reward/Total/Episode", ep_reward, episode)

                    writer.add_scalar("Done/Success", int(terminated), episode)
                    writer.add_scalar("Done/Max Steps Reached", info.get('max_steps_reached', False), episode)
                    writer.add_scalar("Done/Hit Obstacle", info.get('hit_obstacle', False), episode)

                    writer.add_scalar("Distances/Start",  start_dist, episode)
                    writer.add_scalar("Distances/End",    state[-2], episode)
                    writer.add_scalar("Distances/Change", start_dist - state[-2], episode)

        checkpoint_path = f"{self.models_save_dir}/{agent.config.name}/final.pth"
        agent.save(checkpoint_path)


class RecurrentPPOTrainer(BaseTrainer):
    def train(self, num_episodes: Optional[int] = None, eps_step: Optional[int] = None):
        agent = self.agents[0]
        env   = self.envs[0]

        num_episodes = num_episodes or self.config.num_episodes
        episodes = tqdm.tqdm(range(num_episodes), desc="Training") if self.config.useTQDM else range(num_episodes)
        path_planner = PathPlanner(env.map.grid)
        global_step = 0
        with SummaryWriter(log_dir=f"{self.logs_save_dir}/{agent.config.name}") as writer:
            with InterruptHandler(agent, writer, f"{self.models_save_dir}/{agent.config.name}") as handler:
                for episode in episodes:
                    state, _ = env.reset()
                    agent.reset_hidden()

                    states   = []
                    actions  = []
                    rewards  = []
                    dones    = []
                    log_probs = []
                    values   = []

                    done = False
                    terminated = False
                    ep_steps = 0

                    start_dist = state[-2] # state[-1][0]
                    start_pos = env.robot.position# type: ignore
                    end_pos = env.goal_pos# type: ignore
                    start_pos = (int(start_pos[1]), int(start_pos[0]))
                    end_pos = (int(end_pos[1]), int(end_pos[0]))

                    min_steps = path_planner.min_steps(start=start_pos, goal=end_pos, method="wavefront") 
                    info =  {}
                    while not done:
                        action, log_prob, value = agent.select_action(state, training=True)
                        next_state, reward, terminated, truncated, info = env.step(action)
                        done = terminated or truncated

                        states.append(state)
                        actions.append(action)
                        rewards.append(reward)
                        dones.append(done)
                        log_probs.append(log_prob)
                        values.append(value)

                        state = next_state
                        
                        if info:
                            writer.add_scalar("Info/Max Steps Reached", int(info.get('max_steps_reached', False)), episode)
                            for component, value in info.get('reward_components', {}).items():
                                writer.add_scalar(f"Reward/Components/{component}", value, global_step)
                            writer.add_scalar(f"Reward/Total/Steps", info.get('reward', 0), global_step)
                            # writer.add_scalar("Info/Hit Obstacle", int(info.get('hit_obstacle', False)), episode)
                            # writer.add_scalar("Info/Min Dist", info.get('min_dist', 0), episode)
                            # writer.add_scalar("Info/Max Dist", info.get('max_dist', 0), episode)
                            # writer.add_scalar("Info/Avg Dist", info.get('avg_dist', 0), episode)



                        if self.config.render:
                            env.render()


                        ep_steps += 1
                        global_step += 1

                    agent.store_episode(states, actions, rewards, dones, log_probs, values)

                    loss = agent.train_step()

                    
                    if loss is not None:
                        writer.add_scalar("Policy/Loss", loss, episode)
                    writer.add_scalar("Policy/Policy/Loss", agent.last_policy_loss, episode)
                    writer.add_scalar("Policy/Value/Loss",  agent.last_value_loss,  episode)
                    writer.add_scalar("Policy/Entropy", agent.last_entropy, episode)

                    writer.add_scalar("Steps/Episode",           ep_steps,                   episode)
                    writer.add_scalar("Steps/Min",       min_steps,                  episode)
                    writer.add_scalar("Steps/Ratio",     ep_steps/min_steps if min_steps>0 else 0, episode)
                    
                    ep_reward = sum(rewards)
                    writer.add_scalar(f"Reward/Total/Episode", ep_reward, episode)

                    writer.add_scalar(f"Done/Success", int(terminated), episode)
                    writer.add_scalar(f"Done/Max Steps Reached", info.get('max_steps_reached', False), episode)
                    writer.add_scalar(f"Done/Hit Obstacle", info.get('hit_obstacle', False), episode)

                    writer.add_scalar(f"Distances/Start", start_dist, episode)
                    writer.add_scalar(f"Distances/End", state[-2], episode) # state[-1][0], episode)
                    writer.add_scalar(f"Distances/Change", start_dist - state[-2], episode) # state[-1][0], episode)
        
        checkpoint_path = f"{self.models_save_dir}/{agent.config.name}/final.pth"# {self.exec_date}
        agent.save(checkpoint_path)

    def simulate(self, agent: BaseRLAgent, env: gym.Env, num_episodes: int = 10, render: bool = False):
        for episode in range(num_episodes):
            state, _ = env.reset()
            agent.reset_hidden(batch_size=1)

            done = False
            ep_reward = 0.0
            steps = 0

            while not done:
                # acción determinista (sin sampling)
                action, _, _ = agent.select_action(state, training=False)

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                state = next_state
                ep_reward += float(reward)
                steps += 1

                if render:
                    env.render()

            print(f"[SIM] Episode {episode+1}/{num_episodes} - Reward: {ep_reward:.2f} - Steps: {steps}")


class MultiAgentTrainer(BaseTrainer):
    """
    Trainer class for training multiple agents in a shared environment.
    Each agent learns its own policy while interacting with the same environment.
    """

    def __init__(self, logs_save_dir: str, models_save_dir: str, config: Optional[TrainerConfig] = None):
        super().__init__(logs_save_dir, models_save_dir, config)
    
    
    def add_environments(self, *envs: gym.Env):
        if len(self.envs) == 1 or len(envs) > 1:
            raise ValueError("MultiAgentTrainer allows only one environment")
        return super().add_environments(*envs)

    
    
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

    
    def add_agents(self, *agents: BaseRLAgent):
        if len(self.agents) == 1 or len(agents) > 1:
            raise ValueError("MultiEnvironmentTrainer allows only one agent")
        return super().add_agents(*agents)
    

    
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
    
    
    def _inner_training(self, agent: BaseRLAgent, env: Optional[gym.Env], num_episodes: int, eps_step: int):
        global_step = 0
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
                    writer.add_scalar(f"Distances/Start", start_dist, episode)
                    writer.add_scalar(f"Distances/End", state[-2], episode)
                    writer.add_scalar(f"Distances/Change", start_dist - state[-2], episode)
                    
                    
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

