"""
rewards.py

Reward contract shared by every grid environment.

The environment builds a StepContext after each step (it decides whether the
episode ended) and passes it to a RewardFunction, which only scores the step.
Keeping termination out of the reward makes success rates comparable between
experiments that use different rewards.

Author: Martin
"""
from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, NamedTuple, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class StepContext:
    """
    Everything a reward function may use to score one step.

    Distances are in grid cells; positions are (col, row).

    Parameters
    ----------
    action : int
        Action taken this step.
    previous_action : int
        Action taken on the previous step (-1 on the first step of an episode).
    obs : np.ndarray
        Observation after the step, as returned by the environment.
    previous_obs : np.ndarray
        Observation before the step.
    dist_to_goal : float
        Euclidean distance from the robot to the goal after the step.
    previous_dist_to_goal : float
        Same distance before the step.
    min_obstacle_dist : float
        Shortest lidar reading after the step (lidar_max_range when nothing is detected).
    hit_obstacle : bool
        The robot tried to move into a wall or an obstacle and stayed in place.
    reached_goal : bool
        The episode terminated because the robot reached the goal.
    max_steps_reached : bool
        The episode was truncated because it reached max_steps.
    steps : int
        Steps taken so far in the episode, including this one.
    max_steps : int
        Step limit of the episode.
    robot_pos : np.ndarray
        Robot position after the step.
    goal_pos : np.ndarray
        Goal position.
    map_diagonal : float
        Largest map diagonal, for normalising distances.
    lidar_max_range : float
        Maximum lidar range.
    """
    action: int
    previous_action: int
    obs: np.ndarray
    previous_obs: np.ndarray
    dist_to_goal: float
    previous_dist_to_goal: float
    min_obstacle_dist: float
    hit_obstacle: bool
    reached_goal: bool
    max_steps_reached: bool
    steps: int
    max_steps: int
    robot_pos: np.ndarray
    goal_pos: np.ndarray
    map_diagonal: float
    lidar_max_range: float


class RewardResult(NamedTuple):
    """Total reward of a step and the named terms that add up to it."""
    reward: float
    components: Dict[str, float]


class RewardFunction(ABC):
    """
    Base class for reward functions.

    Subclasses implement __call__. Dataclass subclasses get get_config() for
    free, so their parameters can be logged with each run.
    """

    @abstractmethod
    def __call__(self, ctx: StepContext) -> RewardResult:
        """
        Score one step.

        Parameters
        ----------
        ctx : StepContext
            Outcome of the step, built by the environment.

        Returns
        -------
        RewardResult
            Total reward and its named components.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        return type(self).__name__

    def get_config(self) -> Dict[str, Any]:
        """Parameters of this reward, for experiment logs."""
        params = dataclasses.asdict(self) if dataclasses.is_dataclass(self) else {}
        return {"name": self.name, **params}


RewardCallable = Union[RewardFunction, Callable[[StepContext], Union[RewardResult, Tuple[float, Dict[str, float]]]]]


def evaluate_reward(reward_fn: RewardCallable, ctx: StepContext) -> RewardResult:
    """
    Call a reward function and normalise its output to a RewardResult.

    Plain callables may return a bare float or a (reward, components) pair.
    """
    result = reward_fn(ctx)
    if isinstance(result, RewardResult):
        return result
    if isinstance(result, tuple):
        reward, components = result
        return RewardResult(float(reward), dict(components))
    return RewardResult(float(result), {})


def summarize_components(components: Dict[str, float]) -> Dict[str, float]:
    """
    Smallest, largest and spread of the absolute component values.

    Used for the Info/Reward/{min,max,diff} TensorBoard scalars.
    """
    if not components:
        return {"componente_reward_min": 0.0, "componente_reward_max": 0.0, "componente_reward_diff": 0.0}
    magnitudes = [abs(float(v)) for v in components.values()]
    return {
        "componente_reward_min": min(magnitudes),
        "componente_reward_max": max(magnitudes),
        "componente_reward_diff": max(magnitudes) - min(magnitudes),
    }


@dataclass
class ProgressReward(RewardFunction):
    """
    Dense progress shaping plus terminal bonus and penalty.

    reward = progress + step_cost + proximity_penalty + terminal_reward, where

    - progress = progress_scale * (previous_dist - dist) / map_diagonal,
      clipped at 0 when clip_negative_progress is True;
    - step_cost = -step_penalty;
    - proximity_penalty = -max_proximity_penalty * (safety_margin - d) / safety_margin
      when the nearest obstacle d is closer than safety_margin;
    - terminal_reward = +goal_reward on reaching the goal, -collision_penalty on a collision.

    The defaults reproduce the reward of grid_training_10 copy.py;
    clip_negative_progress=True reproduces the one in
    Grid_Robot_Sections_Env_MultiMap before the reward was injected.

    Parameters
    ----------
    progress_scale : float
        Weight of the normalised progress term.
    clip_negative_progress : bool
        If True, moving away from the goal gives 0 progress instead of a penalty.
    step_penalty : float
        Constant cost per step.
    safety_margin : float
        Lidar distance (cells) below which proximity is penalised.
    max_proximity_penalty : float
        Penalty when touching an obstacle; it decreases linearly to 0 at safety_margin.
    goal_reward : float
        Bonus when the goal is reached.
    collision_penalty : float
        Penalty when the robot hits a wall or an obstacle.
    """
    progress_scale: float = 10.0
    clip_negative_progress: bool = False
    step_penalty: float = 0.05
    safety_margin: float = 3.0
    max_proximity_penalty: float = 1.0
    goal_reward: float = 20.0
    collision_penalty: float = 5.0

    def __call__(self, ctx: StepContext) -> RewardResult:
        progress = self.progress_scale * (ctx.previous_dist_to_goal - ctx.dist_to_goal) / ctx.map_diagonal
        if self.clip_negative_progress:
            progress = max(progress, 0.0)

        closeness = max(0.0, self.safety_margin - ctx.min_obstacle_dist) / self.safety_margin if self.safety_margin > 0 else 0.0

        if ctx.reached_goal:
            terminal = self.goal_reward
        elif ctx.hit_obstacle:
            terminal = -self.collision_penalty
        else:
            terminal = 0.0

        components = {
            "progress": float(progress),
            "step_cost": -self.step_penalty,
            "proximity_penalty": -self.max_proximity_penalty * closeness,
            "terminal_reward": terminal,
        }
        return RewardResult(sum(components.values()), components)
