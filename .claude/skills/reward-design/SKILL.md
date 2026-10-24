---
name: reward-design
description: Design or refine a reward_fn for Grid_Robot_Env or Grid_Robot_Sections_Env. Use when the user wants to change robot behavior through the reward.
argument-hint: "[desired behavior]"
---

# Reward Function Design

Design or refine a reward function for `Grid_Robot_Env` or `Grid_Robot_Sections_Env`.

`$ARGUMENTS` may describe the desired behavior (e.g. `faster navigation`, `obstacle avoidance`, `smoother paths`).

## How reward functions work here

Contract lives in `src/tfm_control_ucm/core/rewards.py`. The environment (via
`RewardedStepMixin._finish_step` in `grid_env/environment.py`) decides the end
of the episode **before** calling the reward, then calls
`reward_fn(ctx: StepContext)`. The reward only scores the step; it never
returns `terminated`/`truncated`, so the success rate stays comparable across
rewards.

- Goal reached (`dist < 1` cell) → `terminated`.
- Collision (robot could not move) or step limit → `truncated`.

`StepContext` fields (no need to parse `obs` indices):

```python
ctx.action, ctx.previous_action        # int (-1 on the first step)
ctx.obs, ctx.previous_obs              # raw observations
ctx.dist_to_goal, ctx.previous_dist_to_goal
ctx.min_obstacle_dist                  # nearest lidar reading (lidar_max_range if none)
ctx.hit_obstacle, ctx.reached_goal, ctx.max_steps_reached
ctx.steps, ctx.max_steps
ctx.robot_pos, ctx.goal_pos            # (col, row)
ctx.map_diagonal, ctx.lidar_max_range  # for normalisation
```

Write a reward as a `RewardFunction` subclass (a `@dataclass` gets
`get_config()` for free, so its parameters are logged with the run) returning
`RewardResult(total, components)`. Plain callables also work and may return a
float or `(total, components)`. The env puts `components` in
`info['reward_components']`, which the PPO trainers log as
`Info/Reward/Components/*`.

`ProgressReward` (in `rewards.py`) is the current reward: progress shaping
(optionally clipped at 0), step cost, linear proximity penalty, +goal, −collision.
Tests for rewards are in `tests/test_rewards.py`.

## Design principles for this environment

1. **Dense distance shaping**: `reward += α * (prev_dist - curr_dist)` rewards moving closer. Use normalized distance (`/map_diagonal`) to keep rewards in a stable range (~[-1, 1]).

2. **Goal bonus**: large positive reward (e.g. +10) when `ctx.reached_goal`. The threshold is the env's `_goal_tolerance` (1 cell), not the reward's.

3. **Wall/timeout penalty**: negative reward on `ctx.hit_obstacle` and on `ctx.max_steps_reached`. Keep it proportional to the goal bonus.

4. **Step penalty**: small negative per step (e.g. -0.01) encourages efficiency. Too large → robot learns to end episodes immediately.

5. **Oscillation penalty**: penalize repeating `action` ↔ `prev_action` oscillation (e.g. turn-left then turn-right), which causes spinning.

6. **LIDAR-based obstacle penalty**: `ctx.min_obstacle_dist` = distance to the nearest obstacle. Penalize when this is below a safety threshold.

## Template

```python
from dataclasses import dataclass
from tfm_control_ucm.core.rewards import RewardFunction, RewardResult, StepContext

@dataclass
class OscillationAwareReward(RewardFunction):
    progress_scale: float = 2.0
    goal_reward: float = 10.0
    collision_penalty: float = 1.0
    timeout_penalty: float = 0.5
    safety_margin: float = 2.0
    oscillation_penalty: float = 0.1

    def __call__(self, ctx: StepContext) -> RewardResult:
        c = {
            "progress": self.progress_scale * (ctx.previous_dist_to_goal - ctx.dist_to_goal) / ctx.map_diagonal,
            "goal": self.goal_reward if ctx.reached_goal else 0.0,
            "collision": -self.collision_penalty if ctx.hit_obstacle else 0.0,
            "timeout": -self.timeout_penalty if ctx.max_steps_reached else 0.0,
            "proximity": -0.5 * max(0.0, 1.0 - ctx.min_obstacle_dist / self.safety_margin),
            # turn-left <-> turn-right (GridRobot actions 1 and 2)
            "oscillation": -self.oscillation_penalty
                           if {ctx.previous_action, ctx.action} == {1, 2} else 0.0,
        }
        return RewardResult(sum(c.values()), c)
```

Use it in a **new** numbered train script (old scripts are kept as history):
`env = Grid_Robot_Sections_Env_MultiMap(..., reward_fn=OscillationAwareReward(...))`.
Add a test with a hand-built `StepContext` to `tests/test_rewards.py` (see `make_ctx` there).

## Literature-grounded reward terms

From the papers in the project's reference collection:

**Gao et al. 2025** (HRL mapless navigation, ICRA) uses:
```
r = r_arrival + r_step + r_dist + r_out
r_arrival = 100  if dist < 0.5 m, else 0
r_step    = -1   (per sub-goal step, encourages efficiency)
r_dist    = μ * (d_{A*,t-1} − d_{A*,t})   (A* distance progress)
r_out     = -20  if agent selects an infeasible sub-goal (blocked by obstacle)
```
This is a dense shaping + sparse terminal design. In a discrete grid, `d_{A*}` can be replaced with BFS distance (already computed by `PathPlanner` in `TrainingManager`).

**Xue et al. 2022** (NavACL-Q, Appl. Sciences) notes: sparse terminal-only reward makes DRL converge very slowly when goal-reaching is rare. Their fix is curriculum learning (NavACL-Q) rather than reward shaping. In this project, dense distance shaping is the equivalent approach.

**Thesis constraint** (from `2_estado_del_arte_propuesta.tex`): reward terms and their magnitudes must be documented explicitly. A proximity penalty does not constitute a formal safety guarantee.

## Steps for this command

1. Read the reward used in the most recent training script and its class in `core/rewards.py`.
2. Understand the symptom the user wants to fix (from `$ARGUMENTS` or by asking).
3. Propose a concrete change to the reward function with explanation of the RL reasoning.
4. Apply the change (a new `RewardFunction` or new parameters, in a new train script), add a test, and run `pytest`.
5. Suggest which `TrainingManager` metric to watch to confirm improvement (`Steps/Ratio`, `Success/Episode`, `Distances/Change`).
