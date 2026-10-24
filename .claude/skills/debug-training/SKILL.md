---
name: debug-training
description: Systematically diagnose why an RL training run is not learning (flat reward, exploding loss, no success). Use when a run fails to improve or behaves strangely.
argument-hint: "[test number] [symptom]"
---

# Debug RL Training

Systematically diagnose why a training run is not learning.

`$ARGUMENTS` may specify a test number or a symptom (e.g. `14 reward not improving`).

## Useful scripts

```bash
python scripts/inspect_checkpoint.py checkpoints/Test/{N}/
python scripts/plot_runs.py {N} --smooth 100
python scripts/compare_runs.py {N-1} {N} --tail 0.2
```

### Quick inline sanity check (paste into a Python session)

```python
import torch, sys
sys.path.insert(0, "src")
from tfm_control_ucm.core.grid_env.environment import Grid_Robot_Sections_Env
from tfm_control_ucm.core.grid_env.map import GridMap
from tfm_control_ucm.core.grid_env.robot import GridRobot

# 1. Check observation shape
mapa  = GridMap.load("./src/tfm_control_ucm/maps/map_obstacle_free.json")
robot = GridRobot()
env   = Grid_Robot_Sections_Env(map=mapa, robot=robot, cell_size=10)
obs, _ = env.reset()
print("obs shape:", obs.shape)          # expect (obs_dim,)

# 2. Step through random actions and inspect reward signals
import numpy as np
for _ in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  action={action}  reward={reward:+.3f}  term={terminated}  trunc={truncated}")

# 3. Verify obs_dim formula: num_rays * 2 + 2
# (each LIDAR ray = [distance, angle], plus [dist_to_goal, angle_to_goal])
print("expected obs_dim:", env.get_observation_dim())
```

### Checkpoint quick-read (no full model needed)

```python
import torch
ckpt = torch.load("checkpoints/Test/24/GridAgent/final.pth",
                  map_location="cpu", weights_only=False)
print("keys:", list(ckpt.keys()))
print("epsilon:", ckpt.get("epsilon"))
print("global_step:", ckpt.get("global_step"))
print("config:", ckpt.get("config"))
```

## Background from project literature

This project compares **DQN** (off-policy, discrete actions, experience replay) and **PPO** (on-policy, actor-critic, clipped surrogate). They differ in more than network architecture:
- DQN can reuse stored transitions; PPO discards data after a bounded number of updates (Shakya et al., 2023, *RL Algorithms: A Brief Survey*).
- A fair comparison must report the same interaction budget — total environment steps, not just episodes.
- The navigation task here is partially observable: sectored LIDAR + relative goal angle/distance is a compressed view, not a full state (from thesis §4, *Observación local y observabilidad parcial*).

Key insight from Xue et al. 2022 (*DRL + Curriculum for Mapless Navigation*): sparse rewards (only terminal success) are a primary cause of slow convergence in intralogistics navigation. Their NavACL-Q uses curriculum to surface positive samples. In this project the equivalent mitigation is reward shaping (see `/reward-design`).

## Diagnostic checklist

Work through these in order, stopping when you find the root cause.

### 1. Reward signal
- Read the `reward_fn` passed to the environment constructor. Is the goal-reach reward large enough relative to step penalties?
- Is `terminated` (goal reached) vs `truncated` (wall hit / timeout) correct? `TrainingManager` logs `Success/Episode` = `int(terminated)` only.
- Check `Distances/Change` in TensorBoard — if it averages near 0 the robot is stuck, not moving toward the goal.

### 2. Exploration
- Read `eps_decay` in the config. With exponential decay `ε = ε_end + (ε_start − ε_end) * exp(−step / eps_decay)`, ε reaches ~0.37 * (ε_start − ε_end) after `eps_decay` steps.
- Expected decay schedule: for `num_episodes=50000` and `eps_step=10`, total steps ≈ `50000 * avg_ep_len * 10`. `eps_decay=200_000` is reasonable for ~500-step episodes.
- If `Reward/Total` is random-walk-flat after 10k episodes, epsilon is likely still high — lower `eps_decay`.

### 3. Network and optimizer
- Print `agent.policy_net` to verify layer sizes match `obs_dim`. `obs_dim = num_rays * 2 + 2` (each LIDAR ray is `[dist, angle]` + `[dist_to_goal, angle_to_goal]`).
- Learning rate: 1e-3 is typical for DQN; 3e-4 for PPO. Too high → unstable loss; too low → no progress after 100k steps.
- Gradient clipping is set by `max_grad_norm` (default 10.0 in `RLConfig`). If loss spikes, reduce it.

### 4. Replay buffer (DQN/GridAgent only)
- Buffer is filled before learning starts (`len(buffer) >= batch_size`). With `buffer_size=100_000` and `batch_size=64`, learning starts after 64 steps.
- If using `ReplayBuffer` (deque-based in `agents/utils.py`): sample returns `(states, actions, rewards, next_states, dones)` as numpy arrays — check dtypes, especially `dones` must be float for loss computation.
- Check that `store()` is called with `done = terminated or truncated`.

### 5. Target network (DQN)
- `target_update` (default 1000 global steps) controls how often the target network syncs. Too frequent → unstable; too infrequent → slow learning.
- Verify `double_dqn=True` is set in `RLConfig` for `GridAgent` — plain DQN often overestimates Q-values in this environment.

### 6. PPO-specific
- `gae_lambda` (0.95), `clip_eps` (0.2), `entropy_coeff` (0.01) are the critical knobs.
- `PPOAgent.train_step()` is called once per episode end (when `eps_step=0`). Ensure the buffer has at least `batch_size` transitions before that — for short episodes this may not happen.
- `RecurrentPPOAgent` resets hidden state at episode start — verify `reset_hidden()` is called in the training loop.

### 7. Environment sanity check
- Run `env.reset()` and print the observation shape. Confirm it matches `obs_dim` in the config.
- Step through 5 random actions and print `(reward, terminated, truncated)` for each. Check that the reward function returns sensible values.
- If using multiple environments, `BaseTrainer.add_environments()` asserts all envs share the same obs/action dims.

## Common fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| `Steps/Ratio` never drops below 5 | Sparse reward, no intermediate shaping | Add small `+reward` for reducing distance, large `+reward` for goal |
| Loss NaN after first update | LR too high or reward not normalized | Lower LR; clip rewards to [-1, 1] |
| Agent always turns in circles | Action logits collapse to one action | Increase `entropy_coeff` (PPO); check Q-value diversity with `get_q_value_stats` |
| `Success/Episode` = 0 throughout | `terminated` never True | Verify goal-reach condition in `reward_fn`; check goal placement isn't overlapping obstacles |
| Recurrent agent diverges | Hidden state carried across episodes | Call `agent.reset_hidden()` in the episode reset block |
