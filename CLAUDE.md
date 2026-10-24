# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Master's thesis (TFM) project for robot navigation using reinforcement learning. A robot navigates a 2D grid map toward a goal using LIDAR observations, trained with algorithms including DQN, PPO, SAC, and recurrent PPO (GRU).

**Before doing anything else, read `PROJECT_BRIEFING.md` at the repo root.**
It tracks current/dynamic state (in-flight refactors, what's currently
broken, experiment history, next steps) that this file does not — this file
only covers static architecture. After a session that changes code, finishes
an experiment, or uncovers a project gotcha, refresh it with
`/project-briefing update` (see `.claude/skills/project-briefing/SKILL.md`).

## Reference documents

Key papers are stored in `C:/Users/marti/Documents/Studies/Masters/Control UCM/TFM/`:

| File | What it covers |
|---|---|
| `Articulos/RL_Algorithms_Brief.pdf` | Survey of DQN, PPO, SAC, A2C — use for algorithm comparison rationale |
| `Articulos/RL_Overview.pdf` | Kevin Murphy (2025), 200pp — value-based, policy-based, model-based RL; partial observability (§1.3.4) |
| `Articulos/DQN/applsci-12-03153-v2.pdf` | Xue et al. 2022 — DRL + curriculum learning for mapless AGV navigation with LiDAR (most directly related work) |
| `Articulos/PPO/2503.12036v1_IMPORTNAT.pdf` | Gao et al. 2025 (ICRA) — Hierarchical RL for safe mapless navigation; reward design, congestion estimation, local minima |
| `Articulos/RNN_Review.pdf` | Mienye et al. 2024 — GRU/LSTM for sequential data; motivation for recurrent PPO agents |
| `Articulos/Propuestas_revision/2_estado_del_arte_propuesta.tex` | Thesis state-of-the-art chapter (Spanish) — defines scope, algorithms compared, and what claims can/cannot be made |

The thesis scope (from `.tex`): DQN vs PPO on 2D grid, static obstacles, discrete actions. Does **not** cover multi-agent, dynamic obstacles, physical robots, or formal safety guarantees.

## Setup & Commands

```bash
# Install package in editable mode
pip install -e .

# Run a training script (from the repo root)
python train/grid_training_24.py

# Monitor training with TensorBoard
tensorboard --logdir runs/

# Sync offline W&B runs
wandb sync --sync-all
```

```bash
# Run the unit tests (map, robots, environments, rewards; ~3 s)
pytest
```

The `train/grid_training_XX.py` scripts are the runnable entry points, each numbered script representing an experimental iteration. Old scripts are kept as a history of the code's evolution and are not updated when the API changes; new experiments get a new number.

```bash
# Evaluate a saved checkpoint
python scripts/eval_agent.py checkpoints/Test/24/GridAgent/final.pth --episodes 30

# Plot TensorBoard metrics for specific runs
python scripts/plot_runs.py 21 24 --smooth 100 --save

# Compare final-phase stats across runs (generates markdown table)
python scripts/compare_runs.py 21 22 24 --tail 0.2

# Inspect checkpoint metadata without loading the full model
python scripts/inspect_checkpoint.py checkpoints/Test/24/
```

## Architecture

### Core Layer (`src/tfm_control_ucm/core/`)

**Environment** (`grid_env/`): A custom Gymnasium environment built from three composable pieces:
- `GridMap` — 2D binary occupancy grid loaded from JSON (see `maps/`). Obstacles are `GridObstacle` rectangles.
- `Robot` (abstract) / `GridRobot` / `GridRobotNotTurning` — manages position and discrete movement on the map.
- `GridLidar` — GPU-accelerated raycasting sensor that returns `(distance, angle)` pairs. Supports a ray cache for performance.

`Grid_Robot_Env` (and variants like `Grid_Robot_Sections_Env`) compose these into a Gymnasium env. The observation is the top-N LIDAR readings by distance concatenated with `[dist_to_goal, angle_to_goal]`. The **reward function is passed in at construction time** — there is no hard-coded reward. The contract is in `core/rewards.py`: the env decides the end of the episode (goal → `terminated`; collision or step limit → `truncated`), builds a `StepContext` and calls a `RewardFunction` (e.g. `ProgressReward`), which only scores the step.

**Agents** (`agents/`): Three-level hierarchy:
1. `BaseRLAgent` (abstract) — handles device placement, optimizer init, replay buffer, gradient clipping, save/load.
2. Algorithm-specific agents (`GridAgent`, `PPOAgent`, `PPOGRUAgent`, `RecurrentPPOAgent`, `SACDiscreteAgent`, `A2CAgent`, `DQNAgent`) that implement `select_action`, `train_step`, and `_compute_loss`.
3. `AgentFactory.create_agent(config)` — factory that dispatches on `RLAlgorithm` enum.

Agent configuration uses typed dataclasses in `agents/utils.py`: `RLConfig` (base), `PPOConfig` (adds clip_eps, gae_lambda, entropy/value coefficients), `PPORNNConfig`, `SACConfig`.

**Networks** (`agents/networks.py`):
- `GridAgentNet` — fixed-architecture MLP for DQN (obs→253→56→actions).
- `SimpleQNetwork` — configurable MLP Q-network.
- `PolicyNetwork` / `ValueNetwork` — actor/critic MLPs for policy gradient methods.
- `GRUPolicyNetwork` / `GRUValueNetwork` — recurrent networks for `RecurrentPPOAgent`.

**Training** (`training/TrainingManager.py`):
- `TrainerConfig` — controls episodes, parallelism, early stopping, render, `eps_step` (train every N steps vs end of episode).
- `BaseTrainer` — abstract base with `_inner_training` loop. Writes TensorBoard scalars (reward, loss, epsilon, steps ratio vs optimal path, success rate).
- `MultiEnvironmentTrainer` — concrete trainer; supports cycling through multiple environments. Uses `PathPlanner` (BFS wavefront) to compute the optimal step count per episode for the steps-ratio metric.
- Training writes checkpoints to `checkpoints/Test/XX/` and TensorBoard logs to `runs/Test/XX/`.

**Experiment Manager** (`Manager.py`): `RLManager` + `EnvironmentConfig` + `AgentConfig` — a higher-level orchestration layer for running multiple agent×environment combinations. Separate from `TrainingManager`; used when comparing agents or pairing strategies.

### Configuration (`config/`)
- `training_config.yaml` — episode count, gamma, epsilon schedule, replay buffer params.
- `pytorch_config.yaml` — seed, device, network architecture defaults.

These YAML files are not wired into the current training scripts automatically; scripts configure agents inline via dataclasses.

### Maps (`maps/` and `src/tfm_control_ucm/maps/`)
JSON files define the grid dimensions and obstacle list. Load with `GridMap.load("path/to/map.json")`. Two maps exist: `map_obstacle_free.json` (empty) and `map_simple.json` (with obstacles).

## Key Patterns

**Constructing and running an experiment** (see `train/grid_training_09.py` as the canonical pattern):
```python
mapa = GridMap.load("./src/tfm_control_ucm/maps/map_obstacle_free.json")
robot = GridRobot()
env = Grid_Robot_Sections_Env(map=mapa, robot=robot, cell_size=10)

config = RLConfig(obs_dim=env.get_observation_dim(), action_dim=env.get_action_dim(),
                  algorithm=RLAlgorithm.GRID_AGENT, device=device, ...)
agent = GridAgent(config, policy_net=..., target_net=...)

trainer = MultiEnvironmentTrainer("./runs/Test/XX", "./checkpoints/Test/XX", TrainerConfig(...))
trainer.add_agents(agent).add_environments(env)
trainer.train()
```

**Training numbering convention**: each `train/grid_training_XX.py` and its corresponding `runs/Test/XX/` and `checkpoints/Test/XX/` form one experiment. The current branch is test 24.

**Replay buffer**: `BaseRLAgent` uses a simple circular `ReplayBuffer` (deque-based, in `agents/utils.py`). The `GridAgent_2_FIXED` legacy class uses TorchRL's `LazyTensorStorage`-backed `ReplayBuffer` — the newer `GridAgent` uses the simpler one.

**Device selection** (standard pattern across all training scripts):
```python
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
```
