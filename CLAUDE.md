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
- `Robot` (abstract) / `GridRobot` / `GridRobotNotTurning` — manages position and discrete movement on the map; `heading` (0 for robots without orientation) and `random_reset(map, rng)`.
- `GridLidar` — GPU-accelerated raycasting sensor that returns `(distance, angle)` pairs, angles relative to the robot heading. Supports a ray cache for performance.

`BaseGridEnv` composes these into a Gymnasium env and holds everything shared (map selection, seeded start/goal sampling, end-of-episode rules, rendering; `reset()` info gives `map`, `optimal_steps`, `dist2goal`). Subclasses only define the observation: `Grid_Robot_Env` (top-N LIDAR readings by distance), `Grid_Robot_Sections_Env` (nearest reading per angular section) and `Grid_Robot_Sections_Env_MultiMap` (sections over several maps), each followed by `[dist_to_goal, angle_to_goal]`. All angles are egocentric (0 = where the robot faces, counter-clockwise). The **reward function is passed in at construction time** — there is no hard-coded reward. The contract is in `core/rewards.py`: the env decides the end of the episode (goal → `terminated`; collision or step limit → `truncated`), builds a `StepContext` and calls a `RewardFunction` (e.g. `ProgressReward`), which only scores the step.

**Agents** (`agents/`): Three-level hierarchy:
1. `BaseRLAgent` (abstract) — handles device placement, optimizer init, replay buffer, gradient clipping, save/load, and the episode hooks the trainer uses (`on_episode_start`, `act`, `observe`, `on_episode_end`, `episode_metrics`, `train_on_step`). Defaults are off-policy (replay buffer, train every `eps_step`); the PPO agents keep the episode and train in `on_episode_end`.
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
- `BaseTrainer` — holds the **single training loop** for every algorithm (`_train_agent`/`_run_episode`), talking to agents only through their hooks and to envs only through `reset`/`step` info. Same TensorBoard tags for DQN and PPO (`Reward/Total`, `Success/Episode`, `Success/Map/<map>`, `Steps/Ratio/Episode`, `Loss/Episode`, …; list at the top of `TrainingManager.py`) plus `Config/Agent` and `Config/Reward` text.
- `SingleTrainer` (one agent, one env) and `MultiEnvironmentTrainer` (one agent, several envs; blocks per env or alternating) only decide which env each episode uses. `PPOTrainer`, `PPOGRUTrainer`, `RecurrentPPOTrainer` are deprecated aliases of `SingleTrainer`.
- Training writes checkpoints to `checkpoints/Test/XX/` and TensorBoard logs to `runs/Test/XX/`.

**Experiment Manager** (`Manager.py`): `RLManager` + `EnvironmentConfig` + `AgentConfig` — a higher-level orchestration layer for running multiple agent×environment combinations. Separate from `TrainingManager`; used when comparing agents or pairing strategies.

### Configuration (`config/`)
- `training_config.yaml` — episode count, gamma, epsilon schedule, replay buffer params.
- `pytorch_config.yaml` — seed, device, network architecture defaults.

These YAML files are not wired into the current training scripts automatically; scripts configure agents inline via dataclasses.

### Maps (`maps/` and `src/tfm_control_ucm/maps/`)
JSON files define the grid dimensions and obstacle list. Load with `GridMap.load("path/to/map.json")`. `./maps/` (used by current experiments) has `map_obstacle_free.json`, `map_simple.json`, `map_1_obstacle.json` and `map_4_square_obstacle.json`; `src/tfm_control_ucm/maps/` only has the first two.

## Key Patterns

**Constructing and running an experiment** (see `train/grid_training_25.py` as the canonical pattern):
```python
mapas = [GridMap.load(p) for p in ["./maps/map_obstacle_free.json", "./maps/map_simple.json"]]
env = Grid_Robot_Sections_Env_MultiMap(maps=mapas, robot=GridRobotNotTurning(),
                                       reward_fn=ProgressReward(...), num_sections=8, lidar_config=...)

config = PPOConfig(obs_dim=env.get_observation_dim(), action_dim=env.get_action_dim(),
                   algorithm=RLAlgorithm.PPO, device=device, ...)
agent = PPOAgent(config, policy_network=..., value_network=...)

trainer = SingleTrainer("./runs/Test/XX", "./checkpoints/Test/XX", TrainerConfig(...))
trainer.add_agents(agent).add_environments(env)
trainer.train()
```

**Training numbering convention**: each `train/grid_training_XX.py` and its corresponding `runs/Test/XX/` and `checkpoints/Test/XX/` form one experiment. The current branch is test 24.

**Replay buffer**: `BaseRLAgent` uses a simple circular `ReplayBuffer` (deque-based, in `agents/utils.py`). The `GridAgent_2_FIXED` legacy class uses TorchRL's `LazyTensorStorage`-backed `ReplayBuffer` — the newer `GridAgent` uses the simpler one.

**Device selection** (standard pattern across all training scripts):
```python
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
```
