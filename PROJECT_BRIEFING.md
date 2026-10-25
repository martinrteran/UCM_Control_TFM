# Project Briefing — read this first

> **Purpose**: let a fresh agent get productive in this repo in one read, without
> re-exploring the whole codebase. This file tracks *dynamic* state (what's in
> flight, what's broken, what's next). Static architecture docs live in
> `CLAUDE.md` — read that too, but this file is the "what's actually going on
> right now" layer on top of it.
>
> **Keep this current**: after any non-trivial session (new experiment, a
> refactor, a bug fix), run `/project-briefing update` (see
> `.claude/skills/project-briefing/SKILL.md`) or edit this file by hand. Bump the
> "Last updated" line below.

**Last updated**: 2026-10-25 (by Claude, decoupling plan phases 0–4) · **Branch**: `Put-all-together` (pushed) · **Latest commit**: Phase 4 commit after `56e863d` "Phase 3 - BaseGridEnv and egocentric observation"

**Decoupling plan** — evaluation + 6-phase plan (with diagrams) in the Claude doc
"Evaluación del desacoplamiento" (https://claude.ai/artifact/XgaBeVnuccLX64zzihFyJS).
Phases 0 (tests), 1 (bug fixes), 2 (reward contract + `grid_training_25.py`),
3 (`BaseGridEnv`, egocentric observation) and 4 (one training loop with agent
hooks) are done and pushed. Left: phase 5 (delete commented-out code, legacy
classes and copy scripts). **Next experiment: train test 25.**

---

## 1. What this project is

Master's thesis (TFM): a robot navigates a 2D occupancy-grid map toward a goal
using simulated LIDAR, trained with DQN/PPO/A2C/SAC and a recurrent (GRU) PPO
variant. Thesis scope is narrow on purpose: **DQN vs PPO, static obstacles,
discrete actions, single robot** — no multi-agent, no dynamic obstacles, no
physical robot, no formal safety guarantees (see
`2_estado_del_arte_propuesta.tex`, referenced from `CLAUDE.md`).

Full architecture tour (classes, configs, commands) is in `CLAUDE.md` at repo
root — this file assumes you've skimmed that.

---

## 2. Current state — READ THIS BEFORE RUNNING ANYTHING

Code is committed and pushed. Still uncommitted on purpose (waiting for a user
decision): 13 old TensorBoard logs under `runs/Test/09`–`20` show as modified
only because `.gitattributes` now routes `*.tfevents.*` through Git LFS (content
unchanged, ~140 MB of LFS if committed), and the untracked `runs/Test/10_v2`
(one 239 MB event file) and `checkpoints/Test/10_v2`.

**Convention (user decision, 2026-10-24): old `train/grid_training_NN.py`
scripts are never fixed in place.** They are a record of how the code evolved.
When the API changes, write a new numbered script instead.

**Results before test 25 are not directly comparable with test 25+**: the PPO
critic loss was wrong (§2g), the goal angle was mirrored (§2g), the episode-end
rules moved into the env, and every algorithm now logs the same tags.

### 2a. Reward contract: `reward_fn(ctx: StepContext)`, env decides episode end

`src/tfm_control_ucm/core/rewards.py` defines the contract:
- `StepContext` (frozen dataclass): action, previous_action, obs, previous_obs,
  `dist_to_goal`, `previous_dist_to_goal`, `min_obstacle_dist`, `hit_obstacle`,
  `reached_goal`, `max_steps_reached`, steps, max_steps, robot_pos, goal_pos,
  map_diagonal, lidar_max_range. Rewards never parse `obs` indices.
- `RewardFunction` (ABC): `__call__(ctx) -> RewardResult(reward, components)`;
  dataclass subclasses get `get_config()` (name + params) for experiment logs.
  Plain callables also accepted (may return a float or `(reward, components)`).
- `ProgressReward`: progress `progress_scale*(prev−dist)/diag` (optionally
  clipped at 0), `−step_penalty`, linear proximity penalty below
  `safety_margin`, `+goal_reward`, `−collision_penalty`. Defaults = the
  `grid_training_10 copy.py` reward; `clip_negative_progress=True` = the
  pre-refactor MultiMap reward.

`BaseGridEnv._finish_step` does the end of every `step()`: **the env decides**
`terminated` (goal within `_goal_tolerance = 1` cell) and `truncated`
(collision or step limit), builds the `StepContext`, calls the reward, and fills
`info` with `reached_goal`, `hit_obstacle`, `max_steps_reached`, `dist2goal`,
`min_dist`, `map`, `reward`, `reward_components`, `componente_reward_{min,max,diff}`.
`reset()` info has `map`, `optimal_steps` (wavefront BFS, turns not counted) and
`dist2goal`.

Open design questions (not changed): collisions are `truncated`, and
`PPOAgent` passes only `terminated` as `dones` to GAE, so a collision
bootstraps from the value of the state where the robot stopped. Also,
`PPOAgent` samples random 64-transition batches from a 100k replay buffer of
past episodes (stale log-probs and advantages) instead of updating on the
latest rollout — closer to off-policy than textbook PPO.

### 2b. Train/eval scripts

- `train/grid_training_25.py` — **the working script**. Same setup as 24 (PPO,
  4 maps, random map per episode) with `ProgressReward` built from CLI args,
  `--seed`, `--render/--no-render`, `SingleTrainer`, and writes the full config
  (args, reward params, maps, lidar) to `runs/Test/<N>/config.json`.
  Smoke-tested for 8 episodes; not trained yet.
- `train/grid_training_22.py`–`_24.py` and the `* copy.py` scripts use the old
  11-positional-arg reward or no reward at all and fail at env construction.
  **Kept broken on purpose** (historical record, see convention above).
- `scripts/eval_agent.py` — rebuilds the env from `runs/Test/<N>/config.json`
  or from the checkpoint dims (action_dim → robot, obs_dim → sections), rebuilds
  any MLP policy from the state_dict shapes, uses the agent hooks, and reports
  success/collision/timeout overall and per map. Runs without `config.json`
  are evaluated with `_legacy_goal_angle` (test 22 on `map_4_square_obstacle`:
  80 % success with it, 0 % without). Recurrent agents not supported.
- `scripts/plot_runs.py` / `compare_runs.py` read the unified tags and fall back
  to the old PPO tags (`Done/Success`, `Steps/Ratio`, `Done/Reward/Episode`…).
  `matplotlib` is not installed in `.venv`. `compare_runs.py 22` takes more
  than 5 min because it reloads the full event file once per metric.
- `TrainerConfig` asserts `num_episodes > 1000`; for a quick smoke test call
  `trainer.train(num_episodes=3)` directly.

### 2c. Env knobs that used to be attributes

Scripts 22–24 set `env._max_proximity_penalty`, `env._safety_margin`,
`env._step_penalty`. Nothing reads them any more; they are `ProgressReward`
parameters now (`max_proximity_penalty`, `safety_margin`, `step_penalty`).

### 2d. Training: one loop, agent hooks

`BaseTrainer._train_agent` / `_run_episode` is the only training loop. It talks
to agents through `BaseRLAgent` hooks — `on_episode_start`, `act(state, training)`,
`observe(s, a, r, s2, terminated, truncated)`, `on_episode_end(last_state)`,
`episode_metrics()`, class attribute `train_on_step` — and to envs only through
`reset`/`step` info (no more `env.map`, `env.robot` or `agent.value_net`).

| Class | Role |
|---|---|
| `SingleTrainer` | 1 agent, 1 env |
| `MultiEnvironmentTrainer` | 1 agent, several envs: blocks per env (`onePerEnv`, `cyclesPerEnv`) or alternating every episode |
| `PPOTrainer`, `PPOGRUTrainer`, `RecurrentPPOTrainer` | deprecated aliases of `SingleTrainer` (old scripts still import them) |

Default hooks = off-policy (replay buffer, `train_step` every `eps_step` steps,
or at episode end if `eps_step=0`). `PPOAgent`, `PPOGRUAgent` and
`RecurrentPPOAgent` override them with exactly the data flow their old trainers
had (values and log-probs recorded in `act`, GAE or sequence building and the
update in `on_episode_end`). Tags are the same for every algorithm (list at the
top of `TrainingManager.py`), plus `Config/Agent` and `Config/Reward` text.
`MultiAgentTrainer` and `MultiTrainer` were dropped (they were commented out).

### 2e. `Manager.py` (`RLManager`/`EnvironmentConfig`/`AgentConfig`) looks unused

It's a generic, env-agnostic experiment runner with its own `__main__` demo
using stub classes (`StubEnv`, `QLearningAgent`). Nothing in `train/` imports
it. Treat it as exploratory/abandoned scaffolding unless you find evidence
otherwise.

### 2f. Maps: two locations

`./maps/` (used by tests 22+) has 4 maps: `map_obstacle_free.json`,
`map_simple.json`, `map_1_obstacle.json`, `map_4_square_obstacle.json`;
`src/tfm_control_ucm/maps/` only has the first two. `map_simple` has the most
obstacles (14, overlapping: 121 occupied cells).

### 2g. Tests and bug fixes (decoupling phases 0–4)

`pytest` (from the repo root, ~10 s) runs 84 tests in `tests/`
(`test_map`, `test_robot`, `test_environment`, `test_rewards`, `test_heading`,
`test_training`, `test_scripts`; 1 skipped without `matplotlib`). They build tiny
maps and inject their own reward, so they don't depend on any train script.

Bugs fixed (each reproduced by a test first):
- **PPO critic loss**: `ValueNetwork` returns `(B, 1)` and returns are `(B,)`, so
  `returns - values` was a `(B, B)` matrix — every return was compared with every
  value and the critic learned the batch mean. Affects all PPO runs (tests 14–24).
- **Mirrored goal angle**: `atan2(row_diff, col_diff)` (goal north = −π/2) while
  lidar angles use north = +π/2. Now one egocentric convention for both;
  `BaseGridEnv._legacy_goal_angle` reproduces the old one for old checkpoints.
- MultiMap map selection (last map almost never picked, inverted sequential
  mode — from the uncommitted refactor, never used by a trained run) and diagonal.
- `_previous_obs` aliased the returned observation.
- `Grid_Robot_Env` crashed on `reset()`; both robots crashed on numpy actions;
  `GridRobot` validation; the `num_sections` check used `and`.
- `GridMap`: border off-by-one, overlapping-obstacle removal, list aliasing, `str()`.
- `PPOAgent` crashed the trainer if the first episode ended before a full batch.
- `DQNAgent`: the target net ignored a custom `q_network` (state_dict mismatch) and
  it read `config.use_double_dqn` instead of `double_dqn` (crashed on the first
  update). No script used `DQNAgent` (they use `GridAgent`).

Still open: `GridAgent_2_FIXED`, the commented-out `Grid_Robot_Env_Image_lidar`
and other commented-out code (phase 5).

---

## 3. Experiment history (`train/grid_training_NN.py` ↔ `runs/Test/NN` ↔ `checkpoints/Test/NN`)

Numbering convention: each `train/grid_training_NN.py` is one experiment;
`runs/Test/NN/` (TensorBoard) and `checkpoints/Test/NN/` (`.pth`) are its
outputs. Read `.claude/skills/analyze-run/SKILL.md` for how to pull metrics out of
a given run. Summary from commit messages + script contents:

| Test | Algorithm / Agent | Map | Commit note | Outcome |
|---|---|---|---|---|
| 02–11 | `GridAgent` (DQN), `Simple Q Network` | obstacle_free / simple | early prototyping | superseded, mostly historical |
| 12–13 | DQN variants | simple | "Simulation 12", "Training 13" | prototyping |
| 14–18 | PPO (`PPOAgent` + `SimpleQNetwork`/`ValueNetwork`) | simple / obstacles | "Test 17 - Suicide Agent", "Test 18 - SemiSuccessful" | mixed; reward shaping issues (agent ending episodes early = "suicide") |
| 19 | PPO, 1-obstacle map, proximity-penalty + safety-margin + 8-section LIDAR sweep | `map_1_obstacle` | "Test-19-1 obstacle, working" | working baseline |
| 20 | PPO + timing instrumentation | — | — | — |
| 21 (`_GRU`) | `PPOGRUAgent` (non-recurrent-buffer GRU) | `map_4_square_obstacle` | "Test 21 - GRU Unsuccessful" | **failed** |
| 21 (`_GRU_RNN`) | `RecurrentPPOAgent` + `RecurrentPPOBuffer` (sequence-chunked GAE) | `map_4_square_obstacle` | "Test 21 - GRU RNN - Successfully" | **worked** — the sequence-buffer variant succeeded where the non-sequence GRU agent didn't |
| 22 | PPO, `Grid_Robot_Sections_Env`, `n_sections=8` | `map_4_square_obstacle` | "Test 22 - no improvement" | plateaued |
| 23 | PPO, same env/config as 22 | `map_4_square_obstacle` | "Test 23 - Before Training" | **committed before training ran** — likely never executed to completion (no `runs/Test/23` on disk) |
| 24 | PPO, `Grid_Robot_Sections_Env_MultiMap` (4 maps, random per episode), `n`/CLI-args via `argparse` | all 4 maps | "Teest 24 - Sim 23 - Change Name Maps" | **never ran**; kept as history (no `reward_fn`) |
| 25 | Same as 24 on the decoupled API: `ProgressReward(clip_negative_progress=True)`, env decides episode end, egocentric angles, fixed PPO critic, unified trainer, `config.json` per run | all 4 maps | "Test 25 - Decoupled reward, bug fixes and unit tests" (+ phases 3–4) | **ready, not trained** (8-episode smoke test only) |

Takeaway: the sequence-chunked recurrent buffer (test 21 GRU_RNN) is the only
GRU-based success so far; the project was moving toward multi-map
generalization (test 24) when the reward-injection refactor landed and froze
further training.

---

## 4. Architecture quick-reference (file:line)

Only the parts not fully obvious from `CLAUDE.md`, or useful jump points:

**Environments** — `src/tfm_control_ucm/core/grid_env/environment.py`
- `BaseGridEnv` (line 41) — everything shared: validation, map selection (`_select_map`, seeded), start/goal sampling, `reset`/`step`, `_finish_step` (episode end + reward), egocentric `_angle_to_goal`, `_optimal_steps`, rendering. Subclasses implement `_build_observation_space`, `_get_observation`, `_render_frame`.
- `Grid_Robot_Env` (line 305) — `_min_num_rays` farthest rays, 2D observation `(N+1, 2)`.
- `Grid_Robot_Sections_Env` (line 351) — nearest reading per angular section, flat observation; what every PPO experiment since test 19 uses.
- `Grid_Robot_Sections_Env_MultiMap` (line 430) — Sections over several maps (`random_map=True` uniform, `False` round-robin); drops the renderer when the map changes.
- Commented-out `Grid_Robot_Env_Image_lidar` (line 450+) is dead code (phase 5).

**Robot / Map / Sensor** — `grid_env/{robot,map,sensor}.py`
- `Robot` (ABC): `NUM_ACTIONS` enforced via `__init_subclass__`; `heading` (radians, 0 = east, counter-clockwise; 0 for robots without orientation); `random_reset(map, rng)`.
- `GridRobot` — egocentric actions 0=forward, 1=turn left, 2=turn right. Starts with a random orientation; the lidar and the goal angle turn with it. Usable since phase 3 but not trained yet.
- `GridRobotNotTurning` — absolute N/S/E/W moves (`NUM_ACTIONS=4`), heading 0. **What all PPO train scripts (19–25) use.**
- `GridLidar.scan(map, pos, heading)` returns `[dist, angle relative to heading]`. The cache (`with_cache=True`) is reused for any heading when `fov = 2π`; otherwise rotated rays are cast on the fly.

**Agents** — `src/tfm_control_ucm/core/agents/{agent.py,base_rl_agent.py,utils.py,networks.py}`
- All concrete agents live in **`agent.py`**: `GridAgent_2_FIXED` (legacy, non-`BaseRLAgent`), `GridAgent` (line 268), `DQNAgent` (631), `A2CAgent` (743), `PPOAgent` (850), `SACDiscreteAgent` (1028), `PPOGRUAgent` (1145), `RecurrentPPOAgent` (1314).
- Episode hooks: defaults in `base_rl_agent.py` (around line 176); overridden by the three PPO agents.
- `AgentFactory.create_agent()` only dispatches `DQN/DDQN`, `A2C`, `GRID_AGENT` — PPO, SAC and GRU agents are built directly in train scripts.
- `PPOGRUAgent` vs `RecurrentPPOAgent`: both PPO+GRU; `RecurrentPPOAgent` chunks each episode into `seq_len` sequences (`RecurrentPPOBuffer`) — the one that worked in test 21.
- Config dataclasses (utils.py): `RLConfig` (base) → `DQNConfig`, `SACConfig`, `PPOConfig` → `PPORNNConfig`. `DQNAgent` needs a `DQNConfig` (it reads `eps_start`, `double_dqn`…); `GridAgent` reads them with `getattr` defaults.

**Training** — `src/tfm_control_ucm/core/training/TrainingManager.py` (see §2d)
- `BaseTrainer._run_episode` (line 142) and `_log_episode` (line 207).
- `InterruptHandler` (training/utils.py) — Ctrl+C saves an emergency checkpoint and exits cleanly; the Ctrl+\ "soft reset" is POSIX-only (no-op on Windows).

**Utils**
- `utils/path_planner.py`: pure-Python `PathPlanner` (used by the env for `optimal_steps`) and a `TorchPathPlanner` GPU variant.

---

## 5. Observation / action cheat sheet (current PPO envs, test 19+)

- Robot: `GridRobotNotTurning`, 4 absolute actions (N/S/E/W), heading 0.
- Observation (`Grid_Robot_Sections_Env[_MultiMap]`): `num_sections` angular LIDAR buckets (each `[nearest_dist, angle]`; empty sections get `[max_range, section_center_angle]`) followed by `[dist_to_goal, angle_to_goal]`. `obs_dim = num_sections*2 + 2`. All angles egocentric, counter-clockwise, 0 = robot heading (east for `GridRobotNotTurning`).
- `num_sections = 4 * n` (CLI `--n`, default 2 → 8 sections), must be ≥ 4 and a multiple of 4. `lidar num_rays = n_sections*15` so each section has several candidate rays.
- Reward: `ProgressReward` from `core/rewards.py`, injected by the train script (§2a). Episode end decided by the env: goal within 1 cell → `terminated`; collision or step limit → `truncated`.

---

## 6. Config files are decorative

`config/training_config.yaml` and `config/pytorch_config.yaml` exist but
**nothing in `train/` loads them** — every script configures agents inline
via dataclasses with its own CLI args (`argparse`, only in `grid_training_24.py`
so far) or hard-coded literals (22, 23, GRU variants). Don't trust the YAML
values as reflecting current hyperparameters.

---

## 7. Where to look for more

- `CLAUDE.md` — static architecture, package layout, key patterns, setup commands.
- `.claude/skills/*/SKILL.md` — `analyze-run`, `debug-training`, `new-experiment`, `new-network`, `reward-design`, `tune`, `rl-theory`, `docstring`, `project-briefing` — task-specific runbooks. `reward-design` documents the `StepContext`/`RewardFunction` contract and has a template.
- Papers under `C:/Users/marti/Documents/Studies/Masters/Control UCM/TFM/Articulos/` (see `CLAUDE.md` table) — cited directly inside `debug-training.md`, `reward-design.md`, `new-network.md` for design rationale (e.g. local-minima failure mode from Gao et al. 2025, partial-observability motivation for GRU from Murphy 2025 §1.3.4).
