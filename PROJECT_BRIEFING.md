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

**Last updated**: 2026-10-24 (by Claude, decoupling plan phases 0–2) · **Branch**: `Put-all-together` · **Latest commit**: `c849f55` "Teest 24 - Sim 23 - Change Name Maps"

**Decoupling plan in progress** — evaluation + 6-phase plan (with diagrams) in the
Claude doc "Evaluación del desacoplamiento"
(https://claude.ai/artifact/XgaBeVnuccLX64zzihFyJS). Phases 0 (tests), 1 (bug
fixes) and 2 (reward contract + `train/grid_training_25.py`) are done, uncommitted.
Next: phase 3 (`BaseGridEnv`, real `Robot.heading`) and phase 4 (one training loop
with agent hooks), which can go in parallel. See §2a and §2g.

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

## 2. Current working-tree state — READ THIS BEFORE RUNNING ANYTHING

The repo is **mid-refactor, uncommitted**. `git status` shows these modified
but not committed:

```
M src/tfm_control_ucm/core/agents/agent.py
M src/tfm_control_ucm/core/agents/base_rl_agent.py
M src/tfm_control_ucm/core/agents/utils.py
M src/tfm_control_ucm/core/grid_env/environment.py
M src/tfm_control_ucm/core/grid_env/map.py
M src/tfm_control_ucm/core/grid_env/robot.py
M src/tfm_control_ucm/core/grid_env/sensor.py
M src/tfm_control_ucm/core/grid_env/utils.py
M src/tfm_control_ucm/core/training/TrainingManager.py
M .gitignore   requirements.txt (+pytest)
D (staged) 49 tracked *.pyc files — removed from the index only, still on disk
?? .claude/   .idea/   CLAUDE.md   PROJECT_BRIEFING.md   scripts/
?? tests/   pytest.ini   src/tfm_control_ucm/core/rewards.py   train/grid_training_25.py
?? "simulations/grid_simulation_10 copy.py"
?? "train/grid_training_10 copy.py"
```

**Convention (user decision, 2026-10-24): old `train/grid_training_NN.py`
scripts are never fixed in place.** They are a record of how the code evolved.
When the API changes, write a new numbered script instead (next: 25).

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

`RewardedStepMixin` (top of `environment.py`, inherited by all three envs)
does the end of every `step()`: **the env decides** `terminated` (goal within
`_goal_tolerance = 1` cell) and `truncated` (collision or step limit), builds the
`StepContext`, calls the reward, and fills `info` with `reached_goal`,
`hit_obstacle`, `max_steps_reached`, `dist2goal`, `min_dist`, `reward`,
`reward_components`, `componente_reward_{min,max,diff}` — the keys
`PPOTrainer` logs. It is the seed of phase 3's `BaseGridEnv`.

Open design question (not changed): collisions are `truncated`, and
`PPOTrainer` passes only `terminated` as `dones` to GAE, so a collision
bootstraps from the value of the state where the robot stopped.

### 2b. Train/eval scripts

- `train/grid_training_25.py` — **the working script**. Same setup as 24 (PPO,
  4 maps, random map per episode) with `ProgressReward` built from CLI args,
  `--seed`, `--render/--no-render`, and writes the full config (args, reward
  params, maps, lidar) to `runs/Test/25/config.json`. Smoke-tested for 5
  episodes; not trained yet.
- `train/grid_training_22.py`–`_24.py` and the `* copy.py` scripts use the old
  11-positional-arg reward or no reward at all and fail at env construction.
  **Kept broken on purpose** (historical record, see convention above).
- `scripts/eval_agent.py` still builds `Grid_Robot_Sections_Env` without
  `reward_fn` → broken; needs `reward_fn=ProgressReward(...)` (or any callable).
- `TrainerConfig` asserts `num_episodes > 1000`; for a quick smoke test call
  `trainer.train(num_episodes=3)` directly.

### 2c. Env knobs that used to be attributes

Scripts 22–24 set `env._max_proximity_penalty`, `env._safety_margin`,
`env._step_penalty`. Nothing reads them any more; they are `ProgressReward`
parameters now (`max_proximity_penalty`, `safety_margin`, `step_penalty`).

### 2d. `TrainingManager.py` — trainer classes don't match `CLAUDE.md`

`CLAUDE.md` describes a `MultiEnvironmentTrainer` as *the* concrete trainer.
That class (and `MultiAgentTrainer`, `MultiTrainer`) is **commented out** in
the current `TrainingManager.py`. The classes actually active today:

| Class | Status | Used by |
|---|---|---|
| `BaseTrainer` | abstract, has `_inner_training` (DQN-style loop) + `simulate()` | base for `SingleTrainer` |
| `SingleTrainer` | concrete, 1 agent + 1 env, calls `_inner_training` | not seen in recent train scripts |
| `PPOTrainer` | concrete, full custom PPO loop (GAE, rollout, own logging) | `grid_training_22/23/24.py` |
| `PPOGRUTrainer` | concrete, PPO loop for `PPOGRUAgent` (non-recurrent-buffer GRU) | `grid_training_21_GRU.py` |
| `RecurrentPPOTrainer` | concrete, PPO loop for `RecurrentPPOAgent` (`RecurrentPPOBuffer`, sequence-chunked) | `grid_training_21_GRU_RNN.py` |
| `MultiEnvironmentTrainer`, `MultiAgentTrainer`, `MultiTrainer` | **commented out**, not usable | — |

If you need multi-map training today, look at `Grid_Robot_Sections_Env_MultiMap`
(random map per episode, used in `grid_training_24.py`) rather than a
multi-env trainer — that's how multi-map support was actually implemented.

### 2e. `Manager.py` (`RLManager`/`EnvironmentConfig`/`AgentConfig`) looks unused

It's a generic, env-agnostic experiment runner with its own `__main__` demo
using stub classes (`StubEnv`, `QLearningAgent`). Nothing in `train/` imports
it. Treat it as exploratory/abandoned scaffolding unless you find evidence
otherwise — don't assume it's wired into the real pipeline the way `CLAUDE.md`
implies ("higher-level orchestration layer... used when comparing agents").

### 2f. Maps: two locations, four maps now

`train/grid_training_22+.py` load maps from **`./maps/`** (repo-root dir),
which now has **4** maps: `map_obstacle_free.json`, `map_simple.json`,
`map_1_obstacle.json`, `map_4_square_obstacle.json`. The older
`src/tfm_control_ucm/maps/` only has the first two — `CLAUDE.md`'s "two maps
exist" line is stale. Use `./maps/` for anything matching current experiments.
`map_simple` has the most obstacles (14, overlapping: 121 occupied cells).

### 2g. Tests and bug fixes (decoupling phases 0–1)

`pytest` (from the repo root, ~3 s) runs 55 tests in `tests/`:
`test_map.py`, `test_robot.py`, `test_environment.py` (incl. episode-end rules),
`test_rewards.py`. They build tiny maps and
inject a trivial `reward_fn`, so they don't depend on any train script. Run them
before and after touching `grid_env/`.

Fixed in the working tree (all were real failures reproduced by the tests):
- `Grid_Robot_Sections_Env_MultiMap._get_random_map` — was
  `int(np.random.uniform(0, n-1))`, so the last map was almost never picked;
  `random_map=False` was inverted. Now `self.np_random.integers(n)` (seeded by
  `reset(seed=...)`) or a round-robin `_map_idx`. This bug came from the
  uncommitted refactor; no trained run ever used it (test 24 never ran).
- `MultiMap._map_diagonal` / observation bounds now use the largest map.
- `_previous_obs` aliased the returned observation in both sections envs → `.copy()`.
- `Grid_Robot_Env` crashed on first `reset()` (`[::-1]` on a torch tensor) → lidar
  output converted to numpy.
- Both robots crashed on numpy integer actions (`action_space.sample()` gives
  `np.int8`); `GridRobot.move` now rejects actions outside `[0, 3)`;
  `GridRobot.__init__` validation fixed.
- `GridMap`: obstacles touching the right/bottom border were rejected (off-by-one);
  removing one of two overlapping obstacles freed the other's cells; constructor
  aliased the caller's obstacle list; `str(map)` raised `KeyError`.
- `PPOAgent` only created `last_entropy`/`last_policy_loss`/`last_value_loss` in
  its first `train_step`, so `PPOTrainer` crashed if the first episode ended
  before filling a 64-step batch (e.g. an early collision). Now initialised to 0
  in `__init__`, like the GRU agents. This bug is also in the committed code.

Still open: `MultiMap.reset()` no longer resets the pygame renderer when the map
changes (only matters with `render_mode="human"`); multi-env trainers still
commented out (§2d).

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
| 25 | Same as 24 on the decoupled API: `ProgressReward(clip_negative_progress=True)`, env decides episode end, `config.json` per run | all 4 maps | not committed yet | **ready, not trained** (5-episode smoke test only) |

Takeaway: the sequence-chunked recurrent buffer (test 21 GRU_RNN) is the only
GRU-based success so far; the project was moving toward multi-map
generalization (test 24) when the reward-injection refactor landed and froze
further training.

---

## 4. Architecture quick-reference (file:line)

Only the parts not fully obvious from `CLAUDE.md`, or useful jump points:

**Environments** — `src/tfm_control_ucm/core/grid_env/environment.py`
- `Grid_Robot_Env` (line 33) — original single-ray-list env, works with either `Robot` subclass now (was `GridRobotNotTurning`-only).
- `Grid_Robot_Sections_Env` (line 187) — buckets LIDAR rays into `num_sections` angular sections, each giving nearest-hit `[dist, angle]`; this is what every PPO experiment since test 19 actually uses.
- `Grid_Robot_Sections_Env_MultiMap` (line 367) — same as above but picks a random (or round-robin) map from `maps: Iterable[GridMap]` each `reset()`; used by test 24.
- A large commented-out `Grid_Robot_Env_Image_lidar` block (line 569+) is dead/experimental (2D occupancy-image observation) — ignore unless resurrecting it.

**Robot / Map / Sensor** — `grid_env/{robot,map,sensor}.py`
- `Robot` (ABC) requires subclasses declare `NUM_ACTIONS`; enforced via `__init_subclass__` (robot.py:83).
- `GridRobot` — egocentric actions: 0=forward, 1=turn left, 2=turn right (others raise `ValueError`). `NUM_ACTIONS=3`. The envs accept it but always scan the lidar with heading 0 and never randomise its orientation — usable only after phase 3.
- `GridRobotNotTurning` — absolute N/S/E/W moves, no orientation state. `NUM_ACTIONS=4`. **This is what all PPO train scripts (19–24) actually use** — orientation-based turning was abandoned in favor of absolute moves.
- `GridLidar` (sensor.py) — raycasting with an optional full-map distance cache (`with_cache=True`, built once per unique `gridmap.name`, chunked for memory). Noise is Gaussian, applied post-cache-lookup, zeroed on max-range (no-hit) rays.

**Agents** — `src/tfm_control_ucm/core/agents/{agent.py,base_rl_agent.py,utils.py,networks.py}`
- All concrete agents live in **`agent.py`** (not split into per-algorithm files despite `CLAUDE.md` implying otherwise): `GridAgent_2_FIXED` (legacy, non-`BaseRLAgent`, TorchRL `ReplayBuffer`+`LazyTensorStorage`), `GridAgent`, `DQNAgent`, `A2CAgent`, `PPOAgent`, `SACDiscreteAgent`, `PPOGRUAgent`, `RecurrentPPOAgent`.
- `AgentFactory.create_agent()` (agent.py:1357) only actually dispatches `DQN/DDQN`, `A2C`, `GRID_AGENT` — **PPO, SAC, GRU variants are not in the factory**, they're always constructed directly in train scripts.
- `PPOGRUAgent` vs `RecurrentPPOAgent`: both PPO+GRU, but `PPOGRUAgent` keeps a flat Python-list buffer and computes GAE in one shot per episode (`compute_gae` method, agent.py:1166); `RecurrentPPOAgent` uses `RecurrentPPOBuffer` (utils.py:252) which chunks each episode into `seq_len`-sized sequences before GAE+padding — this is the one that worked in test 21.
- Config dataclasses (utils.py): `RLConfig` (base) → `DQNConfig`, `SACConfig`, `PPOConfig` → `PPORNNConfig`. Note **`DQNConfig` exists but `GridAgent`/`DQNAgent` actually read `eps_start/eps_end/eps_decay/target_update/double_dqn` via `getattr(config, ..., default)` on the base `RLConfig`** — in practice train scripts pass a plain `RLConfig` or `PPOConfig` with those extra kwargs tacked on dynamically (Python dataclasses don't enforce this, so it silently works via `getattr` defaults).

**Training** — `src/tfm_control_ucm/core/training/TrainingManager.py` (see §2d for which trainers are live)
- TensorBoard tags differ between `BaseTrainer._inner_training` (DQN-style: `Loss/Episode`, `Policy/Epsilon/Episode`, `Steps/Ratio/Episode`, `Reward/Total`, `Success/Episode`, `Distances/*`) and `PPOTrainer`/`PPOGRUTrainer`/`RecurrentPPOTrainer` (PPO-style: `Agent/Loss`, `Agent/Entropy`, `Steps/Ratio` (no `/Episode` suffix), `Done/Success`, `Done/Max Steps Reached`, `Done/Hit Obstacle`, plus per-step `Info/Reward/Components/*` if the env's `info` dict has `reward_components`). **`scripts/plot_runs.py`/`compare_runs.py` need to know which tag family a given test number used** — don't assume `Reward/Total` exists for PPO runs.
- `PathPlanner.min_steps(..., method="wavefront")` (used for the `Steps/Ratio` metric) is recomputed **once per episode** (`path_planner = PathPlanner(env.map.grid)` inside the episode loop in `PPOTrainer`/`PPOGRUTrainer`/`RecurrentPPOTrainer` — note: *inside* the episode loop, not hoisted out, so it rebuilds every episode; cheap for small grids but worth knowing if profiling).
- `InterruptHandler` (training/utils.py) — Ctrl+C saves an emergency checkpoint and exits cleanly; Ctrl+\\ is wired for a "soft reset" signal but POSIX-only (no-op on Windows, which is this project's dev OS).

**Utils**
- `utils/path_planner.py` has both a pure-Python `PathPlanner` (BFS/DFS/Dijkstra/greedy/A*/wavefront) and a `TorchPathPlanner` GPU variant (BFS/A*/wavefront only) — `TrainingManager` uses the plain `PathPlanner`, not the Torch one.

---

## 5. Observation / action cheat sheet (current PPO envs, test 19+)

- Robot: `GridRobotNotTurning`, 4 absolute actions (N/S/E/W), no orientation in observation.
- Observation (`Grid_Robot_Sections_Env[_MultiMap]`): `num_sections` angular LIDAR buckets (each `[nearest_dist, angle]`, `-1`-padded sentinel value is **not** used — empty sections get `[max_range, section_center_angle]`) concatenated with `[dist_to_goal, angle_to_goal]`. `obs_dim = num_sections*2 + 2`.
- `num_sections = 4 * n` (CLI `--n`, default 2 in test 24 → 8 sections). `lidar num_rays` is set much higher than `num_sections` (e.g. `n_sections*15`) so each section reliably has multiple candidate rays to pick the nearest from.
- Reward: `ProgressReward` from `core/rewards.py`, injected by the train script (§2a). Episode end is decided by the env: goal within 1 cell → `terminated`; collision or step limit → `truncated`.

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
