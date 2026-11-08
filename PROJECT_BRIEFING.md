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

**Last updated**: 2026-11-08 08:45 (by Claude, results of tests 33-39; seeds 351-354 of test 35 training). **Headline**: test 35 (test 31 + momentum in the observation) is the best agent, **confirmed over 5 seeds**: greedy 99.4 ± 0.4 % vs 95.7 ± 0.4 % for test 31's 5 seeds (Welch t = 16), loops 6.2 vs 42.6, sampled 99.7 ± 0.2 %, 0 collisions: the greedy loops were partial observability, and remembering the last move removes them. **DQN** (tests 36-39) is 10-41 points below PPO in greedy mode because the Q-greedy policy loops; lr 1e-4 (test 39) made it worse. **Test 40** (DQN with the momentum observation) jumps to 91.1 % greedy (loops 344 → 88) but stays below PPO with the same observation (test 35: 99.6 %): the fair DQN-vs-PPO pair is 40 vs 35. **Over 5 seeds each: DQN 87.9 ± 3.8 % greedy vs PPO 99.4 ± 0.4 % (Welch t = 6.7), sampled 93.8 ± 2.3 vs 99.7 ± 0.2 %; DQN fails by greedy loops (120 ± 38 of 1000), worst on simple_map (75.9 %), and varies ~10× more across seeds.** **Textbook PPO (test 41, 5 seeds): greedy 93.1 ± 1.1 %, sampled 98.1 ± 1.0 % — below PPOAgent (35) but above DQN (40, t = 2.9 greedy / 4.0 sampled), so PPO > DQN also holds with textbook PPO; it converges prematurely (approx KL 0.03-0.07 per update).** Hypothesis not tested yet: the goal angle jumps from +π to −π behind the robot (sin/cos encoding in a new env subclass). · **Branch**: `Put-all-together` · **Latest commit**: "MaskedDQNAgent and test 36" (after `a8341e5` "Slip trainer and GridMap.save fix"). **Test 33 was launched 2026-11-07 19:27** (seed 0, logs in `logs/test33_train.{out,err}`). Still untracked: the outputs of tests 26–32 and 311–314 (`runs/`, `checkpoints/`, `plots/`, `logs/`)

**Plan to February 2027** (delivery of code + thesis; user's RTX 5060 Ti; no ROS expected): slippery floors in the
grid (tests 33-34, §2j) → **DQN on the current API** (no DQN run exists since test 25; the thesis is "DQN vs PPO")
→ final configurations with 3-5 seeds by ~20 Dec → writing in January. Gazebo/Isaac: out of the core scope;
at most a time-boxed Isaac Lab demo in January if everything above is done.

**Decoupling plan** — evaluation + 6-phase plan (with diagrams) in the Claude doc
"Evaluación del desacoplamiento" (https://claude.ai/artifact/XgaBeVnuccLX64zzihFyJS).
Phases 0 (tests), 1 (bug fixes), 2 (reward contract + `grid_training_25.py`),
3 (`BaseGridEnv`, egocentric observation) and 4 (one training loop with agent
hooks) are done and pushed. Left: phase 5 (delete commented-out code, legacy
classes and copy scripts).

**Where the experiments stand**: tests 26-30 (curriculum, geodesic reward, GRU memory, 30-cell lidar) all
left map_simple at ~78 %. **Test 31 (invalid-action masking) is the best agent, confirmed over 5 training seeds**
(`plots/seeds_31/seeds_stats.md`): eval success greedy **95.7 ± 0.4 %** (95 % CI 95.3-96.2), sampled **99.3 ± 0.4 %**
(98.8-99.8), map_simple 90.4 ± 1.6 % / 98.1 ± 1.4 %, 0 collisions in every seed. So the map_simple ceiling of
tests 25-30 was **collisions, not local minima**. The only remaining failure is **greedy loops** (42.6 ± 3.3 of
1000; sampling cuts them to 6.6 ± 3.8). **Test 32** (= 31 with signed progress, 1 seed) did not remove them:
94.1 % greedy and 59 loops, 4.5 and 5.0 std of the 31 seeds away, so it is probably *worse* in greedy mode
(sampled 98.8 %, within noise); confirming needs seeds 1-4 of test 32. Working explanation of the loops: partial
observability (memoryless deterministic policy on aliased observations; §2i.1).

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
  map_diagonal, lidar_max_range, `geodesic_dist_to_goal`, `previous_geodesic_dist_to_goal` (BFS distance
  around obstacles, from the goal distance field the env computes once per reset; inf if unreachable).
  Rewards never parse `obs` indices.
- `RewardFunction` (ABC): `__call__(ctx) -> RewardResult(reward, components)`;
  dataclass subclasses get `get_config()` (name + params) for experiment logs.
  Plain callables also accepted (may return a float or `(reward, components)`).
- `ProgressReward`: progress `progress_scale*(prev−dist)/diag` (optionally
  clipped at 0), `−step_penalty`, linear proximity penalty below
  `safety_margin`, `+goal_reward`, `−collision_penalty`; `distance="euclidean"|"geodesic"` picks the
  distance of the progress term (geodesic falls back to Euclidean if unreachable). Defaults = the
  `grid_training_10 copy.py` reward; `clip_negative_progress=True` = the
  pre-refactor MultiMap reward.

`BaseGridEnv._finish_step` does the end of every `step()`: **the env decides**
`terminated` (goal within `_goal_tolerance = 1` cell) and `truncated`
(collision or step limit), builds the `StepContext`, calls the reward, and fills
`info` with `reached_goal`, `hit_obstacle`, `max_steps_reached`, `dist2goal`,
`min_dist`, `map`, `reward`, `reward_components`, `componente_reward_{min,max,diff}`.
`reset()` info has `map`, `optimal_steps` (wavefront BFS, turns not counted) and
`dist2goal`.

Open design questions (not changed; see §2i for why they matter): collisions are `truncated`, and
`PPOAgent` passes only `terminated` as `dones` to GAE, so a collision
bootstraps from the value of the state where the robot stopped. Also,
`PPOAgent` samples random 64-transition batches from a 100k replay buffer of
past episodes (stale log-probs and advantages) instead of updating on the
latest rollout — closer to off-policy than textbook PPO.

The masking uses the true occupancy of the neighbouring cells (privileged map information): in the thesis it is
an external action constraint (a shield), not learned obstacle avoidance and not a formal safety guarantee.

### 2a-bis. Curriculum (`core/curriculum.py`)

`SuccessCurriculum(stages, window)` with `CurriculumStage(maps, max_goal_distance, success_threshold, min_episodes, max_episodes)`.
The **environment owns it** (`BaseGridEnv(curriculum=...)`): the current stage limits the maps (by `GridMap.name`,
uniform among them) and the start-goal distance, and `_finish_step` reports each episode's outcome
(`curriculum.record(reached_goal)`). The stage advances when the success rate over the last `window` episodes of
the stage reaches its threshold (after `min_episodes`), or after `max_episodes`. Trainer and agents don't know
about it; the trainer logs `Curriculum/Stage` and `Config/Curriculum` if the env has one.

### 2a-ter. Action masking

`Robot.valid_actions(map)` (bool per action: `GridRobotNotTurning` = target cell free; `GridRobot` = front
free, turns always valid). Every env `reset`/`step` info has `action_mask`. With
`TrainerConfig(action_masking=True)` the trainer passes it to `agent.act(..., action_mask=...)`; only
`PPOAgent` supports it (masked logits when acting, mask stored as a 9th buffer field and applied in
`_compute_loss`); other agents raise `NotImplementedError`. `eval_agent.py` applies it when the run's
`config.json` has `"action_masking": true`.

### 2a-quater. Slippery floors (new classes; `GridMap`, the robots, `BaseGridEnv`, `StepContext` and `PygameRenderer` are unchanged)

**User decision (2026-11-07)**: new features go into **new classes** (subclasses / new modules), not into the
existing map, robot and environment classes. A first version that modified them was reverted.

| New | File | Extends |
|---|---|---|
| `GridTerrain`, `SlipperyGridMap`, `friction_at(map, r, c)` | `core/grid_env/slippery_map.py` | `GridMap` |
| `SlipperyRobotMixin`, `SlipperyGridRobotNotTurning`, `SlipperyGridRobot` | `core/grid_env/slippery_robot.py` | `GridRobotNotTurning`, `GridRobot` |
| `Slippery_Grid_Robot_Sections_Env_MultiMap` | `core/grid_env/slippery_environment.py` | `Grid_Robot_Sections_Env_MultiMap` |
| `SlipperyPygameRenderer` | `renderers/slippery_renderer.py` | `PygameRenderer` |
| `SlipSingleTrainer`, `SlipMultiEnvironmentTrainer` (`SlipLoggingMixin`, `SlipCounter`) | `core/training/slip_trainer.py` | `SingleTrainer`, `MultiEnvironmentTrainer` |
| `SlipModel`, `classify_maneuver`, `SlipEvent`, `NO_SLIP` | `core/slip.py` | — |

`TrainingManager.py` is unchanged too. The slip trainers wrap each episode's env in `SlipCounter` (pass-through
that reads `info` of every step), run the parent's loop and then log, per episode and only when the env reports
slips: `Slip/Count`, `Slip/Rate` (slips/steps), `Slip/Collision`, `Slip/Kind/<kind>` (each kind of the slip model),
plus the text `Config/Slip`. `export_episode_metrics.py` exports them to `episodes.csv` like any other tag. The
plain `SingleTrainer` on a slippery env trains the same but logs no `Slip/*` (tested). Shared files that did
change: the eval scripts, and `GridMap.save()` / `GridObstacle.get_json_repr()` (bug fix, user asked, see below).

- **Map**: `GridTerrain(corner, size, friction, name)` patches fill `SlipperyGridMap.friction` (float, default 1.0 =
  normal floor; later patch wins on overlap). Optional `"terrains"` key in the JSON: plain map files load as a
  `SlipperyGridMap` with grip 1 everywhere. The occupancy `grid` is untouched, so the lidar, the BFS
  (`optimal_steps`, geodesic distance) and the action mask ignore friction. `str()` draws `~` for slippery free
  cells; `SlipperyPygameRenderer` tints them blue. `SlipperyGridMap.save()` adds the terrains. The plain
  `GridMap.save()` used to crash (numpy arrays and `GridObstacle` objects in the JSON); fixed on 2026-11-07,
  with a round-trip test in `test_map.py`.
- **Robot**: the slipping happens in `SlipperyRobotMixin.move()`. It keeps the **momentum** (last displacement,
  `(0, 0)` stopped; turning in place stops it), uses the robot's own RNG (`set_rng`), and returns False **only on a
  collision** (a wheel spin returns True), so `BaseGridEnv.step()`'s `hit_obstacle = not moved` is unchanged.
  `slip_model=None` = the parent's moves, momentum still tracked. Outcome in `last_slip`, `last_slip_collision`,
  `last_friction`.
- **Slip model** (`SlipModel`, given to the robot): each move is classified against the momentum.
  Each move is classified against it — straight → *overshoot* (2 cells), start → *stay* (wheels spin, **not** a
  collision), turn 90° → *diagonal* (through the momentum cell), reverse → *drag* (1 cell in the old direction);
  optional *lateral* drift. `p = clip(gain·(demand − μ), 0, max_prob)` with μ of the cell the step starts on;
  defaults demand 0.3/0.4/0.6/0.8, gain 2, cap 0.9 (ice μ=0.2: 0.2/0.4/0.8/0.9; wet μ=0.5: 0/0/0.2/0.6; μ=1: 0).
  The demands are model parameters (only their order is justified). Moving into a wall is still a plain collision.
  A slide into an obstacle: `into_obstacle="collide"` (default, user decision) = collision (`truncated`), the
  robot stays on the last free cell; `"stop"` = no collision.
- **Consequence for masking**: the mask only checks the intended cell, so with `collide` collisions are possible
  again in masked runs (random masked policy on the slippery maps: 107/300 episodes end in a slide collision).
- **RNG**: the slippery env's `reset(seed)` seeds the robot's slip generator and the sensor-noise generator with
  `SeedSequence(seed).spawn(2)`, apart from `np_random`, so the map / start / goal sequence of a seeded evaluation
  is the same with or without slips (tested).
- **Observation** `terrain_obs` of the slippery env: `"none"` (default), `"proprio"` (+4, egocentric: momentum
  forward/left, slipped on the last step, noisy μ of the current cell — what odometry + IMU give a real robot; user
  asked for "the most realistic"), `"neighbors"` (+8: also noisy μ of front/left/back/right, like a camera terrain
  classifier). Noise `terrain_obs_noise` (std, default 0.05). The env keeps the lidar + goal part in
  `self._base_obs` and hands that to the parent's `_start_episode` / `_finish_step`, so the reward's `ctx.obs` /
  `ctx.previous_obs` and `render` see the usual layout.
- **Reward**: no friction term (user decision); `StepContext` is unchanged, so rewards cannot see the slip.
  `info` (only when the robot has a slip model) has `slipped`, `slip_kind`, `slip_maneuver`, `slip_probability`,
  `slip_collision`, `friction`. `eval_agent.py` builds the slippery env when `config.json` has `"terrain_obs"`
  (with `"slip"` → `SlipModel`, null → no slips) and counts slips / slide collisions; so does `eval_policy_modes.py`.
- **Maps**: `maps/*_slippery.json` = the 4 maps with the same obstacles plus ice/wet patches (15-24 % of free
  cells, several next to walls and obstacles); names end in `_slippery`, so per-map tags differ from test 31's.
- Not done: a friction-aware planner (Dijkstra with expected cost) for `optimal_steps`; the goal only counts if the
  robot *ends* the step on it (sliding over it does not).

### 2a-quinquies. `MaskedDQNAgent` (`core/agents/masked_dqn.py`, new; `GridAgent` unchanged, `DQNAgent` only bug-fixed, §2g)

Double DQN with invalid-action masking, the DQN counterpart of PPO + masking (test 31). Why a new agent:
`DQNAgent` raised on masks, its target update checked `global_step % target_update` only on training steps (so it
fired irregularly), its target maximised over invalid actions, and the shared `ReplayBuffer` rebuilds arrays from
dicts on every sample. `MaskedDQNAgent`: ε-greedy among the valid actions (linear decay over `eps_decay`
*training* steps; `training=False` is greedy and does not move ε); target `r + γ·Q_target(s', argmax_valid Q(s'))`
with the next-state mask stored with the transition (it arrives with the next `act()`, so a non-terminal transition
enters the buffer one step later); terminated **or** truncated = terminal (as `DQNAgent`: makes crashes final; small
time-limit bias); hard target copy every `target_update` gradient updates; no update before `learning_starts`;
`MaskedReplayBuffer` = preallocated numpy ring buffer. Config `MaskedDQNConfig(DQNConfig)` (+`learning_starts`,
`eps_decay` default 500k). Logs `Policy/Epsilon/Episode`, `Agent/Q/Mean`, `Agent/DQN/Loss`, `Agent/DQN/Updates`.
`eval_agent.py` loads `dqn`/`ddqn` checkpoints as `MaskedDQNAgent` (`grid_agent` still as `GridAgent`), so
`eval_policy_modes.py` works; its "sampled" mode is ε-greedy with ε = `eps_end` for DQN. Tests: `test_masked_dqn.py`.
Options added 2026-11-07 (off by default, test 36 runs without them): `soft_target_update` (Polyak with `tau` after
every update) and `bootstrap_truncated` (step-limit truncation bootstraps; only correct when truncation cannot be a
collision — masked runs without slips).

**Observed DQN instability (2026-11-07, test 36 at episode ~1650)**: success rose to 77 % while ε was high
(episodes 250-750) and fell to 23 % as ε reached 0.05 (77 % timeouts, steps ratio 10 = greedy loops), with mean Q
up to 37 — above any reachable return (≈ 20 goal + ≤ 10 progress) while episode returns were 1-4: overestimation.
Sanity check on a 30×30 empty map: `MaskedDQNAgent` **and** the plain `DQNAgent` both reach ~100 % greedy in 150
episodes and then degrade (to 29-48 %) once ε is small, with clipped **and** signed progress → not a code bug of
one agent; a stability problem of DQN on this task (plus the greedy loops of §2i.1). A sweep of fixes (lr 1e-4,
soft target, target every 250 updates, ε_end 0.1, bootstrap on truncation) was run on that map; see §3 / results.

### 2a-sexies. `StandardPPOAgent` (textbook PPO, `core/agents/standard_ppo.py`) and step-info trainers

Answers §2i.2 (`PPOAgent` is not textbook PPO; `PPOAgent` itself is unchanged). `StandardPPOAgent`: rollouts of
`rollout_steps` (2048) env steps across episodes → GAE(λ) with `V(s')` per step (terminal = no bootstrap; time limit
and mid-episode rollout end bootstrap; the recursion is cut at every episode end) → `n_epochs` (10) of shuffled
minibatches of `batch_size` (64): clipped surrogate on per-minibatch normalised advantages, value loss 0.5·MSE
(optional `clip_value`), entropy bonus, grad clip, optional `target_kl` early stop → rollout discarded. Masking:
masks stored with the rollout, masked log-probs/entropy in the update. The update runs inside `observe()` when the
rollout is full (`train_on_step = False`, independent of `eps_step`); `on_episode_end` returns the loss of an
update done during the episode. Logs `Agent/Entropy`, `Agent/Policy/Loss`, `Agent/Value/Loss`, `Agent/PPO/ApproxKL`,
`ClipFraction`, `ExplainedVariance`, `Updates`. Checkpoints use the `PPOAgent` format (algorithm "ppo"), so
`eval_agent.py` evaluates them as `PPOAgent` (same greedy/sampled behaviour; tested).

Collisions vs time limit: the env ends both as `truncated` (§2i.3) and `TrainingManager` passes no info to agents,
so `core/training/step_info_trainer.py` adds `StepInfoSingleTrainer` / `StepInfoMultiEnvironmentTrainer` /
`SlipStepInfoSingleTrainer` (with slip logging): `StepInfoRelay` wraps each episode's env and calls
`agent.on_step_info(info)` after `env.step()`, before the loop's `observe()`. `StandardPPOAgent` then treats
`info['hit_obstacle']` as terminal; without relay, `truncation_is_terminal` (default False) decides.
Smoke run (600 episodes, test 41 setup): 93 % training success from episode ~150 (PPOAgent needs ~1500),
241 steps/s on the GPU (10 epochs per 2048 steps), approx KL 0.018, clip fraction 0.19, explained variance 0.79.
Tests: `tests/test_standard_ppo.py` (GAE cases, collision vs time limit, rollout/epochs, target_kl, masking, eval).

### 2b. Train/eval scripts

- `train/grid_training_25.py` — **the working script**. Same setup as 24 (PPO,
  4 maps, random map per episode) with `ProgressReward` built from CLI args,
  `--seed`, `--render/--no-render`, `SingleTrainer`, and writes the full config
  (args, reward params, maps, lidar) to `runs/Test/<N>/config.json`.
  Base of every later script (26-32 are "test 25 + one change").
- `train/grid_training_26.py` — test 25 + curriculum + `TrainerConfig(log_steps=False)`; curriculum CLI args
  `--curriculum_window`, `--stage_threshold`, `--stage_min_episodes`, `--stage_max_episodes`; `config.json` includes
  the curriculum.
- `train/grid_training_27.py` (geodesic progress, `--progress_distance`) and `train/grid_training_28.py`
  (`RecurrentPPOAgent`, `--hidden_dim`, `--seq_len`) — both test 25 + one change, `log_steps=False`.
- `train/grid_training_29.py` (28 with the recurrent training fixed), `_30.py` (30-cell lidar),
  `_31.py` (action masking, `TrainerConfig(action_masking=True)`).
- `train/grid_training_32.py` — test 31 with `clip_negative_progress=False` (default of the CLI flag;
  `--clip_negative_progress` restores 31). Trained (§3).
- `train/grid_training_33.py` — test 31 on the slippery maps (`SlipperyGridMap`, `SlipperyGridRobotNotTurning`,
  `Slippery_Grid_Robot_Sections_Env_MultiMap`, `SlipSingleTrainer`) with `SlipModel()` (`collide`), observation of 31
  (`--terrain_obs none`). Slip CLI args `--demand_*`, `--slip_gain`, `--slip_max_prob`, `--slip_kinds`,
  `--lateral_demand`, `--into_obstacle`. `config.json` adds `slip`, `terrain_obs`, `terrain_obs_noise`. Not trained.
- `train/grid_training_34.py` — test 33 with `--terrain_obs proprio` (obs_dim 22). `--no-slippery --test_number 35`
  = test 31 maps, no slips, same observation (31 + momentum only: does the momentum alone remove greedy loops?).
  Not trained.
- `train/grid_training_36.py` — **DQN side of the thesis comparison**: test 31 with `MaskedDQNAgent` (§2a-quinquies)
  instead of PPO; same maps, reward, masking, lidar, step limit, seed, network body (`SimpleQNetwork` 3×64) and
  40k episodes; DQN's own hyperparameters (lr 5e-4, buffer 100k, batch 64, ε 1 → 0.05 linear over 500k steps,
  target copy every 1000 updates, learning starts at 5000 transitions, one update every 4 steps, Double DQN).
  `config.json` has `"algorithm": "dqn"`. Smoke run: 463 steps/s on the GPU, 0 collisions.
- Seed runs reuse the scripts with `--seed S --test_number NNS` (e.g. `grid_training_31.py --seed 1 --test_number 311`):
  outputs go to `runs/Test/311`, `checkpoints/Test/311`, and the analysis scripts accept those numbers.
- `train/grid_training_22.py`–`_24.py` and the `* copy.py` scripts use the old
  11-positional-arg reward or no reward at all and fail at env construction.
  **Kept broken on purpose** (historical record, see convention above).
- `scripts/eval_agent.py` — rebuilds the env from `runs/Test/<N>/config.json`
  or from the checkpoint dims (action_dim → robot, obs_dim → sections), rebuilds
  any MLP policy from the state_dict shapes, uses the agent hooks, and reports
  success/collision/timeout overall and per map. Runs without `config.json`
  are evaluated with `_legacy_goal_angle` (test 22 on `map_4_square_obstacle`:
  80 % success with it, 0 % without). GRU checkpoints (`gru.*` weights) load as `RecurrentPPOAgent`.
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

`pytest` (from the repo root, ~12 s) runs 188 tests in `tests/` (incl. `test_masked_dqn`, `test_dqn_agent`)
(`test_map`, `test_robot`, `test_environment`, `test_rewards`, `test_heading`,
`test_training`, `test_scripts`, `test_curriculum`, `test_action_masking`, `test_slip`). `matplotlib` and `pandas` are installed in `.venv` (not in
`requirements.txt`). They build tiny
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
- `DQNAgent` (2026-11-07, user allowed fixing broken agents in place; `test_dqn_agent.py`): the target network is
  copied every `target_update` *gradient updates* (it checked `global_step % target_update` on training steps only,
  so with training every 4 steps it fired irregularly), and `training=False` no longer advances `global_step` /
  ε. Masking is still only in `MaskedDQNAgent`.

Still open: `GridAgent_2_FIXED`, the commented-out `Grid_Robot_Env_Image_lidar`
and other commented-out code (phase 5).

### 2h. How to analyse a finished run; multi-seed runs

Seed runs reuse a script with `--test_number NNS` (`31S` = test 31 with seed S; outputs in `runs/Test/31S`,
`checkpoints/Test/31S`). Seeds 1-4 of test 31 are done (tests 311-314, ~1 h 45 min each on the GPU). The same
loop for test 32 (not run yet), PowerShell from the repo root with `.venv` active:

```powershell
foreach ($s in 1..4) {
    python train/grid_training_32.py --seed $s --test_number "32$s" > "logs/test32${s}_train.out" 2> "logs/test32${s}_train.err"
}
```

Then evaluate and compare (`--seed` of `analyze_tests.py` is the *evaluation* seed and stays 0, so every training
seed is evaluated on the same 1000 episodes; existing evaluations in `plots/evals/` are reused):

```bash
python scripts/analyze_tests.py 32 321 322 323 324 --labels "32 s0" "32 s1" "32 s2" "32 s3" "32 s4" --out plots/seeds_32
```

`summary.md` gives one row per seed with z-tests against the first; for the thesis report mean ± std across seeds
and a t interval (n = 5: t = 2.776), as in `plots/seeds_31/seeds_stats.md` (generated from the eval JSONs; no
script in the repo yet). Note: `runs/Test/311/PPO Agent/` also holds a 50 KB event file from an aborted first
start; the export uses the latest event file, so it is harmless.

Pieces, if needed on their own:
- `scripts/export_episode_metrics.py "runs/Test/<N>/PPO Agent"` → `runs/Test/<N>/episodes.csv` (one pass over the event file).
- `scripts/eval_policy_modes.py <N>` → `plots/evals/eval_<N>_e1000_s0.json` (greedy vs sampled, per map, loops, path ratio;
  applies action masking when `config.json` has it). ~2 min per mode on CPU (measured on Linux for tests 31 and 32).
- `scripts/plot_episode_csv.py runs/Test/<N>/episodes.csv --out plots/test<N>` → single-run plots + `summary_by_phase.csv`.
- `scripts/compare_episode_csv.py <csv...> --labels ... --out ...` → comparison plots + phase table.
- `scripts/eval_agent.py "checkpoints/Test/<N>/PPO Agent/final.pth" --episodes 100` → per-episode printout.

The evaluations are paired: with the same evaluation seed every test sees the same sequence of maps and
start/goal positions. The z-tests measure evaluation noise of one trained agent, not training variability; for
test 31 the training-seed spread is small (std 0.4 points in total success, 1.6 in greedy map_simple).

### 2i. Code review findings (2026-11-04), not fixed yet

1. **Clipped progress gives cycles a positive return, but that is not what causes the loops** (verified by running
   `ProgressReward` with the test 25 parameters, 40x60 map, diagonal 72.1): one cell towards the goal +0.089, one
   cell away -0.050, so an A-B-A cycle earns +0.039 per 2 steps (+7.7 over 400 steps); unclipped -0.100 per cycle.
   The clipped term is not a potential difference, so the policy-invariance result of potential-based shaping
   (Ng, Harada & Russell 1999) does not hold. Test 32 removed the clip and loops did not fall: 59 vs 42.6 ± 3.3
   over the 5 seeds of test 31 (one seed of 32, so not yet conclusive; see §2h).
   Working explanation: the observation (local lidar + goal vector) aliases nearby cells, and a memoryless
   deterministic (greedy) policy can cycle between them; sampling breaks the cycles (31: 41 -> 6 loops). In
   POMDPs stochastic memoryless policies can be strictly better than deterministic ones (Singh, Jaakkola &
   Jordan 1994). Options: report the sampled policy as the main mode; add minimal memory (previous action or
   visit count in the observation) or GRU + masking.
2. **`PPOAgent` is not textbook PPO** (§2a): one gradient step of 64 samples per episode from a 100k buffer of past
   episodes (stale `old_log_probs` and advantages), no fresh rollout, no epochs, no advantage normalisation;
   `PPOConfig.rollout_steps` and `clip_ratio` are unused. For a "DQN vs PPO" thesis either name it as an
   off-policy variant or add a standard PPO (rollout -> K epochs -> clear buffer) and compare.
3. **Collisions are `truncated`** (§2a): the crash costs -5 + gamma*V(crash state). Gymnasium semantics: an episode
   end caused by the MDP (crash) is `terminated`. Masked runs (31, 32) are not affected.
4. **One training seed per configuration** except test 31 (5 seeds): claims between other tests hold for those
   seeds only; final thesis configurations need 3-5 seeds and interval estimates (Agarwal et al. 2021).
5. **Latent bug**: `PolicyNetwork.forward` returns softmax probabilities but `PPOAgent` and `SACDiscreteAgent` read
   them with `Categorical(logits=...)` (double softmax: with 4 actions the top probability is capped at
   e/(e+3) = 0.475). Only hits agents built without explicit networks; tests 25-32 pass `SimpleQNetwork` (logits).
6. Minor: observations not normalised (goal distance up to ~72 vs lidar <= 10, angles in radians); `lr=1e-3` high
   for PPO; `select_action` increments `global_step` in evaluation too; `_sample_goal` loops forever if no free cell
   is farther than `_min_separation_distance`; `ReplayBuffer` stores dicts and rebuilds arrays on every sample.
7. Linux-only failures (Windows unaffected): `core/training/utils.py` imports `msilib` (Windows-only module), and
   on POSIX `InterruptHandler.__enter__` registers `self._handle_soft_reset`, which does not exist.

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
| 25 | Same as 24 on the decoupled API: `ProgressReward(clip_negative_progress=True)`, env decides episode end, egocentric angles, fixed PPO critic, unified trainer, seed 0, 40k episodes (5 h 49 min, GPU) | all 4 maps | "Test 25 - Decoupled reward, bug fixes and unit tests" (+ phases 3–4) | **success 93 % in the last 8k training episodes; eval 400 eps: greedy 90 %, sampled 94 %. map_simple is the weak map (~78 %)**. Plots and phase table in `plots/test25/` |
| 26 | Test 25 + `SuccessCurriculum` (5 stages: empty map, goals <= 20 -> + 1_obstacle, <= 30 -> + 4_obstacles -> + simple_map, <= 30 -> all maps, any distance = test 25's setting; 0.90/0.85 success over 500 episodes, min 500 / max 6000 per stage), `log_steps=False`, seed 0; 40k episodes in 3 h 18 min | all 4 maps | not committed yet | **stages started at episodes 0/1620/2120/3416/3916. Reaches 80 % success with ~half the environment steps of test 25, but the final agent is the same: eval 1000 eps sampled 93.9 % (25: 93.9 %), greedy 89.3 % (25: 91.1 %, n.s.), map_simple 77.8 % in both.** Plots: `plots/compare_25_26/` |
| 27 | Test 25 with `ProgressReward(distance="geodesic")` (progress along the BFS shortest path), `log_steps=False`, no curriculum, seed 0; 2 h 52 min | all 4 maps | not committed yet | **No gain. Eval 1000 eps: sampled 93.4 % (25: 93.9 %, n.s.), map_simple 78.6 % (25: 77.8 %, n.s.); greedy worse, 85.0 % vs 91.1 % (z = -4.2), 137 loops vs 76.** Plots: `plots/compare_25_27_28/` |
| 28 | Test 25 with `RecurrentPPOAgent` (GRU hidden 40, `seq_len` 32), same reward as 25, `log_steps=False`, seed 0; 4 h 37 min | all 4 maps | not committed yet | **Worse and unstable: collapsed from ~90 % to ~25 % success around episode 25k for ~7k episodes, then recovered. Eval: sampled 85.2 %, greedy 77.4 % (224 loops), map_simple ~60 %. RecurrentPPOAgent (and PPOGRUAgent) never clip gradients (`max_grad_norm` ignored) — likely cause; not a fair verdict on memory.** |
| 29 | Test 28 with the recurrent training fixed (gradient clipping and padding mask in `RecurrentPPOAgent.train_step`) and lr 3e-4; seed 0; 2 h 51 min | all 4 maps | not committed yet | **Stable (no collapse), but still below the MLP: eval 1000 eps sampled 89.8 % (25: 93.9 %, z = -3.4), greedy 88.3 % (z = -2.1); map_simple ~70 %. Near-deterministic (entropy 0.27), successful paths optimal (median ratio 1.00), but 95-111 loops in both modes: memory as implemented does not remove loops.** Plots: `plots/compare_25_29_30/` |
| 30 | Test 25 with a 30-cell lidar (was 10; safety margin kept at 6.67 cells), `log_steps=False`, seed 0; 2 h 41 min | all 4 maps | not committed yet | **Same as 25: sampled 93.7 % (z = -0.2), greedy 90.4 % (z = -0.5), map_simple 79.4 % (n.s.). Perception range is not the bottleneck.** |
| 31 | Test 25 with invalid-action masking (`TrainerConfig(action_masking=True)`: PPO samples only among `info["action_mask"]` = `Robot.valid_actions(map)`, acting and learning with the masked policy), `log_steps=False`, seed 0 | all 4 maps | not committed yet | **Best agent. Eval 1000 eps: sampled 99.4 % (25: 93.9 %, z = +6.8), map_simple 98.8 % (z = +7.3); greedy 95.9 % (25: 91.1 %, z = +4.4), map_simple 92.3 %; 0 collisions; greedy 41 timeouts, all loops (25: 76). Faster learning: 84.7 % success in episodes 0-10k (25: 72.2 %), 99.5 % in 30-40k.** The map_simple ceiling was collisions, not local minima. Plots: `plots/compare_25_31/` |
| 32 | Test 31 with signed progress (`clip_negative_progress=False`), seed 0; 1 h 44 min | all 4 maps | not committed yet | **Hypothesis refuted: no fewer loops. Eval 1000 eps vs 31: greedy 94.1 % (31: 95.9 %, z = -1.85, p = 0.08), 59 loops (31: 41); sampled 98.8 % (31: 99.4 %, p = 0.24), map_simple 96.4 %; 0 collisions; same final entropy (0.51). Faster early (88.5 % in episodes 0-10k vs 84.7 %), slightly lower late (98.9 % vs 99.5 %).** Plots: `plots/compare_25_31_32/` |
| 311-314 | Test 31, seeds 1-4 (`grid_training_31.py --seed S --test_number 31S`), ~1 h 45 min each | all 4 maps | not committed yet | **Test 31 is robust to the seed. Over seeds 0-4: greedy 95.7 ± 0.4 % (range 95.2-96.1), sampled 99.3 ± 0.4 % (98.9-99.8), map_simple 90.4 ± 1.6 % greedy / 98.1 ± 1.4 % sampled, loops 42.6 ± 3.3 greedy / 6.6 ± 3.8 sampled, 0 collisions; training success in episodes 30-40k 99.3 ± 0.3 %.** Table: `plots/seeds_31/seeds_stats.md`; per-seed summary and plots: `plots/seeds_31/` |
| 321-324 | Test 32, seeds 1-4 | all 4 maps | — | **optional, not run**: would confirm whether signed progress is worse in greedy mode (§2h) |
| 33 | Test 31 on the slippery maps, `SlipModel()` defaults, slide into obstacle = collision, observation unchanged; seed 0, 2 h 35 min | 4 `_slippery` maps | "Slippery floors…" | **Slips cost success. Eval 1000 eps vs 31: greedy 92.6 % (31: 95.9 %, z = −3.2), 15 collisions (all slides), 59 loops; sampled 93.4 % (31: 99.4 %, z = −7.2), 54 slide collisions — sampling turns more, and turns slip into diagonals (1824 vs 915), so sampled ≈ greedy here. simple_map 84.3 / 88.3 %. Greedy's most frequent slip is drag (reversing on ice): fits the back-and-forth loops. Training plateau 92-93 % from episode 15k.** Plots: `plots/compare_31_33/` |
| 34 | Test 33 + `terrain_obs="proprio"` (momentum, slipped flag, noisy μ here), seed 0; 2 h 52 min | 4 `_slippery` maps | "Slippery floors…" | **Proprioception helps. Eval vs 33: greedy 95.0 % (33: 92.6 %, z = +2.2), loops 14 (33: 59); sampled 96.0 % (93.4 %, z = +2.6), simple_map 89.5 / 94.0 %. 36 % fewer slips (greedy 2112 vs 3309): far fewer drags (665 vs 1374) and diagonals (429 vs 915), same overshoots — it learned to avoid reversing and turning on ice. Greedy slide collisions went up (35 vs 15), sampled down (37 vs 55).** The drop in loops matches the memory hypothesis of §2i.1 (momentum = previous move); test 35 (`grid_training_34.py --no-slippery`) would separate it from the slip effect. 1 seed, z ≈ 2.2-2.6. Plots: `plots/compare_33_34/` |
| 35 | `grid_training_34.py --no-slippery`: test 31 + proprio observation (momentum = last move; slip flag 0, μ ≈ 1), no slips, seed 0 | all 4 maps | — | **New best agent: momentum removes the greedy loops. Eval vs 31: greedy 99.6 % (31: 95.9 %, z = +5.6; 31's 5 seeds: 95.7 ± 0.4 %), loops 4 (31: 41), simple_map 98.4 %; sampled 99.7 %; 0 collisions. Confirms §2i.1: the loops were partial observability of a memoryless deterministic policy; remembering the last move is enough. Greedy ≈ sampled now.** Plots: `plots/compare_31_35/` |
| 40 | `grid_training_40.py`: test 36 (masked Double DQN, lr 5e-4) with the observation of test 35 (`terrain_obs="proprio"`, plain maps, no slips), seed 0; 3 h 30 min | all 4 maps | "Test 40…" | **Momentum also cures most of DQN's loops: greedy 91.1 % (36: 65.6 %, z ≈ +13.8), loops 88 (36: 344), obstacle_free 100 %; sampled 94.9 % (36: 80.5 %); 0 collisions. Best DQN so far, but still below PPO with the same observation (35: 99.6 % greedy, z = −9.0; 99.7 % sampled, z = −6.6) and below PPO without it (31: 95.9 %).** Same early dip as every DQN (mean Q 25, success 17 % around episode 1000). Fair DQN-vs-PPO pair: 40 vs 35. Plots: `plots/compare_35_40_36_31/` |
| 41, 411-414 | `grid_training_41.py`: test 35 with `StandardPPOAgent` (textbook PPO: rollout 2048, 10 epochs, minibatch 64, lr 3e-4, GAE 0.95, clip 0.2, ent 0.01, grad clip 0.5) + `StepInfoSingleTrainer`, seeds 0-4 (~3 h 50 each, 5 in parallel) | all 4 maps | "StandardPPOAgent…" | **Between PPOAgent and DQN. Greedy 93.1 ± 1.1 % (35: 99.4 ± 0.4, t = 11.8; 40 DQN: 87.9 ± 3.8, t = 2.9), loops 69 ± 11 (35: 6; 40: 120); sampled 98.1 ± 1.0 % (35: 99.7, t = 3.6; 40: 93.8, t = 4.0); simple_map 85.2 / 95.2 %; 0 collisions. Learns much faster (96 % training success in episodes 0-8k vs 86 % for 35) but converges prematurely: entropy 0.41 → 0.30 by episode 8k and flat (35: 0.87 → 0.38 slowly), training success stuck at ~98 % (35 reaches 99.8 %), approx KL per update 0.03-0.07 (high; usual target ≤ 0.015-0.02).** Plots: `plots/seeds_41/`, `plots/seeds_35_41_40/` (3 groups); tables `plots/seeds_41/seeds_stats.md` (vs 35), `seeds_stats_vs_40.md`. Next candidate: test 41 with `--target_kl 0.015` |
| 401-404 | Test 40, seeds 1-4 (`grid_training_40.py --seed S --test_number 40S`), ~3 h 45 each (4 in parallel) | all 4 maps | — | **DQN vs PPO with equal observation, 5 seeds each (`plots/seeds_40/seeds_stats.md`, vs 35's seeds): greedy 87.9 ± 3.8 % [83.2, 92.7] vs 99.4 ± 0.4 % (Welch t = 6.7, df 4.1), loops 120 ± 38 vs 6 ± 4; sampled 93.8 ± 2.3 % vs 99.7 ± 0.2 % (t = 5.9); simple_map 75.9 ± 6.1 vs 98.0 ± 0.8 % greedy; 0 collisions in all 10 runs. DQN's seed spread is ~10× PPO's. Seed 0 (91.1 %) was among the better DQN seeds.** `seed_stats.py 40 401 402 403 404 --vs 35 351 352 353 354` |
| 351-354 | Test 35, seeds 1-4 (`grid_training_34.py --no-slippery --seed S --test_number 35S`), ~3 h each (4 in parallel + test 40) | all 4 maps | — | **Confirmed over 5 seeds (`plots/seeds_35/seeds_stats.md`, vs the 5 seeds of 31): greedy 99.4 ± 0.4 % [98.9, 99.8] vs 95.7 ± 0.4 % (Welch t = 16.0, df 8); loops 6.2 ± 3.8 vs 42.6 ± 3.3 (t = 16.3); simple_map 98.0 ± 0.8 vs 90.4 ± 1.6 %; sampled 99.7 ± 0.2 vs 99.3 ± 0.4 % (t = 2.2, df 5); 0 collisions in every seed.** `seed_stats.py 35 351 352 353 354 --vs 31 311 312 313 314` (p values need scipy, not installed) |
| 36 | Test 31 with `MaskedDQNAgent` (Double DQN + masking) instead of PPO, seed 0; 4 h 30 min (with 3 other runs in parallel) | all 4 maps | "MaskedDQNAgent…" | **Far below PPO. Eval greedy 65.6 % (31: 95.9 %, z = −17.2), 0 collisions, 344 timeouts all loops (even obstacle_free 83 %); sampled (ε-greedy 0.05) 80.5 %. Training: dip to 28 % around episode 1500 (mean Q 37 > reachable returns), then 90-92 % (ε = 0.05) for the rest — the final checkpoint is worse than the training plateau: DQN snapshots vary a lot.** Masking removes collisions but not the greedy loops. Plots: `plots/compare_31_36_37_38/` |
| 37 | `grid_training_37.py`: plain `DQNAgent` (fixed), classic DQN target, **no masking**, ε exponential (decay 150k), seed 0 | all 4 maps | — | **Eval greedy 86.4 % (25 PPO no mask: 91.1 %, z = −3.3), 2 collisions, 134 loops (obstacle_free 94 %); sampled 85.9 %, 103 collisions (ε moves into walls). simple_map ≈ PPO without masking (72.6 vs 74.6 % greedy).** Plots: `plots/compare_25_37_38/`, `plots/compare_31_25_37/` |
| 38 | `grid_training_37.py --double_dqn --test_number 38`: as 37 with Double DQN | all 4 maps | — | **Eval greedy 77.0 % (z = −8.6 vs 25), 3 collisions, 227 loops; sampled 83.9 %. Training plateau ≈ 85 % like 37, so 37 > 38 > 36 in eval is snapshot noise of unstable runs (1 seed each), not evidence against Double DQN or masking.** |
| 39 | `grid_training_39.py`: test 36 with lr 1e-4 (sweep on a 30×30 empty map: the only change that kept greedy success > 90 % all along; faster target updates made it worse), seed 0; 5 h 9 min | all 4 maps | — | **Lower lr did not help: eval greedy 54.9 % (36: 65.6 %), 451 loops; sampled 71.5 %. Training also had the early dip (mean Q 22) and learned slower (63 % at episodes 3-5k vs 87-91 % for 36).** The sweep on the empty map did not transfer. Since test 35 shows the DQN failure mode (greedy loops) is cured in PPO by the momentum observation, the natural next DQN test is test 36 + `terrain_obs="proprio"` (the env of test 35) |

Test 25 takeaway: success climbs to ~83 % in 4k episodes and to 93 % by 40k, still rising (entropy 0.55 nats
at the end, map_simple still improving). Successful paths are near-optimal (median steps/BFS 1.08). Failures:
the greedy policy loops (35 of its 36 timeouts revisit one cell >= 20 times; sampling actions removes them,
4_obstacles goes 89 % -> 100 %), and map_simple stays at ~78 % with either policy. That was first read as the
local-minimum failure mode of Gao et al. 2025, but test 31 (masking) lifts map_simple to 98.8 % (sampled): those
failures were collisions near walls, not local minima. The event file
is 1.9 GB because of per-step logging (`TrainerConfig.log_steps`); `scripts/export_episode_metrics.py` +
`scripts/plot_episode_csv.py` analyse it from a CSV.

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
- Slippery env only: optional terrain values after the goal pair (`terrain_obs`, §2a-quater): `proprio` +4, `neighbors` +8.

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
