# Debug session — 2026-10-24

**Scope**: diagnosed "what's wrong with the RL code and how to improve it" against
the mid-refactor working tree on branch `Put-all-together` (latest commit
`c849f55` "Teest 24 - Sim 23 - Change Name Maps").

Supplements `PROJECT_BRIEFING.md`; does not replace it. Branch state (uncommitted
reward-injection refactor, dead env attributes, commented-out `MultiEnvironmentTrainer`,
etc.) still described there — read it first.

---

## 1. Only recent run that actually produced TensorBoard output: `runs/Test/10_v2`

Script: `train/grid_training_10 copy.py` (untracked). First script to use the
new injected-`reward_fn` API end-to-end.

Checkpoint: `checkpoints/Test/10_v2/Grid_Agent/final.pth`
- 50 000 episodes, **38 496 189** global steps consumed
- `obs_dim = 10` (num_sections = 4), `action_dim = 3` → **`GridRobot`** (egocentric)
- `lr = 1e-3`, `gamma = 0.99`, `eps_start=1.0 → eps_end=0.005`, `eps_decay = 1 000 000`
- `double_dqn = False`, `target_update = 1000`, `buffer_size = 100 000`
- Policy net: `GridAgentNet` 10 → 253 → 56 → 3

### Metrics (TB scalars, last 20 % of training)

| Tag | Value |
|---|---|
| `Success/Episode` | **~1.1 %** (random-walk baseline — did not learn) |
| `Steps/Ratio/Episode` | **~28×** optimal (optimal ≈ 36, agent ≈ 824) |
| `Reward/Total` | **−75** overall, **−59** mean over all episodes → **got worse** |
| `Distances/Change` | **−2.6** (net 2.6 closer out of ~28.6 start distance) |
| `Policy/Epsilon` | **0.005** (fully decayed) |
| `Loss/Episode` | ~0.015, flat (converged to a bad policy) |

### Root causes

1. **Partial observability × egocentric actions.** `GridRobot` has 3 actions
   (forward / turn-left / turn-right) but **orientation is not in the observation**
   (obs is sectored LIDAR + `[dist_to_goal, angle_to_goal]`). Plain MLP DQN cannot
   solve this — a true POMDP. The project already discovered this; **all PPO
   experiments from test 19 onward use `GridRobotNotTurning` (4 absolute N/S/E/W
   actions)**.
2. **Reward balance is upside-down.** Inline `reward_fn` in that script has
   `progress ≤ ~0.15 / step`, `step_cost = −0.05`, `proximity_penalty ∈ [0, −1.0]`,
   `goal = +20`, `hit_obstacle = −5`. Over a 1000-step episode the accumulated
   penalty (≈ −50) **exceeds the goal bonus (+20)**, so a wall-hit (−5) is
   actually cheaper than searching — policy learned to shorten episodes by
   hitting walls. The negative `Reward/Total` trajectory (−59 → −75) is
   consistent with that.
3. **Epsilon decay far too fast for the budget.** `eps_decay = 1e6` on a run that
   consumed **38e6** global steps → epsilon reached the floor after ~2 % of
   training; the remaining 98 % was greedy exploitation of a half-trained Q.

---

## 2. `train/grid_training_24.py` — currently BLOCKED

Confirmed by reading the file (not just the briefing).

### Blocker (line 53)

```python
env = grid_env.Grid_Robot_Sections_Env_MultiMap(
    maps=[mapa_0, mapa_1, mapa_4, mapa_simple],
    robot=robot_simple, cell_size=10,
    max_iteration_steps=args.max_steps,
    num_sections=n_sections, lidar_config=lidar_config,
    render_mode="human"
)
```
No `reward_fn=` kwarg. `environment.py:384` makes it keyword-only required.
Immediate `TypeError`.

Lines 55–57 are also dead:
```python
env._max_proximity_penalty = args.max_proximity_penalty
env._safety_margin = env.lidar.max_range * args.safety_margin_factor
env._step_penalty = args.step_penalty
```
Nothing reads these under the new API.

### Other issues worth fixing while you're there

| Where | Issue | Fix |
|---|---|---|
| line 23 — `lr=1e-3` | Too hot for PPO (standard 3e-4). Plausible cause of test 22 plateau. | Drop default to 3e-4. |
| line 20 — `max_steps=400` | Tight on `map_4_square_obstacle`; truncation may dominate. | Verify from logs; raise to 600–800 if truncations > 30 %. |
| line 74 — `policy_net = SimpleQNetwork` | Named Q, used as PPO policy head emitting logits. Works but confusing. | Rename or use `PolicyNetwork`. |
| line 79 — `eps_step=4` | DQN concept; `PPOTrainer` does full-episode rollouts and ignores it. | Remove from PPO configs. |
| lines 35, 37 — `type=bool` argparse | `--render false` parses to `True`. Classic trap. | Use `action='store_true'`. |
| line 53 — `render_mode="human"` hardcoded | Harmless (`PPOTrainer` only calls `env.render()` when `t_config.render=True`), but inconsistent. | Make it `"human" if args.render else None`. |
| no `--seed` | PPO runs are high-variance; without seeds, can't attribute changes vs. luck. | Add `--seed`, apply to torch/np/env RNG. |
| multi-map random per `reset()` | Episode curves become un-interpretable (map-dependent `Steps/Min`). | Log `info["map_name"]` per episode; emit per-map TB tags or use round-robin. |

---

## 3. Action plan (ranked)

1. **Extract `reward_fn` into a reusable module.** Promote the function from
   `train/grid_training_10 copy.py` → `src/tfm_control_ucm/core/grid_env/rewards.py`
   as `make_default_reward(**knobs)` returning a closure. The three CLI knobs
   already in `grid_training_24.py` become its arguments.
2. **Wire it into `grid_training_24.py`:** import `make_default_reward`,
   construct a `reward_fn` from args, pass as `reward_fn=` to env ctor, delete
   dead `env._*` assignments.
3. **Fix `scripts/eval_agent.py` identically.**
4. **Rebalance reward before any run.** Rule of thumb: `goal_reward ≥ 10 × max_steps × step_penalty`.
   For `max_steps=400`, `step_penalty=0.05` that's `goal_reward ≥ 200` (currently 20).
   Alternative: cut `step_penalty` to 5e-3.
5. **Hyperparameter hygiene on 24:** `lr=3e-4` default, add `--seed`, fix
   `type=bool` traps, per-map TB tags.
6. **Only then run as `test 25`** (new number — do not overwrite 24's dirs).
   Suggested order: single map (`map_obstacle_free`) sanity run → `map_simple`
   → full multi-map.

### Smaller follow-ups (not blocking)

- Add `Done/Reason` scalar (0 = success / 1 = hit / 2 = timeout) to
  `BaseTrainer._inner_training` and the three PPO trainers so failure modes are
  distinguishable in TB.
- `Grid_Robot_Env` and `Grid_Robot_Sections_Env` don't call Gym `super().__init__()`
  and don't declare `observation_space` / `action_space`. Fine for hand-rolled
  trainers; will trip wrappers (`RecordEpisodeStatistics`, `AsyncVectorEnv`, SB3)
  if you ever use them.

---

## 4. Open questions for next session

- Does the thesis scope require a **DQN** result (per `2_estado_del_arte_propuesta.tex`:
  "DQN vs PPO")? If yes, we also need a DQN rerun with `GridRobotNotTurning` +
  rebalanced reward + corrected `eps_decay` (≈ 5e6 or episode-indexed schedule).
- Multi-map (test 24) or single-map first? Multi-map amplifies variance; a
  single-map success baseline makes multi-map a cleaner generalization test.
- Keep the inline `reward_fn` in `train/grid_training_10 copy.py` or delete the
  "copy" file once a module version exists?
