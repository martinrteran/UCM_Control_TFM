---
name: tune
description: Recommend hyperparameter changes for a training symptom or run number, with the RL reasoning and the TensorBoard metric to verify. Use when training underperforms.
argument-hint: "[test number] [symptom]"
---

# Tune Hyperparameters

Given a training symptom or a run number, recommend specific hyperparameter changes and explain the RL reasoning.

`$ARGUMENTS`: symptom description or test number, e.g. `24 success never rises` or `loss diverges after 5k steps`.

## How to use

1. If a run number is given, read `train/grid_training_{N}.py` for the current config.
2. Match the symptom(s) to the table below.
3. Propose a **minimal diff** — change one or two knobs at a time so the next experiment isolates the cause.
4. State what TensorBoard metric to watch to confirm the fix worked.

---

## Symptom → knob mapping

### Agent never reaches the goal (`Success/Episode ≈ 0` throughout)

| Cause | Fix |
|---|---|
| Reward too sparse (terminal only) | Add distance-shaping term: `α·(prev_dist - curr_dist)/map_diagonal`. Start with `α=2.0`. |
| Goal bonus too small relative to step cost | Increase goal bonus to ≥10× the per-step penalty. |
| Epsilon decays too fast | Increase `eps_decay`. Rule of thumb: ε should reach ~0.1 after ≈50% of total steps. |
| `eps_step` too small — trains before buffer is warm | Use `eps_step=0` (train at end of episode) for the first 10k episodes, then switch. |
| Environment resets with goal too close or too far | Check `_min_separation_distance` in `Grid_Robot_Env` (default: 5 cells). |

### Reward improves then collapses

| Cause | Fix |
|---|---|
| Target network update too frequent | Increase `target_update` (try 2000–5000 steps). |
| Learning rate too high | Halve `lr` (try 5e-4 → 2.5e-4). |
| Replay buffer too small — old data crowds out recent | Increase `buffer_size` (try 200k). |
| PPO policy ratio explodes | Decrease `clip_eps` (0.2 → 0.1) or add gradient clipping (`max_grad_norm=5.0`). |

### Loss diverges (NaN or growing unboundedly)

| Cause | Fix |
|---|---|
| Learning rate too high | Lower by 10×. |
| Reward not bounded | Clip rewards: `reward = np.clip(reward, -5.0, 10.0)`. |
| Gradient norm exploding | Lower `max_grad_norm` (default 10.0 → 1.0 for PPO, 5.0 for DQN). |
| DQN without target network soft-start | Ensure buffer is pre-filled before training (`len(buffer) >= batch_size`). |

### Agent learns to move but oscillates / circles (`Steps/Ratio` stuck >5)

| Cause | Fix |
|---|---|
| No oscillation penalty | Add: penalise alternating turn-left / turn-right actions. |
| Entropy too low (PPO) — collapses to one action | Increase `entropy_coeff` (0.01 → 0.05). |
| Too few LIDAR rays — can't distinguish corridor geometry | Increase `num_rays` (e.g. 8 → 16 or 30). |
| Network too small to represent the navigation policy | Use `GridAgentNet` (253→56) instead of `SimpleQNetwork` with small `hidden_dim`. |

### Training is very slow (good metrics but only after 100k+ episodes)

| Cause | Fix |
|---|---|
| On-policy PPO sample inefficiency | Switch to DQN/GridAgent for faster convergence in small discrete environments. |
| `eps_step` too large — too few gradient updates per episode | Reduce `eps_step` (e.g. 10 → 1, meaning update every step). |
| Batch size too small | Increase `batch_size` (64 → 256) to reduce gradient variance. |
| Single map — policy doesn't generalise, overfits to one layout | Add `map_simple.json` as a second environment in `MultiEnvironmentTrainer`. |

### Recurrent PPO (GRU) diverges or won't learn

| Cause | Fix |
|---|---|
| Hidden state not reset between episodes | Add `agent.reset_hidden()` after `env.reset()` in the training loop. |
| Sequence length too long — gradients vanish | Reduce `seq_len` in `PPORNNConfig` (try 16–32). |
| No gradient clipping on recurrent net | Set `max_grad_norm=1.0` for GRU agents — recurrent gradients are more sensitive. |
| Value and policy hidden dims mismatched | Both `GRUPolicyNetwork` and `GRUValueNetwork` must have the same `hidden_dim` for `reset_hidden()` to work. |

---

## Reference hyperparameter ranges (from project papers)

| Algorithm | lr | γ | batch | buffer | eps_decay (steps) | clip_eps | entropy_coeff |
|---|---|---|---|---|---|---|---|
| DQN / GridAgent | 1e-3 | 0.99 | 64 | 100k | 50k–200k | — | — |
| PPO (MLP) | 3e-4 | 0.99 | 64–256 | — | — | 0.1–0.2 | 0.01–0.05 |
| PPO-GRU | 1e-4 | 0.99 | 64 | — | — | 0.1 | 0.01 |
| SAC (discrete) | 3e-4 | 0.99 | 256 | 100k | — | — | α=0.2 |

Sources: Shakya et al. 2023 (RL survey); Xue et al. 2022 (NavACL-Q, SAC for intralogistics navigation).

---

## Experiment discipline

From the thesis scope (`2_estado_del_arte_propuesta.tex`): each experiment must document all changed hyperparameters explicitly. When proposing a change:
- Change **one variable at a time** unless two are causally linked (e.g. `lr` + `batch_size`).
- Run for at least 20k episodes before concluding.
- Compare with `python scripts/compare_runs.py {N-1} {N} --tail 0.25`.

---

## Embedded tuning workflow (copy-paste pattern)

### Step 1 — inspect what the previous run used

```bash
python scripts/inspect_checkpoint.py checkpoints/Test/{N-1}/
```

Output includes the full config dict so you can see every hyperparameter that was active. Use this as the baseline for your change.

### Step 2 — make ONE change in the training script

Minimal diff for the most common tuning scenario (epsilon decay too fast):

```python
# BEFORE
config = RLConfig(
    ...,
    eps_decay=50_000,   # ← too fast for 50k episodes
)

# AFTER
config = RLConfig(
    ...,
    eps_decay=200_000,  # ε reaches ~0.1 after ~50% of steps
)
```

### Step 3 — verify the fix is working early (at ~5k episodes)

```python
# In a Jupyter cell or quick script — read mid-training TensorBoard data
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
acc = EventAccumulator("runs/Test/25/GridAgent")
acc.Reload()

eps = acc.Scalars("Policy/Epsilon/Episode")
print(f"Epsilon at episode 5000: {eps[4999].value:.4f}")   # should be ~0.6 if eps_decay=200k
success = acc.Scalars("Success/Episode")
last_100 = [e.value for e in success[-100:]]
print(f"Success rate (last 100 ep): {sum(last_100)/len(last_100):.3f}")
```

### Step 4 — final comparison

```bash
# Generates markdown table for the thesis
python scripts/compare_runs.py {N-1} {N} --tail 0.25
```

Expected output format:
```
| Run                   | Reward (mean)   | Success rate   | Steps ratio    |
|-----------------------|-----------------|----------------|----------------|
| Test 24 / GridAgent   | 41.234 ± 8.123  | 0.712 ± 0.091  | 1.821 ± 0.512  |
| Test 25 / GridAgent   | 52.100 ± 5.400  | 0.891 ± 0.045  | 1.213 ± 0.187  |
```
