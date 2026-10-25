---
name: analyze-run
description: Analyze the results of a training run (TensorBoard logs in runs/Test/N): reward, success rate, steps ratio, loss and convergence. Use when the user asks how a run went or to review a test number.
argument-hint: "[test number]"
---

# Analyze Training Run

Analyze the results of a training run from this project.

`$ARGUMENTS` should be a test number (e.g. `14`) or left empty to analyze the most recent run.

## Steps

1. **Find the run directory**: `runs/Test/{N}/`. If no N given, list `runs/Test/` and pick the highest-numbered folder.

2. **List what was trained**: glob `runs/Test/{N}/**/events.out.tfevents.*` — each subdirectory name is the agent name logged from `agent.config.name`.

3. **Read the corresponding training script** `train/grid_training_{N}.py` to understand the configuration (algorithm, hyperparams, maps, `TrainerConfig`).

4. **Report what TensorBoard metrics were logged** (from `BaseTrainer._log_episode` in `TrainingManager.py`, same for every algorithm since test 25; older PPO runs used `Done/Success`, `Steps/Ratio`, `Done/Reward/Episode`, which `plot_runs.py`/`compare_runs.py` read as aliases):
   - `Loss/Episode` or `Loss/Global step`
   - `Policy/Epsilon/Episode`
   - `Steps/Episode`, `Steps/Min/Episode`, `Steps/Ratio/Episode` (actual steps ÷ optimal BFS steps — lower is better)
   - `Reward/Total`, `Reward/Avg`
   - `Success/Episode` (1 = reached goal, 0 = truncated/hit wall)
   - `Distances/Start`, `Distances/End`, `Distances/Change`

5. **Check for a saved checkpoint**: `checkpoints/Test/{N}/*/final.pth`. If it exists, load it with `torch.load(..., weights_only=False)` and report the saved keys (epsilon, global_step, episode_count, algorithm).

6. **Summarize** what was explored in this run, what the key hyperparameter differences were from the prior experiment, and (if the user asks) suggest what to try next.

## Useful scripts

```bash
python scripts/plot_runs.py {N}           # visualise all metrics for one run
python scripts/compare_runs.py {N} {M}   # side-by-side stats table
python scripts/inspect_checkpoint.py checkpoints/Test/{N}/
python scripts/eval_agent.py checkpoints/Test/{N}/{AgentName}/final.pth --episodes 30
```

### `plot_runs.py` — what it reads and plots

The script reads these exact TensorBoard tags (from `BaseTrainer._log_episode` in `TrainingManager.py`, same for every algorithm since test 25; older PPO runs used `Done/Success`, `Steps/Ratio`, `Done/Reward/Episode`, which `plot_runs.py`/`compare_runs.py` read as aliases):

```python
METRICS = {
    "Reward/Total":           "Total reward / episode",
    "Success/Episode":        "Success rate (1 = reached goal)",
    "Steps/Ratio/Episode":    "Steps / optimal steps (lower = better)",
    "Loss/Episode":           "Training loss",
    "Policy/Epsilon/Episode": "Epsilon (exploration)",
}
```

Key function — loads a single agent's event file:

```python
def load_run(run_dir: Path) -> dict[str, tuple[list, list]]:
    acc = EventAccumulator(str(run_dir))
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    result = {}
    for tag in METRICS:
        if tag in tags:
            events = acc.Scalars(tag)
            steps = [e.step for e in events]
            vals  = [e.value for e in events]
            result[tag] = (steps, vals)
    return result
```

Smoothing applies a rolling average over a window (default 50 episodes). Faint raw data is overlaid. Output goes to `plots/run_comparison.png` with `--save`.

### `compare_runs.py` — thesis table generation

Computes `mean ± std` over the **last N% of training** (converged phase). Output is a plain-text table plus a Markdown table ready to paste into the thesis:

```
| Run              | Reward (mean)  | Success rate   | Steps ratio    |
|------------------|----------------|----------------|----------------|
| Test 21 / GRU    | 48.234 ± 5.123 | 0.823 ± 0.071  | 1.421 ± 0.312  |
| Test 24 / GridAg | 52.100 ± 3.400 | 0.891 ± 0.045  | 1.213 ± 0.187  |
```

### `inspect_checkpoint.py` — what it prints

```
──────────────────────────────────────────────────────────────
  checkpoints/Test/24/GridAgent/final.pth
──────────────────────────────────────────────────────────────
  algorithm            = grid_agent
  epsilon              = 0.0501
  global_step          = 847320
  episode_count        = 50000
  config:
    obs_dim              = 32
    action_dim           = 3
    lr                   = 0.001
    double_dqn           = True
  policy_state_dict layers:
    fc1.weight: (253, 32)
    fc2.weight: (56, 253)
    fc3.weight: (3, 56)
  file size            = 1.24 MB
```

## Interpreting metrics

- `Steps/Ratio/Episode` trending toward 1.0 means the agent is approaching optimal path length — the primary success signal.
- `Success/Episode` should rise over training; if it stays near 0, the reward signal or exploration is likely broken.
- `Distances/Change` (start_dist − end_dist) being negative means the agent moves away from the goal.
- Epsilon stuck above ~0.5 after many episodes suggests `eps_decay` is too high or `global_step` is not incrementing.
- Q-values all collapsing to one action → check reward shaping, learning rate, or double-DQN flag.

## Literature context

The thesis scope (from `Propuestas_revision/2_estado_del_arte_propuesta.tex`):
- Comparing **DQN vs PPO** on a 2D grid with static obstacles and discrete actions.
- Does NOT cover: multi-agent, physical validation, dynamic obstacles, or safety guarantees.
- Primary metrics: success rate + steps ratio (actual / BFS-optimal). Reward is secondary.

Gao et al. 2025 (*Hierarchical RL for Safe Mapless Navigation*, ICRA) identifies **local minima** as the main failure mode for LIDAR-based reactive policies. If `Success/Episode` stays near zero after many episodes, the robot is likely stuck in a geometry local minimum — consider adding a stuck-detection mechanism or a hierarchical sub-goal layer.

The `Steps/Ratio` metric is equivalent to the SPL (Success weighted by Path Length) metric used in Gao et al. 2025 (Eq. 7). Values close to 1.0 indicate near-optimal paths on successful episodes.
