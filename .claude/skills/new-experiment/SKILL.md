---
name: new-experiment
description: Scaffold the next numbered training script train/grid_training_N.py from the most recent one. Use when starting a new experiment.
argument-hint: "[what changes vs previous experiment]"
---

# New Training Experiment

Scaffold the next numbered training experiment for this project.

## Steps

1. **Determine the next experiment number**: glob `train/grid_training_*.py`, parse the numbers, find max N, the new script is N+1. Ignore files with non-numeric suffixes (e.g. `_GRU`, `copy`).

2. **Read the most recent numbered script** to understand the current configuration (map, robot, env class, agent, network, hyperparams, TrainerConfig).

3. **Ask the user** (if not already specified in `$ARGUMENTS`): what changes from the previous experiment? Examples: different reward function, new map, different network, different algorithm, changed hyperparameters.

4. **Create `train/grid_training_{N+1}.py`** based on the previous script with the requested changes applied. Follow these conventions from the codebase:
   - Device selection: `torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")`
   - Load maps with: `GridMap.load("./src/tfm_control_ucm/maps/<map>.json")`
   - Trainer paths: `"./runs/Test/{N+1}"` and `"./checkpoints/Test/{N+1}"`
   - Agent name in config matches the TensorBoard label
   - `lidar_config` dict uses keys: `num_rays`, `max_range`, `fov`, `noise_std`, `with_cache`, `device`

5. **Remind the user** to commit under the naming convention used in recent git log entries (e.g. `Test {N+1} - <short description>`).

## Notes

- Maps available: `map_obstacle_free.json` (empty), `map_simple.json` (obstacles)
- Env classes in `environment.py`: `Grid_Robot_Env`, `Grid_Robot_Sections_Env`
- Agents: `GridAgent` (DQN/DDQN), `PPOAgent`, `PPOGRUAgent`, `RecurrentPPOAgent`, `SACDiscreteAgent`
- Networks: `GridAgentNet`, `SimpleQNetwork`, `PolicyNetwork`/`ValueNetwork`, GRU variants
- `TrainerConfig` requires `num_episodes > 1000`; `eps_step=0` trains at end-of-episode, `eps_step=N` trains every N steps
