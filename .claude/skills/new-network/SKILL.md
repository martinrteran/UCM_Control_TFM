---
name: new-network
description: Add a new PyTorch network architecture to agents/networks.py following the project's conventions and observation space. Use when a new network type is needed.
argument-hint: "<network type and purpose>"
---

# New Neural Network

Add a new PyTorch network architecture to `src/tfm_control_ucm/core/agents/networks.py`.

`$ARGUMENTS` should describe the network type and purpose (e.g. `dueling DQN head`, `attention over LIDAR rays`, `GRU policy network`).

## Steps

1. **Read `agents/networks.py`** to understand existing architectures and naming conventions.

2. **Design the network** for this project's observation space:
   - Input shape: `(batch, obs_dim)` where `obs_dim = num_rays * 2 + 2`
   - Each LIDAR ray contributes `[distance, angle]`; last two values are `[dist_to_goal, angle_to_goal]`
   - Actions: discrete (3 by default — forward, turn left, turn right)

3. **Implement the class** following these conventions from the file:
   - Class-level `name` string attribute (used for checkpointing/logging)
   - Accept `obs_dim` and `action_dim` (and optionally `hidden_dim`, `num_layers`) as constructor params
   - Implement `reset_parameters()` if the network will be used with `GridAgent` (which calls it on soft resets)
   - For recurrent networks: `hidden_dim` attribute on the class (required by `PPOGRUAgent.reset_hidden()` and `RecurrentPPOAgent.reset_hidden()`)
   - For GRU/LSTM: forward signature `(x, h) -> (output, h_new)` where x is `(batch, seq, features)` or `(batch, features)` for step-by-step inference

4. **PyTorch best practices** for this codebase:
   - Use `nn.Sequential` for simple feed-forward stacks
   - Activation: ReLU throughout (matches existing `GridAgentNet`, `SimpleQNetwork`)
   - Policy heads: return **logits** (not softmax probabilities) — `PPOAgent` and `PPOGRUAgent` pass logits to `torch.distributions.Categorical(logits=...)`
   - Value heads: return raw scalar `(batch, 1)` — no activation
   - Dueling DQN: split final layer into `value_stream` (→1) and `advantage_stream` (→action_dim); combine as `V + A - A.mean()`

5. **Add the class** to `networks.py` and export it from `agents/__init__.py` if needed.

6. **Show usage** — a minimal example instantiating the network with the env's obs/action dims and verifying `forward()` output shape with a random batch.

## Literature-informed architecture choices

From the papers in the project library:

**Partial observability and recurrence** (Murphy 2025, *RL Overview*, §1.3.4): sectored LIDAR + goal angle/distance is a compressed observation — the robot cannot distinguish all global states. A GRU or LSTM policy helps when temporal context matters (e.g. detecting whether the robot is stuck). The project already has `PPOGRUAgent` and `RecurrentPPOAgent`; their networks require `(batch, seq, features)` input and expose a `hidden_dim` attribute.

**Dueling DQN** (referenced in Gao et al. 2025 as "Dueling Double DQN"): splits the final layer into a value stream V(s) → scalar and an advantage stream A(s,a) → (action_dim,), combined as `V + A − mean(A)`. Reduces variance in Q-value estimation; useful when many actions have similar value.

**Obstacle encoding** (Gao et al. 2025): they sort LIDAR returns by distance and take the top-N closest obstacles, then encode each as `[distance, angle]`. This matches the observation format in `Grid_Robot_Env._get_observation()` exactly — the top-N by *farthest* distance is taken there, so confirm the sort direction when debugging navigation near obstacles.

## Architecture reference

| Network | Use with | Output |
|---|---|---|
| `GridAgentNet` | `GridAgent` (DQN) | Q-values `(batch, action_dim)` |
| `SimpleQNetwork` | `DQNAgent` | Q-values `(batch, action_dim)` |
| `PolicyNetwork` | `PPOAgent`, `A2CAgent` | Logits `(batch, action_dim)` |
| `ValueNetwork` | `PPOAgent`, `A2CAgent` | Value `(batch, 1)` |
| GRU variants | `PPOGRUAgent`, `RecurrentPPOAgent` | `(output, hidden)` |
