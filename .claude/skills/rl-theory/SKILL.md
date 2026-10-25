---
name: rl-theory
description: Explain an RL concept, formula or algorithm decision (GAE, PPO clipping, double DQN, entropy bonus, etc.) in the context of this codebase.
argument-hint: "<concept or question>"
---

# RL Theory

Explain an RL concept, formula, or algorithm decision in the context of this codebase.

`$ARGUMENTS` is the concept or question, e.g.:
- `GAE` / `gae_lambda`
- `why PPO clips the ratio`
- `double DQN vs DQN`
- `entropy bonus`
- `why reset hidden state between episodes`
- `temporal difference error`
- `replay buffer decorrelation`

## How to answer

1. Give the mathematical definition first — use the notation from the project's reference papers.
2. Show exactly where and how it appears in this codebase (file:line references).
3. Connect it to the training metrics logged in TensorBoard so the user knows what to watch.
4. Cite the relevant paper from the project's library when applicable.

---

## Core RL concepts and their codebase locations

### Bellman equation and TD error
```
δ_t = r_t + γ · V(s_{t+1}) · (1 - done_t) - V(s_t)
```
Used in GAE and A2C/PPO critic update. In code: `compute_gae()` in [base_rl_agent.py](src/tfm_control_ucm/core/agents/base_rl_agent.py).

### DQN Q-target
```
y_t = r_t + γ · (1 - done_t) · max_{a'} Q_target(s_{t+1}, a')
```
Implemented in `GridAgent._compute_loss()` and `DQNAgent._compute_loss()` in [agent.py](src/tfm_control_ucm/core/agents/agent.py). Loss: Smooth L1 (Huber) between current Q and target Q.

### Double DQN (DDQN)
Standard DQN overestimates Q-values because the same network selects and evaluates the next action.
Double DQN decouples selection (policy net) from evaluation (target net):
```
y_t = r_t + γ · Q_target(s_{t+1},  argmax_{a'} Q_policy(s_{t+1}, a'))
```
Controlled by `double_dqn: bool` in `RLConfig`. In code: the `if self.double_dqn:` branch in `GridAgent._compute_loss()`.
Reference: Shakya et al. 2023, §3.1 on DQN improvements.

### Epsilon-greedy exploration and decay
```
ε_t = ε_end + (ε_start - ε_end) · exp(-t / eps_decay)
```
`t` = `global_step` (incremented every `select_action` call, not every episode).
Watch: `Policy/Epsilon/Episode` in TensorBoard. If ε stays high after 10k episodes, `eps_decay` is too large relative to the total steps budget.
In code: `GridAgent.select_action()` in [agent.py](src/tfm_control_ucm/core/agents/agent.py).

### Generalized Advantage Estimation (GAE)
```
Â_t = Σ_{l=0}^{T-t} (γλ)^l · δ_{t+l}
δ_t = r_t + γ · V(s_{t+1}) · (1 - done_t) - V(s_t)
```
`λ` = `gae_lambda` in `PPOConfig`. High λ (→1) = low bias, high variance. Low λ (→0) = one-step TD, low variance, higher bias.
Returns: `advantages = GAE`, `returns = advantages + V(s)` (used as critic target).
In code: `PPOGRUAgent.compute_gae()` and `RecurrentPPOAgent.buffer.build_sequences()` in [agent.py](src/tfm_control_ucm/core/agents/agent.py); standalone `compute_gae()` in [base_rl_agent.py](src/tfm_control_ucm/core/agents/base_rl_agent.py).

### PPO clipped surrogate objective
```
L^{CLIP} = E_t[ min(r_t · Â_t,  clip(r_t, 1-ε, 1+ε) · Â_t) ]
r_t = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)   (probability ratio)
```
`ε` = `clip_eps` in `PPOConfig` (default 0.2). The clip prevents the policy from changing too much in one update — the "trust region" idea without second-order optimization.
Full loss: `L = L^CLIP + c_v · L^value - c_e · H(π)` where `c_v` = `value_coeff`, `c_e` = `entropy_coeff`.
In code: `PPOAgent._compute_loss()` in [agent.py](src/tfm_control_ucm/core/agents/agent.py).
Reference: Shakya et al. 2023 §3.3; Murphy 2025 §3.3.

### Entropy bonus
```
H(π) = -Σ_a π(a|s) log π(a|s)
```
Added to the PPO loss as `- entropy_coeff · H(π)`. Encourages exploration by penalising confident (low-entropy) policies.
If the agent collapses to always choosing one action, `entropy_coeff` is too low.
Watch: `last_entropy` attribute on the agent after each `train_step`.
In code: `PPOAgent._compute_loss()` — `entropy = dist.entropy().mean()`.

### Experience replay and decorrelation
Storing transitions in a circular buffer and sampling randomly breaks temporal correlations between consecutive frames — sampling correlated sequences would make gradient updates non-i.i.d. and destabilise training.
Buffer in code: `ReplayBuffer` (deque-based) in [agents/utils.py](src/tfm_control_ucm/core/agents/utils.py). Capacity = `buffer_size` (default 100 000). Learning starts when `len(buffer) >= batch_size`.
Reference: Shakya et al. 2023 §3.1 (Experience Replay).

### Target network
A second copy of the Q-network held fixed for `target_update` steps. Without it, the Bellman target and the predicted Q-value move together — the "deadly triad" instability.
Updated by hard copy: `target_net.load_state_dict(policy_net.state_dict())`.
In code: `GridAgent.update_target_network()`. Frequency = `target_update` in `RLConfig`.
Soft update (Polyak averaging) is used in `SACDiscreteAgent`: `θ_target ← τ·θ + (1-τ)·θ_target` via `BaseRLAgent.soft_update()`.

### Recurrent hidden state and episode boundaries
GRU/LSTM agents (PPOGRUAgent, RecurrentPPOAgent) carry `h_policy` and `h_value` across steps. At episode reset the hidden state **must** be zeroed — otherwise information leaks between episodes and the policy conditions on phantom history.
In code: `agent.reset_hidden()` should be called at the start of each episode in the training loop. Check the relevant `train/` script to confirm it is called after `env.reset()`.

### Partial observability in this environment
The observation `obs = [lidar_0, ..., lidar_{N-1}, dist_to_goal, angle_to_goal]` is a compressed local view. Global map position, full obstacle layout, and velocity are all hidden. This is a POMDP (Partially Observable MDP).
Formal treatment: Murphy 2025 §1.3.4. Practical implication: a recurrent policy can help because it can integrate history to infer unobserved state — but this is only necessary if the observable information genuinely differs across similar-looking states.

### Steps/Ratio metric (SPL equivalent)
```
Steps_Ratio = actual_steps / optimal_BFS_steps
```
Computed per episode by the environment (`reset()` info `optimal_steps`, `PathPlanner.min_steps()` wavefront BFS) and logged by `BaseTrainer._log_episode()`.
Equivalent to SPL (Success weighted by Path Length) from Gao et al. 2025, Eq. 7:
```
SPL = (1/N) Σ S_i · p̄_i / max(p_i, p̄_i)
```
Values close to 1 on successful episodes mean near-optimal navigation.

---

## Algorithm pseudocode (embedded from project papers)

### DQN training loop (Shakya et al. 2023, §3.1 — adapted for this codebase)

```
Initialize policy_net Q_θ and target_net Q_θ' with θ' ← θ
Initialize replay buffer D with capacity buffer_size
ε ← ε_start

For each episode:
  s ← env.reset()
  For each step t:
    a ← argmax Q_θ(s) with prob (1−ε), else random   # select_action()
    s', r, done ← env.step(a)
    D.add(s, a, r, s', done)                          # store()
    ε ← ε_end + (ε_start−ε_end)·exp(−global_step / eps_decay)

    if len(D) ≥ batch_size:                            # train_step()
      (s_b, a_b, r_b, s'_b, d_b) ← D.sample(batch_size)

      if double_dqn:
        a* ← argmax_{a'} Q_θ(s'_b, a')               # policy net selects
        y  ← r_b + γ·(1−d_b)·Q_θ'(s'_b, a*)         # target net evaluates
      else:
        y ← r_b + γ·(1−d_b)·max_{a'} Q_θ'(s'_b, a')

      L ← SmoothL1(Q_θ(s_b, a_b), y)
      θ ← θ − α·∇L  (with gradient clipping, max_grad_norm)

    if global_step % target_update == 0:
      θ' ← θ                                          # hard update
```

### PPO training loop (Shakya et al. 2023, §3.3 — adapted for this codebase)

```
Initialize policy π_θ and value V_φ
π_old ← π_θ

For each episode:
  Collect trajectory τ = {s_t, a_t, r_t, V(s_t)} using π_θ
  Compute GAE advantages Â_t and returns R_t  (compute_gae)

  For K update epochs:
    For each mini-batch:
      r_t ← π_θ(a_t|s_t) / π_old(a_t|s_t)           # probability ratio

      L^CLIP ← E[min(r_t·Â_t, clip(r_t, 1−ε, 1+ε)·Â_t)]
      L^VF   ← MSE(V_φ(s_t), R_t)
      L^H    ← H(π_θ)  (entropy bonus)

      L ← −L^CLIP + value_coeff·L^VF − entropy_coeff·L^H
      (θ, φ) ← (θ, φ) − α·∇L  (with gradient clipping)

  π_old ← π_θ
```

### Reward decomposition (Gao et al. 2025, ICRA — *Hierarchical RL for Safe Mapless Navigation*)

The paper's reward design (closest to this project's setup):

```
r_total = r_arrival + r_step + r_dist + r_out

r_arrival = +100   if dist_to_goal < threshold
r_step    = -1     per step (encourages efficiency)
r_dist    = μ · Δd_{A*}   where Δd = prev_dist − curr_dist, μ=2.0
r_out     = -20    on obstacle collision
```

This matches the template in `/reward-design`. The `μ·Δd` term is the distance-shaping component; it should be normalized by the map diagonal to keep it bounded.

### Xue et al. 2022 — Key hyperparameters for mapless AGV navigation

From *applsci-12-03153-v2.pdf* (Table 3 in the paper — DRL + curriculum learning):

| Parameter | Value used | Effect |
|---|---|---|
| LIDAR rays | 30 | Full 360° coverage at 12° intervals |
| Hidden units | 256 → 256 | Larger than `GridAgentNet` (253→56) |
| Replay buffer | 1 000 000 | 10× larger than project default |
| Batch size | 256 | 4× larger than project default |
| Learning rate | 1e-4 | More conservative than project default |
| Curriculum stages | 3 | Easy → medium → full map |
| Episode timeout | 500 steps | Similar to this project |

Note: their environment uses continuous LIDAR (not top-N sorted), no rotation actions (holonomic robot). The large buffer and batch size matter more for their denser environments.
