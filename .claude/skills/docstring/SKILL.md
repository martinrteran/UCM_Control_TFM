---
name: docstring
description: Write or improve NumPy-style docstrings for a function, method or class in this project. Use when asked to document code.
argument-hint: "[file path or symbol]"
---

# Docstring

Write or improve docstrings for the selected code (function, method, or class).

`$ARGUMENTS` is optional: a file path or symbol name to target. Without arguments, act on the active selection or the most recently discussed code.

## Style rules for this project

Use **NumPy docstring style** — it is standard for scientific Python and matches what tools like Sphinx and VSCode render best.

```python
def example(param1: int, param2: np.ndarray, optional: float = 1.0) -> dict:
    """
    One-line imperative summary (no period).

    Optional extended description. Explain *why* the method does what it does,
    not *what* it does (the signature and body already show that). Include
    mathematical notation when it clarifies the RL algorithm, e.g.:

        Q(s, a) = r + γ · max_{a'} Q(s', a')

    Parameters
    ----------
    param1 : int
        Description of the parameter.
    param2 : np.ndarray of shape (obs_dim,)
        Include shape for tensors and arrays.
    optional : float, optional
        Description. Default 1.0.

    Returns
    -------
    dict
        Keys: ``loss`` (float), ``q_mean`` (float).

    Raises
    ------
    ValueError
        When the buffer has fewer samples than batch_size.

    Notes
    -----
    Algorithm-specific context goes here: e.g. why double-DQN is used,
    what the GAE formula is, why the clipping bound is applied.
    """
```

## What to document (and what not to)

**Always document:**
- Public methods on `BaseRLAgent` subclasses (`select_action`, `train_step`, `_compute_loss`, `store`, `save`, `load`)
- `__init__` for every agent and network class — list every hyperparameter, its type, and its effect
- Reward functions (parameters are unusual; document them all)
- `TrainerConfig` and `RLConfig` dataclass fields

**Keep short (one-liner only):**
- Private helpers that are obvious from the name (`_to_tensor`, `_clip_grad_norm`)
- Properties that just return a stored value

**Never write:**
- Comments that restate the code (`# increment step counter`)
- Docstrings on `__repr__`

## Project-specific type conventions

| Type hint | Meaning in this codebase |
|---|---|
| `np.ndarray of shape (obs_dim,)` | Flat observation from `env.get_observation_dim()` |
| `np.ndarray of shape (N, 2)` | LIDAR scan: each row is `[distance, angle]` |
| `torch.Tensor of shape (B, obs_dim)` | Batched states from replay buffer |
| `RLConfig \| PPOConfig \| PPORNNConfig` | Config dataclass — say which subclass is required |
| `BaseRLAgent` | Accept any subclass; note if a specific interface is required |

## RL math notation to use

Prefer standard notation consistent with the project's reference papers (Shakya et al. 2023, Murphy 2025):

- Policy: `π(a | s)` or `π_θ`
- Q-function: `Q(s, a)` or `Q_θ`
- Value function: `V(s)` or `V_φ`
- Advantage: `A(s, a) = Q(s, a) - V(s)`
- GAE: `Â_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}` where `δ_t = r_t + γV(s_{t+1}) - V(s_t)`
- PPO clip: `L^{CLIP} = E[min(r_t A_t, clip(r_t, 1-ε, 1+ε) A_t)]`
- DQN target: `y = r + γ(1-done) · max_{a'} Q_target(s', a')`
- Double DQN target: `y = r + γ(1-done) · Q_target(s', argmax_{a'} Q_policy(s', a'))`

## Steps

1. Read the target function/class in full (if not already in context).
2. Identify the algorithm-specific logic that a future reader needs explained (not just the parameter types).
3. Write the NumPy-style docstring. Include math notation for any loss or update formula.
4. Apply the edit — do not add a trailing summary or ask for confirmation unless the behavior is genuinely ambiguous.
5. If the class has multiple public methods, offer to document them all in one pass.

---

## Worked Examples

### Example 1 — `compute_gae` in `base_rl_agent.py`

**BEFORE** (current state: only inline parameter list, no descriptions, no math):

```python
def compute_gae(rewards, values, dones, gamma, lam):
    """
    rewards: np.array [T]
    values: np.array [T+1] (incluye valor del estado final)
    dones: np.array [T] (bool)
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * (0.0 if dones[t] else values[t + 1]) - values[t]
        gae = delta + gamma * lam * (0.0 if dones[t] else gae)
        advantages[t] = gae
    returns = advantages + values[:-1]
    return advantages, returns
```

**AFTER** (proper NumPy docstring with math, shapes, and algorithm context):

```python
def compute_gae(rewards, values, dones, gamma, lam):
    """
    Compute Generalized Advantage Estimation (GAE) for a single episode.

    GAE trades off bias vs. variance in the advantage estimate by combining
    multi-step TD errors with an exponential weighting controlled by λ:

        δ_t = r_t + γ · V(s_{t+1}) · (1 - done_t) - V(s_t)
        Â_t = Σ_{l=0}^{T-t} (γλ)^l · δ_{t+l}

    λ→1 recovers Monte-Carlo returns (low bias, high variance).
    λ→0 recovers one-step TD (high bias, low variance).
    Returns are used as the critic regression target: R_t = Â_t + V(s_t).

    Parameters
    ----------
    rewards : np.ndarray of shape (T,)
        Rewards collected in the episode.
    values : np.ndarray of shape (T+1,)
        Value estimates V(s_t) for t=0..T; the extra entry V(s_T) is the
        bootstrap value of the terminal or truncated state.
    dones : np.ndarray of shape (T,), dtype bool
        True when the episode ended at step t (stops GAE accumulation).
    gamma : float
        Discount factor (typically 0.99).
    lam : float
        GAE smoothing parameter λ from `PPOConfig.gae_lambda` (typically 0.95).

    Returns
    -------
    advantages : np.ndarray of shape (T,)
        Normalised advantage estimates Â_t.
    returns : np.ndarray of shape (T,)
        Critic regression targets R_t = Â_t + V(s_t).

    Notes
    -----
    Used by `PPOAgent.train_step()` and `PPOGRUAgent.train_step()`.
    Reference: Schulman et al. 2016 (GAE paper); Shakya et al. 2023 §3.3.
    """
```

---

### Example 2 — `GridAgent.select_action` in `agent.py`

**BEFORE** (one-liner, no formula, no exploration schedule):

```python
def select_action(self, state: np.ndarray):
    """Select action using epsilon-greedy policy."""
    self.global_step += 1
    self.epsilon = self.eps_end + (self.eps_start - self.eps_end) * np.exp(
        -1.0 * self.global_step / self.eps_decay
    )
    ...
```

**AFTER**:

```python
def select_action(self, state: np.ndarray, training: bool = True) -> int:
    """
    Select an action using an ε-greedy policy with exponential decay.

    During training, ε decays according to:

        ε_t = ε_end + (ε_start - ε_end) · exp(-global_step / eps_decay)

    where `global_step` counts every call to this method (not episodes).
    At evaluation time (`training=False`) the greedy action is always taken.

    Parameters
    ----------
    state : np.ndarray of shape (obs_dim,)
        Flat observation from `env.step()` or `env.reset()`.
    training : bool, optional
        If False, bypass exploration and always act greedily. Default True.

    Returns
    -------
    int
        Action index in [0, action_dim).

    Notes
    -----
    Monitor `Policy/Epsilon/Episode` in TensorBoard; ε should reach ~0.1
    after roughly 50 % of the total step budget (`eps_decay` ≈ 0.5 × max_steps).
    """
```

---

### Example 3 — `RLConfig` dataclass field documentation

**BEFORE** (inline comments, mixed Spanish/English, no units):

```python
@dataclass
class RLConfig:
    obs_dim: int = 32        # observation dimension
    action_dim: int = 3      # number of discrete actions
    gamma: float = 0.99      # discount
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 100_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: int = 100_000  # steps
```

**AFTER**:

```python
@dataclass
class RLConfig:
    """
    Base hyperparameter config for all RL agents.

    Passed to `BaseRLAgent.__init__` and accessed via `self.config`.
    Algorithm-specific subclasses: `PPOConfig`, `PPORNNConfig`, `SACConfig`,
    `DQNConfig`.

    Parameters
    ----------
    obs_dim : int
        Flat observation dimension; must equal `env.get_observation_dim()`.
        Formula: `num_rays * 2 + 2` (each ray is `[dist, angle]`, plus
        `[dist_to_goal, angle_to_goal]`).
    action_dim : int
        Number of discrete actions (3 for GridRobot: forward/left/right).
    gamma : float
        Discount factor γ. 0.99 is standard for navigation (long horizon).
    lr : float
        Adam learning rate. Typical: 1e-3 (DQN), 3e-4 (PPO).
    batch_size : int
        Mini-batch size for gradient updates.
    buffer_size : int
        Circular replay buffer capacity (DQN only). 100k transitions ≈ 200 MB.
    eps_start : float
        Initial ε for ε-greedy exploration (DQN only).
    eps_end : float
        Final ε after decay (DQN only). 0.05 keeps 5 % random exploration.
    eps_decay : int
        Decay timescale in global steps. ε reaches `eps_end + 37% range`
        after `eps_decay` steps.
    """
```
