"""
rl_experiment.py
----------------
A flexible configuration and runner class for Reinforcement Learning experiments.
Supports multiple environments, multiple agents, and configurable episodes.

Each agent declares whether it trains on a single environment at a time
("single" mode) or consumes all registered environments simultaneously
("multi" mode, e.g. for multi-task / multi-env agents such as IMPALA or
domain-randomised policies).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

# Type alias kept narrow so IDEs and mypy catch typos early.
EnvMode = Literal["single", "multi"]


# ---------------------------------------------------------------------------
# Data classes for individual components
# ---------------------------------------------------------------------------

@dataclass
class EnvironmentConfig:
    """Configuration for a single RL environment."""

    name: str
    env_id: str                          # e.g. "CartPole-v1", "LunarLander-v2"
    make_fn: Callable[[], Any]           # zero-arg factory that returns an env
    max_steps: int = 1_000
    seed: int | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")

    def build(self) -> Any:
        """Instantiate and return the environment."""
        env = self.make_fn(**self.kwargs)
        if self.seed is not None and hasattr(env, "reset"):
            env.reset(seed=self.seed)
        return env

    def __repr__(self) -> str:
        return (
            f"EnvironmentConfig(name={self.name!r}, env_id={self.env_id!r}, "
            f"max_steps={self.max_steps}, seed={self.seed})"
        )


@dataclass
class AgentConfig:
    """
    Configuration for a single RL agent.

    Parameters
    ----------
    name : str
        Human-readable label for this agent configuration.
    agent_cls : type
        The agent class to instantiate.
    env_mode : {"single", "multi"}
        Controls how environments are presented to this agent at run time.

        ``"single"`` (default)
            The agent trains on **one environment at a time**.  The experiment
            runner pairs the agent with each registered environment separately,
            producing one run per pairing (respecting the experiment's
            ``pairing`` strategy).  The ``build(env)`` method receives a
            single environment instance and the training function is called
            with signature ``train_fn(env, agent, n_episodes)``.

        ``"multi"``
            The agent trains on **all registered environments at once** (e.g.
            multi-task, domain-randomised, or population-based agents such as
            IMPALA).  A single run is created per agent; ``build(envs)``
            receives a *list* of environment instances and the training
            function is called with signature
            ``train_fn(envs, agent, n_episodes)``.

    hyperparams : dict
        Keyword arguments forwarded to ``agent_cls.__init__`` after the env(s).
    policy : str
        An informational label for the policy type (e.g. "epsilon-greedy").
    """

    name: str
    agent_cls: type
    env_mode: EnvMode = "single"
    hyperparams: dict[str, Any] = field(default_factory=dict)
    policy: str = "default"

    def __post_init__(self) -> None:
        if self.env_mode not in ("single", "multi"):
            raise ValueError(
                f"env_mode must be 'single' or 'multi', got {self.env_mode!r}"
            )

    def build(self, env_or_envs: Any) -> Any:
        """
        Instantiate and return the agent.

        Parameters
        ----------
        env_or_envs : env | list[env]
            A single environment (``env_mode="single"``) or a list of
            environments (``env_mode="multi"``).
        """
        return self.agent_cls(env_or_envs, **self.hyperparams)

    def __repr__(self) -> str:
        return (
            f"AgentConfig(name={self.name!r}, agent_cls={self.agent_cls.__name__}, "
            f"env_mode={self.env_mode!r}, policy={self.policy!r}, "
            f"hyperparams={self.hyperparams})"
        )


# ---------------------------------------------------------------------------
# Main experiment configuration class
# ---------------------------------------------------------------------------

class RLManager:
    """
    Manages a collection of environment + agent configurations and the number
    of episodes, then produces experiment runs.

    Run generation strategy
    -----------------------
    Agents with ``env_mode="single"`` are paired with environments according
    to the experiment's ``pairing`` strategy:

    - ``"product"`` (default): every single-mode agent runs on every environment.
    - ``"zip"``: environments and single-mode agents are paired by position
      (both lists must be the same length).

    Agents with ``env_mode="multi"`` always produce exactly **one run** per
    agent, regardless of the ``pairing`` strategy.  That run receives the
    full list of registered environments.

    Parameters
    ----------
    name : str
        A human-readable label for this experiment batch.
    n_episodes : int
        Default number of episodes for every run (overridable per-run).
    pairing : {"product", "zip"}
        Pairing strategy applied to ``"single"``-mode agents only.
    """

    def __init__(
        self,
        name: str = "experiment",
        n_episodes: int = 100,
        pairing: str = "product",
    ) -> None:
        if n_episodes <= 0:
            raise ValueError(f"n_episodes must be positive, got {n_episodes}")
        if pairing not in {"product", "zip"}:
            raise ValueError(f"pairing must be 'product' or 'zip', got {pairing!r}")

        self.name = name
        self.n_episodes = n_episodes
        self.pairing = pairing

        self._envs: list[EnvironmentConfig] = []
        self._agents: list[AgentConfig] = []

    # ------------------------------------------------------------------
    # Environment management
    # ------------------------------------------------------------------

    def add_environment(self, env_config: EnvironmentConfig) -> "RLManager":
        """Register an environment configuration. Returns self for chaining."""
        if not isinstance(env_config, EnvironmentConfig):
            raise TypeError(f"Expected EnvironmentConfig, got {type(env_config)}")
        self._envs.append(env_config)
        return self

    def add_environments(self, *env_configs: EnvironmentConfig) -> "RLManager":
        """Register multiple environment configurations at once."""
        for ec in env_configs:
            self.add_environment(ec)
        return self

    def remove_environment(self, name: str) -> "RLManager":
        """Remove an environment by name."""
        self._envs = [e for e in self._envs if e.name != name]
        return self

    @property
    def environments(self) -> list[EnvironmentConfig]:
        return list(self._envs)

    # ------------------------------------------------------------------
    # Agent management
    # ------------------------------------------------------------------

    def add_agent(self, agent_config: AgentConfig) -> "RLManager":
        """Register an agent configuration. Returns self for chaining."""
        if not isinstance(agent_config, AgentConfig):
            raise TypeError(f"Expected AgentConfig, got {type(agent_config)}")
        self._agents.append(agent_config)
        return self

    def add_agents(self, *agent_configs: AgentConfig) -> "RLManager":
        """Register multiple agent configurations at once."""
        for ac in agent_configs:
            self.add_agent(ac)
        return self

    def remove_agent(self, name: str) -> "RLManager":
        """Remove an agent by name."""
        self._agents = [a for a in self._agents if a.name != name]
        return self

    @property
    def agents(self) -> list[AgentConfig]:
        return list(self._agents)

    # ------------------------------------------------------------------
    # Episode configuration
    # ------------------------------------------------------------------

    def set_episodes(self, n: int) -> "RLManager":
        """Update the default episode count. Returns self for chaining."""
        if n <= 0:
            raise ValueError(f"n_episodes must be positive, got {n}")
        self.n_episodes = n
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _single_mode_runs(
        self, single_agents: list[AgentConfig], episodes: int
    ) -> list[dict[str, Any]]:
        """Build run descriptors for all single-mode agents."""
        if self.pairing == "product":
            pairs = list(itertools.product(self._envs, single_agents))
        else:  # "zip"
            if len(self._envs) != len(single_agents):
                raise ValueError(
                    f"'zip' pairing requires equal numbers of envs and single-mode "
                    f"agents, got {len(self._envs)} envs and "
                    f"{len(single_agents)} single-mode agents."
                )
            pairs = list(zip(self._envs, single_agents))

        return [
            {
                "env_configs": [env_cfg],   # list of one for uniform interface
                "agent_config": agent_cfg,
                "n_episodes": episodes,
            }
            for env_cfg, agent_cfg in pairs
        ]

    def _multi_mode_runs(
        self, multi_agents: list[AgentConfig], episodes: int
    ) -> list[dict[str, Any]]:
        """Build run descriptors for all multi-mode agents."""
        return [
            {
                "env_configs": list(self._envs),  # full environment list
                "agent_config": agent_cfg,
                "n_episodes": episodes,
            }
            for agent_cfg in multi_agents
        ]

    # ------------------------------------------------------------------
    # Run generation
    # ------------------------------------------------------------------

    def get_runs(self, n_episodes: int | None = None) -> list[dict[str, Any]]:
        """
        Return a list of run descriptors.

        Each descriptor is a plain dict with keys:

        ``run_id``
            Unique string identifier for this run.
        ``env_configs``
            A list of :class:`EnvironmentConfig` objects — one item for
            ``"single"`` agents, all registered environments for ``"multi"``
            agents.
        ``agent_config``
            The :class:`AgentConfig` for this run.
        ``n_episodes``
            Episode count that will be used for this run.

        Parameters
        ----------
        n_episodes : int | None
            Override the instance-level episode count for this call only.
        """
        if not self._envs:
            raise RuntimeError("No environments have been added.")
        if not self._agents:
            raise RuntimeError("No agents have been added.")

        episodes = n_episodes if n_episodes is not None else self.n_episodes

        single_agents = [a for a in self._agents if a.env_mode == "single"]
        multi_agents  = [a for a in self._agents if a.env_mode == "multi"]

        raw_runs: list[dict[str, Any]] = []
        if single_agents:
            raw_runs.extend(self._single_mode_runs(single_agents, episodes))
        if multi_agents:
            raw_runs.extend(self._multi_mode_runs(multi_agents, episodes))

        # Stamp each run with a unique ID.
        for idx, run in enumerate(raw_runs):
            run["run_id"] = f"{self.name}_run{idx:03d}"

        return raw_runs

    def execute(
        self,
        train_fn: Callable[[Any, Any, int], Any],
        n_episodes: int | None = None,
        verbose: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Build every (env(s), agent) combination and call
        ``train_fn(env_or_envs, agent, n_episodes)``.

        The first argument passed to ``train_fn`` is:

        - A **single environment instance** when ``agent.env_mode == "single"``.
        - A **list of environment instances** when ``agent.env_mode == "multi"``.

        Parameters
        ----------
        train_fn : callable
            ``train_fn(env_or_envs, agent, n_episodes) -> result``
        n_episodes : int | None
            Per-call episode override.
        verbose : bool
            Print progress info.

        Returns
        -------
        list of dicts — each run descriptor augmented with a ``"result"`` key.
        """
        runs = self.get_runs(n_episodes=n_episodes)
        results = []

        for run in runs:
            agent_cfg: AgentConfig = run["agent_config"]
            env_cfgs: list[EnvironmentConfig] = run["env_configs"]

            # Build environment(s)
            if agent_cfg.env_mode == "single":
                built_envs = env_cfgs[0].build()          # single env instance
                env_label  = env_cfgs[0].name
            else:
                built_envs = [ec.build() for ec in env_cfgs]   # list of envs
                env_label  = f"[{', '.join(ec.name for ec in env_cfgs)}]"

            agent = agent_cfg.build(built_envs)

            if verbose:
                print(
                    f"[{run['run_id']}] "
                    f"env_mode={agent_cfg.env_mode!r}  "
                    f"env(s)={env_label}  "
                    f"agent={agent_cfg.name!r}  "
                    f"episodes={run['n_episodes']}"
                )

            result = train_fn(built_envs, agent, run["n_episodes"])

            # Close environment(s)
            for env in (built_envs if isinstance(built_envs, list) else [built_envs]):
                if hasattr(env, "close"):
                    env.close()

            results.append({**run, "result": result})

        return results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of this configuration."""
        single_agents = [a for a in self._agents if a.env_mode == "single"]
        multi_agents  = [a for a in self._agents if a.env_mode == "multi"]

        if self.pairing == "product":
            n_single_runs = len(self._envs) * len(single_agents)
        else:
            n_single_runs = min(len(self._envs), len(single_agents))
        n_multi_runs = len(multi_agents)

        lines = [
            f"RLExperimentConfig: {self.name!r}",
            f"  Episodes      : {self.n_episodes}",
            f"  Pairing       : {self.pairing}  (applies to single-mode agents only)",
            f"  Environments  ({len(self._envs)}):",
        ]
        for e in self._envs:
            lines.append(
                f"    • {e.name} ({e.env_id})  "
                f"max_steps={e.max_steps}  seed={e.seed}"
            )
        lines.append(f"  Agents ({len(self._agents)}):")
        for a in self._agents:
            lines.append(
                f"    • {a.name} [{a.agent_cls.__name__}]  "
                f"env_mode={a.env_mode!r}  policy={a.policy!r}  "
                f"params={a.hyperparams}"
            )
        lines.append(
            f"  Total runs    : {n_single_runs + n_multi_runs}  "
            f"({n_single_runs} single-env + {n_multi_runs} multi-env)"
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"RLExperimentConfig(name={self.name!r}, n_episodes={self.n_episodes}, "
            f"pairing={self.pairing!r}, "
            f"envs={len(self._envs)}, agents={len(self._agents)})"
        )


# ---------------------------------------------------------------------------
# Quick usage example (runs when executed directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    # ---- Minimal stub classes so the demo works without gym installed ----

    class StubEnv:
        def __init__(self, env_id: str, **kwargs):
            self.env_id = env_id
            self.observation_space = [0, 1, 2, 3]
            self.action_space = [0, 1]

        def reset(self, seed=None):
            return [0.0, 0.0, 0.0, 0.0]

        def step(self, action):
            obs = [random.random()] * 4
            reward = random.random()
            done = random.random() < 0.05
            return obs, reward, done, {}

        def close(self):
            pass

    # Single-env agent (e.g. Q-Learning, DQN)
    class QLearningAgent:
        def __init__(self, env, lr=0.1, gamma=0.99, **kwargs):
            self.env = env          # single env
            self.lr = lr
            self.gamma = gamma

        def act(self, obs):
            return random.choice(self.env.action_space)

        def learn(self, obs, action, reward, next_obs, done):
            pass

    # Multi-env agent (e.g. IMPALA, domain-randomised policy)
    class MultiEnvAgent:
        def __init__(self, envs: list, lr=3e-4, **kwargs):
            self.envs = envs        # list of envs
            self.lr = lr

        def act(self, obs, env_idx: int = 0):
            return random.choice(self.envs[env_idx].action_space)

        def learn(self, transitions):
            pass  # simplified stub

    # ---- Training loops ----

    def single_env_train(env, agent, n_episodes: int):
        """Standard single-environment training loop."""
        rewards = []
        for _ in range(n_episodes):
            obs = env.reset()
            ep_r, done = 0.0, False
            while not done:
                action = agent.act(obs)
                obs, r, done, _ = env.step(action)
                agent.learn(obs, action, r, obs, done)
                ep_r += r
            rewards.append(ep_r)
        return {"avg_reward": round(sum(rewards) / len(rewards), 4)}

    def multi_env_train(envs: list, agent, n_episodes: int):
        """Multi-environment training loop — round-robin across envs each episode."""
        rewards = []
        for ep in range(n_episodes):
            env = envs[ep % len(envs)]          # cycle through environments
            obs = env.reset()
            ep_r, done = 0.0, False
            while not done:
                action = agent.act(obs)
                obs, r, done, _ = env.step(action)
                ep_r += r
            rewards.append(ep_r)
        return {
            "avg_reward": round(sum(rewards) / len(rewards), 4),
            "n_envs": len(envs),
        }

    def dispatch_train(env_or_envs, agent, n_episodes: int):
        """Single dispatcher that routes to the right loop by agent type."""
        if isinstance(env_or_envs, list):
            return multi_env_train(env_or_envs, agent, n_episodes)
        return single_env_train(env_or_envs, agent, n_episodes)

    # ---- Build experiment ----

    exp = RLManager(name="demo_exp", n_episodes=30, pairing="product")

    exp.add_environments(
        EnvironmentConfig(
            name="CartPole",
            env_id="CartPole-v1",
            make_fn=lambda: StubEnv("CartPole-v1"),
            max_steps=500,
            seed=42,
        ),
        EnvironmentConfig(
            name="LunarLander",
            env_id="LunarLander-v2",
            make_fn=lambda: StubEnv("LunarLander-v2"),
            max_steps=1000,
            seed=7,
        ),
        EnvironmentConfig(
            name="MountainCar",
            env_id="MountainCar-v0",
            make_fn=lambda: StubEnv("MountainCar-v0"),
            max_steps=200,
            seed=99,
        ),
    )

    exp.add_agents(
        # Trains on one env at a time — will produce 3 runs (one per env)
        AgentConfig(
            name="QLearning",
            agent_cls=QLearningAgent,
            env_mode="single",
            hyperparams={"lr": 0.05, "gamma": 0.95},
            policy="epsilon-greedy",
        ),
        # Trains on ALL envs simultaneously — will produce 1 run
        AgentConfig(
            name="IMPALA",
            agent_cls=MultiEnvAgent,
            env_mode="multi",
            hyperparams={"lr": 3e-4},
            policy="vtrace",
        ),
    )

    print(exp.summary())
    print()

    results = exp.execute(dispatch_train, verbose=True)

    print("\n=== Results ===")
    for r in results:
        env_names = [ec.name for ec in r["env_configs"]]
        print(f"  {r['run_id']:25s}  envs={env_names}  {r['result']}")