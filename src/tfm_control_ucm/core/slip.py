"""
slip.py

Friction-dependent slipping for the grid environments.

The map stores the grip of every cell (GridMap.friction, mu in [0, 1], 1 = normal
floor). The robot remembers its momentum, the last displacement it made ((0, 0)
when it is stopped). Each step the move it attempts is classified against that
momentum, and every maneuver demands a different grip and slips in its own way:

    maneuver   momentum -> action   slip when it happens            demand d
    straight   E -> E               overshoot: slides one more cell 0.3
    start      stopped -> E         stay: the wheels spin, no move  0.4
    turn       E -> N               diagonal: ends at N + E         0.6
    reverse    E -> W               drag: momentum carries it E     0.8

Following Coulomb (the floor can give at most mu * N of force), the robot only
slips when the maneuver demands more grip than the floor has:

    p_slip = clip(gain * (d - mu), 0, max_prob)

so cells with mu >= d never slip and the default floor (mu = 1) is deterministic.
The demands are model parameters, not measured values: what is justified is their
order (going straight needs less grip than starting, turning or reversing).

Optionally ("lateral" in kinds) a move that did not slip can still drift sideways,
like the classic slippery gridworlds, with probability clip(gain * (lateral_demand - mu), 0, max_prob).

The environment decides what a slide into a wall means (into_obstacle):
"collide" ends the episode as a collision, "stop" leaves the robot on the last free cell.

Author: Martin
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, NamedTuple, Tuple

import numpy as np

Vec = Tuple[int, int]

MANEUVERS = ("straight", "start", "turn", "reverse")
SLIP_OF_MANEUVER = {"straight": "overshoot", "start": "stay", "turn": "diagonal", "reverse": "drag"}
SLIP_KINDS = ("overshoot", "stay", "diagonal", "drag", "lateral")
INTO_OBSTACLE = ("collide", "stop")


def classify_maneuver(momentum: Vec, intended: Vec) -> str:
    """
    Name of the maneuver that goes from the current momentum to the intended move.

    Parameters
    ----------
    momentum : (int, int)
        Last displacement of the robot (dc, dr); (0, 0) when stopped.
    intended : (int, int)
        Displacement the action asks for; must not be (0, 0).

    Returns
    -------
    str
        "start", "straight", "reverse" or "turn".
    """
    if tuple(momentum) == (0, 0): return "start"
    if tuple(intended) == tuple(momentum): return "straight"
    if tuple(intended) == (-momentum[0], -momentum[1]): return "reverse"
    return "turn"


class SlipEvent(NamedTuple):
    """
    What happens to one move.

    kind : "none" or one of SLIP_KINDS.
    maneuver : one of MANEUVERS.
    probability : probability the maneuver had of slipping.
    path : displacements applied one after another (each one must land on a free cell).
    momentum : momentum after the move if the whole path is free.
    """
    kind: str
    maneuver: str
    probability: float
    path: Tuple[Vec, ...]
    momentum: Vec


@dataclass
class SlipModel:
    """
    Maneuver-dependent slip probabilities from the floor's grip (see the module docstring).

    Parameters
    ----------
    demand_straight, demand_start, demand_turn, demand_reverse : float
        Grip each maneuver needs; it slips only on cells with a lower mu.
    gain : float
        Slope k of p = clip(k * (d - mu), 0, max_prob).
    max_prob : float
        Upper bound of the slip probability.
    kinds : tuple of str
        Slip kinds that can happen (subset of SLIP_KINDS). A maneuver whose kind is
        left out never slips. "lateral" adds the sideways drift.
    lateral_demand : float
        Demand of the sideways drift, used only if "lateral" is in kinds.
    into_obstacle : str
        "collide": a slide into a wall or an obstacle is a collision (ends the episode);
        "stop": the robot stops on the last free cell.
    """
    demand_straight: float = 0.3
    demand_start: float = 0.4
    demand_turn: float = 0.6
    demand_reverse: float = 0.8
    gain: float = 2.0
    max_prob: float = 0.9
    kinds: Tuple[str, ...] = ("overshoot", "stay", "diagonal", "drag")
    lateral_demand: float = 0.3
    into_obstacle: str = "collide"

    def __post_init__(self):
        self.kinds = tuple(self.kinds)
        unknown = set(self.kinds) - set(SLIP_KINDS)
        if unknown: raise ValueError(f"Unknown slip kinds {sorted(unknown)}; valid: {SLIP_KINDS}")
        if self.into_obstacle not in INTO_OBSTACLE: raise ValueError(f"into_obstacle must be one of {INTO_OBSTACLE}")
        if self.gain < 0: raise ValueError("gain must be >= 0")
        if not 0.0 <= self.max_prob <= 1.0: raise ValueError("max_prob must be in [0, 1]")

    @property
    def name(self) -> str:
        return type(self).__name__

    def get_config(self) -> Dict[str, Any]:
        """Parameters of this model, for experiment logs (SlipModel(**{k: v for k != 'name'}) rebuilds it)."""
        params = dataclasses.asdict(self)
        params["kinds"] = list(self.kinds)
        return {"name": self.name, **params}

    def demand(self, maneuver: str) -> float:
        return {"straight": self.demand_straight, "start": self.demand_start,
                "turn": self.demand_turn, "reverse": self.demand_reverse}[maneuver]

    def probability(self, demand: float, friction: float) -> float:
        """p = clip(gain * (demand - friction), 0, max_prob): 0 when the floor gives enough grip."""
        return float(np.clip(self.gain * (demand - friction), 0.0, self.max_prob))

    def slip_probability(self, friction: float, momentum: Vec, intended: Vec) -> float:
        """Probability that the maneuver slips (0 if its kind is disabled)."""
        maneuver = classify_maneuver(momentum, intended)
        if SLIP_OF_MANEUVER[maneuver] not in self.kinds: return 0.0
        return self.probability(self.demand(maneuver), friction)

    def sample(self, friction: float, momentum: Vec, intended: Vec, rng: np.random.Generator) -> SlipEvent:
        """
        Decide what happens to a move.

        Parameters
        ----------
        friction : float
            Grip of the cell the robot stands on when the step starts.
        momentum : (int, int)
            Last displacement of the robot; (0, 0) when stopped.
        intended : (int, int)
            Displacement the action asks for (not (0, 0); turning in place never slips).
        rng : np.random.Generator
            Generator of the slips (the environment's own, separate from the one
            that picks maps and positions).

        Returns
        -------
        SlipEvent
        """
        momentum = (int(momentum[0]), int(momentum[1]))
        intended = (int(intended[0]), int(intended[1]))
        maneuver = classify_maneuver(momentum, intended)
        kind = SLIP_OF_MANEUVER[maneuver]
        p = self.slip_probability(friction, momentum, intended)

        if p > 0.0 and rng.random() < p:
            if kind == "overshoot": return SlipEvent(kind, maneuver, p, (intended, intended), intended)
            if kind == "stay": return SlipEvent(kind, maneuver, p, (), (0, 0))
            if kind == "diagonal": return SlipEvent(kind, maneuver, p, (momentum, intended), intended) # Through the cell the momentum points to
            return SlipEvent(kind, maneuver, p, (momentum,), momentum) # drag

        if "lateral" in self.kinds:
            p_lateral = self.probability(self.lateral_demand, friction)
            if p_lateral > 0.0 and rng.random() < p_lateral:
                dc, dr = intended
                side = (dr, dc) if rng.random() < 0.5 else (-dr, -dc) # Perpendicular to the intended move
                return SlipEvent("lateral", maneuver, p_lateral, (side,), side)

        return SlipEvent("none", maneuver, p, (intended,), intended)
