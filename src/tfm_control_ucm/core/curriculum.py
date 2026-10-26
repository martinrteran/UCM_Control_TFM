"""
curriculum.py

Curriculum learning for the grid environments (Xue et al. 2022 use a
curriculum to make goal-reaching frequent enough to learn from).

A curriculum is a list of stages. Each stage restricts which maps the
environment may pick and how far the goal may be from the start. The
environment reports the outcome of every episode; the curriculum moves to the
next stage when the success rate over the last `window` episodes of the
current stage reaches the stage's threshold. The last stage is kept until the
end of training.

The environment owns the curriculum (BaseGridEnv(curriculum=...)), so the
trainer and the agents do not need to know about it.

Author: Martin
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class CurriculumStage:
    """
    One stage of a curriculum.

    Parameters
    ----------
    maps : Sequence[str]
        Names (GridMap.name) of the maps that can be picked in this stage, uniformly.
    max_goal_distance : float, optional
        Largest Euclidean start-goal distance in cells; None means no limit.
    success_threshold : float
        Success rate over the last `window` episodes needed to leave the stage.
    min_episodes : int
        Episodes to play in the stage before it can be left.
    max_episodes : int, optional
        Leave the stage after this many episodes even below the threshold, so a
        stage that is too hard cannot block the rest of the curriculum.
    """
    maps: Sequence[str]
    max_goal_distance: Optional[float] = None
    success_threshold: float = 0.8
    min_episodes: int = 500
    max_episodes: Optional[int] = None

    def __post_init__(self):
        if not self.maps: raise ValueError("A curriculum stage needs at least one map")
        if self.max_goal_distance is not None and self.max_goal_distance <= 0: raise ValueError("max_goal_distance must be positive")
        if not 0.0 < self.success_threshold <= 1.0: raise ValueError("success_threshold must be in (0, 1]")
        if self.min_episodes < 0: raise ValueError("min_episodes must be non-negative")
        if self.max_episodes is not None and self.max_episodes < max(1, self.min_episodes): raise ValueError("max_episodes must be at least min_episodes")


class SuccessCurriculum:
    """
    Moves through the stages when the recent success rate is high enough.

    Parameters
    ----------
    stages : Sequence[CurriculumStage]
        Stages in order; the last one is kept once reached.
    window : int, optional
        Number of recent episodes of the current stage used for the success rate.
    """

    def __init__(self, stages: Sequence[CurriculumStage], window: int = 500):
        if not stages: raise ValueError("A curriculum needs at least one stage")
        if window <= 0: raise ValueError("window must be positive")
        self.stages: List[CurriculumStage] = list(stages)
        self.window = window
        self.stage_idx = 0
        self.episodes_in_stage = 0
        self.total_episodes = 0
        self.stage_started_at: List[int] = [0] # Episode at which each reached stage started
        self._recent: deque = deque(maxlen=window)

    @property
    def stage(self) -> CurriculumStage:
        return self.stages[self.stage_idx]

    @property
    def is_last_stage(self) -> bool:
        return self.stage_idx == len(self.stages) - 1

    def success_rate(self) -> float:
        """Success rate over the recent episodes of the current stage (0 if none yet)."""
        return sum(self._recent) / len(self._recent) if self._recent else 0.0

    def record(self, success: bool) -> bool:
        """
        Record the outcome of an episode.

        Returns
        -------
        bool
            True if this episode made the curriculum move to the next stage.
        """
        self.total_episodes += 1
        self.episodes_in_stage += 1
        self._recent.append(bool(success))
        if self.is_last_stage: return False
        ready = (self.episodes_in_stage >= self.stage.min_episodes
                 and len(self._recent) == self.window
                 and self.success_rate() >= self.stage.success_threshold)
        timed_out = self.stage.max_episodes is not None and self.episodes_in_stage >= self.stage.max_episodes
        if ready or timed_out:
            self.stage_idx += 1
            self.episodes_in_stage = 0
            self._recent.clear()
            self.stage_started_at.append(self.total_episodes)
        return ready or timed_out

    def check_maps(self, map_names: Sequence[str]):
        """Raise if a stage names a map the environment does not have."""
        missing = {name for stage in self.stages for name in stage.maps} - set(map_names)
        if missing: raise ValueError(f"Curriculum maps not in the environment: {', '.join(sorted(missing))}")

    def get_config(self) -> Dict[str, Any]:
        """Parameters of this curriculum, for experiment logs."""
        return {"name": type(self).__name__, "window": self.window,
                "stages": [{**asdict(s), "maps": list(s.maps)} for s in self.stages]}
