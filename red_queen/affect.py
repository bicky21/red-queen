"""Experimental affect-related signals.

These are plain numbers computed from the state and recent outcomes. They
are NOT feelings or emotions. They exist so we can measure, by experiment,
whether modulating action scores this way changes behaviour.

Signals:
  urgency     - total goal deficit, capped at 1.0
  concern     - per-condition deficit (which problem is worst)
  frustration - count of consecutive actions that did not reduce distance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from .goal import DEFICIT_KEYS, RecoveryGoal
from .state import SystemState


@dataclass(frozen=True)
class AffectConfig:
    enabled: bool = True
    concern_gain: float = 1.0          # extra weight on the worst deficits
    urgency_cost_relief: float = 0.5   # urgent -> action cost matters less
    frustration_penalty: float = 0.05  # discourage repeating a failing action


@dataclass(frozen=True)
class AffectSignals:
    urgency: float
    concern: Dict[str, float] = field(default_factory=dict)
    frustration: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {"urgency": self.urgency, "concern": dict(self.concern), "frustration": self.frustration}


NEUTRAL_WEIGHTS: Dict[str, float] = {key: 1.0 for key in DEFICIT_KEYS}


def compute_signals(goal: RecoveryGoal, state: SystemState, frustration: int) -> AffectSignals:
    deficits = goal.deficits(state)
    return AffectSignals(
        urgency=min(1.0, sum(deficits.values())),
        concern=dict(deficits),
        frustration=max(0, int(frustration)),
    )


def update_frustration(previous: int, distance_before: float, distance_after: float) -> int:
    """Reset on progress; otherwise grow by one."""
    return 0 if distance_after < distance_before else previous + 1


def deficit_weights(config: AffectConfig, signals: AffectSignals) -> Dict[str, float]:
    if not config.enabled:
        return dict(NEUTRAL_WEIGHTS)
    return {key: 1.0 + config.concern_gain * signals.concern.get(key, 0.0) for key in DEFICIT_KEYS}
