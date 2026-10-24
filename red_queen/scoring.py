"""One-step action scoring.

score = weighted_gain - cost_term - repeat_penalty

  weighted_gain  = weighted distance now - weighted distance after the action
  cost_term      = action cost (reduced under urgency when affect is on)
  repeat_penalty = frustration * penalty, only for repeating the last action
                   and only when affect is on

Ties are broken by action name so the ranking is fully deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional

from .actions import ACTION_NAMES, ACTIONS, apply_action
from .affect import AffectConfig, AffectSignals, deficit_weights
from .goal import RecoveryGoal
from .state import SystemState

_ROUND = 9  # keeps float noise from deciding ties


@dataclass(frozen=True)
class ScoredAction:
    name: str
    score: float
    gain: float
    cost_term: float
    penalty: float
    predicted: SystemState
    memory_discount: float = 0.0  # M2.1: fraction of positive gain removed by memory


# M2.1: memory may remove at most this fraction of an action's expected gain.
MAX_MEMORY_DISCOUNT = 0.9


def clamp_discount(value: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if value != value:  # NaN
        return 0.0
    return max(0.0, min(MAX_MEMORY_DISCOUNT, value))


class ActionScorer:
    def __init__(self, goal: RecoveryGoal, affect: AffectConfig) -> None:
        self.goal = goal
        self.affect = affect

    def score_all(
        self,
        state: SystemState,
        signals: AffectSignals,
        last_action: Optional[str] = None,
        discounts: Optional[Mapping[str, float]] = None,
    ) -> List[ScoredAction]:
        """`discounts` (M2.1) is advisory memory: for each action, the expected
        chance it has no effect. It only shrinks a *positive* gain, is capped at
        MAX_MEMORY_DISCOUNT, and unknown action names are ignored. With
        `discounts=None` scoring is exactly the M1 behaviour."""
        weights = deficit_weights(self.affect, signals)
        cost_multiplier = (
            1.0 - self.affect.urgency_cost_relief * signals.urgency if self.affect.enabled else 1.0
        )
        current = self.goal.distance(state, weights)

        scored = []
        for name in ACTION_NAMES:
            predicted = apply_action(state, name)
            gain = round(current - self.goal.distance(predicted, weights), _ROUND)
            discount = 0.0
            if discounts and gain > 0:
                discount = clamp_discount(discounts.get(name, 0.0))
                gain = round(gain * (1.0 - discount), _ROUND)
            cost_term = round(ACTIONS[name].cost * cost_multiplier, _ROUND)
            penalty = 0.0
            if self.affect.enabled and name == last_action and signals.frustration > 0:
                penalty = round(self.affect.frustration_penalty * signals.frustration, _ROUND)
            score = round(gain - cost_term - penalty, _ROUND)
            scored.append(ScoredAction(name, score, gain, cost_term, penalty, predicted, discount))

        scored.sort(key=lambda s: (-s.score, s.name))
        return scored
