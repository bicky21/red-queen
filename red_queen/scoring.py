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
from typing import List, Optional

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


class ActionScorer:
    def __init__(self, goal: RecoveryGoal, affect: AffectConfig) -> None:
        self.goal = goal
        self.affect = affect

    def score_all(
        self,
        state: SystemState,
        signals: AffectSignals,
        last_action: Optional[str] = None,
    ) -> List[ScoredAction]:
        weights = deficit_weights(self.affect, signals)
        cost_multiplier = (
            1.0 - self.affect.urgency_cost_relief * signals.urgency if self.affect.enabled else 1.0
        )
        current = self.goal.distance(state, weights)

        scored = []
        for name in ACTION_NAMES:
            predicted = apply_action(state, name)
            gain = round(current - self.goal.distance(predicted, weights), _ROUND)
            cost_term = round(ACTIONS[name].cost * cost_multiplier, _ROUND)
            penalty = 0.0
            if self.affect.enabled and name == last_action and signals.frustration > 0:
                penalty = round(self.affect.frustration_penalty * signals.frustration, _ROUND)
            score = round(gain - cost_term - penalty, _ROUND)
            scored.append(ScoredAction(name, score, gain, cost_term, penalty, predicted))

        scored.sort(key=lambda s: (-s.score, s.name))
        return scored
