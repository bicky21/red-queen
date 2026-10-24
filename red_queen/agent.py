"""The M1 recovery agent.

Loop:  observe -> (goal met? stop) -> compute signals -> score actions
       -> act through the environment -> observe again -> ... -> verify

Execution is bounded by `max_steps`. Success is decided by the
independent verifier, never by the agent's own belief.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from .affect import AffectConfig, compute_signals, update_frustration
from .environment import SimulatedEnvironment
from .goal import RecoveryGoal
from .scoring import ActionScorer
from .state import SystemState
from .verification import GoalVerifier, VerificationResult

STOP_GOAL = "goal_believed_satisfied"
STOP_MAX_STEPS = "max_steps_reached"


@dataclass(frozen=True)
class StepRecord:
    step: int
    observed: Dict[str, Any]
    chosen: str
    score: float
    ranking: Tuple[Tuple[str, float], ...]
    affect: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "observed": dict(self.observed),
            "chosen": self.chosen,
            "score": self.score,
            "ranking": [list(item) for item in self.ranking],
            "affect": self.affect,
        }


@dataclass(frozen=True)
class EpisodeResult:
    scenario: str
    affect_enabled: bool
    success: bool
    stop_reason: str
    actions: Tuple[str, ...]
    steps: Tuple[StepRecord, ...]
    final_state: SystemState
    verification: VerificationResult
    event_count: int
    events: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def action_count(self) -> int:
        return len(self.actions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "affect_enabled": self.affect_enabled,
            "success": self.success,
            "stop_reason": self.stop_reason,
            "action_count": self.action_count,
            "actions": list(self.actions),
            "steps": [s.to_dict() for s in self.steps],
            "final_state": self.final_state.to_dict(),
            "verification": self.verification.to_dict(),
            "event_count": self.event_count,
            "events": [dict(e) for e in self.events],
        }


class RecoveryAgent:
    def __init__(
        self,
        goal: RecoveryGoal,
        affect: AffectConfig = AffectConfig(),
        max_steps: int = 20,
    ) -> None:
        if max_steps < 0:
            raise ValueError("max_steps must be >= 0")
        self.goal = goal
        self.affect = affect
        self.max_steps = max_steps
        self.scorer = ActionScorer(goal, affect)
        self.verifier = GoalVerifier(goal)

    def run(self, env: SimulatedEnvironment, scenario: str = "") -> EpisodeResult:
        actions: List[str] = []
        records: List[StepRecord] = []
        frustration = 0
        last_action = None

        while len(actions) < self.max_steps:
            observed = env.observe()
            if self.goal.is_satisfied(observed):
                break
            signals = compute_signals(self.goal, observed, frustration)
            ranking = self.scorer.score_all(observed, signals, last_action)
            best = ranking[0]

            records.append(
                StepRecord(
                    step=len(actions) + 1,
                    observed=observed.to_dict(),
                    chosen=best.name,
                    score=best.score,
                    ranking=tuple((r.name, r.score) for r in ranking),
                    affect=signals.to_dict(),
                )
            )
            distance_before = self.goal.distance(observed)
            env.execute(best.name)
            actions.append(best.name)
            frustration = update_frustration(frustration, distance_before, self.goal.distance(env.observe()))
            last_action = best.name

        stop_reason = STOP_GOAL if self.goal.is_satisfied(env.observe()) else STOP_MAX_STEPS
        verification = self.verifier.verify(env)
        return EpisodeResult(
            scenario=scenario,
            affect_enabled=self.affect.enabled,
            success=verification.satisfied,
            stop_reason=stop_reason,
            actions=tuple(actions),
            steps=tuple(records),
            final_state=env.observe(),
            verification=verification,
            event_count=len(env.events),
            events=tuple(e.to_dict() for e in env.events),
        )
