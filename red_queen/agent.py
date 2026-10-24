"""The reactive recovery agent (M1; policy-aware since M3).

Loop:  observe -> (goal met? stop) -> compute signals -> score actions
       -> keep policy-allowed actions -> act through the PolicyExecutor
       -> observe again -> ... -> verify

Execution is bounded by `max_steps`, enforced again at the execution
boundary. Success is decided by the independent verifier, never by the
agent's own belief. With the default policy the behaviour is exactly M2.1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Protocol, Set, Tuple

from .affect import AffectConfig, compute_signals, update_frustration
from .environment import SimulatedEnvironment
from .goal import RecoveryGoal
from .policy import DEFAULT_POLICY, Policy, PolicyDenied, PolicyExecutor
from .scoring import ActionScorer
from .state import SystemState
from .verification import GoalVerifier, VerificationResult

class Advisor(Protocol):
    """M2.1 advisory interface: per-action discount in [0, 1] for a step."""

    def discounts(self, step: int) -> Mapping[str, float]: ...


STOP_GOAL = "goal_believed_satisfied"
STOP_MAX_STEPS = "max_steps_reached"
STOP_NO_ALLOWED = "no_allowed_action"          # M3
STOP_POLICY = "policy_denials_exceeded"        # M3


@dataclass(frozen=True)
class StepRecord:
    step: int
    observed: Dict[str, Any]
    chosen: str
    score: float
    ranking: Tuple[Tuple[str, float], ...]
    affect: Dict[str, Any]
    memory_discounts: Optional[Dict[str, float]] = None  # M2.1; None = no advisor

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "step": self.step,
            "observed": dict(self.observed),
            "chosen": self.chosen,
            "score": self.score,
            "ranking": [list(item) for item in self.ranking],
            "affect": self.affect,
        }
        if self.memory_discounts is not None:
            out["memory_discounts"] = dict(self.memory_discounts)
        return out


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
    stats: Dict[str, Any] = field(default_factory=dict)  # M3: agent/policy/planning counters

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
            "stats": dict(self.stats),
        }


class RecoveryAgent:
    def __init__(
        self,
        goal: RecoveryGoal,
        affect: AffectConfig = AffectConfig(),
        max_steps: int = 20,
        advisor: Optional["Advisor"] = None,
        policy: Policy = DEFAULT_POLICY,
    ) -> None:
        if max_steps < 0:
            raise ValueError("max_steps must be >= 0")
        self.goal = goal
        self.affect = affect
        self.max_steps = max_steps
        self.advisor = advisor
        self.policy = policy
        self.scorer = ActionScorer(goal, affect)
        self.verifier = GoalVerifier(goal)

    def run(self, env: SimulatedEnvironment, scenario: str = "") -> EpisodeResult:
        actions: List[str] = []
        records: List[StepRecord] = []
        frustration = 0
        last_action = None
        worked: Set[str] = set()  # actions observed to behave as modelled this episode
        executor = PolicyExecutor(env, self.policy, self.max_steps)
        filtered = 0
        stop_reason: Optional[str] = None

        while len(actions) < self.max_steps:
            observed = env.observe()
            if self.goal.is_satisfied(observed):
                break
            signals = compute_signals(self.goal, observed, frustration)
            discounts = None
            if self.advisor is not None:
                # Current observation wins: an action that already worked in
                # this episode gets no memory discount.
                raw = self.advisor.discounts(len(actions) + 1)
                discounts = {a: d for a, d in raw.items() if a not in worked}
            ranking = self.scorer.score_all(observed, signals, last_action, discounts)
            allowed = [r for r in ranking if self.policy.check(r.name).allowed]
            filtered += len(ranking) - len(allowed)
            if not allowed:
                stop_reason = STOP_NO_ALLOWED
                break
            best = allowed[0]

            records.append(
                StepRecord(
                    step=len(actions) + 1,
                    observed=observed.to_dict(),
                    chosen=best.name,
                    score=best.score,
                    ranking=tuple((r.name, r.score) for r in ranking),
                    affect=signals.to_dict(),
                    memory_discounts=(
                        None if discounts is None
                        else {r.name: r.memory_discount for r in ranking if r.memory_discount > 0}
                    ),
                )
            )
            distance_before = self.goal.distance(observed)
            try:
                executor.execute(best.name)
            except PolicyDenied:  # should be impossible after filtering; fail closed
                records.pop()
                stop_reason = STOP_POLICY
                break
            actions.append(best.name)
            if env.observe() == best.predicted:
                worked.add(best.name)
            frustration = update_frustration(frustration, distance_before, self.goal.distance(env.observe()))
            last_action = best.name

        if self.goal.is_satisfied(env.observe()):
            stop_reason = STOP_GOAL
        elif stop_reason is None:
            stop_reason = STOP_MAX_STEPS
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
            stats={
                "agent": "reactive",
                "policy": self.policy.name,
                "policy_filtered": filtered,
                "policy_denials": len(executor.denials),
            },
        )
