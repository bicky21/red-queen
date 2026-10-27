"""M3 goal-directed planning agent.

Loop:
  Goal -> Observe -> (goal met? stop) -> Plan -> Validate plan against policy
       -> Act (next step, through PolicyExecutor) -> Observe again
       -> prediction matches? continue plan : discard plan and replan
       -> ... -> Verify (independent)

M3.3: an optional advisor (same interface as the M2.1 memory advisor)
supplies remembered per-step fault risk. It only breaks ties between plans
of the shortest length, and an action observed to work in this episode
loses its risk (observation wins). Without an advisor, behaviour is M3.

Bounds: max_steps (actions), max_replans (planning calls after the first),
max_policy_denials (rejected plans or executions), and the planner's own
depth and node limits. Every limit ends the episode with a named stop
reason. Success is decided only by the independent verifier.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Set

from .agent import STOP_GOAL, STOP_MAX_STEPS, STOP_POLICY, EpisodeResult, StepRecord
from .environment import SimulatedEnvironment
from .goal import RecoveryGoal
from .planner import BoundedPlanner
from .policy import DEFAULT_POLICY, Policy, PolicyDenied, PolicyExecutor
from .state import SystemState
from .verification import GoalVerifier

STOP_NO_PLAN = "no_plan_found"
STOP_MAX_REPLANS = "max_replans_reached"


class PlanningAgent:
    def __init__(
        self,
        goal: RecoveryGoal,
        planner: Optional[BoundedPlanner] = None,
        policy: Policy = DEFAULT_POLICY,
        max_steps: int = 20,
        max_replans: int = 20,
        max_policy_denials: int = 3,
        advisor=None,
    ) -> None:
        if max_steps < 0 or max_replans < 0 or max_policy_denials < 1:
            raise ValueError("invalid bounds")
        self.goal = goal
        self.planner = planner if planner is not None else BoundedPlanner(goal)
        self.policy = policy
        self.max_steps = max_steps
        self.max_replans = max_replans
        self.max_policy_denials = max_policy_denials
        self.advisor = advisor
        self.verifier = GoalVerifier(goal)

    def run(self, env: SimulatedEnvironment, scenario: str = "") -> EpisodeResult:
        executor = PolicyExecutor(env, self.policy, self.max_steps)
        actions: List[str] = []
        records: List[StepRecord] = []
        queue: List[str] = []
        expected: List[SystemState] = []
        plans = mismatches = nodes = plan_rejections = 0
        stop_reason: Optional[str] = None
        worked: Set[str] = set()
        plan_risks: List[float] = []

        while len(actions) < self.max_steps:
            observed = env.observe()
            if self.goal.is_satisfied(observed):
                break

            if not queue:
                if plans > self.max_replans:
                    stop_reason = STOP_MAX_REPLANS
                    break
                if self.advisor is None:
                    plan = self.planner.plan(observed, self.policy.allowed)
                else:
                    offset, seen = len(actions), frozenset(worked)

                    def risk(depth: int, offset=offset, seen=seen) -> Mapping[str, float]:
                        raw = self.advisor.discounts(offset + depth + 1)
                        return {a: d for a, d in raw.items() if a not in seen}

                    plan = self.planner.plan(observed, self.policy.allowed, risk=risk)
                    plan_risks.append(getattr(plan, "risk", 0.0))
                plans += 1
                nodes += plan.nodes_expanded
                if not plan.reaches_goal:
                    stop_reason = STOP_NO_PLAN
                    break
                # Validate: never trust a plan just because the planner produced it.
                if any(not self.policy.check(a).allowed for a in plan.actions) or \
                        len(plan.actions) != len(plan.predicted):
                    plan_rejections += 1
                    if plan_rejections + len(executor.denials) >= self.max_policy_denials:
                        stop_reason = STOP_POLICY
                        break
                    continue
                queue, expected = list(plan.actions), list(plan.predicted)

            action, predicted = queue.pop(0), expected.pop(0)
            try:
                executor.execute(action)
            except PolicyDenied:
                queue, expected = [], []
                if plan_rejections + len(executor.denials) >= self.max_policy_denials:
                    stop_reason = STOP_POLICY
                    break
                continue
            actions.append(action)
            records.append(
                StepRecord(
                    step=len(actions),
                    observed=observed.to_dict(),
                    chosen=action,
                    score=0.0,
                    ranking=(),
                    affect={},
                )
            )
            if env.observe() != predicted:  # observation disagrees with the plan
                mismatches += 1
                queue, expected = [], []
            else:
                worked.add(action)

        if self.goal.is_satisfied(env.observe()):
            stop_reason = STOP_GOAL
        elif stop_reason is None:
            stop_reason = STOP_MAX_STEPS
        verification = self.verifier.verify(env)
        return EpisodeResult(
            scenario=scenario,
            affect_enabled=False,
            success=verification.satisfied,
            stop_reason=stop_reason,
            actions=tuple(actions),
            steps=tuple(records),
            final_state=env.observe(),
            verification=verification,
            event_count=len(env.events),
            events=tuple(e.to_dict() for e in env.events),
            stats={
                "agent": "planner",
                "policy": self.policy.name,
                "plans": plans,
                "replans": max(0, plans - 1),
                "prediction_mismatches": mismatches,
                "nodes_expanded": nodes,
                "plan_rejections": plan_rejections,
                "policy_denials": len(executor.denials),
                **({} if self.advisor is None else {"memory": True, "plan_risks": plan_risks}),
            },
        )
