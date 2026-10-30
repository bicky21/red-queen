"""M5.2 redundant sensors: two independent sensors against sensor bias.

M5.1 showed that a constant sensor bias cannot be detected from one sensor:
every reading is shifted the same way, so readings stay consistent. M5.2
reads TWO independent sensors with the same noise bound each step and fuses
their observation boxes:

  "intersect" - belief = A box ∩ B box (tight). Sound only if BOTH sensors
                are within their bound. If the boxes do not overlap, at
                least one sensor is faulty: the agent halts
                (stop reason "sensor_fault_detected") without claiming success.
  "hull"      - belief = smallest box containing A box and B box. Sound if AT
                LEAST ONE sensor is within its bound, so a single biased
                sensor can never cause a false stop. Costs more actions.
  "latch"     - intersect until the first disagreement, then hull for the
                rest of the episode.

Planning and stopping are as in the M5 interval agent: plan from the worst
corner of the belief, execute one action through the PolicyExecutor, stop
only when every state in the belief meets the goal. Success is decided by
the independent GoalVerifier on the true state.

Stated assumption for the hull guarantee: at most ONE sensor is faulty at a
time, and the noise bound is known. A bias shared by both sensors (common
mode) defeats every fusion rule here (measured in the M5.2 study).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .agent import STOP_GOAL, STOP_MAX_STEPS, STOP_POLICY, EpisodeResult, StepRecord
from .belief_agent import intersect
from .goal import RecoveryGoal
from .planner import BoundedPlanner
from .planning_agent import STOP_NO_PLAN
from .policy import DEFAULT_POLICY, Policy, PolicyDenied, PolicyExecutor
from .sensors import Box, NoisySensor, SensedEnvironment, observation_box
from .state import SystemState
from .verification import GoalVerifier

FUSIONS = ("intersect", "hull", "latch")
STOP_SENSOR_FAULT = "sensor_fault_detected"


def hull(a: Box, b: Box) -> Box:
    """Smallest box containing both (service status must agree; it is read exactly)."""
    if a.service_up != b.service_up:
        raise ValueError("sensors disagree on the exactly-read service status")
    return Box(min(a.b_lo, b.b_lo), max(a.b_hi, b.b_hi), min(a.t_lo, b.t_lo), max(a.t_hi, b.t_hi), a.service_up)


class RedundantBeliefAgent:
    def __init__(
        self,
        goal: RecoveryGoal,
        noise: int,
        fusion: str = "hull",
        planner: Optional[BoundedPlanner] = None,
        policy: Policy = DEFAULT_POLICY,
        max_steps: int = 20,
        max_policy_denials: int = 3,
    ) -> None:
        if fusion not in FUSIONS:
            raise ValueError(f"fusion must be one of {FUSIONS}")
        if max_steps < 0 or max_policy_denials < 1:
            raise ValueError("invalid bounds")
        observation_box(SystemState(50, 50, True), noise)  # validates the noise bound
        self.goal = goal
        self.noise = noise
        self.fusion = fusion
        self.planner = planner if planner is not None else BoundedPlanner(goal)
        self.policy = policy
        self.max_steps = max_steps
        self.max_policy_denials = max_policy_denials
        self.verifier = GoalVerifier(goal)

    def run(self, env, sensor_a: NoisySensor, sensor_b: NoisySensor, scenario: str = "") -> EpisodeResult:
        if sensor_a is sensor_b:
            raise ValueError("redundancy needs two separate sensors")
        sensed_a, sensed_b = SensedEnvironment(env, sensor_a), SensedEnvironment(env, sensor_b)
        executor = PolicyExecutor(sensed_a, self.policy, self.max_steps)
        actions: List[str] = []
        records: List[StepRecord] = []
        beliefs: List[Dict] = []
        disagreements: List[int] = []
        latched = False
        plans = nodes = 0
        stop_reason: Optional[str] = None

        while True:
            ra, rb = sensed_a.observe(), sensed_b.observe()
            box_a, box_b = observation_box(ra, self.noise), observation_box(rb, self.noise)
            both = intersect(box_a, box_b)
            if both is None and (not disagreements or disagreements[-1] != sensed_a.step):
                disagreements.append(sensed_a.step)
            if self.fusion == "hull" or (self.fusion == "latch" and (latched or both is None)):
                latched = latched or both is None
                belief = hull(box_a, box_b)
            elif both is None:  # intersect: the sensors cannot both be right
                stop_reason = STOP_SENSOR_FAULT
                break
            else:
                belief = both
            beliefs.append({"step": sensed_a.step, **belief.to_dict()})
            if belief.satisfies(self.goal):
                stop_reason = STOP_GOAL
                break
            if len(actions) >= self.max_steps:
                stop_reason = STOP_MAX_STEPS
                break
            plan = self.planner.plan(belief.worst(), self.policy.allowed)
            plans += 1
            nodes += plan.nodes_expanded
            if not plan.reaches_goal or not plan.actions:
                stop_reason = STOP_NO_PLAN
                break
            action = plan.actions[0]
            try:
                executor.execute(action)
            except PolicyDenied:
                if len(executor.denials) >= self.max_policy_denials:
                    stop_reason = STOP_POLICY
                    break
                continue
            actions.append(action)
            records.append(StepRecord(step=len(actions), observed={"a": ra.to_dict(), "b": rb.to_dict()},
                                      chosen=action, score=0.0, ranking=(), affect={}))

        verification = self.verifier.verify(env)  # truth, after the episode
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
                "agent": f"redundant_{self.fusion}",
                "policy": self.policy.name,
                "noise": self.noise,
                "plans": plans,
                "nodes_expanded": nodes,
                "disagreement_steps": disagreements,
                "latched": latched,
                "policy_denials": len(executor.denials),
                "beliefs": beliefs,
            },
        )
