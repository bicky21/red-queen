"""M5 belief agent: acting safely when the state is only seen through noise.

The agent never sees the true state. Each step it reads the sensor and keeps
a belief Box (an interval for battery and temperature; the service status
is read exactly). It then:

  1. stops if the STOP box satisfies the goal for every state inside it;
  2. otherwise plans (BoundedPlanner, BFS) from the box's WORST corner
     (lowest battery, highest temperature) and executes only the first
     action through the PolicyExecutor;
  3. reads again and repeats (receding horizon).

Why the worst corner is enough: every action's effect is monotone in
battery (up is better) and temperature (down is better), and the goal is
monotone in the same directions. So a plan that reaches the goal from the
worst corner reaches it from every state in the box (checked exhaustively
in validation V134).

Modes:
  "interval"        belief = the current reading's observation box only.
                    Always contains the truth (bounded noise), so stopping
                    is sound. Forgets everything between steps.
  "filter"          belief = model prediction of the last belief, intersected
                    with the new observation box; reset to the observation
                    box when they do not overlap. Tighter, but if a fault or
                    disturbance happens and the boxes still overlap, the
                    belief can exclude the truth. It stops on this belief.
  "filter_verified" plans on the filter belief, but stops only when the
                    current observation box (always sound) satisfies the goal.

Assumption (stated, not hidden): the agent knows the sensor noise bound.
Success is decided only by the independent GoalVerifier on the true state.

M5.1 `adapt_noise=True`: after every action the agent checks whether the
new reading can be EXPLAINED from the previous reading with the assumed
bound by some known event: the action worked or was blocked (a fault),
optionally followed by a known disturbance. If no explanation fits, the
bound must be too small (or something unknown happened), so the agent
widens its bound to the smallest value that explains the readings (capped
at MAX_NOISE) and uses it from then on. It only ever widens.
If the assumed bound is correct and only known events happen, the true
explanation always fits, so nothing changes (proved in the docs, checked in
V143). A constant sensor bias shifts every reading equally and can NOT be
detected this way (measured in the M5.1 study).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .actions import apply_action
from .environment import DISTURBANCES
from .agent import STOP_GOAL, STOP_MAX_STEPS, STOP_POLICY, EpisodeResult, StepRecord
from .goal import RecoveryGoal
from .planner import BoundedPlanner
from .planning_agent import STOP_NO_PLAN
from .policy import DEFAULT_POLICY, Policy, PolicyDenied, PolicyExecutor
from .sensors import MAX_NOISE, Box, NoisySensor, SensedEnvironment, observation_box
from .state import SystemState
from .verification import GoalVerifier

BELIEF_MODES = ("interval", "filter", "filter_verified")


def predict_box(box: Box, action: str) -> Box:
    """Model prediction of a box after one action (exact for monotone effects).

    Battery effects do not depend on temperature and vice versa, so applying
    the action to the low and high corners gives the predicted intervals.
    The service status is not predicted; it is read exactly next step.
    """
    lo = apply_action(SystemState(box.b_lo, box.t_lo, box.service_up), action)
    hi = apply_action(SystemState(box.b_hi, box.t_hi, box.service_up), action)
    return Box(lo.battery, hi.battery, lo.temperature, hi.temperature, box.service_up)


# M5.1: known events that can change battery/temperature after an action.
LEVEL_DISTURBANCES = ("battery_drain", "heat_spike")  # service_crash only changes the service


def _apply_to_box(box: Box, fn) -> Box:
    lo = fn(SystemState(box.b_lo, box.t_lo, box.service_up))
    hi = fn(SystemState(box.b_hi, box.t_hi, box.service_up))
    return Box(lo.battery, hi.battery, lo.temperature, hi.temperature, box.service_up)


def explanation_boxes(prev: Box, action: str) -> List[Box]:
    """Where the state can be after `action` under every known explanation:
    action worked or was blocked, then no disturbance or one known one."""
    out = []
    for base in (predict_box(prev, action), prev):
        out.append(base)
        out.extend(_apply_to_box(base, DISTURBANCES[d]) for d in LEVEL_DISTURBANCES)
    return out


def _gap(lo1: int, hi1: int, lo2: int, hi2: int) -> int:
    return max(0, lo1 - hi2, lo2 - hi1)


def unexplained_gap(prev: Box, new: Box, action: str) -> int:
    """0 if some known explanation is consistent with both boxes; otherwise
    the smallest distance by which every explanation misses."""
    return min(max(_gap(e.b_lo, e.b_hi, new.b_lo, new.b_hi), _gap(e.t_lo, e.t_hi, new.t_lo, new.t_hi))
               for e in explanation_boxes(prev, action))


def needed_noise(prev_reading: SystemState, reading: SystemState, action: str, noise: int,
                 cap: int = MAX_NOISE) -> int:
    """Smallest bound >= noise (<= cap) that explains the two readings."""
    n = noise
    while n < cap and unexplained_gap(observation_box(prev_reading, n), observation_box(reading, n), action) > 0:
        n += 1
    return n


def intersect(a: Box, b: Box) -> Optional[Box]:
    """Overlap of two boxes, using b's service status; None if disjoint."""
    b_lo, b_hi = max(a.b_lo, b.b_lo), min(a.b_hi, b.b_hi)
    t_lo, t_hi = max(a.t_lo, b.t_lo), min(a.t_hi, b.t_hi)
    if b_lo > b_hi or t_lo > t_hi:
        return None
    return Box(b_lo, b_hi, t_lo, t_hi, b.service_up)


class BeliefAgent:
    def __init__(
        self,
        goal: RecoveryGoal,
        noise: int,
        mode: str = "interval",
        planner: Optional[BoundedPlanner] = None,
        policy: Policy = DEFAULT_POLICY,
        max_steps: int = 20,
        max_policy_denials: int = 3,
        adapt_noise: bool = False,
    ) -> None:
        if mode not in BELIEF_MODES:
            raise ValueError(f"mode must be one of {BELIEF_MODES}")
        if max_steps < 0 or max_policy_denials < 1:
            raise ValueError("invalid bounds")
        observation_box(SystemState(50, 50, True), noise)  # validates the noise bound
        self.goal = goal
        self.noise = noise
        self.mode = mode
        self.planner = planner if planner is not None else BoundedPlanner(goal)
        self.policy = policy
        self.max_steps = max_steps
        self.max_policy_denials = max_policy_denials
        self.adapt_noise = bool(adapt_noise)
        self.verifier = GoalVerifier(goal)

    def run(self, env, sensor: NoisySensor, scenario: str = "") -> EpisodeResult:
        sensed = SensedEnvironment(env, sensor)  # every decision below uses this only
        executor = PolicyExecutor(sensed, self.policy, self.max_steps)
        actions: List[str] = []
        records: List[StepRecord] = []
        beliefs: List[Dict] = []
        belief: Optional[Box] = None
        last_action: Optional[str] = None
        plans = nodes = resets = fallbacks = 0
        stop_reason: Optional[str] = None
        noise = self.noise                       # M5.1: may only grow when adapt_noise
        widenings: List[Dict] = []
        unexplained_at_cap = 0
        prev_reading: Optional[SystemState] = None
        checked_step = 0

        while True:
            reading = sensed.observe()
            if self.adapt_noise and prev_reading is not None and last_action is not None \
                    and sensed.step != checked_step:
                checked_step = sensed.step
                need = needed_noise(prev_reading, reading, last_action, noise)
                if need == MAX_NOISE and unexplained_gap(observation_box(prev_reading, need),
                                                         observation_box(reading, need), last_action) > 0:
                    unexplained_at_cap += 1
                if need > noise:
                    widenings.append({"step": sensed.step, "from": noise, "to": need})
                    noise = need
                    belief = None  # the old belief used a bound now known to be too small
            obs = observation_box(reading, noise)
            if self.mode == "interval" or belief is None or last_action is None:
                belief = obs
            else:
                merged = intersect(predict_box(belief, last_action), obs)
                if merged is None:  # the model was wrong (fault/disturbance): trust the reading
                    resets += 1
                    belief = obs
                else:
                    belief = merged
            beliefs.append({"step": sensed.step, **belief.to_dict()})
            stop_box = obs if self.mode == "filter_verified" else belief
            if stop_box.satisfies(self.goal):
                stop_reason = STOP_GOAL
                break
            if len(actions) >= self.max_steps:
                stop_reason = STOP_MAX_STEPS
                break
            plan_box = belief
            if belief.satisfies(self.goal):  # filter_verified only: belief says done, reading does not
                plan_box = obs
                fallbacks += 1
            plan = self.planner.plan(plan_box.worst(), self.policy.allowed)
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
            last_action = action
            prev_reading = reading
            records.append(StepRecord(step=len(actions), observed=reading.to_dict(), chosen=action,
                                      score=0.0, ranking=(), affect={}))

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
                "agent": f"belief_{self.mode}",
                "policy": self.policy.name,
                "noise": self.noise,
                **({"adapt_noise": True, "noise_final": noise, "widenings": widenings,
                    "unexplained_at_cap": unexplained_at_cap} if self.adapt_noise else {}),
                "plans": plans,
                "nodes_expanded": nodes,
                "belief_resets": resets,
                "stop_fallbacks": fallbacks,
                "policy_denials": len(executor.denials),
                "beliefs": beliefs,
            },
        )
