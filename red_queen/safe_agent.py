"""M5.3 consolidated safety layer: SafeEstimator + SafeAgent.

M5 to M5.2 added three separate protections in three separate agents:
  M5   interval belief, stop only when every state in the box meets the goal
  M5.1 widen a sensor's noise bound when its readings cannot be explained
  M5.2 read two sensors and use the hull of their boxes

SafeEstimator combines them in ONE state-estimation component, separate from
planning and execution, and adds two things:

  * an operator report: a list of alerts (bound widened, sensors disagree,
    reading unexplainable) with step numbers and numeric evidence only;
  * a defined response when a sensor's consecutive readings cannot be
    explained by any known event even at the maximum noise bound:
      on_unexplainable="halt"    - the estimate is declared untrustworthy and
                                   the agent stops with reason "safe_halt",
                                   claiming nothing;
      on_unexplainable="isolate" - that sensor is excluded for the rest of the
                                   episode and the other sensor is used; the
                                   agent halts only when no trusted sensor is
                                   left (always, with a single sensor).
    "isolate" was added after the first study run showed "halt" stopping
    even though the second sensor was healthy (post-hoc; see the M5.3 record).
  * M5.4 `cross_isolate=True` (pre-registered in docs/PREREG_M5_4.md): when
    both sensors are trusted, their boxes do not overlap, and exactly one of
    them has needed its noise bound widened, that one is isolated
    (alert sensor_isolated, reason "cross_sensor"). Off by default unless the
    pre-registered decision rule adopts it.

Configurations reproduce the earlier agents exactly (checked per run in V156):
  SafeAgent(sensors=1, adapt=False) == BeliefAgent "interval"            (M5)
  SafeAgent(sensors=1, adapt=True)  == BeliefAgent "interval", adapt     (M5.1)
  SafeAgent(sensors=2, adapt=False) == RedundantBeliefAgent "hull"       (M5.2)
  SafeAgent(sensors=2, adapt=True)  -> new: hull of per-sensor adapted boxes

M6 (pre-registered in docs/PREREG_M6.md): optional untrusted disturbance
memory. With `advisor` (anything offering remembered_disturbances(), e.g.
the M3.5 MemoryAdvisor) and `predict_disturbances=True`, each plan from the
belief's worst corner also predicts remembered disturbances at future steps
(the M3.5 planner mode, with its fallback). Memory never reaches the
estimator or the stopping rule. `disturbance_guard="on_mismatch"`: if the new
belief does not overlap the box predicted from the previous belief (action
worked + remembered disturbance), disturbance prediction is switched off
for the rest of the episode.

M7 (pre-registered in docs/PREREG_M7.md): `run(..., observer=None)`. A
read-only introspection hook, called at each decision point (after the plan
is computed, and at the moment the agent decides to stop) with a snapshot of
the agent's OWN state only: belief box and width, alert counts, trusted
sensors, noise bounds, worst-corner plan length (None if no plan reaches the
goal), steps left, and whether it is stopping. Never the true state. The
observer's return value is ignored and its exceptions are caught and
counted, so it cannot change behaviour.

Stated assumptions: at most one sensor is biased; the KINDS of events are
known (blocked actions, the two level disturbances). Success is decided by
the independent GoalVerifier on the true state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .agent import STOP_GOAL, STOP_MAX_STEPS, STOP_POLICY, EpisodeResult, StepRecord
from .belief_agent import _apply_to_box, intersect, needed_noise, predict_box, unexplained_gap
from .environment import DISTURBANCES
from .goal import RecoveryGoal
from .planner import BoundedPlanner
from .planning_agent import STOP_NO_PLAN
from .policy import DEFAULT_POLICY, Policy, PolicyDenied, PolicyExecutor
from .redundant_agent import hull
from .sensors import MAX_NOISE, Box, SensedEnvironment, observation_box
from .state import SystemState
from .verification import GoalVerifier

STOP_SAFE_HALT = "safe_halt"
MEMORY_GUARDS = (None, "on_mismatch")  # M6
ALERT_KINDS = ("bound_widened", "sensors_disagree", "reading_unexplainable", "sensor_isolated")
ON_UNEXPLAINABLE = ("halt", "isolate")


@dataclass(frozen=True)
class Alert:
    step: int
    kind: str
    sensor: str  # "A", "B" or "A+B"
    detail: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"step": self.step, "kind": self.kind, "sensor": self.sensor, "detail": dict(self.detail)}


@dataclass
class Estimate:
    belief: Optional[Box]          # None only when the estimate is untrustworthy (halt)
    halt: bool
    new_alerts: List[Alert] = field(default_factory=list)


class SafeEstimator:
    """State estimation only: readings in, belief box (or halt) out. Never acts."""

    def __init__(self, noise: int, sensors: int = 2, adapt: bool = True, on_unexplainable: str = "isolate",
                 cross_isolate: bool = False) -> None:
        if sensors not in (1, 2):
            raise ValueError("sensors must be 1 or 2")
        if on_unexplainable not in ON_UNEXPLAINABLE:
            raise ValueError(f"on_unexplainable must be one of {ON_UNEXPLAINABLE}")
        observation_box(SystemState(50, 50, True), noise)  # validates the noise bound
        self.sensors = sensors
        self.adapt = bool(adapt)
        self.on_unexplainable = on_unexplainable
        if cross_isolate and sensors != 2:
            raise ValueError("cross_isolate needs two sensors")
        self.cross_isolate = bool(cross_isolate)
        self.initial_noise = noise
        self.noise = [noise] * sensors
        self.trusted = [True] * sensors
        self.alerts: List[Alert] = []
        self._prev: Optional[List[SystemState]] = None
        self._checked_step = 0

    def update(self, readings: Sequence[SystemState], step: int, last_action: Optional[str]) -> Estimate:
        if len(readings) != self.sensors:
            raise ValueError(f"expected {self.sensors} readings")
        new: List[Alert] = []
        halt = False
        names = ("A", "B")
        if self.adapt and self._prev is not None and last_action is not None and step != self._checked_step:
            self._checked_step = step
            for i, (prev, cur) in enumerate(zip(self._prev, readings)):
                if not self.trusted[i]:
                    continue
                need = needed_noise(prev, cur, last_action, self.noise[i])
                if need == MAX_NOISE and unexplained_gap(observation_box(prev, need),
                                                         observation_box(cur, need), last_action) > 0:
                    new.append(Alert(step, "reading_unexplainable", names[i],
                                     {"previous": prev.to_dict(), "reading": cur.to_dict(), "action": last_action}))
                    others = any(self.trusted[j] for j in range(self.sensors) if j != i)
                    if self.on_unexplainable == "isolate" and others:
                        self.trusted[i] = False
                        new.append(Alert(step, "sensor_isolated", names[i],
                                         {"remaining": self.sensors - 1, "reason": "unexplainable"}))
                        continue
                    halt = True
                if need > self.noise[i]:
                    new.append(Alert(step, "bound_widened", names[i], {"from": self.noise[i], "to": need}))
                    self.noise[i] = need
        boxes = [observation_box(r, n) for r, n in zip(readings, self.noise)]
        if self.sensors == 2 and all(self.trusted) and intersect(boxes[0], boxes[1]) is None:
            new.append(Alert(step, "sensors_disagree", "A+B",
                             {"A": boxes[0].to_dict(), "B": boxes[1].to_dict()}))
            grown = [n > self.initial_noise for n in self.noise]
            if self.cross_isolate and grown.count(True) == 1:
                i = grown.index(True)
                self.trusted[i] = False
                new.append(Alert(step, "sensor_isolated", names[i], {"remaining": 1, "reason": "cross_sensor"}))
        self.alerts.extend(new)
        if halt:
            return Estimate(None, True, new)
        use = [b for b, ok in zip(boxes, self.trusted) if ok]
        belief = use[0] if len(use) == 1 else hull(use[0], use[1])
        return Estimate(belief, False, new)

    def action_taken(self, readings: Sequence[SystemState]) -> None:
        """Remember the readings taken just before an executed action."""
        self._prev = list(readings)

    def report(self) -> Dict[str, Any]:
        counts = {k: sum(a.kind == k for a in self.alerts) for k in ALERT_KINDS}
        return {"noise_bounds": list(self.noise), "trusted": list(self.trusted), "alert_counts": counts,
                "alerts": [a.to_dict() for a in self.alerts]}


class SafeAgent:
    def __init__(
        self,
        goal: RecoveryGoal,
        noise: int,
        sensors: int = 2,
        adapt: bool = True,
        on_unexplainable: str = "isolate",
        cross_isolate: bool = False,
        planner: Optional[BoundedPlanner] = None,
        policy: Policy = DEFAULT_POLICY,
        max_steps: int = 20,
        max_policy_denials: int = 3,
        advisor=None,
        predict_disturbances: bool = False,
        disturbance_guard: Optional[str] = None,
    ) -> None:
        if max_steps < 0 or max_policy_denials < 1:
            raise ValueError("invalid bounds")
        SafeEstimator(noise, sensors, adapt, on_unexplainable, cross_isolate)  # validates the configuration now
        if predict_disturbances and not callable(getattr(advisor, "remembered_disturbances", None)):
            raise ValueError("predict_disturbances needs an advisor with remembered_disturbances()")
        if disturbance_guard not in MEMORY_GUARDS:
            raise ValueError(f"disturbance_guard must be one of {MEMORY_GUARDS}")
        if disturbance_guard is not None and not predict_disturbances:
            raise ValueError("disturbance_guard needs predict_disturbances=True")
        self.advisor = advisor
        self.predict_disturbances = bool(predict_disturbances)
        self.disturbance_guard = disturbance_guard
        self.goal = goal
        self.noise = noise
        self.sensors = sensors
        self.adapt = bool(adapt)
        self.on_unexplainable = on_unexplainable
        self.cross_isolate = bool(cross_isolate)
        self.planner = planner if planner is not None else BoundedPlanner(goal)
        self.policy = policy
        self.max_steps = max_steps
        self.max_policy_denials = max_policy_denials
        self.verifier = GoalVerifier(goal)

    def run(self, env, sensors: Sequence, scenario: str = "", observer=None) -> EpisodeResult:
        sensors = list(sensors)
        if len(sensors) != self.sensors:
            raise ValueError(f"expected {self.sensors} sensor(s)")
        if self.sensors == 2 and sensors[0] is sensors[1]:
            raise ValueError("redundancy needs two separate sensors")
        sensed = [SensedEnvironment(env, s) for s in sensors]  # decisions use readings only
        estimator = SafeEstimator(self.noise, self.sensors, self.adapt, self.on_unexplainable,
                                  self.cross_isolate)
        executor = PolicyExecutor(sensed[0], self.policy, self.max_steps)
        actions: List[str] = []
        records: List[StepRecord] = []
        last_action: Optional[str] = None
        plans = nodes = 0
        stop_reason: Optional[str] = None
        remembered: Dict[int, str] = {}
        if self.predict_disturbances:  # untrusted: keep only known disturbance names at positive int steps
            raw = self.advisor.remembered_disturbances()
            remembered = {k: v for k, v in dict(raw).items()
                          if isinstance(k, int) and not isinstance(k, bool) and k >= 1 and v in DISTURBANCES}
        remembered_at_start = dict(remembered)
        guard_tripped_at: Optional[int] = None
        acted_from: Optional[Box] = None  # belief the last action was planned from
        observer_errors = 0

        def introspect(belief: Box, plan_len: Optional[int], stopping: bool) -> None:
            nonlocal observer_errors
            if observer is None:
                return
            report = estimator.report()
            snapshot = {"step": len(actions), "belief": belief.to_dict(), "width": belief.width(),
                        "alert_counts": dict(report["alert_counts"]), "trusted": list(report["trusted"]),
                        "noise_bounds": list(report["noise_bounds"]), "plan_len": plan_len,
                        "steps_left": self.max_steps - len(actions), "stopping": stopping}
            try:
                observer(snapshot)
            except Exception:  # an observer can never affect the agent
                observer_errors += 1

        while True:
            readings = [s.observe() for s in sensed]
            est = estimator.update(readings, sensed[0].step, last_action)
            if est.halt:
                stop_reason = STOP_SAFE_HALT
                break
            if self.disturbance_guard and remembered and acted_from is not None and last_action is not None:
                expected = predict_box(acted_from, last_action)
                d = remembered.get(len(actions))
                if d is not None:
                    expected = _apply_to_box(expected, DISTURBANCES[d])
                if intersect(expected, est.belief) is None:
                    guard_tripped_at = len(actions)
                    remembered.clear()  # memory (or the model) proved wrong: stop predicting
            if est.belief.satisfies(self.goal):
                introspect(est.belief, 0, True)
                stop_reason = STOP_GOAL
                break
            if len(actions) >= self.max_steps:
                stop_reason = STOP_MAX_STEPS
                break
            if remembered:
                offset = len(actions)
                plan = self.planner.plan(est.belief.worst(), self.policy.allowed,
                                         disturbances=lambda depth, offset=offset: remembered.get(offset + depth + 1))
            else:
                plan = self.planner.plan(est.belief.worst(), self.policy.allowed)
            plans += 1
            nodes += plan.nodes_expanded
            introspect(est.belief, len(plan.actions) if plan.reaches_goal else None, False)
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
            acted_from = est.belief
            estimator.action_taken(readings)
            records.append(StepRecord(step=len(actions), observed={n: r.to_dict() for n, r in zip("AB", readings)},
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
                "agent": f"safe_{self.sensors}s_{'adapt' if self.adapt else 'fixed'}",
                "on_unexplainable": self.on_unexplainable,
                "cross_isolate": self.cross_isolate,
                "policy": self.policy.name,
                "noise": self.noise,
                "plans": plans,
                "nodes_expanded": nodes,
                "policy_denials": len(executor.denials),
                "operator_report": estimator.report(),
                **({"remembered_disturbances": {str(k): v for k, v in sorted(remembered_at_start.items())}}
                   if self.predict_disturbances else {}),
                **({"guard_tripped_at": guard_tripped_at} if self.disturbance_guard else {}),
                **({"observer_errors": observer_errors} if observer is not None else {}),
            },
        )
