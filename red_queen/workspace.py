"""M8 global workspace (pre-registered in docs/PREREG_M8.md).

Global Workspace Theory, as indicator properties (Butlin, Long et al. 2023):
  GWT-1 several specialised modules work in parallel
  GWT-2 a limited-capacity workspace (bottleneck)
  GWT-3 the workspace contents are broadcast to ALL modules
  GWT-4 what enters the workspace depends on the current state

Each cycle (one per decision point): every module proposes messages with a
salience in [0, 1]; the workspace selects the top `capacity` (ties: source,
then kind); every module receives every selected message; the selection is
logged as the episode's workspace stream.

The only behavioural effect allowed: MemoryModule switches disturbance
memory OFF when it receives a trigger broadcast. The workspace can only
remove memory's influence; it never adds actions and never touches the
estimator, the policy or the stopping rule. No claim of consciousness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .self_model import SelfModel, features

MEMORY_OFF_KINDS = frozenset({"expect_failure", "sensor_isolated", "reading_unexplainable"})
SENSOR_SALIENCE = {"sensor_isolated": 0.9, "reading_unexplainable": 0.9, "sensors_disagree": 0.6,
                   "bound_widened": 0.5}
MEMORY_SALIENCE = 0.4
STATUS_SALIENCE = 0.1
SELF_THRESHOLD = 0.3  # M7.1


@dataclass(frozen=True)
class Message:
    source: str
    kind: str
    salience: float
    content: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.salience) <= 1.0:
            raise ValueError("salience must be in [0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "kind": self.kind, "salience": round(float(self.salience), 4),
                "content": dict(self.content)}


class Module:
    name = "module"

    def propose(self, snapshot: Mapping[str, Any], context: Mapping[str, Any]) -> List[Message]:
        return []

    def receive(self, message: Message) -> None:
        pass


class SelfModule(Module):
    name = "self"

    def __init__(self, model: SelfModel, threshold: float = SELF_THRESHOLD) -> None:
        if not model.fitted:
            raise ValueError("self-model is not fitted")
        self.model = model
        self.threshold = threshold
        self.received = 0

    def propose(self, snapshot, context):
        p = self.model.predict(features(snapshot))
        if p < self.threshold:
            return [Message(self.name, "expect_failure", 1.0 - p, {"confidence": round(p, 4)})]
        return []

    def receive(self, message):
        self.received += 1


class SensorModule(Module):
    name = "sensor"

    def __init__(self) -> None:
        self.seen: Dict[str, int] = {}
        self.received = 0

    def propose(self, snapshot, context):
        out = []
        counts = snapshot["alert_counts"]
        for kind in sorted(SENSOR_SALIENCE):
            new = counts.get(kind, 0) - self.seen.get(kind, 0)
            if new > 0:
                out.append(Message(self.name, kind, SENSOR_SALIENCE[kind], {"new": new}))
        self.seen = dict(counts)
        return out

    def receive(self, message):
        self.received += 1


class MemoryModule(Module):
    name = "memory"

    def __init__(self) -> None:
        self.memory_off = False
        self.off_reason: Optional[str] = None
        self.received = 0

    def propose(self, snapshot, context):
        d = context.get("next_disturbance")
        if d and not self.memory_off:
            return [Message(self.name, "disturbance_expected", MEMORY_SALIENCE, {"disturbance": d})]
        return []

    def receive(self, message):
        self.received += 1
        if message.kind in MEMORY_OFF_KINDS and not self.memory_off:
            self.memory_off = True
            self.off_reason = message.kind


class PlannerModule(Module):
    name = "planner"

    def __init__(self) -> None:
        self.received = 0

    def propose(self, snapshot, context):
        status = "stopping" if snapshot["stopping"] else ("no_plan" if snapshot["plan_len"] is None else "plan")
        return [Message(self.name, "status", STATUS_SALIENCE,
                        {"status": status, "plan_len": snapshot["plan_len"], "steps_left": snapshot["steps_left"]})]

    def receive(self, message):
        self.received += 1


class OperatorModule(Module):
    name = "operator"

    def __init__(self) -> None:
        self.stream: List[Dict[str, Any]] = []
        self.received = 0

    def receive(self, message):
        self.received += 1
        self.stream.append(message.to_dict())


class Workspace:
    """Limited-capacity selection + global broadcast. Create one per episode."""

    def __init__(self, modules: Sequence[Module], capacity: Optional[int] = 1) -> None:
        if capacity is not None and (isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1):
            raise ValueError("capacity must be an int >= 1 or None")
        names = [m.name for m in modules]
        if len(set(names)) != len(names):
            raise ValueError("module names must be unique")
        self.modules = list(modules)
        self.capacity = capacity
        self.cycles: List[Dict[str, Any]] = []

    def cycle(self, snapshot: Mapping[str, Any], context: Mapping[str, Any]) -> List[Message]:
        proposals: List[Message] = []
        participants = 0
        for m in self.modules:
            participants += 1
            proposals.extend(m.propose(snapshot, context))
        ranked = sorted(proposals, key=lambda msg: (-float(msg.salience), msg.source, msg.kind))
        selected = ranked if self.capacity is None else ranked[: self.capacity]
        for msg in selected:
            for m in self.modules:  # global broadcast: every module receives every selected message
                m.receive(msg)
        self.cycles.append({"step": snapshot["step"], "participants": participants, "proposed": len(proposals),
                            "broadcast": [m.to_dict() for m in selected]})
        return selected

    def module(self, name: str) -> Module:
        for m in self.modules:
            if m.name == name:
                return m
        raise KeyError(name)

    def wants_memory_off(self) -> bool:
        try:
            return bool(self.module("memory").memory_off)
        except KeyError:
            return False

    def report(self) -> Dict[str, Any]:
        op = None
        try:
            op = self.module("operator")
        except KeyError:
            pass
        return {"capacity": self.capacity, "cycles": len(self.cycles),
                "proposed": sum(c["proposed"] for c in self.cycles),
                "broadcast": sum(len(c["broadcast"]) for c in self.cycles),
                "stream": list(op.stream) if op is not None else None}


def standard_workspace(self_model: SelfModel, capacity: Optional[int] = 1) -> Workspace:
    """The pre-registered module set."""
    return Workspace([SelfModule(self_model), SensorModule(), MemoryModule(), PlannerModule(), OperatorModule()],
                     capacity)
