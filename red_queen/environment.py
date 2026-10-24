"""Deterministic simulated environment with scheduled disturbances.

The environment owns the true state. Actions arrive only through
`execute`, which accepts allowlisted action names and logs every event.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from .actions import ACTIONS, UnknownActionError, apply_action

# (action name, first step, last step) - inclusive, 1-based action steps.
BlockedWindow = Tuple[str, int, int]
from .state import SystemState

DISTURBANCES: Dict[str, Callable[[SystemState], SystemState]] = {
    "battery_drain": lambda s: s.with_changes(battery=s.battery - 40),
    "heat_spike": lambda s: s.with_changes(temperature=s.temperature + 45),
    "service_crash": lambda s: s.with_changes(service_up=False),
}


class UnknownDisturbanceError(ValueError):
    """Raised when a scenario schedules a disturbance that does not exist."""


@dataclass(frozen=True)
class Event:
    step: int
    kind: str  # "action", "blocked", "disturbance" or "rejected"
    name: str
    before: Dict[str, Any]
    after: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "kind": self.kind,
            "name": self.name,
            "before": dict(self.before),
            "after": dict(self.after),
        }


class SimulatedEnvironment:
    def __init__(
        self,
        initial_state: SystemState,
        disturbances: Optional[Mapping[int, str]] = None,
        blocked: Iterable[BlockedWindow] = (),
    ) -> None:
        schedule = dict(disturbances or {})
        for step, name in schedule.items():
            if name not in DISTURBANCES:
                raise UnknownDisturbanceError(f"Unknown disturbance: {name!r}")
            if not isinstance(step, int) or step < 1:
                raise ValueError(f"Disturbance step must be an int >= 1, got {step!r}")
        windows = tuple(tuple(w) for w in blocked)
        for window in windows:
            if len(window) != 3:
                raise ValueError(f"Blocked window must be (action, first, last), got {window!r}")
            action, first, last = window
            if action not in ACTIONS:
                raise UnknownActionError(f"Cannot block unknown action: {action!r}")
            if not (isinstance(first, int) and isinstance(last, int) and 1 <= first <= last):
                raise ValueError(f"Blocked window steps must satisfy 1 <= first <= last, got {window!r}")
        self._state = initial_state
        self._step = 0
        self._schedule = schedule
        self._blocked = windows
        self.events: List[Event] = []

    def is_blocked(self, action_name: str, step: int) -> bool:
        """True if this action has no effect when executed at this step.

        The agent is not told about blocked windows; it can only notice
        that an action produced no change.
        """
        return any(a == action_name and first <= step <= last for a, first, last in self._blocked)

    @property
    def step(self) -> int:
        return self._step

    def observe(self) -> SystemState:
        """Return the true current state (immutable, so safe to hand out)."""
        return self._state

    def execute(self, action_name: str) -> SystemState:
        """Run one allowlisted action, then any disturbance due at this step."""
        if action_name not in ACTIONS:
            snapshot = self._state.to_dict()
            self.events.append(Event(self._step, "rejected", str(action_name), snapshot, snapshot))
            raise UnknownActionError(f"Rejected non-allowlisted action: {action_name!r}")

        self._step += 1
        before = self._state
        if self.is_blocked(action_name, self._step):
            kind = "blocked"  # scheduled fault: the action has no effect
        else:
            kind = "action"
            self._state = apply_action(before, action_name)
        self.events.append(Event(self._step, kind, action_name, before.to_dict(), self._state.to_dict()))

        disturbance = self._schedule.get(self._step)
        if disturbance is not None:
            before = self._state
            self._state = DISTURBANCES[disturbance](before)
            self.events.append(
                Event(self._step, "disturbance", disturbance, before.to_dict(), self._state.to_dict())
            )
        return self._state
