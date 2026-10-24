"""Deterministic simulated environment with scheduled disturbances.

The environment owns the true state. Actions arrive only through
`execute`, which accepts allowlisted action names and logs every event.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional

from .actions import ACTIONS, UnknownActionError, apply_action
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
    kind: str  # "action", "disturbance" or "rejected"
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
    ) -> None:
        schedule = dict(disturbances or {})
        for step, name in schedule.items():
            if name not in DISTURBANCES:
                raise UnknownDisturbanceError(f"Unknown disturbance: {name!r}")
            if not isinstance(step, int) or step < 1:
                raise ValueError(f"Disturbance step must be an int >= 1, got {step!r}")
        self._state = initial_state
        self._step = 0
        self._schedule = schedule
        self.events: List[Event] = []

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
        self._state = apply_action(before, action_name)
        self.events.append(Event(self._step, "action", action_name, before.to_dict(), self._state.to_dict()))

        disturbance = self._schedule.get(self._step)
        if disturbance is not None:
            before = self._state
            self._state = DISTURBANCES[disturbance](before)
            self.events.append(
                Event(self._step, "disturbance", disturbance, before.to_dict(), self._state.to_dict())
            )
        return self._state
