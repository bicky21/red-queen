"""Immutable state of the simulated system.

The state is the single authoritative description of the simulated world.
It is frozen so no component can change it behind the environment's back.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict

MIN_LEVEL = 0
MAX_LEVEL = 100
_LEVEL_FIELDS = ("battery", "temperature")


def clamp_level(value: int) -> int:
    """Keep a numeric level inside [MIN_LEVEL, MAX_LEVEL]."""
    return max(MIN_LEVEL, min(MAX_LEVEL, int(value)))


@dataclass(frozen=True)
class SystemState:
    """Battery (0-100), temperature (0-100) and whether the service is up."""

    battery: int
    temperature: int
    service_up: bool

    def __post_init__(self) -> None:
        if not isinstance(self.service_up, bool):
            raise TypeError("service_up must be a bool")
        for name in _LEVEL_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
            if not MIN_LEVEL <= value <= MAX_LEVEL:
                raise ValueError(f"{name}={value} outside [{MIN_LEVEL}, {MAX_LEVEL}]")

    def with_changes(self, **changes: Any) -> "SystemState":
        """Return a new state; numeric levels are clamped to the valid range."""
        for name in _LEVEL_FIELDS:
            if name in changes:
                changes[name] = clamp_level(changes[name])
        return replace(self, **changes)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
