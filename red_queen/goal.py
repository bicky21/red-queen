"""The recovery goal: a desired region of the state space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

from .state import SystemState

DEFICIT_KEYS = ("battery", "temperature", "service")


@dataclass(frozen=True)
class RecoveryGoal:
    min_battery: int = 60
    max_temperature: int = 50
    require_service: bool = True

    def is_satisfied(self, state: SystemState) -> bool:
        return (
            state.battery >= self.min_battery
            and state.temperature <= self.max_temperature
            and (state.service_up or not self.require_service)
        )

    def deficits(self, state: SystemState) -> Dict[str, float]:
        """How far each condition is from being met, on a 0-1 scale."""
        return {
            "battery": max(0, self.min_battery - state.battery) / 100.0,
            "temperature": max(0, state.temperature - self.max_temperature) / 100.0,
            "service": 1.0 if (self.require_service and not state.service_up) else 0.0,
        }

    def distance(self, state: SystemState, weights: Optional[Mapping[str, float]] = None) -> float:
        """Weighted sum of deficits. Zero exactly when the goal is satisfied."""
        deficits = self.deficits(state)
        if weights is None:
            return sum(deficits.values())
        return sum(weights[key] * deficits[key] for key in DEFICIT_KEYS)
