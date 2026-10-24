"""Independent goal verification.

The verifier reads the environment's true state itself and checks each
goal condition explicitly. It does not use the agent's beliefs, the
scorer, or the distance function, so a bug there cannot hide a failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .environment import SimulatedEnvironment
from .goal import RecoveryGoal


@dataclass(frozen=True)
class VerificationResult:
    satisfied: bool
    checks: Dict[str, bool]
    state: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"satisfied": self.satisfied, "checks": dict(self.checks), "state": dict(self.state)}


class GoalVerifier:
    def __init__(self, goal: RecoveryGoal) -> None:
        self.goal = goal

    def verify(self, env: SimulatedEnvironment) -> VerificationResult:
        truth = env.observe()
        checks = {
            "battery_ok": truth.battery >= self.goal.min_battery,
            "temperature_ok": truth.temperature <= self.goal.max_temperature,
            "service_ok": truth.service_up or not self.goal.require_service,
        }
        return VerificationResult(all(checks.values()), checks, truth.to_dict())
