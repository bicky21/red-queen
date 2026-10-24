"""Fixed, named validation scenarios. No randomness anywhere."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .environment import SimulatedEnvironment
from .state import SystemState


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    initial: SystemState
    disturbances: Tuple[Tuple[int, str], ...] = ()
    max_steps: int = 20

    def make_env(self) -> SimulatedEnvironment:
        return SimulatedEnvironment(self.initial, dict(self.disturbances))


SCENARIOS: Tuple[Scenario, ...] = (
    Scenario("nominal", "Already healthy; no action needed.", SystemState(80, 30, True)),
    Scenario("low_battery", "Battery well below target.", SystemState(20, 30, True)),
    Scenario("overheating", "Temperature far above limit.", SystemState(70, 90, True)),
    Scenario("service_down", "Service stopped, otherwise healthy.", SystemState(70, 30, False)),
    Scenario(
        "overheated_service_down",
        "Service down and too hot to restart; must cool first.",
        SystemState(70, 90, False),
    ),
    Scenario("compound_failure", "Low battery, hot, service down.", SystemState(15, 85, False)),
    Scenario(
        "disturbance_mid_recovery",
        "Heat spike after step 1 and a service crash after step 3.",
        SystemState(30, 40, True),
        disturbances=((1, "heat_spike"), (3, "service_crash")),
    ),
)

SCENARIOS_BY_NAME: Dict[str, Scenario] = {s.name: s for s in SCENARIOS}


def get_scenario(name: str) -> Scenario:
    try:
        return SCENARIOS_BY_NAME[name]
    except KeyError:
        raise KeyError(f"Unknown scenario: {name!r}") from None
