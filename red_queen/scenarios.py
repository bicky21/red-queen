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
    blocked: Tuple[Tuple[str, int, int], ...] = ()  # M1.1 scheduled action faults

    def make_env(self) -> SimulatedEnvironment:
        return SimulatedEnvironment(self.initial, dict(self.disturbances), self.blocked)


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

# M1.1: actions that silently fail on a fixed schedule. The agent's model
# does not know about these faults, so this is the first source of model error.
FAULT_SCENARIOS: Tuple[Scenario, ...] = (
    Scenario(
        "flaky_restart",
        "Service down; restart has no effect during steps 1-3.",
        SystemState(70, 30, False),
        blocked=(("restart_service", 1, 3),),
    ),
    Scenario(
        "flaky_restart_low_battery",
        "Service down and battery low; restart has no effect during steps 1-3.",
        SystemState(40, 30, False),
        blocked=(("restart_service", 1, 3),),
    ),
    Scenario(
        "charger_outage",
        "Battery low; charge has no effect during steps 1-2.",
        SystemState(20, 30, True),
        blocked=(("charge", 1, 2),),
    ),
)

M1_SCENARIO_NAMES: Tuple[str, ...] = tuple(s.name for s in SCENARIOS)

SCENARIOS = SCENARIOS + FAULT_SCENARIOS

# M3.2: extended validation scenarios. Kept OUT of SCENARIOS so every earlier
# experiment and check keeps exactly its previous inputs and results. Several
# were picked from the M3.2 fault-grid search because the agents disagree there.
EXTENDED_SCENARIOS: Tuple[Scenario, ...] = (
    Scenario("battery_empty", "Battery at zero, otherwise healthy.", SystemState(0, 30, True)),
    Scenario("max_heat_service_down", "Temperature at maximum and service down.", SystemState(60, 100, False)),
    Scenario(
        "deep_battery_flaky_restart",
        "Battery very low, service down, restart blocked steps 1-3.",
        SystemState(10, 30, False),
        blocked=(("restart_service", 1, 3),),
    ),
    Scenario(
        "flaky_restart_then_crash",
        "Restart blocked at step 1, then the service crashes again after step 2.",
        SystemState(30, 60, False),
        disturbances=((2, "service_crash"),),
        blocked=(("restart_service", 1, 1),),
    ),
    Scenario(
        "flaky_restart_heat_spike",
        "Restart blocked steps 1-3 and a heat spike after step 1.",
        SystemState(10, 30, False),
        disturbances=((1, "heat_spike"),),
        blocked=(("restart_service", 1, 3),),
    ),
    Scenario(
        "cool_outage_battery_drain",
        "Cool blocked steps 1-3 and a battery drain after step 1.",
        SystemState(10, 60, False),
        disturbances=((1, "battery_drain"),),
        blocked=(("cool", 1, 3),),
    ),
    Scenario(
        "charger_outage_overheating",
        "Hot with low battery; charge blocked steps 1-2.",
        SystemState(20, 90, True),
        blocked=(("charge", 1, 2),),
    ),
    Scenario(
        "double_disturbance",
        "Service down; heat spike after step 2 and battery drain after step 3.",
        SystemState(40, 40, False),
        disturbances=((2, "heat_spike"), (3, "battery_drain")),
    ),
)

ALL_SCENARIOS: Tuple[Scenario, ...] = SCENARIOS + EXTENDED_SCENARIOS

SCENARIOS_BY_NAME: Dict[str, Scenario] = {s.name: s for s in ALL_SCENARIOS}
if len(SCENARIOS_BY_NAME) != len(ALL_SCENARIOS):
    raise RuntimeError("duplicate scenario names")


def get_scenario(name: str) -> Scenario:
    try:
        return SCENARIOS_BY_NAME[name]
    except KeyError:
        raise KeyError(f"Unknown scenario: {name!r}") from None
