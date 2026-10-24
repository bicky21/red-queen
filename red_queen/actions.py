"""The fixed allowlist of simulated actions and their deterministic effects.

Only names in ACTIONS can ever be executed. There is no shell, network,
file or external-service access anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .state import SystemState

# A restart only brings the service up if the system is cool enough.
RESTART_MAX_TEMPERATURE = 80


class UnknownActionError(ValueError):
    """Raised when something tries to use an action outside the allowlist."""


@dataclass(frozen=True)
class ActionSpec:
    name: str
    cost: float
    description: str


ACTIONS: Dict[str, ActionSpec] = {
    spec.name: spec
    for spec in (
        ActionSpec("charge", 0.005, "Battery +20, temperature +5."),
        ActionSpec("cool", 0.005, "Temperature -20."),
        ActionSpec(
            "restart_service",
            0.01,
            f"Battery -10; service starts only if temperature <= {RESTART_MAX_TEMPERATURE}.",
        ),
        ActionSpec("wait", 0.01, "No change."),
    )
}

# Sorted so every loop over actions happens in the same order every run.
ACTION_NAMES: Tuple[str, ...] = tuple(sorted(ACTIONS))


def apply_action(state: SystemState, name: str) -> SystemState:
    """Deterministic transition model shared by the environment and scorer.

    M1 limitation: the agent's model is exact. Later milestones should test
    an agent whose model can be wrong.
    """
    if name not in ACTIONS:
        raise UnknownActionError(f"Unknown action: {name!r}")
    if name == "charge":
        return state.with_changes(battery=state.battery + 20, temperature=state.temperature + 5)
    if name == "cool":
        return state.with_changes(temperature=state.temperature - 20)
    if name == "restart_service":
        started = state.temperature <= RESTART_MAX_TEMPERATURE
        return state.with_changes(battery=state.battery - 10, service_up=state.service_up or started)
    return state  # wait
