"""M3 policy enforcement.

A `Policy` is an explicit allowlist: a subset of the environment's action
allowlist. `PolicyExecutor` is the only path agents use to act. It checks
the policy and an action budget at the execution boundary, so a buggy or
adversarial decision-maker still cannot run a denied action. The
environment keeps its own allowlist underneath (defence in depth).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List

from .actions import ACTIONS
from .environment import SimulatedEnvironment
from .state import SystemState


class PolicyConfigError(ValueError):
    """Raised when a policy names actions the environment does not support."""


class PolicyDenied(PermissionError):
    """Raised by PolicyExecutor when an action may not be executed."""


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str


@dataclass(frozen=True)
class Policy:
    name: str = "default"
    allowed: FrozenSet[str] = frozenset(ACTIONS)

    def __post_init__(self) -> None:
        if not isinstance(self.allowed, frozenset):
            raise PolicyConfigError("allowed must be a frozenset")
        unknown = sorted(a for a in self.allowed if a not in ACTIONS)
        if unknown:
            raise PolicyConfigError(f"Policy {self.name!r} allows unknown actions: {unknown}")

    def check(self, action: str) -> PolicyDecision:
        if action not in ACTIONS:
            return PolicyDecision(False, "not in environment allowlist")
        if action not in self.allowed:
            return PolicyDecision(False, f"not allowed by policy {self.name!r}")
        return PolicyDecision(True, "allowed")


DEFAULT_POLICY = Policy()
POLICIES: Dict[str, Policy] = {
    "default": DEFAULT_POLICY,
    "no_restart": Policy("no_restart", frozenset(ACTIONS) - {"restart_service"}),
    "no_cool": Policy("no_cool", frozenset(ACTIONS) - {"cool"}),
}

# M3.2: extra policy for the extended study only. Kept out of POLICIES so the
# M3 planning experiment keeps exactly its previous inputs and results.
WAIT_ONLY_POLICY = Policy("wait_only", frozenset({"wait"}))
ALL_POLICIES: Dict[str, Policy] = {**POLICIES, "wait_only": WAIT_ONLY_POLICY}


class PolicyExecutor:
    """Execution boundary: policy check + action budget, then the environment."""

    def __init__(self, env: SimulatedEnvironment, policy: Policy, max_actions: int) -> None:
        if max_actions < 0:
            raise ValueError("max_actions must be >= 0")
        self.env = env
        self.policy = policy
        self.max_actions = max_actions
        self.executed = 0
        self.denials: List[Dict[str, Any]] = []

    def execute(self, action: str) -> SystemState:
        decision = self.policy.check(action)
        reason = decision.reason
        if decision.allowed and self.executed >= self.max_actions:
            decision, reason = PolicyDecision(False, "action budget exhausted"), "action budget exhausted"
        if not decision.allowed:
            self.denials.append({"after_actions": self.executed, "action": str(action), "reason": reason})
            raise PolicyDenied(f"{action!r} denied: {reason}")
        self.executed += 1
        return self.env.execute(action)
