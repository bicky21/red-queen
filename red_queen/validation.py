"""M1 validation checks.

Run:  python -m red_queen.validation
Exit code 0 only if every check passes.
"""

from __future__ import annotations

import sys
from typing import Callable, List, Tuple

from .actions import UnknownActionError
from .affect import AffectConfig
from .agent import STOP_MAX_STEPS, RecoveryAgent
from .environment import SimulatedEnvironment
from .experiments import run_affect_comparison, run_scenario
from .goal import RecoveryGoal
from .scenarios import FAULT_SCENARIOS, M1_SCENARIO_NAMES, SCENARIOS, get_scenario
from .state import SystemState
from .sweeps import run_sweep, sweep_configs
from .verification import GoalVerifier

CheckResult = Tuple[bool, str]
# V02-V13 stay tied to the M1 scenarios so their numbering never shifts.
RECOVERY_SCENARIOS = [get_scenario(n) for n in M1_SCENARIO_NAMES if n != "nominal"]

# Frozen M1 behaviour (default affect config), recorded at commit 6f4bef5.
# Any change here is a behaviour regression and must be explained.
M1_FROZEN_SEQUENCES = {
    "nominal": (),
    "low_battery": ("charge", "charge"),
    "overheating": ("cool", "cool"),
    "service_down": ("restart_service",),
    "overheated_service_down": ("cool", "restart_service", "cool"),
    "compound_failure": ("cool", "restart_service", "charge", "cool", "charge", "charge", "cool"),
    "disturbance_mid_recovery": ("charge", "cool", "cool", "restart_service", "charge", "cool"),
}


def check_nominal_no_action() -> CheckResult:
    r = run_scenario(get_scenario("nominal"), affect_enabled=True)
    return r.success and r.action_count == 0, f"success={r.success} actions={r.action_count}"


def make_recovery_check(name: str, affect_enabled: bool) -> Callable[[], CheckResult]:
    def check() -> CheckResult:
        scenario = get_scenario(name)
        r = run_scenario(scenario, affect_enabled)
        ok = r.success and r.action_count <= scenario.max_steps
        return ok, f"success={r.success} actions={r.action_count} final={r.final_state.to_dict()}"

    return check


def check_bounded_failure() -> CheckResult:
    agent = RecoveryAgent(RecoveryGoal(), AffectConfig(enabled=True), max_steps=2)
    r = agent.run(get_scenario("compound_failure").make_env(), "compound_failure_bounded")
    ok = r.action_count == 2 and not r.success and r.stop_reason == STOP_MAX_STEPS
    return ok, f"actions={r.action_count} success={r.success} stop={r.stop_reason}"


def check_determinism() -> CheckResult:
    for scenario in SCENARIOS:
        for enabled in (False, True):
            a = run_scenario(scenario, enabled).to_dict()
            b = run_scenario(scenario, enabled).to_dict()
            if a != b:
                return False, f"trace differs for {scenario.name} affect={enabled}"
    return True, f"{len(SCENARIOS) * 2} episodes reproduced identically"


def check_allowlist_rejection() -> CheckResult:
    env = SimulatedEnvironment(SystemState(50, 50, True))
    before = env.observe()
    for bad in ("shell", "rm -rf /", "", "Charge"):
        try:
            env.execute(bad)
        except UnknownActionError:
            continue
        return False, f"{bad!r} was not rejected"
    ok = env.observe() == before and env.step == 0 and all(e.kind == "rejected" for e in env.events)
    return ok, f"state unchanged={env.observe() == before} rejected_events={len(env.events)}"


def check_restart_blocked_when_hot() -> CheckResult:
    r = run_scenario(get_scenario("overheated_service_down"), affect_enabled=False)
    first = r.actions[0] if r.actions else None
    return first == "cool" and r.success, f"first action={first} success={r.success}"


def check_verifier_independent() -> CheckResult:
    verifier = GoalVerifier(RecoveryGoal())
    bad = verifier.verify(SimulatedEnvironment(SystemState(59, 50, True)))
    good = verifier.verify(SimulatedEnvironment(SystemState(60, 50, True)))
    ok = (not bad.satisfied and bad.checks["battery_ok"] is False) and good.satisfied
    return ok, f"boundary 59->{bad.satisfied}, 60->{good.satisfied}"


def check_disturbances_logged() -> CheckResult:
    r = run_scenario(get_scenario("disturbance_mid_recovery"), affect_enabled=False)
    kinds = [(e["step"], e["name"]) for e in r.events if e["kind"] == "disturbance"]
    ok = kinds == [(1, "heat_spike"), (3, "service_crash")] and r.success
    return ok, f"disturbances={kinds} success={r.success}"


def check_comparison_report_complete() -> CheckResult:
    report = run_affect_comparison()
    expected = len(SCENARIOS) * 2
    ok = len(report["rows"]) == expected and len(report["per_scenario"]) == len(SCENARIOS)
    return ok, f"rows={len(report['rows'])}/{expected}"


# ---- M1.1 checks -------------------------------------------------------


def check_blocked_action_no_effect() -> CheckResult:
    env = SimulatedEnvironment(SystemState(40, 30, False), blocked=(("restart_service", 1, 2),))
    before = env.observe()
    env.execute("restart_service")
    unchanged = env.observe() == before
    env.execute("charge")  # other actions are not affected
    charged = env.observe().battery == 60
    env.execute("restart_service")  # step 3: window over, works again
    works_again = env.observe().service_up
    kinds = [e.kind for e in env.events]
    ok = unchanged and charged and works_again and kinds == ["blocked", "action", "action"]
    return ok, f"unchanged={unchanged} other_ok={charged} after_window={works_again} kinds={kinds}"


def make_fault_recovery_check(name: str, affect_enabled: bool) -> Callable[[], CheckResult]:
    def check() -> CheckResult:
        scenario = get_scenario(name)
        r = run_scenario(scenario, affect_enabled)
        blocked = sum(1 for e in r.events if e["kind"] == "blocked")
        ok = r.success and r.action_count <= scenario.max_steps and blocked > 0
        return ok, f"success={r.success} actions={r.action_count} blocked_events={blocked}"

    return check


def check_frustration_activates() -> CheckResult:
    r = run_scenario(get_scenario("flaky_restart"), affect_enabled=True)
    peak = max((s.affect["frustration"] for s in r.steps), default=0)
    return peak > 0, f"peak frustration={peak}"


def check_sweep_complete() -> CheckResult:
    report = run_sweep()
    expected = len(sweep_configs()) * len(SCENARIOS)
    return len(report["rows"]) == expected, f"rows={len(report['rows'])}/{expected}"


def check_sweep_deterministic() -> CheckResult:
    same = run_sweep() == run_sweep()
    return same, f"two sweeps identical={same}"


def check_m1_regression() -> CheckResult:
    for name, expected in M1_FROZEN_SEQUENCES.items():
        for enabled in (False, True):
            got = run_scenario(get_scenario(name), enabled).actions
            if got != expected:
                return False, f"{name} affect={enabled}: {got} != {expected}"
    return True, f"{len(M1_FROZEN_SEQUENCES) * 2} M1 episodes match frozen sequences"


def build_checks() -> List[Tuple[str, Callable[[], CheckResult]]]:
    checks: List[Tuple[str, Callable[[], CheckResult]]] = [("V01 nominal_no_action", check_nominal_no_action)]
    n = 2
    for enabled in (False, True):
        for s in RECOVERY_SCENARIOS:
            label = f"V{n:02d} recover_{s.name}_affect_{'on' if enabled else 'off'}"
            checks.append((label, make_recovery_check(s.name, enabled)))
            n += 1
    for name, fn in (
        ("bounded_failure_stops", check_bounded_failure),
        ("deterministic_traces", check_determinism),
        ("allowlist_rejects_unknown", check_allowlist_rejection),
        ("restart_blocked_when_hot", check_restart_blocked_when_hot),
        ("verifier_independent_boundary", check_verifier_independent),
        ("disturbances_applied_and_logged", check_disturbances_logged),
        ("comparison_report_complete", check_comparison_report_complete),
    ):
        checks.append((f"V{n:02d} {name}", fn))
        n += 1
    # M1.1 (V21 onward)
    checks.append((f"V{n:02d} blocked_action_has_no_effect", check_blocked_action_no_effect))
    n += 1
    for s in FAULT_SCENARIOS:
        for enabled in (False, True):
            label = f"V{n:02d} recover_{s.name}_affect_{'on' if enabled else 'off'}"
            checks.append((label, make_fault_recovery_check(s.name, enabled)))
            n += 1
    for name, fn in (
        ("frustration_activates_under_faults", check_frustration_activates),
        ("sweep_report_complete", check_sweep_complete),
        ("sweep_deterministic", check_sweep_deterministic),
        ("m1_behaviour_unchanged", check_m1_regression),
    ):
        checks.append((f"V{n:02d} {name}", fn))
        n += 1
    return checks


def main() -> int:
    checks = build_checks()
    passed = 0
    for label, fn in checks:
        try:
            ok, detail = fn()
        except Exception as exc:  # a crash is a failure, never a pass
            ok, detail = False, f"exception: {exc!r}"
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    print(f"\n{passed}/{len(checks)} validation checks passed")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
