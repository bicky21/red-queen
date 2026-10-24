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


# ---- M2 checks ---------------------------------------------------------

# Frozen M1.1 fault-scenario behaviour (default affect config), commit 5c50695.
M1_1_FROZEN_SEQUENCES = {
    "flaky_restart": ("restart_service",) * 4,
    "flaky_restart_low_battery": ("restart_service",) * 4 + ("charge", "charge"),
    "charger_outage": ("charge",) * 4,
}


def _filled_memory(path=":memory:", **kwargs):
    from .memory import EpisodeMemory
    from .memory_cli import record_all

    memory = EpisodeMemory(path, **kwargs)
    record_all(memory)
    return memory


def check_memory_roundtrip() -> CheckResult:
    from .memory import EpisodeMemory

    config = AffectConfig(enabled=True)
    result = run_scenario(get_scenario("disturbance_mid_recovery"), True, affect=config)
    with EpisodeMemory(":memory:") as memory:
        rid = memory.record(result, config)
        rec = memory.get(rid)
        events = memory.events(rid)
    ok = (
        rec is not None and rec.valid and rec.actions == result.actions and rec.success == result.success
        and rec.final_state == result.final_state.to_dict() and events == list(result.events)
    )
    return ok, f"actions_match={rec is not None and rec.actions == result.actions} events={len(events)}"


def check_memory_persists() -> CheckResult:
    import tempfile
    from pathlib import Path

    from .memory import EpisodeMemory

    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "m.sqlite"
        _filled_memory(db).close()
        with EpisodeMemory(db) as reopened:
            n, ok_integrity = reopened.count(), reopened.integrity_ok()
    expected = len(SCENARIOS) * 2
    return n == expected and ok_integrity, f"episodes after reopen={n}/{expected} integrity={ok_integrity}"


def check_lexical_retrieval() -> CheckResult:
    memory = _filled_memory()
    top = memory.search("restart blocked", limit=3)
    memory.close()
    names = [r.scenario for r in top]
    ok = len(top) == 3 and all(n.startswith("flaky_restart") for n in names) and top[0].score == 2
    return ok, f"top3={names}"


def check_retrieval_deterministic() -> CheckResult:
    a, b = _filled_memory(), _filled_memory()
    ra = [(r.episode_id, r.score) for r in a.search("charge blocked failure", limit=10)]
    rb = [(r.episode_id, r.score) for r in b.search("charge blocked failure", limit=10)]
    a.close(), b.close()
    return ra == rb and bool(ra), f"identical={ra == rb} results={len(ra)}"


def check_memory_caps() -> CheckResult:
    memory = _filled_memory(max_episodes=5, max_events_per_episode=2)
    n, ids = memory.count(), [r.episode_id for r in memory.search("scenario", limit=20)]
    max_events = max(memory.event_count(i) for i in ids)
    memory.close()
    expected_ids = list(range(len(SCENARIOS) * 2, len(SCENARIOS) * 2 - 5, -1))
    ok = n == 5 and ids == expected_ids and max_events <= 2
    return ok, f"kept={n} newest_ids={ids} max_events_per_episode={max_events}"


def check_query_limits() -> CheckResult:
    from .memory import MAX_QUERY_CHARS, MAX_RESULTS

    memory = _filled_memory()
    rejected = 0
    for kwargs in ({"query": "a" * (MAX_QUERY_CHARS + 1)}, {"query": "x", "limit": 0},
                   {"query": "x", "limit": MAX_RESULTS + 1}):
        try:
            memory.search(**kwargs)
        except ValueError:
            rejected += 1
    memory.close()
    return rejected == 3, f"rejected {rejected}/3 out-of-bounds queries"


def check_sql_injection_inert() -> CheckResult:
    memory = _filled_memory()
    before = memory.count()
    for q in ("x'; DROP TABLE episodes;--", "\" OR 1=1 --", "%' OR '1'='1"):
        memory.search(q)
    after, ok_integrity = memory.count(), memory.integrity_ok()
    memory.close()
    return before == after and ok_integrity, f"episodes before={before} after={after} integrity={ok_integrity}"


def check_schema_version_enforced() -> CheckResult:
    import sqlite3
    import tempfile
    from pathlib import Path

    from .memory import EpisodeMemory, MemoryStoreError

    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "m.sqlite"
        EpisodeMemory(db).close()
        conn = sqlite3.connect(db)
        with conn:
            conn.execute("UPDATE meta SET value = '999' WHERE key = 'schema_version'")
        conn.close()
        try:
            EpisodeMemory(db).close()
            wrong_version_refused = False
        except MemoryStoreError:
            wrong_version_refused = True
        junk = Path(tmp) / "junk.sqlite"
        junk.write_bytes(b"this is not a database" * 100)
        try:
            EpisodeMemory(junk).close()
            junk_refused = False
        except MemoryStoreError:
            junk_refused = True
    ok = wrong_version_refused and junk_refused
    return ok, f"wrong_version_refused={wrong_version_refused} corrupt_file_refused={junk_refused}"


def check_tampered_record_flagged() -> CheckResult:
    memory = _filled_memory()
    memory._conn.execute("UPDATE episodes SET actions_json = ? WHERE id = 1", ('["shell", "rm -rf /"]',))
    memory._conn.execute("UPDATE episodes SET actions_json = ? WHERE id = 2", ("not json",))
    r1, r2, r3 = memory.get(1), memory.get(2), memory.get(3)
    memory.close()
    ok = (not r1.valid) and (not r2.valid) and r3.valid and r1.source == "memory"
    return ok, f"tampered_invalid={not r1.valid} malformed_invalid={not r2.valid} clean_valid={r3.valid}"


def check_memory_does_not_change_behaviour() -> CheckResult:
    from .memory import EpisodeMemory

    with EpisodeMemory(":memory:") as memory:
        for s in SCENARIOS:
            for enabled in (False, True):
                config = AffectConfig(enabled=enabled)
                first = run_scenario(s, enabled, affect=config)
                memory.record(first, config)
                second = run_scenario(s, enabled, affect=config)  # memory now non-empty
                if first.to_dict() != second.to_dict():
                    return False, f"{s.name} affect={enabled} changed after recording"
    return True, f"{len(SCENARIOS) * 2} episodes identical with and without stored memory"


def check_m1_1_regression() -> CheckResult:
    for name, expected in M1_1_FROZEN_SEQUENCES.items():
        for enabled in (False, True):
            got = run_scenario(get_scenario(name), enabled).actions
            if got != expected:
                return False, f"{name} affect={enabled}: {got} != {expected}"
    return True, f"{len(M1_1_FROZEN_SEQUENCES) * 2} M1.1 episodes match frozen sequences"


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
    # M2 (V32 onward)
    for name, fn in (
        ("memory_roundtrip", check_memory_roundtrip),
        ("memory_persists_across_reopen", check_memory_persists),
        ("lexical_retrieval_ranks_relevant", check_lexical_retrieval),
        ("retrieval_deterministic", check_retrieval_deterministic),
        ("memory_caps_enforced", check_memory_caps),
        ("query_limits_enforced", check_query_limits),
        ("sql_injection_inert", check_sql_injection_inert),
        ("schema_and_corrupt_file_refused", check_schema_version_enforced),
        ("tampered_record_flagged_invalid", check_tampered_record_flagged),
        ("memory_does_not_change_behaviour", check_memory_does_not_change_behaviour),
        ("m1_1_behaviour_unchanged", check_m1_1_regression),
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
