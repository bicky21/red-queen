"""M1 validation checks.

Run:  python -m red_queen.validation
Exit code 0 only if every check passes.
"""

from __future__ import annotations

import sys
from typing import Callable, List, Tuple

from .actions import UnknownActionError
from .affect import AffectConfig
from .agent import STOP_MAX_STEPS, STOP_POLICY, RecoveryAgent
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


# ---- M2.1 checks -------------------------------------------------------


def check_empty_advisor_identical() -> CheckResult:
    from .memory import EpisodeMemory
    from .memory_advice import MemoryAdvisor

    for s in SCENARIOS:
        for enabled in (False, True):
            with EpisodeMemory(":memory:") as memory:
                with_advisor = run_scenario(s, enabled, advisor=MemoryAdvisor(memory, s.name))
            plain = run_scenario(s, enabled)
            if with_advisor.actions != plain.actions or with_advisor.events != plain.events:
                return False, f"{s.name} affect={enabled} differs with empty memory"
    return True, f"{len(SCENARIOS) * 2} episodes identical with empty memory and no advisor"


def check_relevant_memory_outcome() -> CheckResult:
    from .memory_experiment import run_condition

    s = get_scenario("flaky_restart_low_battery")
    empty, _ = run_condition(s, "empty", False)
    relevant, advice = run_condition(s, "relevant", False)
    ok = (relevant.success and empty.action_count == 6 and relevant.action_count == 5
          and relevant.actions[0] == "charge" and advice["records_used"] == 1)
    return ok, f"empty={empty.action_count} relevant={relevant.action_count} first={relevant.actions[0]} advice={advice}"


def check_irrelevant_memory_no_effect() -> CheckResult:
    from .memory_experiment import run_condition

    for s in SCENARIOS:
        empty, _ = run_condition(s, "empty", True)
        irrelevant, advice = run_condition(s, "irrelevant", True)
        if irrelevant.actions != empty.actions or advice["records_used"] != 0:
            return False, f"{s.name}: irrelevant memory changed behaviour or was used ({advice})"
    return True, f"{len(SCENARIOS)} scenarios unchanged; 0 irrelevant records used"


def _memory_for(name: str):
    from .memory import EpisodeMemory

    memory = EpisodeMemory(":memory:")
    config = AffectConfig(enabled=False)
    memory.record(run_scenario(get_scenario(name), False, affect=config), config)
    return memory


def check_invalid_records_ignored() -> CheckResult:
    from .memory_advice import MemoryAdvisor

    memory = _memory_for("flaky_restart_low_battery")
    memory._conn.execute("UPDATE episodes SET actions_json = '[\"shell\"]'")
    advisor = MemoryAdvisor(memory, "flaky_restart_low_battery")
    result = run_scenario(get_scenario("flaky_restart_low_battery"), False, advisor=advisor)
    memory.close()
    s = advisor.summary()
    ok = s["records_used"] == 0 and s["records_ignored_invalid"] == 1 and result.action_count == 6
    return ok, f"advisor={s} actions={result.action_count} (same as empty memory)"


def check_poisoned_memory_bounded() -> CheckResult:
    """Valid-looking records whose events all claim 'blocked' must not stop recovery."""
    from .memory import EpisodeMemory
    from .memory_advice import MemoryAdvisor

    for s in SCENARIOS:
        with EpisodeMemory(":memory:") as memory:
            config = AffectConfig(enabled=False)
            for _ in range(3):
                memory.record(run_scenario(s, False, affect=config), config)
            memory._conn.execute("UPDATE events SET kind = 'blocked' WHERE kind = 'action'")
            for enabled in (False, True):
                r = run_scenario(s, enabled, advisor=MemoryAdvisor(memory, s.name))
                if not r.success:
                    return False, f"{s.name} affect={enabled} failed under poisoned memory"
    return True, f"{len(SCENARIOS) * 2} poisoned-memory episodes all recovered"


def check_discount_capped() -> CheckResult:
    from .affect import compute_signals
    from .goal import RecoveryGoal
    from .scoring import MAX_MEMORY_DISCOUNT, ActionScorer

    goal = RecoveryGoal()
    scorer = ActionScorer(goal, AffectConfig(enabled=False))
    s = SystemState(40, 30, False)
    sig = compute_signals(goal, s, 0)
    base = {r.name: r for r in scorer.score_all(s, sig)}
    hit = {r.name: r for r in scorer.score_all(
        s, sig, discounts={"restart_service": 5.0, "wait": 1.0, "shell": 1.0, "charge": float("nan")})}
    capped = abs(hit["restart_service"].gain - base["restart_service"].gain * (1 - MAX_MEMORY_DISCOUNT)) < 1e-9
    negative_untouched = hit["wait"].score == base["wait"].score
    nan_ignored = hit["charge"].score == base["charge"].score
    no_new_actions = set(hit) == set(base)
    ok = capped and negative_untouched and nan_ignored and no_new_actions
    return ok, (f"capped={capped} non_positive_gain_untouched={negative_untouched} "
                f"nan_ignored={nan_ignored} no_new_actions={no_new_actions}")


def check_observation_overrides_memory() -> CheckResult:
    from .memory_advice import MemoryAdvisor

    memory = _memory_for("low_battery")  # charge worked at steps 1-2
    memory._conn.execute("UPDATE events SET kind = 'blocked'")  # claim charge always fails
    r = run_scenario(get_scenario("low_battery"), False, advisor=MemoryAdvisor(memory, "low_battery"))
    memory.close()
    first, second = r.steps[0].memory_discounts, r.steps[1].memory_discounts
    ok = r.success and "charge" in first and "charge" not in second
    return ok, f"step1 discounts={first} step2 discounts={second} success={r.success}"


def check_memory_experiment() -> CheckResult:
    from .memory_experiment import CONDITIONS, run_memory_experiment

    a = run_memory_experiment()
    b = run_memory_experiment()
    expected = len(SCENARIOS) * 2 * len(CONDITIONS)
    all_ok = all(r["success"] for r in a["rows"])
    ok = a == b and len(a["rows"]) == expected and all_ok
    return ok, f"rows={len(a['rows'])}/{expected} deterministic={a == b} all_succeeded={all_ok}"


# ---- M3 checks ---------------------------------------------------------


def check_reactive_default_policy_unchanged() -> CheckResult:
    from .policy import DEFAULT_POLICY

    frozen = {**M1_FROZEN_SEQUENCES, **M1_1_FROZEN_SEQUENCES}
    for name, expected in frozen.items():
        for enabled in (False, True):
            r = run_scenario(get_scenario(name), enabled, policy=DEFAULT_POLICY)
            if r.actions != expected or r.stats["policy_denials"] or r.stats["policy_filtered"]:
                return False, f"{name} affect={enabled}: {r.actions} stats={r.stats}"
    return True, f"{len(frozen) * 2} reactive episodes match frozen sequences; 0 denials, 0 filtered"


def check_planner_recovers_default() -> CheckResult:
    from .planning_experiment import run_planner

    results = {s.name: run_planner(s) for s in SCENARIOS}
    failed = [n for n, r in results.items() if not r.success]
    return not failed, f"recovered {len(results) - len(failed)}/{len(results)} failed={failed}"


def _brute_force_shortest(state, goal, max_len=7):
    import itertools

    from .actions import ACTION_NAMES, apply_action

    if goal.is_satisfied(state):
        return 0
    for n in range(1, max_len + 1):
        for seq in itertools.product(ACTION_NAMES, repeat=n):
            s = state
            for a in seq:
                s = apply_action(s, a)
            if goal.is_satisfied(s):
                return n
    return None


def check_planner_shortest() -> CheckResult:
    from .goal import RecoveryGoal
    from .planner import BoundedPlanner

    goal = RecoveryGoal()
    planner = BoundedPlanner(goal)
    names = ("low_battery", "overheating", "service_down", "overheated_service_down", "compound_failure")
    for name in names:
        start = get_scenario(name).initial
        plan = planner.plan(start, ACTIONS_ALL())
        oracle = _brute_force_shortest(start, goal)
        if not plan.reaches_goal or len(plan.actions) != oracle:
            return False, f"{name}: planner={len(plan.actions)} oracle={oracle}"
    return True, f"{len(names)} plans match exhaustive-search optimum"


def ACTIONS_ALL():
    from .actions import ACTION_NAMES

    return ACTION_NAMES


def check_planner_replans() -> CheckResult:
    from .planning_experiment import run_planner

    fr = run_planner(get_scenario("flaky_restart"))
    dm = run_planner(get_scenario("disturbance_mid_recovery"))
    ok = (fr.success and fr.stats["replans"] == 3 and fr.stats["prediction_mismatches"] == 3
          and dm.success and dm.stats["replans"] == 2 and dm.stats["prediction_mismatches"] == 2)
    return ok, (f"flaky_restart replans={fr.stats['replans']} mismatches={fr.stats['prediction_mismatches']}; "
                f"disturbance replans={dm.stats['replans']} mismatches={dm.stats['prediction_mismatches']}")


def check_planner_deterministic() -> CheckResult:
    from .planning_experiment import run_planner
    from .policy import POLICIES

    for policy in POLICIES.values():
        for s in SCENARIOS:
            if run_planner(s, policy).to_dict() != run_planner(s, policy).to_dict():
                return False, f"{s.name} policy={policy.name} not reproducible"
    return True, f"{len(SCENARIOS) * len(POLICIES)} planner episodes reproduced identically"


def check_planner_bounds() -> CheckResult:
    from .goal import RecoveryGoal
    from .planner import BoundedPlanner
    from .planning_agent import STOP_NO_PLAN
    from .planning_experiment import run_planner

    goal = RecoveryGoal()
    start = get_scenario("compound_failure").initial
    by_nodes = BoundedPlanner(goal, max_nodes=10).plan(start, ACTIONS_ALL())
    by_depth = BoundedPlanner(goal, max_depth=2).plan(start, ACTIONS_ALL())
    r = run_planner(get_scenario("compound_failure"), planner=BoundedPlanner(goal, max_nodes=10))
    ok = (not by_nodes.reaches_goal and by_nodes.hit_node_limit and by_nodes.nodes_expanded <= 10
          and not by_depth.reaches_goal and r.action_count == 0 and r.stop_reason == STOP_NO_PLAN
          and not r.success)
    return ok, (f"node_limit: nodes={by_nodes.nodes_expanded} hit={by_nodes.hit_node_limit}; "
                f"depth_limit found={by_depth.reaches_goal}; agent actions={r.action_count} stop={r.stop_reason}")


class _RoguePlanner:
    """Test double: always proposes a plan that uses a denied action."""

    def plan(self, start, allowed):
        from .actions import apply_action
        from .planner import Plan

        nxt = apply_action(start, "restart_service")
        return Plan(("restart_service",), (nxt,), True, 1, False)


def check_rogue_plan_rejected() -> CheckResult:
    from .goal import RecoveryGoal
    from .planning_agent import PlanningAgent
    from .policy import POLICIES

    agent = PlanningAgent(RecoveryGoal(), planner=_RoguePlanner(), policy=POLICIES["no_restart"])
    r = agent.run(get_scenario("service_down").make_env(), "service_down")
    ok = (r.action_count == 0 and r.event_count == 0 and r.stop_reason == STOP_POLICY
          and r.stats["plan_rejections"] == 3)
    return ok, f"actions={r.action_count} env_events={r.event_count} stop={r.stop_reason} rejections={r.stats['plan_rejections']}"


def check_executor_boundary() -> CheckResult:
    from .policy import POLICIES, PolicyDenied, PolicyExecutor

    env = SimulatedEnvironment(SystemState(50, 50, False))
    ex = PolicyExecutor(env, POLICIES["no_restart"], max_actions=1)
    before = env.observe()
    outcomes = []
    for action in ("restart_service", "shell", "charge", "charge"):
        try:
            ex.execute(action)
            outcomes.append("ran")
        except PolicyDenied:
            outcomes.append("denied")
    ok = (outcomes == ["denied", "denied", "ran", "denied"] and env.step == 1
          and env.observe() == before.with_changes(battery=70, temperature=55)
          and [d["reason"] for d in ex.denials][-1] == "action budget exhausted")
    return ok, f"outcomes={outcomes} env_steps={env.step} denials={len(ex.denials)}"


def check_policy_config_validated() -> CheckResult:
    from .policy import Policy, PolicyConfigError

    rejected = 0
    for bad in (frozenset({"charge", "shell"}), {"charge"}, ["cool"]):
        try:
            Policy("bad", bad)
        except PolicyConfigError:
            rejected += 1
    return rejected == 3, f"rejected {rejected}/3 invalid policy configs"


def check_denied_never_executed() -> CheckResult:
    from .planning_experiment import run_planning_experiment

    report = run_planning_experiment()
    leaked = sum(r["denied_actions_executed"] for r in report["rows"])
    return leaked == 0, f"denied actions reaching environment across {len(report['rows'])} runs: {leaked}"


def check_impossible_goal_detected() -> CheckResult:
    from .planning_agent import STOP_NO_PLAN
    from .planning_experiment import run_agent
    from .policy import POLICIES

    s, pol = get_scenario("service_down"), POLICIES["no_restart"]
    p, r = run_agent("planner", s, pol), run_agent("reactive", s, pol)
    ok = (not p.success and p.action_count == 0 and p.stop_reason == STOP_NO_PLAN
          and not r.success and r.action_count == 20)
    return ok, f"planner actions={p.action_count} stop={p.stop_reason}; reactive actions={r.action_count} stop={r.stop_reason}"


def check_planning_experiment() -> CheckResult:
    from .planning_experiment import AGENTS, run_planning_experiment
    from .policy import POLICIES

    a, b = run_planning_experiment(), run_planning_experiment()
    expected = len(SCENARIOS) * len(POLICIES) * len(AGENTS)
    return a == b and len(a["rows"]) == expected, f"rows={len(a['rows'])}/{expected} deterministic={a == b}"


def check_m2_1_results_unchanged() -> CheckResult:
    from .memory_experiment import run_memory_experiment

    s = run_memory_experiment()["summary"]
    got = (s["empty"]["total_actions"], s["irrelevant"]["total_actions"], s["relevant"]["total_actions"])
    return got == (70, 70, 68), f"memory experiment total actions empty/irrelevant/relevant={got} (expected 70/70/68)"


# ---- M3.2 checks -------------------------------------------------------

# Frozen M3.2 fault-grid totals (default policy, 1680 cases), recorded at build time.
M3_2_GRID_TOTALS = {"reactive": 8741, "planner": 8428, "planner_reversed": 8713}


def check_extended_scenarios_isolated() -> CheckResult:
    from .scenarios import ALL_SCENARIOS, EXTENDED_SCENARIOS, SCENARIOS_BY_NAME

    ok = (len(SCENARIOS) == 10 and len(EXTENDED_SCENARIOS) == 8
          and len(ALL_SCENARIOS) == 18 and len(SCENARIOS_BY_NAME) == 18
          and all(get_scenario(s.name) is s for s in EXTENDED_SCENARIOS)
          and not set(s.name for s in EXTENDED_SCENARIOS) & set(s.name for s in SCENARIOS))
    return ok, f"core={len(SCENARIOS)} extended={len(EXTENDED_SCENARIOS)} lookup={len(SCENARIOS_BY_NAME)}"


def check_all_agents_recover_extended() -> CheckResult:
    from .scenario_study import AGENTS, run_agent
    from .scenarios import ALL_SCENARIOS

    failed = [(s.name, a) for s in ALL_SCENARIOS for a in AGENTS
              if not run_agent(a, s.make_env, max_steps=s.max_steps).success]
    total = len(ALL_SCENARIOS) * len(AGENTS)
    return not failed, f"recovered {total - len(failed)}/{total} failed={failed}"


def check_plan_length_order_invariant() -> CheckResult:
    from .goal import RecoveryGoal
    from .planner import BoundedPlanner
    from .scenario_study import GRID_STARTS, REVERSED_ORDER
    from .scenarios import ALL_SCENARIOS

    goal = RecoveryGoal()
    fwd, rev = BoundedPlanner(goal), BoundedPlanner(goal, action_order=REVERSED_ORDER)
    starts = list(GRID_STARTS) + [s.initial for s in ALL_SCENARIOS]
    differing_plans = 0
    for start in starts:
        a, b = fwd.plan(start, ACTIONS_ALL()), rev.plan(start, ACTIONS_ALL())
        if len(a.actions) != len(b.actions) or a.reaches_goal != b.reaches_goal:
            return False, f"length differs at {start.to_dict()}: {a.actions} vs {b.actions}"
        differing_plans += a.actions != b.actions
    return True, f"{len(starts)} start states: equal plan lengths; {differing_plans} with different action order"


def check_action_order_validated() -> CheckResult:
    from .goal import RecoveryGoal
    from .planner import BoundedPlanner

    rejected = 0
    for bad in (("charge", "cool"), ("charge", "cool", "restart_service", "shell"),
                ("charge", "charge", "cool", "wait")):
        try:
            BoundedPlanner(RecoveryGoal(), action_order=bad)
        except ValueError:
            rejected += 1
    return rejected == 3, f"rejected {rejected}/3 invalid action orders"


def check_reactive_optimal_fault_free() -> CheckResult:
    from .goal import RecoveryGoal
    from .planner import BoundedPlanner

    goal, planner = RecoveryGoal(), BoundedPlanner(RecoveryGoal())
    states = [SystemState(b, t, up) for b in range(0, 101, 5) for t in range(0, 101, 5) for up in (False, True)]
    worse = []
    for s in states:
        r = RecoveryAgent(goal, AffectConfig(enabled=False), 20).run(SimulatedEnvironment(s))
        shortest = len(planner.plan(s, ACTIONS_ALL()).actions)
        if not r.success or r.action_count != shortest:
            worse.append((s.to_dict(), r.action_count, shortest))
    return not worse, f"{len(states)} fault-free start states; reactive not optimal in {len(worse)}"


def check_grid_totals() -> CheckResult:
    from .scenario_study import run_grid_part

    g = run_grid_part()
    totals = {a: s["total_actions"] for a, s in g["summary"].items()}
    all_ok = all(s["successes"] == s["episodes"] for s in g["summary"].values())
    ok = totals == M3_2_GRID_TOTALS and all_ok and g["cases"] == 1680
    return ok, f"cases={g['cases']} totals={totals} all_succeeded={all_ok}"


def check_wait_only_policy() -> CheckResult:
    from .planning_agent import STOP_NO_PLAN
    from .policy import WAIT_ONLY_POLICY
    from .scenario_study import run_agent

    s = get_scenario("low_battery")
    p = run_agent("planner", s.make_env, WAIT_ONLY_POLICY)
    r = run_agent("reactive", s.make_env, WAIT_ONLY_POLICY)
    ok = (p.action_count == 0 and p.stop_reason == STOP_NO_PLAN and not p.success
          and not r.success and set(r.actions) == {"wait"} and r.action_count == 20)
    return ok, f"planner actions={p.action_count} stop={p.stop_reason}; reactive {r.action_count}x{set(r.actions)}"


def check_extended_policy_compliance() -> CheckResult:
    from .policy import ALL_POLICIES
    from .scenario_study import run_named_part

    part = run_named_part()
    leaked = [(r["scenario"], r["policy"], r["agent"]) for r in part["rows"]
              if not set(r["actions"]) <= ALL_POLICIES[r["policy"]].allowed]
    return not leaked, f"{len(part['rows'])} runs; executed actions outside policy: {len(leaked)}"


def check_scenario_study_deterministic() -> CheckResult:
    from .scenario_study import run_named_part

    a, b = run_named_part(), run_named_part()
    return a == b and len(a["rows"]) == 216, f"rows={len(a['rows'])}/216 deterministic={a == b}"


def check_m3_results_unchanged() -> CheckResult:
    from .planning_experiment import run_planning_experiment

    s = run_planning_experiment()["summary"]
    got = tuple(s[k]["total_actions"] for k in ("default/reactive", "default/planner", "no_restart/reactive",
                                                 "no_restart/planner", "no_cool/reactive", "no_cool/planner"))
    expected = (35, 33, 128, 11, 97, 16)
    return got == expected, f"M3 planning totals={got} (expected {expected})"


# ---- M3.3 checks -------------------------------------------------------

# Frozen M3.3 fault-grid totals (default policy, 1680 cases), recorded at build time.
M3_3_GRID_TOTALS = {
    "reactive": 8741, "reactive_memory": 8134, "planner": 8428, "planner_reversed": 8713,
    "planner_memory": 8271, "planner_reversed_memory": 8219,
}


def _m33_starts():
    from .scenario_study import GRID_STARTS
    from .scenarios import ALL_SCENARIOS

    return list(GRID_STARTS) + [s.initial for s in ALL_SCENARIOS]


def check_zero_risk_identical() -> CheckResult:
    from .goal import RecoveryGoal
    from .memory import EpisodeMemory
    from .memory_advice import MemoryAdvisor
    from .planner import BoundedPlanner
    from .planning_agent import PlanningAgent
    from .scenario_study import REVERSED_ORDER
    from .scenarios import ALL_SCENARIOS

    goal = RecoveryGoal()
    for order in (ACTIONS_ALL(), REVERSED_ORDER):
        planner = BoundedPlanner(goal, action_order=order)
        for start in _m33_starts():
            if planner.plan(start, ACTIONS_ALL()).actions != planner.plan(start, ACTIONS_ALL(), risk=lambda d: {}).actions:
                return False, f"zero-risk plan differs at {start.to_dict()}"
    for s in ALL_SCENARIOS:
        with EpisodeMemory(":memory:") as memory:
            a = PlanningAgent(goal, advisor=MemoryAdvisor(memory, s.name)).run(s.make_env(), s.name)
        b = PlanningAgent(goal).run(s.make_env(), s.name)
        if a.actions != b.actions or a.events != b.events:
            return False, f"{s.name}: empty-memory planner differs from no-memory planner"
    return True, f"{len(_m33_starts()) * 2} zero-risk plans and {len(ALL_SCENARIOS)} empty-memory episodes identical"


def check_risk_never_changes_length() -> CheckResult:
    from .goal import RecoveryGoal
    from .planner import BoundedPlanner

    goal = RecoveryGoal()
    planner = BoundedPlanner(goal)
    risks = [lambda d: {a: 0.9 for a in ACTIONS_ALL()}]
    risks += [lambda d, a=a: {a: 0.9} for a in ACTIONS_ALL()]
    risks += [lambda d: {"charge": 0.9} if d % 2 else {"cool": 0.9, "restart_service": 0.5}]
    checked = 0
    for start in _m33_starts():
        base = planner.plan(start, ACTIONS_ALL())
        for risk in risks:
            p = planner.plan(start, ACTIONS_ALL(), risk=risk)
            checked += 1
            if len(p.actions) != len(base.actions) or p.reaches_goal != base.reaches_goal:
                return False, f"length changed at {start.to_dict()}: {base.actions} -> {p.actions}"
    return True, f"{checked} risk-weighted plans: length and reachability unchanged"


def check_risk_input_sanitised() -> CheckResult:
    from .goal import RecoveryGoal
    from .planner import BoundedPlanner
    from .scoring import MAX_MEMORY_DISCOUNT

    goal = RecoveryGoal()
    planner = BoundedPlanner(goal)
    start = get_scenario("compound_failure").initial
    allowed = ("charge", "cool", "restart_service")
    bad_inputs = [
        lambda d: {"charge": float("nan"), "cool": -5, "restart_service": 99, "shell": 1.0},
        lambda d: None,
        lambda d: {"charge": "high"},
        lambda d: {"wait": 0.9},  # not allowed here
    ]
    for risk in bad_inputs:
        p = planner.plan(start, allowed, risk=risk)
        if not p.reaches_goal or not set(p.actions) <= set(allowed) or p.risk > MAX_MEMORY_DISCOUNT * len(p.actions):
            return False, f"bad risk input broke the plan: {p}"
    return True, f"{len(bad_inputs)} malformed risk inputs handled; plans valid and within cap"


def check_risk_picks_lower_risk_plan() -> CheckResult:
    from .goal import RecoveryGoal
    from .planner import BoundedPlanner
    from .scenario_study import REVERSED_ORDER

    goal = RecoveryGoal()
    planner = BoundedPlanner(goal, action_order=REVERSED_ORDER)
    start = get_scenario("deep_battery_flaky_restart").initial
    plain = planner.plan(start, ACTIONS_ALL())
    risky = planner.plan(start, ACTIONS_ALL(), risk=lambda d: {"restart_service": 0.9} if d < 3 else {})
    ok = plain.actions[0] == "restart_service" and risky.actions[-1] == "restart_service" and risky.risk == 0.0
    return ok, f"reversed order alone: {plain.actions}; with risk: {risky.actions} (risk={risky.risk})"


def check_memory_planning_grid() -> CheckResult:
    from .memory_planning_study import run_grid_part

    g = run_grid_part()
    totals = {a: s["total_actions"] for a, s in g["summary"].items()}
    all_ok = all(s["successes"] == s["episodes"] for s in g["summary"].values())
    improves = (totals["planner_memory"] < totals["planner"]
                and totals["planner_reversed_memory"] < totals["planner_reversed"])
    ok = totals == M3_3_GRID_TOTALS and all_ok and improves and g["cases"] == 1680
    return ok, f"totals={totals} all_succeeded={all_ok} memory_improves_both_orders={improves}"


def _poisoned_memory(name, make_env):
    from .memory_planning_study import prior_memory

    memory = prior_memory(name, make_env)
    memory._conn.execute("UPDATE events SET kind = 'blocked' WHERE kind = 'action'")
    return memory


def check_poisoned_memory_planner() -> CheckResult:
    from .memory_planning_study import run_agent
    from .scenarios import ALL_SCENARIOS

    for s in ALL_SCENARIOS:
        with _poisoned_memory(s.name, s.make_env) as memory:
            for agent in ("planner_memory", "planner_reversed_memory"):
                if not run_agent(agent, s.name, s.make_env, memory).success:
                    return False, f"{s.name} {agent} failed under poisoned memory"
    return True, f"{len(ALL_SCENARIOS) * 2} poisoned-memory planner episodes all recovered"


def check_invalid_memory_ignored_by_planner() -> CheckResult:
    from .memory_planning_study import prior_memory, run_agent

    s = get_scenario("deep_battery_flaky_restart")
    with prior_memory(s.name, s.make_env) as memory:
        memory._conn.execute("UPDATE episodes SET actions_json = '[\"shell\"]'")
        a = run_agent("planner_reversed_memory", s.name, s.make_env, memory)
    b = run_agent("planner_reversed", s.name, s.make_env, None)
    return a.actions == b.actions, f"tampered memory: {a.action_count} actions; no memory: {b.action_count}"


class _RecordingPlanner:
    def __init__(self):
        from .goal import RecoveryGoal
        from .planner import BoundedPlanner

        self.inner = BoundedPlanner(RecoveryGoal())
        self.first_step_risk = []

    def plan(self, start, allowed, risk=None):
        self.first_step_risk.append(sorted(risk(0)) if risk else None)
        return self.inner.plan(start, allowed, risk=risk)


class _FlatAdvisor:
    def discounts(self, step):
        return {a: 0.5 for a in ACTIONS_ALL()}


def check_observation_wins_in_planner() -> CheckResult:
    from .goal import RecoveryGoal
    from .planning_agent import PlanningAgent

    rec = _RecordingPlanner()
    env = SimulatedEnvironment(SystemState(10, 30, False), {2: "battery_drain"})
    r = PlanningAgent(RecoveryGoal(), planner=rec, advisor=_FlatAdvisor()).run(env)
    ok = (r.success and len(rec.first_step_risk) >= 2 and "charge" in rec.first_step_risk[0]
          and "charge" not in rec.first_step_risk[1])
    return ok, f"risk keys per planning call={rec.first_step_risk} success={r.success}"


def check_memory_planning_named_deterministic() -> CheckResult:
    from .memory_planning_study import run_named_part

    a, b = run_named_part(), run_named_part()
    all_ok = all(s["successes"] == s["episodes"] for s in a["summary"].values())
    return a == b and all_ok, f"named part deterministic={a == b} all_succeeded={all_ok}"


# ---- M3.4 checks -------------------------------------------------------


def check_slack_validated() -> CheckResult:
    from .goal import RecoveryGoal
    from .planner import BoundedPlanner

    rejected = 0
    for bad in (-1, 4, True, 1.5, "2"):
        try:
            BoundedPlanner(RecoveryGoal(), slack=bad)
        except ValueError:
            rejected += 1
    return rejected == 5, f"rejected {rejected}/5 invalid slack values"


def check_slack_zero_and_no_risk_identical() -> CheckResult:
    from .goal import RecoveryGoal
    from .planner import BoundedPlanner

    goal = RecoveryGoal()
    risk = lambda d: {"restart_service": 0.9} if d < 3 else {"charge": 0.3}  # noqa: E731
    for start in _m33_starts():
        base_no_risk = BoundedPlanner(goal).plan(start, ACTIONS_ALL())
        base_risk = BoundedPlanner(goal).plan(start, ACTIONS_ALL(), risk=risk)
        for k in (1, 2, 3):
            if BoundedPlanner(goal, slack=k).plan(start, ACTIONS_ALL()).actions != base_no_risk.actions:
                return False, f"slack={k} without risk changed the plan at {start.to_dict()}"
        if BoundedPlanner(goal, slack=0).plan(start, ACTIONS_ALL(), risk=risk).actions != base_risk.actions:
            return False, f"slack=0 differs from M3.3 at {start.to_dict()}"
    return True, f"{len(_m33_starts())} starts: slack ignored without risk; slack=0 equals M3.3"


def _risk_table(seed):
    import random

    rng = random.Random(seed)
    table = {(d, a): rng.choice([0, 0, 0, 0.3, 0.6, 0.9]) for d in range(14) for a in ACTIONS_ALL()}
    return lambda d, t=table: {a: t[(d, a)] for a in ACTIONS_ALL()}


def check_slack_plans_valid_and_bounded() -> CheckResult:
    from .actions import apply_action
    from .goal import RecoveryGoal
    from .planner import BoundedPlanner

    goal = RecoveryGoal()
    checked = 0
    for seed in range(40):
        risk = _risk_table(seed)
        for start in _m33_starts()[seed % 6::6]:
            shortest = BoundedPlanner(goal).plan(start, ACTIONS_ALL(), risk=risk)
            p = BoundedPlanner(goal, slack=3).plan(start, ACTIONS_ALL(), risk=risk)
            checked += 1
            s = start
            for i, (a, predicted) in enumerate(zip(p.actions, p.predicted)):
                s = apply_action(s, a)
                if s != predicted or (i < len(p.actions) - 1 and goal.is_satisfied(s)):
                    return False, f"invalid slack plan at {start.to_dict()}: {p.actions}"
            if (p.reaches_goal != shortest.reaches_goal or len(p.actions) > len(shortest.actions) + 3
                    or (p.actions and not goal.is_satisfied(s))):
                return False, f"slack plan out of bounds at {start.to_dict()}: {p.actions}"
    return True, f"{checked} slack plans: model-consistent, end at goal, within +3 steps"


def check_slack_changes_only_when_cheaper() -> CheckResult:
    """Slack may change a plan only if the new plan's expected cost
    (length + summed risk) is strictly lower. Frozen rate: 1 of 1800."""
    from .goal import RecoveryGoal
    from .planner import BoundedPlanner

    goal = RecoveryGoal()
    starts = _m33_starts()
    total = changed = 0
    for seed in range(300):
        risk = _risk_table(1000 + seed)
        for start in starts[seed % 7::7]:
            total += 1
            a = BoundedPlanner(goal).plan(start, ACTIONS_ALL(), risk=risk)
            b = BoundedPlanner(goal, slack=3).plan(start, ACTIONS_ALL(), risk=risk)
            if a.actions != b.actions:
                changed += 1
                if not len(b.actions) + b.risk < len(a.actions) + a.risk - 1e-9:
                    return False, f"slack changed plan without lower expected cost at {start.to_dict()}"
    return changed == 1, (f"{total} random-risk plans; changed by slack=3: {changed} (expected 1), "
                          f"each change strictly cheaper")


def check_slack_no_effect_on_fault_cases() -> CheckResult:
    from .scenario_study import grid_cases
    from .scenarios import ALL_SCENARIOS
    from .slack_study import run_case

    cases = [(s.name, s.make_env) for s in ALL_SCENARIOS]
    for i, (start, faults, disturbances) in enumerate(grid_cases()):
        if faults and not disturbances:
            cases.append((f"grid_{i:04d}",
                          lambda s=start, f=faults: SimulatedEnvironment(s, {}, f)))
    diffs = 0
    for name, make_env in cases:
        c = run_case(name, make_env)
        if not all(v["success"] for v in c.values()):
            return False, f"{name}: an agent failed"
        diffs += sum(c[f"planner_memory_k{k}"]["action_count"] != c["planner_memory_k0"]["action_count"]
                     for k in (1, 2, 3))
        diffs += c["planner_reversed_memory_k3"]["action_count"] != c["planner_reversed_memory_k0"]["action_count"]
    return diffs == 0, f"{len(cases)} cases (18 named + fault-only grid); slack changed action count {diffs} times"


def check_slack_keeps_impossible_goal_detection() -> CheckResult:
    from .goal import RecoveryGoal
    from .memory_advice import MemoryAdvisor
    from .memory_planning_study import prior_memory
    from .planner import BoundedPlanner
    from .planning_agent import STOP_NO_PLAN, PlanningAgent
    from .policy import POLICIES, WAIT_ONLY_POLICY

    results = []
    for name, policy in (("service_down", POLICIES["no_restart"]), ("overheating", POLICIES["no_cool"]),
                         ("low_battery", WAIT_ONLY_POLICY)):
        s = get_scenario(name)
        with prior_memory(s.name, s.make_env) as memory:
            agent = PlanningAgent(RecoveryGoal(), planner=BoundedPlanner(RecoveryGoal(), slack=3),
                                  policy=policy, advisor=MemoryAdvisor(memory, s.name))
            r = agent.run(s.make_env(), s.name)
        results.append((name, r.action_count, r.stop_reason))
        if r.action_count != 0 or r.stop_reason != STOP_NO_PLAN:
            return False, f"{results}"
    return True, f"impossible goals still detected with slack=3: {results}"


def check_slack_node_budget_fallback() -> CheckResult:
    from .goal import RecoveryGoal
    from .planner import BoundedPlanner

    goal = RecoveryGoal()
    start = get_scenario("compound_failure").initial
    risk = lambda d: {"charge": 0.9} if d < 2 else {}  # noqa: E731
    needed = BoundedPlanner(goal).plan(start, ACTIONS_ALL(), risk=risk).nodes_expanded
    shortest = BoundedPlanner(goal, max_nodes=needed).plan(start, ACTIONS_ALL(), risk=risk)
    limited = BoundedPlanner(goal, max_nodes=needed, slack=3).plan(start, ACTIONS_ALL(), risk=risk)
    fell_back = limited.nodes_expanded == shortest.nodes_expanded  # fallback returns the shortest plan object
    unlimited = BoundedPlanner(goal, slack=3).plan(start, ACTIONS_ALL(), risk=risk)
    searched_more = unlimited.nodes_expanded > 2 * needed  # proves the budget really was too small
    ok = shortest.reaches_goal and limited.actions == shortest.actions and fell_back and searched_more
    return ok, (f"budget={needed}: fell_back={fell_back} plan={limited.actions}; "
                f"unbounded slack search used {unlimited.nodes_expanded} nodes")


def check_slack_poisoned_memory_recovers() -> CheckResult:
    from .goal import RecoveryGoal
    from .memory_advice import MemoryAdvisor
    from .planner import BoundedPlanner
    from .planning_agent import PlanningAgent
    from .scenarios import ALL_SCENARIOS

    for s in ALL_SCENARIOS:
        with _poisoned_memory(s.name, s.make_env) as memory:
            r = PlanningAgent(RecoveryGoal(), planner=BoundedPlanner(RecoveryGoal(), slack=3),
                              advisor=MemoryAdvisor(memory, s.name)).run(s.make_env(), s.name)
            if not r.success:
                return False, f"{s.name} failed with slack=3 and poisoned memory"
    return True, f"{len(ALL_SCENARIOS)} poisoned-memory episodes with slack=3 all recovered"


# ---- M3.5 checks -------------------------------------------------------

# Frozen M3.5 grid totals (default policy, 1680 cases), recorded at build time.
M3_5_GRID_TOTALS = {
    "reactive_memory": 8134, "planner_memory": 8271, "planner_reversed_memory": 8219,
    "planner_dist_memory": 8062, "planner_reversed_dist_memory": 7857,
}


def _insert_event(memory, step, kind, name):
    eid = memory._conn.execute("SELECT id FROM episodes ORDER BY id LIMIT 1").fetchone()[0]
    memory._conn.execute(
        "INSERT INTO events (episode_id, step, kind, name, before_json, after_json) VALUES (?, ?, ?, ?, '{}', '{}')",
        (eid, step, kind, name))


def check_disturbance_extraction() -> CheckResult:
    from .memory_advice import MemoryAdvisor
    from .memory_planning_study import prior_memory

    s = get_scenario("disturbance_mid_recovery")
    with prior_memory(s.name, s.make_env) as memory:
        adv = MemoryAdvisor(memory, s.name)
        remembered = adv.remembered_disturbances()
        summary_keys = sorted(adv.summary())
    ok = (remembered == {1: "heat_spike", 3: "service_crash"}
          and summary_keys == ["events_ignored", "nonzero_entries", "records_ignored_invalid", "records_used"])
    return ok, f"remembered={remembered} summary keys unchanged={summary_keys}"


def check_disturbance_input_sanitised() -> CheckResult:
    from .goal import RecoveryGoal
    from .memory_advice import MemoryAdvisor
    from .memory_planning_study import prior_memory
    from .planner import BoundedPlanner

    s = get_scenario("low_battery")
    with prior_memory(s.name, s.make_env) as memory:
        for step, name in ((2, "meteor"), (-1, "heat_spike"), (0, "battery_drain")):
            _insert_event(memory, step, "disturbance", name)
        memory._conn.execute("INSERT INTO events (episode_id, step, kind, name, before_json, after_json) "
                             "VALUES (1, 'x', 'disturbance', 'heat_spike', '{}', '{}')")
        adv = MemoryAdvisor(memory, s.name)
        remembered, ignored = adv.remembered_disturbances(), adv.disturbance_events_ignored

    def boom(depth):
        raise RuntimeError("broken advisor")

    planner, start = BoundedPlanner(RecoveryGoal()), get_scenario("compound_failure").initial
    base = planner.plan(start, ACTIONS_ALL())
    junk_ok = all(planner.plan(start, ACTIONS_ALL(), disturbances=fn).actions == base.actions
                  for fn in (boom, lambda d: "meteor", lambda d: 42, lambda d: None))
    ok = remembered == {} and ignored == 4 and junk_ok
    return ok, f"remembered={remembered} ignored={ignored} junk_disturbance_inputs_harmless={junk_ok}"


def check_disturbance_off_identical() -> CheckResult:
    from .goal import RecoveryGoal
    from .memory_advice import MemoryAdvisor
    from .memory_planning_study import prior_memory
    from .planner import BoundedPlanner
    from .planning_agent import PlanningAgent
    from .scenarios import ALL_SCENARIOS

    goal = RecoveryGoal()
    planner = BoundedPlanner(goal)
    for start in _m33_starts():
        if planner.plan(start, ACTIONS_ALL(), disturbances=lambda d: None).actions != \
                planner.plan(start, ACTIONS_ALL(), risk=lambda d: {}).actions:
            return False, f"no-disturbance prediction changed the plan at {start.to_dict()}"
    for s in ALL_SCENARIOS:
        with prior_memory(s.name, s.make_env) as memory:
            a = PlanningAgent(goal, advisor=MemoryAdvisor(memory, s.name)).run(s.make_env(), s.name)
            b = PlanningAgent(goal, advisor=MemoryAdvisor(memory, s.name),
                              predict_disturbances=False).run(s.make_env(), s.name)
        if a.actions != b.actions or a.events != b.events or "remembered_disturbances" in b.stats:
            return False, f"{s.name}: predict_disturbances=False differs from M3.3"
    return True, f"{len(_m33_starts())} plans and {len(ALL_SCENARIOS)} episodes identical with prediction off"


def check_prediction_avoids_replans() -> CheckResult:
    from .disturbance_study import run_agent
    from .memory_planning_study import prior_memory

    s = get_scenario("disturbance_mid_recovery")
    with prior_memory(s.name, s.make_env) as memory:
        with_pred = run_agent("planner_dist_memory", s.name, s.make_env, memory)
        without = run_agent("planner_memory", s.name, s.make_env, memory)
    ok = (with_pred.success and with_pred.stats["prediction_mismatches"] == 0
          and without.stats["prediction_mismatches"] == 2)
    return ok, (f"mismatches with prediction={with_pred.stats['prediction_mismatches']} "
                f"(without={without.stats['prediction_mismatches']}); actions {with_pred.action_count} "
                f"vs {without.action_count}")


def check_disturbance_fallback() -> CheckResult:
    from .goal import RecoveryGoal
    from .planner import BoundedPlanner

    planner, start = BoundedPlanner(RecoveryGoal()), get_scenario("overheating").initial
    base = planner.plan(start, ACTIONS_ALL())
    p = planner.plan(start, ACTIONS_ALL(), disturbances=lambda d: "heat_spike")  # never cool enough
    ok = p.reaches_goal and p.actions == base.actions
    return ok, f"impossible-under-prediction falls back to {p.actions} (no-prediction plan {base.actions})"


def check_disturbance_grid() -> CheckResult:
    from .disturbance_study import run_case
    from .scenario_study import grid_cases

    totals = {a: 0 for a in M3_5_GRID_TOTALS}
    all_ok = True
    for i, (start, faults, disturbances) in enumerate(grid_cases()):
        make_env = lambda s=start, f=faults, d=disturbances: SimulatedEnvironment(s, dict(d), f)  # noqa: E731
        c = run_case(f"grid_{i:04d}", make_env)
        for a in totals:
            totals[a] += c[a]["action_count"]
            all_ok &= c[a]["success"]
    beats = (totals["planner_dist_memory"] < totals["reactive_memory"]
             and totals["planner_reversed_dist_memory"] < totals["reactive_memory"])
    ok = totals == M3_5_GRID_TOTALS and all_ok and beats
    return ok, f"totals={totals} all_succeeded={all_ok} both_orders_beat_reactive_memory={beats}"


def check_fake_disturbances_bounded() -> CheckResult:
    from .disturbance_study import run_agent
    from .memory_planning_study import prior_memory
    from .scenarios import ALL_SCENARIOS

    worst = 0
    for s in ALL_SCENARIOS:
        for fake in ("battery_drain", "heat_spike", "service_crash"):
            with prior_memory(s.name, s.make_env) as memory:
                for step in range(1, 21):
                    _insert_event(memory, step, "disturbance", fake)
                r = run_agent("planner_dist_memory", s.name, s.make_env, memory)
            if not r.success:
                return False, f"{s.name} failed with fake {fake} at every step"
            worst = max(worst, r.action_count)
    return True, f"{len(ALL_SCENARIOS) * 3} episodes with fake disturbances at every step recovered; max actions={worst}"


def check_disturbance_policy_preserved() -> CheckResult:
    from .goal import RecoveryGoal
    from .memory_advice import MemoryAdvisor
    from .memory_planning_study import prior_memory
    from .planning_agent import STOP_NO_PLAN, PlanningAgent
    from .policy import ALL_POLICIES
    from .scenarios import ALL_SCENARIOS

    runs = 0
    for policy in ALL_POLICIES.values():
        for s in ALL_SCENARIOS:
            with prior_memory(s.name, s.make_env) as memory:
                r = PlanningAgent(RecoveryGoal(), policy=policy, advisor=MemoryAdvisor(memory, s.name),
                                  predict_disturbances=True).run(s.make_env(), s.name)
            runs += 1
            if not set(r.actions) <= policy.allowed:
                return False, f"{s.name}/{policy.name}: executed action outside policy"
            if not r.success and (r.action_count > 3 or r.stop_reason != STOP_NO_PLAN):
                return False, f"{s.name}/{policy.name}: failed without early no-plan stop ({r.stop_reason})"
    return True, f"{runs} runs: actions within policy; every failure stopped early with no_plan_found"


def check_disturbance_named_deterministic() -> CheckResult:
    from .disturbance_study import run_case
    from .scenarios import ALL_SCENARIOS

    a = {s.name: run_case(s.name, s.make_env) for s in ALL_SCENARIOS}
    b = {s.name: run_case(s.name, s.make_env) for s in ALL_SCENARIOS}
    all_ok = all(v["success"] for c in a.values() for v in c.values())
    return a == b and all_ok, f"named part deterministic={a == b} all_succeeded={all_ok}"


def check_prediction_requires_advisor() -> CheckResult:
    from .goal import RecoveryGoal
    from .planning_agent import PlanningAgent

    rejected = 0
    for advisor in (None, object()):
        try:
            PlanningAgent(RecoveryGoal(), advisor=advisor, predict_disturbances=True)
        except ValueError:
            rejected += 1
    return rejected == 2, f"rejected {rejected}/2 configs without a disturbance-capable advisor"


# ---- M3.6 checks -------------------------------------------------------

# Frozen M3.6 subset totals (every 8th grid case = 210 cases), recorded at build time.
M3_6_SUBSET_TOTALS = {
    0.0: {"reactive_memory": 1026, "planner_memory": 1049, "planner_dist_memory": 1024,
          "planner_reversed_dist_memory": 995, "planner_dist_guard": 1031, "planner_reversed_dist_guard": 1006},
    1.0: {"reactive_memory": 1094, "planner_memory": 1070, "planner_dist_memory": 1098,
          "planner_reversed_dist_memory": 1119, "planner_dist_guard": 1082, "planner_reversed_dist_guard": 1110},
    "unrelated": {"reactive_memory": 1063, "planner_memory": 1059, "planner_dist_memory": 1076,
                  "planner_reversed_dist_memory": 1082, "planner_dist_guard": 1070,
                  "planner_reversed_dist_guard": 1076},
}
_M36_CACHE = {}


def _m36_subset():
    if "report" not in _M36_CACHE:
        from .noisy_memory_study import run_noisy_memory_study

        _M36_CACHE["report"] = run_noisy_memory_study(stride=8)
    return _M36_CACHE["report"]


def check_distort_properties() -> CheckResult:
    from .environment import DISTURBANCES
    from .memory_noise import FAULTABLE, NOISE_LEVELS, distort
    from .scenario_study import grid_cases

    cases = list(grid_cases())
    for i, (_, faults, dist) in enumerate(cases):
        if distort(faults, dist, 0.0, str(i)) != (tuple(faults), tuple(sorted(dist))):
            return False, f"level 0 changed case {i}"
        for level in NOISE_LEVELS:
            a = distort(faults, dist, level, str(i))
            if a != distort(faults, dist, level, str(i)):
                return False, f"case {i} level {level} not deterministic"
            nf, nd = a
            if any(f[0] not in FAULTABLE or not 1 <= f[1] <= f[2] for f in nf) or \
                    any(s < 1 or n not in DISTURBANCES for s, n in nd):
                return False, f"case {i} level {level} produced invalid output {a}"
    try:
        distort((), (), 1.5, "x")
        bad_level_refused = False
    except ValueError:
        bad_level_refused = True
    return bad_level_refused, (f"{len(cases)} cases x {len(NOISE_LEVELS)} levels: deterministic, valid; "
                               f"level 0 = exact memory; invalid level refused={bad_level_refused}")


def check_distortion_grows_with_level() -> CheckResult:
    from .memory_noise import NOISE_LEVELS, distort
    from .scenario_study import grid_cases

    cases = list(grid_cases())
    changed = []
    for level in NOISE_LEVELS:
        changed.append(sum(1 for i, (_, f, d) in enumerate(cases)
                           if distort(f, d, level, str(i)) != (tuple(f), tuple(sorted(d)))))
    ok = changed[0] == 0 and all(a < b for a, b in zip(changed, changed[1:])) and changed[-1] > len(cases) // 2
    return ok, f"cases changed per level {dict(zip(NOISE_LEVELS, changed))} of {len(cases)}"


def check_guard_none_identical() -> CheckResult:
    from .goal import RecoveryGoal
    from .memory_advice import MemoryAdvisor
    from .memory_planning_study import prior_memory
    from .planning_agent import PlanningAgent
    from .scenarios import ALL_SCENARIOS

    goal = RecoveryGoal()
    for s in ALL_SCENARIOS:
        with prior_memory(s.name, s.make_env) as memory:
            a = PlanningAgent(goal, advisor=MemoryAdvisor(memory, s.name), predict_disturbances=True).run(
                s.make_env(), s.name)
            b = PlanningAgent(goal, advisor=MemoryAdvisor(memory, s.name), predict_disturbances=True,
                              disturbance_guard=None).run(s.make_env(), s.name)
        if a.to_dict() != b.to_dict() or "guard_tripped_at" in b.stats:
            return False, f"{s.name}: guard=None differs from M3.5"
    return True, f"{len(ALL_SCENARIOS)} episodes identical to M3.5 with guard=None"


def check_guard_config_validated() -> CheckResult:
    from .goal import RecoveryGoal
    from .memory import EpisodeMemory
    from .memory_advice import MemoryAdvisor
    from .planning_agent import PlanningAgent

    rejected = 0
    with EpisodeMemory(":memory:") as memory:
        adv = MemoryAdvisor(memory, "x")
        for kwargs in ({"predict_disturbances": True, "disturbance_guard": "sometimes"},
                       {"predict_disturbances": False, "disturbance_guard": "on_any_mismatch"}):
            try:
                PlanningAgent(RecoveryGoal(), advisor=adv, **kwargs)
            except ValueError:
                rejected += 1
    return rejected == 2, f"rejected {rejected}/2 invalid guard configurations"


def check_guard_trips_only_when_wrong() -> CheckResult:
    from .goal import RecoveryGoal
    from .memory_advice import MemoryAdvisor
    from .memory_planning_study import prior_memory
    from .planning_agent import PlanningAgent

    goal = RecoveryGoal()
    s = get_scenario("disturbance_mid_recovery")
    with prior_memory(s.name, s.make_env) as memory:  # exact memory
        exact = PlanningAgent(goal, advisor=MemoryAdvisor(memory, s.name), predict_disturbances=True,
                              disturbance_guard="on_any_mismatch").run(s.make_env(), s.name)
    wrong_env = lambda: SimulatedEnvironment(SystemState(30, 40, True), {2: "battery_drain"})  # noqa: E731
    with prior_memory(s.name, wrong_env) as memory:  # memory of a different disturbance
        wrong = PlanningAgent(goal, advisor=MemoryAdvisor(memory, s.name), predict_disturbances=True,
                              disturbance_guard="on_any_mismatch").run(s.make_env(), s.name)
    ok = (exact.success and exact.stats["guard_tripped_at"] is None
          and wrong.success and wrong.stats["guard_tripped_at"] is not None)
    return ok, (f"exact memory: tripped={exact.stats['guard_tripped_at']}; "
                f"wrong memory: tripped at step {wrong.stats['guard_tripped_at']}, success={wrong.success}")


def check_noisy_memory_never_fails() -> CheckResult:
    report = _m36_subset()
    runs = sum(l["cases"] * len(l["successes"]) for l in report["levels"])
    failed = sum(l["cases"] - s for l in report["levels"] for s in l["successes"].values())
    return failed == 0, f"{runs} runs over 5 memory-noise levels; failures={failed}"


def check_noisy_subset_frozen() -> CheckResult:
    report = _m36_subset()
    got = {l["level"]: {a: l["total_actions"][a] for a in M3_6_SUBSET_TOTALS[0.0]}
           for l in report["levels"] if l["level"] in M3_6_SUBSET_TOTALS}
    return got == M3_6_SUBSET_TOTALS, f"subset totals at levels 0.0 / 1.0 / unrelated match frozen values: {got == M3_6_SUBSET_TOTALS}"


def check_guard_improves_worst_case() -> CheckResult:
    report = _m36_subset()
    levels = report["levels"]
    out = []
    for guard, plain, base in (("planner_dist_guard", "planner_dist_memory", "planner"),
                               ("planner_reversed_dist_guard", "planner_reversed_dist_memory", "planner_reversed")):
        worst_guard = max(l["total_actions"][guard] - l["total_actions"][base] for l in levels)
        worst_plain = max(l["total_actions"][plain] - l["total_actions"][base] for l in levels)
        exact = next(l for l in levels if l["level"] == 0.0)
        cost = (exact["total_actions"][guard] - exact["total_actions"][plain]) / exact["total_actions"][plain]
        out.append((guard, worst_plain, worst_guard, round(cost * 100, 2)))
        if not (worst_guard < worst_plain and cost <= 0.02):
            return False, f"{out}"
    return True, "worst-case regret vs no memory (no guard -> guard), cost at exact memory %: " + \
        "; ".join(f"{g}: {a} -> {b}, {c}%" for g, a, b, c in out)


# ---- M4 checks ---------------------------------------------------------

M4_GRID_TOTALS = {"planner_reversed": 1107, "baseline": 1068, "accepted": 471}


def _pipeline(text, start, policy, remaining=20):
    """parse + validate exactly as the gate does; returns 'accepted' or the rejection reason."""
    from .goal import RecoveryGoal
    from .planner import BoundedPlanner
    from .proposals import ProposalRejected, parse_proposal, validate_proposal

    goal = RecoveryGoal()
    ref = BoundedPlanner(goal).plan(start, policy.allowed)
    try:
        actions = parse_proposal(text)
        validate_proposal(actions, start, goal, policy, remaining, len(ref.actions) if ref.reaches_goal else None)
    except ProposalRejected as exc:
        return exc.reason
    return "accepted"


def check_parser_rejects_bad_outputs() -> CheckResult:
    from .goal import RecoveryGoal
    from .policy import POLICIES
    from .proposals import AdversarialProposer, MalformedProposer, ProposalContext

    start = get_scenario("service_down").initial
    policy = POLICIES["no_restart"]
    expected_malformed = ["not_json", "not_json", "not_object", "actions_not_list", "non_string_action",
                          "empty", "unknown_keys", "duplicate_keys", "rationale_invalid", "not_json"]
    expected_adversarial = ["unknown_action", "unknown_action", "unknown_action", "unknown_keys",
                            "denied_by_policy", "too_many_actions", "too_long_text", "not_text", "not_text",
                            "not_json"]
    ctx = ProposalContext.build(start, RecoveryGoal(), policy, 0, 20)
    m, a = MalformedProposer(), AdversarialProposer()
    got_m = [_pipeline(m.propose(ctx), start, policy) for _ in expected_malformed]
    got_a = [_pipeline(a.propose(ctx), start, policy) for _ in expected_adversarial]
    missing = _pipeline('{"rationale": "no actions"}', start, policy)
    ok = got_m == expected_malformed and got_a == expected_adversarial and missing == "missing_actions"
    return ok, f"malformed reasons ok={got_m == expected_malformed}; adversarial reasons ok={got_a == expected_adversarial}" + \
        ("" if ok else f" got_m={got_m} got_a={got_a}")


def check_valid_proposal_accepted() -> CheckResult:
    import json

    from .policy import DEFAULT_POLICY

    start = get_scenario("compound_failure").initial
    plan = ["cool", "cool", "charge", "charge", "charge", "cool", "restart_service"]
    with_rationale = json.dumps({"actions": plan, "rationale": "cool first"})
    reordered = json.dumps({"actions": ["charge", "charge", "charge", "cool", "cool", "cool", "restart_service"]})
    results = [_pipeline(json.dumps({"actions": plan}), start, DEFAULT_POLICY),
               _pipeline(with_rationale, start, DEFAULT_POLICY), _pipeline(reordered, start, DEFAULT_POLICY)]
    return results == ["accepted"] * 3, f"valid proposals: {results}"


def check_validator_rules() -> CheckResult:
    import json

    from .policy import DEFAULT_POLICY, POLICIES

    j = lambda acts: json.dumps({"actions": acts})  # noqa: E731
    cases = [
        (j(["restart_service"]), get_scenario("service_down").initial, POLICIES["no_restart"], 20, "denied_by_policy"),
        (j(["charge", "charge"]), get_scenario("low_battery").initial, DEFAULT_POLICY, 1, "exceeds_step_budget"),
        (j(["wait", "wait", "wait", "charge", "charge"]), get_scenario("low_battery").initial, DEFAULT_POLICY, 20,
         "longer_than_allowed"),
        (j(["charge", "charge", "wait"]), get_scenario("low_battery").initial, DEFAULT_POLICY, 20, "passes_through_goal"),
        (j(["charge"]), get_scenario("low_battery").initial, DEFAULT_POLICY, 20, "does_not_reach_goal"),
        (j(["charge"]), get_scenario("service_down").initial, POLICIES["no_restart"], 20, "no_reference_plan"),
    ]
    got = [_pipeline(text, start, pol, rem) for text, start, pol, rem, _ in cases]
    want = [c[-1] for c in cases]
    return got == want, f"validator reasons: {got}"


def check_gate_bounds() -> CheckResult:
    from .goal import RecoveryGoal
    from .memory import EpisodeMemory
    from .memory_advice import MemoryAdvisor
    from .planning_agent import PlanningAgent
    from .policy import DEFAULT_POLICY
    from .proposals import CrashingProposer, PlannerProposer, ProposalGate

    goal = RecoveryGoal()
    gate = ProposalGate(CrashingProposer(), goal, DEFAULT_POLICY, max_calls=10, max_rejections=3)
    start = get_scenario("low_battery").initial
    outcomes = [gate.try_propose(start, 0, 20, 2) for _ in range(6)]
    disabled_ok = gate.calls == 3 and gate.disabled_after_call == 3 and all(o is None for o in outcomes)
    good = ProposalGate(PlannerProposer(), goal, DEFAULT_POLICY, max_calls=2)
    calls_ok = sum(good.try_propose(start, 0, 20, 2) is not None for _ in range(5)) == 2 and good.calls == 2
    refused = 0
    for kwargs in ({"max_calls": -1}, {"max_rejections": 0}, {"max_extra_steps": 9}):
        try:
            ProposalGate(PlannerProposer(), goal, DEFAULT_POLICY, **kwargs)
        except ValueError:
            refused += 1
    with EpisodeMemory(":memory:") as memory:
        for kwargs in ({"proposer": PlannerProposer(), "advisor": MemoryAdvisor(memory, "x")},
                       {"proposer": object()}):
            try:
                PlanningAgent(goal, **kwargs)
            except ValueError:
                refused += 1
    ok = disabled_ok and calls_ok and refused == 5
    return ok, f"crash->disabled after 3 calls={disabled_ok}; call cap={calls_ok}; invalid configs refused={refused}/5"


_M4_CACHE = {}


def _m4_report():
    if "r" not in _M4_CACHE:
        from .proposal_study import run_proposal_study

        _M4_CACHE["r"] = run_proposal_study()
    return _M4_CACHE["r"]


def check_proposal_safety_totals() -> CheckResult:
    from .proposal_study import safety_totals

    s = safety_totals(_m4_report())
    ok = s["runs"] == 1974 and all(v == 0 for k, v in s.items() if k != "runs")
    return ok, f"{s}"


def check_correct_proposer_followed() -> CheckResult:
    from .scenario_study import run_agent
    from .scenarios import ALL_SCENARIOS
    from .proposal_study import run_pair
    from .policy import DEFAULT_POLICY

    g = _m4_report()["grid"]["by_proposer"]["planner_reversed"]
    for s in ALL_SCENARIOS:
        _, r = run_pair("planner_reversed", s.make_env, DEFAULT_POLICY, s.name)
        ref = run_agent("planner_reversed", s.make_env)
        if r.actions != ref.actions:
            return False, f"{s.name}: proposal run differs from reversed-order planner"
    got = {"planner_reversed": g["total_actions"], "baseline": g["baseline_actions"], "accepted": g["proposals_accepted"]}
    ok = got == M4_GRID_TOTALS and g["actions_from_proposals"] > 0
    return ok, f"18 named runs match the reversed-order planner; grid {got}"


def check_rationale_has_no_effect() -> CheckResult:
    from .policy import ALL_POLICIES
    from .proposal_study import run_pair
    from .scenarios import ALL_SCENARIOS

    for policy in ALL_POLICIES.values():
        for s in ALL_SCENARIOS:
            _, a = run_pair("planner_reversed", s.make_env, policy, s.name)
            _, b = run_pair("valid_plan_hostile_rationale", s.make_env, policy, s.name)
            if a.actions != b.actions or a.events != b.events:
                return False, f"{s.name}/{policy.name}: hostile rationale changed behaviour"
    return True, f"{len(ALL_SCENARIOS) * len(ALL_POLICIES)} runs identical with and without hostile rationale"


class _MutatingProposer:
    name = "mutating"

    def propose(self, context):
        import json

        object.__setattr__(context, "allowed_actions", ("charge", "cool", "restart_service", "wait"))
        return json.dumps({"actions": ["restart_service"]})


def check_context_isolation() -> CheckResult:
    from .goal import RecoveryGoal
    from .planning_agent import PlanningAgent
    from .policy import POLICIES
    from .proposals import ProposalContext, render_prompt

    s = get_scenario("service_down")
    r = PlanningAgent(RecoveryGoal(), policy=POLICIES["no_restart"], proposer=_MutatingProposer()).run(
        s.make_env(), s.name)
    ctx = ProposalContext.build(s.initial, RecoveryGoal(), POLICIES["no_restart"], 0, 20)
    plain = all(isinstance(v, (tuple, int, str, bool)) for v in vars(ctx).values())
    prompt = render_prompt(ctx)
    prompt_ok = prompt == render_prompt(ctx) and "restart_service" not in prompt and "cool" in prompt
    s2 = get_scenario("low_battery")
    r2 = PlanningAgent(RecoveryGoal(), policy=POLICIES["no_restart"], proposer=_MutatingProposer()).run(
        s2.make_env(), s2.name)
    ok = (r.action_count == 0 and r2.success and "restart_service" not in r2.actions
          and r2.stats["proposals_accepted"] == 0 and plain and prompt_ok)
    return ok, (f"context mutation cannot widen policy (denied action executed: "
                f"{'restart_service' in r2.actions}); context plain={plain}; prompt deterministic and policy-filtered={prompt_ok}")


def check_proposal_study_deterministic() -> CheckResult:
    from .proposal_study import run_proposal_study

    same = run_proposal_study() == _m4_report()
    return same, f"two study runs identical={same}"


# ---- M4.1 checks -------------------------------------------------------


def check_loopback_only() -> CheckResult:
    from .llm_adapter import LocalModelError, check_local_url

    good = ["http://127.0.0.1:11434", "http://localhost:11434", "http://[::1]:11434"]
    bad = ["http://10.0.0.1:11434", "https://localhost:11434", "http://example.com",
           "http://127.0.0.1.evil.com:11434", "http://localhost:11434/api/generate", "http://u:p@localhost:1"]
    accepted = sum(1 for u in good if check_local_url(u))
    refused = 0
    for u in bad:
        try:
            check_local_url(u)
        except LocalModelError:
            refused += 1
    return accepted == 3 and refused == len(bad), f"loopback accepted {accepted}/3; non-local refused {refused}/{len(bad)}"


def check_no_proxy_no_redirect() -> CheckResult:
    import os
    from unittest import mock

    from .fake_ollama import FakeOllama
    from .llm_adapter import LocalModelError, OllamaProposer, _request
    from .proposals import ProposalContext
    from .goal import RecoveryGoal
    from .policy import DEFAULT_POLICY

    ctx = ProposalContext.build(SystemState(20, 30, True), RecoveryGoal(), DEFAULT_POLICY, 0, 20)
    with FakeOllama("smart") as f, mock.patch.dict(os.environ, {"HTTP_PROXY": "http://10.255.255.1:9",
                                                                   "http_proxy": "http://10.255.255.1:9"}):
        proxy_ignored = "actions" in OllamaProposer("fake", f.url, timeout=5).propose(ctx)
    with FakeOllama("redirect") as f:
        try:
            OllamaProposer("fake", f.url, timeout=5).propose(ctx)
            redirect_refused = False
        except LocalModelError:
            redirect_refused = True
    try:
        _request("http://127.0.0.1:1", "/api/pull", {}, 1.0, 100)
        endpoint_refused = False
    except LocalModelError:
        endpoint_refused = True
    ok = proxy_ignored and redirect_refused and endpoint_refused
    return ok, f"proxy env ignored={proxy_ignored}; redirect refused={redirect_refused}; other endpoints refused={endpoint_refused}"


def check_timeout_and_size_caps() -> CheckResult:
    import time

    from .fake_ollama import FakeOllama
    from .goal import RecoveryGoal
    from .llm_adapter import LocalModelError, OllamaProposer
    from .policy import DEFAULT_POLICY
    from .proposals import ProposalContext

    ctx = ProposalContext.build(SystemState(20, 30, True), RecoveryGoal(), DEFAULT_POLICY, 0, 20)
    results = {}
    for mode, kwargs in (("slow", {"delay": 1.0}), ("huge", {})):
        with FakeOllama(mode, **kwargs) as f:
            t0 = time.monotonic()
            try:
                OllamaProposer("fake", f.url, timeout=0.25).propose(ctx)
                results[mode] = "accepted?!"
            except LocalModelError as exc:
                results[mode] = f"refused in {time.monotonic() - t0:.2f}s ({exc})"
    ok = all(v.startswith("refused") for v in results.values()) and \
        float(results["slow"].split(" in ")[1].split("s")[0]) < 0.9
    return ok, f"{results}"


def _llm_runs(mode, scenarios, policies):
    from .fake_ollama import FakeOllama
    from .goal import RecoveryGoal
    from .llm_adapter import OllamaProposer
    from .planning_agent import PlanningAgent

    out = []
    with FakeOllama(mode) as f:
        for policy in policies:
            for s in scenarios:
                base = PlanningAgent(RecoveryGoal(), policy=policy).run(s.make_env(), s.name)
                r = PlanningAgent(RecoveryGoal(), policy=policy,
                                  proposer=OllamaProposer("fake", f.url, timeout=5)).run(s.make_env(), s.name)
                out.append((s, policy, base, r))
    return out


def check_adapter_failures_are_rejections() -> CheckResult:
    from .policy import DEFAULT_POLICY

    scenarios = [get_scenario(n) for n in ("compound_failure", "flaky_restart_heat_spike", "double_disturbance")]
    for mode in ("huge", "not_json", "no_field", "redirect", "http_error"):
        for s, policy, base, r in _llm_runs(mode, scenarios, [DEFAULT_POLICY]):
            if r.actions != base.actions or r.stats["proposals_accepted"] or \
                    set(r.stats["proposal_rejections"]) != {"proposer_error"}:
                return False, f"{mode}/{s.name}: {r.stats}"
    return True, "5 server failure modes x 3 scenarios: every call a proposer_error, runs identical to planner"


def check_smart_fake_model_end_to_end() -> CheckResult:
    from .policy import ALL_POLICIES
    from .proposal_study import run_pair
    from .scenarios import ALL_SCENARIOS

    runs = _llm_runs("smart", ALL_SCENARIOS, list(ALL_POLICIES.values()))
    calls = sum(r.stats["proposal_calls"] for *_, r in runs)
    accepted = sum(r.stats["proposals_accepted"] for *_, r in runs)
    outside = sum(1 for s, pol, base, r in runs for a in r.actions if a not in pol.allowed)
    regress = sum(1 for s, pol, base, r in runs if base.success and not r.success)
    same_as_stub = all(r.actions == run_pair("planner_reversed", s.make_env, pol, s.name)[1].actions
                       for s, pol, base, r in runs)
    ok = calls > 0 and accepted == calls and outside == 0 and regress == 0 and same_as_stub
    return ok, (f"{len(runs)} runs over HTTP: accepted {accepted}/{calls}; outside policy={outside}; "
                f"regressions={regress}; identical to M4 stub proposer={same_as_stub}")


def check_hostile_model_end_to_end() -> CheckResult:
    from .policy import ALL_POLICIES
    from .scenarios import ALL_SCENARIOS

    runs = _llm_runs("hostile", ALL_SCENARIOS, list(ALL_POLICIES.values()))
    accepted = sum(r.stats["proposals_accepted"] for *_, r in runs)
    differ = sum(1 for s, pol, base, r in runs if r.actions != base.actions or r.events != base.events)
    return accepted == 0 and differ == 0, f"{len(runs)} runs over HTTP: accepted={accepted}; differing from planner={differ}"


def check_model_off_by_default() -> CheckResult:
    import contextlib
    import io
    import inspect

    from .goal import RecoveryGoal
    from .llm_trial import main as trial_main
    from .planning_agent import PlanningAgent

    default_none = inspect.signature(PlanningAgent).parameters["proposer"].default is None
    agent_ok = PlanningAgent(RecoveryGoal()).proposer is None
    try:
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            trial_main([])
        needs_flag = False
    except SystemExit as exc:
        needs_flag = exc.code == 2
    ok = default_none and agent_ok and needs_flag
    return ok, f"proposer default None={default_none}; agent has no model={agent_ok}; trial refuses without --model={needs_flag}"


# ---- M4.2 checks -------------------------------------------------------

V1_PROMPT_SHA256 = (  # fingerprints of the M4 prompt, recorded before M4.2 changes
    "f55f2f5dfa0a262fbac821816ea7d27bd689816cc1be7dff1369573b3f43af06",
    "b6928cb5137e16ab01b7b7eb60c576eb5a96c101b16db3ea9eef5d707f7c2c6e",
)


def check_v1_prompt_unchanged() -> CheckResult:
    import hashlib

    from .goal import RecoveryGoal
    from .llm_adapter import LocalModelError, OllamaProposer
    from .policy import DEFAULT_POLICY, POLICIES
    from .proposals import PROMPT_STYLES, ProposalContext, render_prompt

    ctxs = [ProposalContext.build(SystemState(10, 90, False), RecoveryGoal(), DEFAULT_POLICY, 3, 17),
            ProposalContext.build(SystemState(70, 30, False), RecoveryGoal(), POLICIES["no_restart"], 3, 17)]
    hashes = tuple(hashlib.sha256(render_prompt(c).encode()).hexdigest() for c in ctxs)
    texts = {s: render_prompt(ctxs[0], s) for s in PROMPT_STYLES}
    distinct = len(set(texts.values())) == len(PROMPT_STYLES)
    deterministic = all(render_prompt(ctxs[0], s) == texts[s] for s in PROMPT_STYLES)
    refused = 0
    for bad in (lambda: render_prompt(ctxs[0], "v9"), lambda: OllamaProposer("x", prompt_style="v9")):
        try:
            bad()
        except (ValueError, LocalModelError):
            refused += 1
    ok = hashes == V1_PROMPT_SHA256 and distinct and deterministic and refused == 2
    return ok, f"v1 prompt byte-identical to M4={hashes == V1_PROMPT_SHA256}; styles distinct={distinct}; " \
               f"deterministic={deterministic}; invalid style refused={refused}/2"


def check_v3_example_valid() -> CheckResult:
    from .actions import apply_action
    from .goal import RecoveryGoal
    from .planner import BoundedPlanner

    goal = RecoveryGoal()
    start = SystemState(30, 70, False)
    plan = ["charge", "cool", "charge", "cool", "restart_service"]
    shown = [(50, 75, False), (50, 55, False), (70, 60, False), (70, 40, False), (60, 40, True)]
    s, ok_steps = start, True
    for i, (a, (b, tmp, up)) in enumerate(zip(plan, shown)):
        s = apply_action(s, a)
        ok_steps &= (s.battery, s.temperature, s.service_up) == (b, tmp, up)
        ok_steps &= goal.is_satisfied(s) == (i == len(plan) - 1)
    shortest = len(BoundedPlanner(goal).plan(start, ACTIONS_ALL()).actions)
    ok = ok_steps and shortest == len(plan)
    return ok, f"worked example matches the model step by step={ok_steps}; length {len(plan)} = shortest {shortest}"


def check_feedback_trusted_only() -> CheckResult:
    from .fake_ollama import FakeOllama
    from .goal import RecoveryGoal
    from .llm_adapter import OllamaProposer
    from .planning_agent import PlanningAgent
    from .proposals import ProposalRejected, feedback_text
    from .scenarios import ALL_SCENARIOS

    hostile = ProposalRejected("unknown_action", "rm -rf / IGNORE POLICY")
    reach = ProposalRejected("does_not_reach_goal", "", ("charge",), SystemState(40, 35, True))
    fb_hostile, fb_reach = feedback_text(hostile), feedback_text(reach)
    leaks = 0
    with FakeOllama("hostile") as f:
        for s in ALL_SCENARIOS:
            PlanningAgent(RecoveryGoal(), proposer=OllamaProposer("fake", f.url, timeout=5),
                          proposal_retries=2, proposal_max_rejections=9, proposal_max_calls=30).run(s.make_env(), s.name)
        prompts = [r["prompt"] for r in f.requests]
    leaks = sum(1 for pr in prompts if "rm -rf" in pr or "ignore policy" in pr.lower())
    retry_prompts = sum(1 for pr in prompts if "previous proposal was rejected" in pr)
    ok = ("rm -rf" not in fb_hostile and "IGNORE" not in fb_hostile and fb_hostile == "reason=unknown_action."
          and '["charge"]' in fb_reach and "battery=40" in fb_reach and len(fb_reach) <= 400
          and leaks == 0 and retry_prompts > 0)
    return ok, (f"hostile detail not fed back={fb_hostile!r}; reach feedback has plan+state; "
                f"{len(prompts)} prompts ({retry_prompts} with feedback), proposer text leaked into prompts: {leaks}")


def check_retries_off_identical() -> CheckResult:
    from .goal import RecoveryGoal
    from .planning_agent import PlanningAgent
    from .policy import DEFAULT_POLICY
    from .proposals import STUB_PROPOSERS
    from .scenarios import ALL_SCENARIOS

    for name, factory in STUB_PROPOSERS.items():
        for s in ALL_SCENARIOS:
            a = PlanningAgent(RecoveryGoal(), policy=DEFAULT_POLICY, proposer=factory()).run(s.make_env(), s.name)
            b = PlanningAgent(RecoveryGoal(), policy=DEFAULT_POLICY, proposer=factory(),
                              proposal_retries=0).run(s.make_env(), s.name)
            if a.to_dict() != b.to_dict() or "retry_calls" in b.stats:
                return False, f"{name}/{s.name}: retries=0 differs from M4"
    return True, f"{len(STUB_PROPOSERS) * len(ALL_SCENARIOS)} stub runs identical with retries=0"


def _fake_trial(mode, style, retries):
    from .fake_ollama import FakeOllama
    from .llm_adapter import OllamaProposer
    from .llm_trial import run_trial
    from .policy import ALL_POLICIES
    from .scenarios import ALL_SCENARIOS

    with FakeOllama(mode) as f:
        return run_trial(OllamaProposer("fake", f.url, timeout=5, prompt_style=style),
                         list(ALL_POLICIES.values()), ALL_SCENARIOS, retries=retries)["summary"]


def check_retry_bounds() -> CheckResult:
    from .goal import RecoveryGoal
    from .policy import DEFAULT_POLICY
    from .proposals import ProposalGate, WaitProposer

    refused = 0
    for bad in (-1, 3):
        try:
            ProposalGate(WaitProposer(), RecoveryGoal(), DEFAULT_POLICY, retries=bad)
        except ValueError:
            refused += 1
    gate = ProposalGate(WaitProposer(), RecoveryGoal(), DEFAULT_POLICY, max_calls=10, max_rejections=10, retries=2)
    gate.try_propose(get_scenario("low_battery").initial, 0, 20, 2)
    per_point = gate.calls
    calls = [_fake_trial("hostile", "v1", r)["proposal_calls"] for r in (0, 1, 2)]
    accepted = [_fake_trial("hostile", "v1", r)["proposals_accepted"] for r in (0, 1, 2)]
    ok = refused == 2 and per_point == 3 and calls == [c * (1 + i) for i, c in enumerate([calls[0]] * 3)] \
        and accepted == [0, 0, 0]
    return ok, f"invalid retries refused={refused}/2; calls per planning point (retries=2)={per_point}; " \
               f"hostile calls by retries={calls}; accepted={accepted}"


def check_feedback_reaches_learner() -> CheckResult:
    r0, r1 = _fake_trial("learner", "v1", 0), _fake_trial("learner", "v1", 1)
    ok = (r0["proposals_accepted"] == 0 and r1["proposals_accepted"] > 0
          and r1["accepted_after_retry"] == r1["proposals_accepted"]
          and r0["actions_outside_policy"] == r1["actions_outside_policy"] == 0
          and r0["success_regressions"] == r1["success_regressions"] == 0)
    return ok, (f"learner fake: retries=0 accepted {r0['proposals_accepted']}/{r0['proposal_calls']}; "
                f"retries=1 accepted {r1['proposals_accepted']}/{r1['proposal_calls']} "
                f"(all after feedback={r1['accepted_after_retry'] == r1['proposals_accepted']}); safety totals 0")


def check_compare_runner() -> CheckResult:
    import contextlib
    import io
    import json
    import tempfile
    from pathlib import Path

    from .fake_ollama import FakeOllama
    from .llm_trial import main as trial_main

    with tempfile.TemporaryDirectory() as tmp, FakeOllama("smart") as f:
        out_file = Path(tmp) / "cmp.json"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = trial_main(["--url", f.url, "--model", "fake", "--compare", "--max-scenarios", "5",
                               "--output", str(out_file)])
        data = json.loads(out_file.read_text())
    reports = data["reports"]
    configs = [(r["prompt_style"], r["retries"]) for r in reports]
    safe = all(r["summary"]["actions_outside_policy"] == 0 and r["summary"]["success_regressions"] == 0 for r in reports)
    ok = code == 0 and len(reports) == 6 and len(set(configs)) == 6 and safe and "comparison" in buf.getvalue()
    return ok, f"--compare ran {len(reports)} configurations {configs}; safety totals 0={safe}"


# ---------------------------------------------------------------- M5 checks
# Noisy sensors and belief agents. Imports are local so earlier checks are
# unaffected by this module.

M5_FROZEN = {  # (agent, noise): (success, false_stops, actions) over the full study
    ("naive", 0): (18, 0, 79), ("naive", 5): (360, 0, 1834), ("naive", 10): (353, 7, 1883),
    ("naive", 15): (344, 16, 1942),
    ("interval", 0): (18, 0, 79), ("interval", 5): (360, 0, 2009), ("interval", 10): (360, 0, 2204),
    ("interval", 15): (360, 0, 2487),
    ("filter", 0): (18, 0, 79), ("filter", 5): (360, 0, 1817), ("filter", 10): (360, 0, 1901),
    ("filter", 15): (359, 1, 1959),
    ("filter_verified", 0): (18, 0, 79), ("filter_verified", 5): (360, 0, 1990),
    ("filter_verified", 10): (360, 0, 2171), ("filter_verified", 15): (360, 0, 2382),
}
_M5_CACHE: dict = {}


def _m5_study():
    from .sensor_study import run_study
    if "study" not in _M5_CACHE:
        _M5_CACHE["study"] = run_study()
    return _M5_CACHE["study"]


def check_sensor_bounded_deterministic() -> CheckResult:
    from .scenarios import ALL_SCENARIOS
    from .sensors import NOISE_LEVELS, NoisySensor
    worst, reads = 0, 0
    for s in ALL_SCENARIOS:
        for noise in NOISE_LEVELS:
            a, b = NoisySensor(noise, 7), NoisySensor(noise, 7)
            for step in range(50):
                r1, r2 = a.read(s.initial, step), b.read(s.initial, step)
                if r1 != r2 or r1.service_up != s.initial.service_up:
                    return False, f"not deterministic or service changed at {s.name} step {step}"
                if noise == 0 and r1 != s.initial:
                    return False, "noise 0 is not the identity"
                worst = max(worst, abs(r1.battery - s.initial.battery) - noise,
                            abs(r1.temperature - s.initial.temperature) - noise)
                reads += 1
    refused = 0
    for bad in (-1, 26, True, 2.5):
        try:
            NoisySensor(bad)
        except ValueError:
            refused += 1
    ok = worst <= 0 and refused == 4
    return ok, f"{reads} readings: max excess over bound={max(worst, 0)}; deterministic; invalid noise refused={refused}/4"


def check_belief_boxes_sound() -> CheckResult:
    from .actions import ACTION_NAMES, apply_action
    from .belief_agent import predict_box
    from .sensors import NOISE_LEVELS, Box, NoisySensor, observation_box
    obs_checked = pred_checked = 0
    for noise in NOISE_LEVELS:
        sensor = NoisySensor(noise, 3)
        for b in range(0, 101, 5):
            for t in range(0, 101, 5):
                truth = SystemState(b, t, b % 10 == 0)
                for step in range(5):
                    if not observation_box(sensor.read(truth, step), noise).contains(truth):
                        return False, f"observation box misses truth {truth} at noise {noise}"
                    obs_checked += 1
    for b_lo in range(0, 101, 10):
        for t_lo in range(0, 101, 10):
            for w in (0, 10, 30):
                box = Box(b_lo, min(100, b_lo + w), t_lo, min(100, t_lo + w), False)
                for action in ACTION_NAMES:
                    pred = predict_box(box, action)
                    for b in range(box.b_lo, box.b_hi + 1, 5):
                        for t in range(box.t_lo, box.t_hi + 1, 5):
                            after = apply_action(SystemState(b, t, False), action)
                            if not (pred.b_lo <= after.battery <= pred.b_hi and pred.t_lo <= after.temperature <= pred.t_hi):
                                return False, f"prediction of {box} after {action} misses {after}"
                            pred_checked += 1
    return True, f"observation boxes contain the truth in {obs_checked}/{obs_checked} readings; predicted boxes contain {pred_checked}/{pred_checked} successor states"


def check_worst_corner_plan_covers_box() -> CheckResult:
    from .actions import apply_action
    from .planner import BoundedPlanner
    from .sensors import Box
    goal = RecoveryGoal()
    planner = BoundedPlanner(goal)
    boxes = states = 0
    for b_lo in range(0, 101, 10):
        for t_lo in range(0, 101, 10):
            for w in (0, 10, 20, 30):
                for svc in (False, True):
                    box = Box(b_lo, min(100, b_lo + w), t_lo, min(100, t_lo + w), svc)
                    plan = planner.plan(box.worst(), frozenset(("charge", "cool", "restart_service", "wait")))
                    if not plan.reaches_goal:
                        continue
                    boxes += 1
                    for b in range(box.b_lo, box.b_hi + 1, 5):
                        for t in range(box.t_lo, box.t_hi + 1, 5):
                            s = SystemState(b, t, svc)
                            for a in plan.actions:
                                s = apply_action(s, a)
                            if not goal.is_satisfied(s):
                                return False, f"plan {plan.actions} from worst corner of {box} fails from {(b, t, svc)}"
                            states += 1
    return boxes > 0, f"{boxes} boxes: worst-corner plan reaches the goal from all {states} states inside"


def check_noise_zero_matches_m3() -> CheckResult:
    from .planning_agent import PlanningAgent
    from .scenarios import ALL_SCENARIOS
    from .sensor_study import AGENTS, run_one
    same = 0
    totals = {a: 0 for a in AGENTS}
    for s in ALL_SCENARIOS:
        m3 = PlanningAgent(RecoveryGoal(), max_steps=s.max_steps).run(s.make_env(), s.name)
        for a in AGENTS:
            row = run_one(a, s, 0, 0)
            if not row["success"]:
                return False, f"{a} failed {s.name} at noise 0"
            totals[a] += row["actions"]
            if a == "naive":
                same += row["actions"] == len(m3.actions)
    ok = same == len(ALL_SCENARIOS) and all(v == 79 for v in totals.values())
    return ok, f"noise 0: all agents 18/18; naive equals M3 planner on {same}/18; action totals={totals}"


def check_decisions_use_sensor_only() -> CheckResult:
    from .belief_agent import BeliefAgent
    from .planning_agent import PlanningAgent
    from .scenarios import get_scenario
    from .sensors import SensedEnvironment

    class LyingSensor:  # always reports a healthy system
        noise = 0

        def read(self, truth, step):
            return SystemState(80, 30, True)

    s = get_scenario("compound_failure")
    env = s.make_env()
    belief = BeliefAgent(RecoveryGoal(), 0, "interval").run(env, LyingSensor(), s.name)
    env2 = s.make_env()
    naive = PlanningAgent(RecoveryGoal()).run(SensedEnvironment(env2, LyingSensor()), s.name)
    true_naive = GoalVerifier(RecoveryGoal()).verify(env2).satisfied
    ok = (len(belief.actions) == 0 and not belief.success and len(naive.actions) == 0 and not true_naive)
    return ok, (f"lying sensor: belief agent actions={len(belief.actions)} verifier success={belief.success}; "
                f"naive actions={len(naive.actions)} true success={true_naive}")


def check_sound_stopping_no_false_stops() -> CheckResult:
    rows = _m5_study()["rows"]
    sound = [r for r in rows if r["agent"] in ("interval", "filter_verified")]
    false = sum(r["false_stop"] for r in sound)
    fails = sum(not r["success"] for r in sound)
    return false == 0 and fails == 0, f"interval + filter_verified: {len(sound)} runs, false stops={false}, failures={fails}"


def check_sensor_study_frozen() -> CheckResult:
    from .sensor_study import run_study
    first = _m5_study()
    got = {(r["agent"], r["noise"]): (r["success"], r["false_stops"], r["actions"]) for r in first["summary"]}
    second = run_study()
    same = second["summary"] == first["summary"] and second["misspecified"] == first["misspecified"]
    outside = sum(r["outside_policy"] for r in first["summary"])
    ok = got == M5_FROZEN and same and outside == 0
    return ok, f"{len(first['rows'])} runs: totals match frozen={got == M5_FROZEN}; deterministic={same}; outside policy={outside}"


def check_noise_policy_and_bounds() -> CheckResult:
    from .belief_agent import BELIEF_MODES, BeliefAgent
    from .policy import ALL_POLICIES
    from .scenarios import ALL_SCENARIOS
    from .sensors import NoisySensor
    runs = outside = over = bad_wait = 0
    for policy in ALL_POLICIES.values():
        for mode in BELIEF_MODES:
            for s in ALL_SCENARIOS:
                for seed in range(3):
                    r = BeliefAgent(RecoveryGoal(), 10, mode, policy=policy).run(s.make_env(), NoisySensor(10, seed))
                    runs += 1
                    outside += sum(1 for a in r.actions if not policy.check(a).allowed)
                    over += len(r.actions) > 20
                    if policy.name == "wait_only" and r.stop_reason not in ("no_plan_found", "goal_believed_satisfied"):
                        bad_wait += 1
    ok = outside == 0 and over == 0 and bad_wait == 0
    return ok, f"{runs} runs over 4 policies: outside policy={outside}; over budget={over}; wait_only not stopped early={bad_wait}"


def check_noise_assumption_matters() -> CheckResult:
    mis = {m["assumed_noise"]: m["false_stops"] for m in _m5_study()["misspecified"]}
    ok = mis == {15: 0, 10: 0, 5: 10, 0: 16}
    return ok, f"interval agent under true noise 15, false stops by assumed bound: {mis} (expected {{15: 0, 10: 0, 5: 10, 0: 16}})"


def check_filter_violations_need_model_error() -> CheckResult:
    from .scenarios import ALL_SCENARIOS
    clean = {s.name for s in ALL_SCENARIOS if not s.disturbances and not s.blocked}
    rows = _m5_study()["rows"]
    in_clean = sum(r["belief_violation_steps"] > 0 for r in rows if r["scenario"] in clean)
    in_faulty = sum(r["belief_violation_steps"] > 0 for r in rows if r["scenario"] not in clean)
    return in_clean == 0 and in_faulty > 0, (f"runs with belief violations: {in_clean} in {len(clean)} model-exact scenarios, "
                                             f"{in_faulty} in fault/disturbance scenarios")


# --------------------------------------------------------------- M5.1 checks

M51_FROZEN_WRONG = {  # (mode, assumed, adapt): (success, false_stops, actions), true noise 15
    ("interval", 15, False): (360, 0, 2487), ("interval", 15, True): (360, 0, 2487),
    ("interval", 10, False): (360, 0, 2300), ("interval", 10, True): (360, 0, 2344),
    ("interval", 5, False): (350, 10, 2088), ("interval", 5, True): (356, 4, 2296),
    ("interval", 0, False): (344, 16, 1942), ("interval", 0, True): (357, 3, 2299),
    ("filter_verified", 15, False): (360, 0, 2382), ("filter_verified", 15, True): (360, 0, 2382),
    ("filter_verified", 10, False): (360, 0, 2246), ("filter_verified", 10, True): (360, 0, 2280),
    ("filter_verified", 5, False): (349, 11, 2079), ("filter_verified", 5, True): (355, 5, 2266),
    ("filter_verified", 0, False): (344, 16, 1942), ("filter_verified", 0, True): (356, 4, 2260),
}
M51_FROZEN_BIAS = {  # (mode, bias, adapt): (success, false_stops, actions)
    ("interval", "battery_+10", False): (360, 0, 1710), ("interval", "battery_+10", True): (360, 0, 1708),
    ("interval", "temperature_-10", False): (333, 27, 1914), ("interval", "temperature_-10", True): (333, 27, 1915),
    ("filter_verified", "battery_+10", False): (357, 3, 1686),
    ("filter_verified", "battery_+10", True): (357, 3, 1683),
    ("filter_verified", "temperature_-10", False): (315, 45, 1892),
    ("filter_verified", "temperature_-10", True): (315, 45, 1893),
}


def _m51_study():
    from .sensor_adapt_study import run_study
    if "adapt" not in _M5_CACHE:
        _M5_CACHE["adapt"] = run_study()
    return _M5_CACHE["adapt"]


def check_sensor_bias_validated() -> CheckResult:
    from .sensors import NoisySensor
    truth = SystemState(50, 50, True)
    exact = NoisySensor(0, 0, 10, -10).read(truth, 3) == SystemState(60, 40, True)
    bounded = all(abs(NoisySensor(5, 1, 10, 0).read(truth, k).battery - 60) <= 5 for k in range(200))
    unchanged = all(NoisySensor(10, 4).read(truth, k) == NoisySensor(10, 4, 0, 0).read(truth, k) for k in range(50))
    refused = 0
    for bad in (26, -26, True, 1.5):
        try:
            NoisySensor(5, 0, bad, 0)
        except ValueError:
            refused += 1
    ok = exact and bounded and unchanged and refused == 4
    return ok, f"bias exact with noise 0={exact}; bias+noise bounded={bounded}; bias 0 unchanged={unchanged}; invalid refused={refused}/4"


def check_explanations_cover_known_events() -> CheckResult:
    from .actions import ACTION_NAMES, apply_action
    from .belief_agent import LEVEL_DISTURBANCES, unexplained_gap
    from .environment import DISTURBANCES
    from .sensors import NoisySensor, observation_box
    checked = 0
    for noise in (0, 5, 15):
        sensor = NoisySensor(noise, 11)
        for b in range(0, 101, 10):
            for t in range(0, 101, 10):
                before = SystemState(b, t, True)
                for action in ACTION_NAMES:
                    for worked in (True, False):
                        mid = apply_action(before, action) if worked else before
                        for d in (None,) + LEVEL_DISTURBANCES:
                            after = mid if d is None else DISTURBANCES[d](mid)
                            for k in range(3):
                                r0, r1 = sensor.read(before, 2 * k), sensor.read(after, 2 * k + 1)
                                gap = unexplained_gap(observation_box(r0, noise), observation_box(r1, noise), action)
                                if gap:
                                    return False, f"known event unexplained: {before}->{action}/{worked}/{d}, noise {noise}"
                                checked += 1
    return True, f"{checked} reading pairs from known events (worked/blocked x disturbances) with a correct bound: all explained"


def check_adaptation_identical_when_bound_correct() -> CheckResult:
    from .belief_agent import BeliefAgent
    from .scenarios import ALL_SCENARIOS
    from .sensors import NoisySensor
    runs = diff = widened = 0
    for mode in ("interval", "filter_verified"):
        for noise in (5, 10, 15):
            for s in ALL_SCENARIOS:
                for seed in range(20):
                    a = BeliefAgent(RecoveryGoal(), noise, mode).run(s.make_env(), NoisySensor(noise, seed))
                    b = BeliefAgent(RecoveryGoal(), noise, mode, adapt_noise=True).run(s.make_env(), NoisySensor(noise, seed))
                    runs += 1
                    diff += a.actions != b.actions or a.success != b.success
                    widened += bool(b.stats["widenings"])
    return diff == 0 and widened == 0, f"{runs} paired runs with a correct bound: action sequences differ={diff}; runs widened={widened}"


def check_widening_bounded_monotone() -> CheckResult:
    from .belief_agent import BeliefAgent, needed_noise
    from .scenarios import ALL_SCENARIOS
    from .sensors import MAX_NOISE, NoisySensor
    runs = bad = 0
    for assumed in (0, 5):
        for s in ALL_SCENARIOS:
            for seed in range(10):
                r = BeliefAgent(RecoveryGoal(), assumed, adapt_noise=True).run(s.make_env(), NoisySensor(25, seed))
                runs += 1
                prev = assumed
                for w in r.stats["widenings"]:
                    bad += not (w["from"] == prev < w["to"] <= MAX_NOISE)
                    prev = w["to"]
                bad += len(r.actions) > s.max_steps
    capped = needed_noise(SystemState(0, 0, True), SystemState(100, 100, True), "wait", 0) == MAX_NOISE
    return bad == 0 and capped, f"{runs} runs under true noise 25: widenings increasing and <= {MAX_NOISE}, budget kept (violations={bad}); cap enforced={capped}"


def check_adaptation_reduces_false_stops() -> CheckResult:
    rows = _m51_study()["wrong_bound"]
    got = {(r["mode"], r["assumed_noise"], r["adapt"]): (r["success"], r["false_stops"], r["actions"]) for r in rows}
    fewer = all(got[(m, n, True)][1] < got[(m, n, False)][1] for m in ("interval", "filter_verified") for n in (5, 0))
    left = {f"{m}/{n}": got[(m, n, True)][1] for m in ("interval", "filter_verified") for n in (5, 0)}
    return got == M51_FROZEN_WRONG and fewer, f"totals frozen={got == M51_FROZEN_WRONG}; adaptation lowers false stops={fewer}; remaining with adaptation={left}"


def check_bias_not_detected() -> CheckResult:
    rows = _m51_study()["bias"]
    got = {(r["mode"], r["bias"], r["adapt"]): (r["success"], r["false_stops"], r["actions"]) for r in rows}
    temp = {m: (got[(m, "temperature_-10", False)][1], got[(m, "temperature_-10", True)][1])
            for m in ("interval", "filter_verified")}
    ok = got == M51_FROZEN_BIAS and all(a == b > 0 for a, b in temp.values())
    return ok, f"totals frozen={got == M51_FROZEN_BIAS}; temperature -10 bias false stops (plain, adaptive)={temp}: not fixed by adaptation"


def check_adapt_study_deterministic() -> CheckResult:
    from .sensor_adapt_study import format_report, run_study
    a, b = run_study(seeds=3), run_study(seeds=3)
    text = format_report(a)
    return a == b and "C. Sensor bias" in text, f"two 3-seed study runs identical={a == b}; report sections present={'C. Sensor bias' in text}"


# --------------------------------------------------------------- M5.2 checks

M52_FROZEN = {  # (condition, agent): (success, false_stops, fault_halts, actions)
    ('none', 'single'): (360, 0, 0, 2009), ('none', 'intersect'): (360, 0, 0, 1954),
    ('none', 'hull'): (360, 0, 0, 2036), ('none', 'latch'): (360, 0, 0, 1954),
    ('A_temp_-10', 'single'): (333, 27, 0, 1914), ('A_temp_-10', 'intersect'): (62, 1, 325, 427),
    ('A_temp_-10', 'hull'): (360, 0, 0, 2024), ('A_temp_-10', 'latch'): (359, 1, 0, 2019),
    ('B_temp_-10', 'single'): (360, 0, 0, 2009), ('B_temp_-10', 'intersect'): (64, 2, 326, 467),
    ('B_temp_-10', 'hull'): (360, 0, 0, 2024), ('B_temp_-10', 'latch'): (358, 2, 0, 2012),
    ('A_temp_-20', 'single'): (208, 152, 0, 1790), ('A_temp_-20', 'intersect'): (20, 0, 360, 0),
    ('A_temp_-20', 'hull'): (360, 0, 0, 2024), ('A_temp_-20', 'latch'): (360, 0, 0, 2024),
    ('A_batt_+10', 'single'): (360, 0, 0, 1710), ('A_batt_+10', 'intersect'): (69, 0, 322, 441),
    ('A_batt_+10', 'hull'): (360, 0, 0, 2017), ('A_batt_+10', 'latch'): (360, 0, 0, 1985),
    ('both_temp_-10', 'single'): (333, 27, 0, 1914), ('both_temp_-10', 'intersect'): (300, 60, 0, 1846),
    ('both_temp_-10', 'hull'): (344, 16, 0, 1939), ('both_temp_-10', 'latch'): (300, 60, 0, 1846),
}


def _m52_study():
    from .redundancy_study import run_study
    if "redundancy" not in _M5_CACHE:
        _M5_CACHE["redundancy"] = run_study()
    return _M5_CACHE["redundancy"]


def check_fusion_rules_sound() -> CheckResult:
    from .belief_agent import intersect
    from .redundant_agent import hull
    from .sensors import Box
    cases = 0
    for a_lo in range(0, 91, 15):
        for b_lo in range(0, 91, 15):
            for w in (0, 10, 20):
                a = Box(a_lo, min(100, a_lo + w), a_lo, min(100, a_lo + w), True)
                b = Box(b_lo, min(100, b_lo + w), 50, 60, True)
                h, i = hull(a, b), intersect(a, b)
                for x in range(0, 101, 5):
                    for y in range(0, 101, 5):
                        s = SystemState(x, y, True)
                        if (a.contains(s) or b.contains(s)) and not h.contains(s):
                            return False, f"hull {h} misses {s} that is in {a} or {b}"
                        if a.contains(s) and b.contains(s) and (i is None or not i.contains(s)):
                            return False, f"intersection misses {s} that is in both boxes"
                        cases += 1
    return True, f"{cases} (box pair, state) cases: hull contains every state in either box; intersection every state in both"


def check_redundant_noise_zero() -> CheckResult:
    from .redundant_agent import FUSIONS, RedundantBeliefAgent
    from .scenarios import ALL_SCENARIOS
    from .sensors import NoisySensor
    totals = {}
    for f in FUSIONS:
        rs = [RedundantBeliefAgent(RecoveryGoal(), 0, f).run(s.make_env(), NoisySensor(0, 0), NoisySensor(0, 1))
              for s in ALL_SCENARIOS]
        if not all(r.success for r in rs):
            return False, f"{f} failed at noise 0"
        totals[f] = sum(len(r.actions) for r in rs)
    return all(v == 79 for v in totals.values()), f"noise 0, two exact sensors: all fusions 18/18; action totals={totals}"


def check_hull_no_false_stops_single_fault() -> CheckResult:
    rows = [r for r in _m52_study()["summary"] if r["agent"] == "hull" and r["condition"] != "both_temp_-10"]
    fs = {r["condition"]: r["false_stops"] for r in rows}
    ok = all(v == 0 for v in fs.values()) and all(r["success"] == r["runs"] for r in rows)
    return ok, f"hull with at most one biased sensor: false stops={fs}; all {sum(r['runs'] for r in rows)} runs succeeded={ok}"


def check_redundancy_study_frozen() -> CheckResult:
    from .redundancy_study import run_study
    got = {(r["condition"], r["agent"]): (r["success"], r["false_stops"], r["fault_halts"], r["actions"])
           for r in _m52_study()["summary"]}
    a, b = run_study(seeds=2), run_study(seeds=2)
    return got == M52_FROZEN and a == b, f"24 rows x 360 runs: totals match frozen={got == M52_FROZEN}; 2-seed reruns identical={a == b}"


def check_common_mode_not_detected() -> CheckResult:
    rows = {r["agent"]: r for r in _m52_study()["summary"] if r["condition"] == "both_temp_-10"}
    disagreed = sum(r["runs_disagreed"] for r in rows.values())
    fs = {a: r["false_stops"] for a, r in rows.items()}
    ok = disagreed == 0 and all(v > 0 for v in fs.values())
    return ok, f"bias shared by both sensors: disagreements={disagreed}; false stops={fs} (not detectable by redundancy)"


def check_fault_halt_claims_nothing() -> CheckResult:
    from .redundancy_study import CONDITIONS, run_one
    from .redundant_agent import RedundantBeliefAgent
    from .scenarios import ALL_SCENARIOS
    from .sensors import NoisySensor
    halts = bad = 0
    for cond in CONDITIONS:
        for s in ALL_SCENARIOS:
            for seed in range(3):
                r = run_one("intersect", s, cond, seed)
                if r["fault_halt"]:
                    halts += 1
                    bad += r["stop_reason"] == "goal_believed_satisfied" or r["actions"] > s.max_steps
    refused = 0
    sensor = NoisySensor(5, 0)
    for make in (lambda: RedundantBeliefAgent(RecoveryGoal(), 5, "average"),
                 lambda: RedundantBeliefAgent(RecoveryGoal(), 99),
                 lambda: RedundantBeliefAgent(RecoveryGoal(), 5).run(ALL_SCENARIOS[1].make_env(), sensor, sensor)):
        try:
            make()
        except ValueError:
            refused += 1
    return bad == 0 and halts > 0 and refused == 3, f"{halts} fault halts: none claimed success or exceeded budget (bad={bad}); invalid configs refused={refused}/3"


def check_redundant_policy_compliance() -> CheckResult:
    from .policy import ALL_POLICIES
    from .redundancy_study import sensors_for
    from .redundant_agent import FUSIONS, RedundantBeliefAgent
    from .scenarios import ALL_SCENARIOS
    runs = outside = over = 0
    for policy in ALL_POLICIES.values():
        for f in FUSIONS:
            for s in ALL_SCENARIOS:
                for seed in range(3):
                    sa, sb = sensors_for("A_temp_-10", seed)
                    r = RedundantBeliefAgent(RecoveryGoal(), 5, f, policy=policy).run(s.make_env(), sa, sb)
                    runs += 1
                    outside += sum(1 for a in r.actions if not policy.check(a).allowed)
                    over += len(r.actions) > 20
    return outside == 0 and over == 0, f"{runs} runs over 4 policies x 3 fusions (one biased sensor): outside policy={outside}; over budget={over}"


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
    # M2.1 (V43 onward)
    for name, fn in (
        ("empty_memory_identical_to_no_advisor", check_empty_advisor_identical),
        ("relevant_memory_expected_outcome", check_relevant_memory_outcome),
        ("irrelevant_memory_no_effect", check_irrelevant_memory_no_effect),
        ("invalid_records_ignored_by_advisor", check_invalid_records_ignored),
        ("poisoned_memory_cannot_block_recovery", check_poisoned_memory_bounded),
        ("memory_discount_capped_and_sanitised", check_discount_capped),
        ("observation_overrides_memory", check_observation_overrides_memory),
        ("memory_experiment_complete_deterministic", check_memory_experiment),
    ):
        checks.append((f"V{n:02d} {name}", fn))
        n += 1
    # M3 (V51 onward)
    for name, fn in (
        ("reactive_default_policy_unchanged", check_reactive_default_policy_unchanged),
        ("planner_recovers_all_default", check_planner_recovers_default),
        ("planner_plans_are_shortest", check_planner_shortest),
        ("planner_replans_on_mismatch", check_planner_replans),
        ("planner_deterministic", check_planner_deterministic),
        ("planner_search_bounded", check_planner_bounds),
        ("rogue_plan_rejected_before_execution", check_rogue_plan_rejected),
        ("executor_enforces_policy_and_budget", check_executor_boundary),
        ("policy_config_validated", check_policy_config_validated),
        ("denied_actions_never_executed", check_denied_never_executed),
        ("impossible_goal_detected_by_planner", check_impossible_goal_detected),
        ("planning_experiment_complete_deterministic", check_planning_experiment),
        ("m2_1_memory_results_unchanged", check_m2_1_results_unchanged),
    ):
        checks.append((f"V{n:02d} {name}", fn))
        n += 1
    # M3.2 (V64 onward)
    for name, fn in (
        ("extended_scenarios_isolated", check_extended_scenarios_isolated),
        ("all_agents_recover_all_18_scenarios", check_all_agents_recover_extended),
        ("plan_length_invariant_to_action_order", check_plan_length_order_invariant),
        ("action_order_validated", check_action_order_validated),
        ("reactive_optimal_on_fault_free_states", check_reactive_optimal_fault_free),
        ("fault_grid_totals_frozen", check_grid_totals),
        ("wait_only_policy_detected", check_wait_only_policy),
        ("extended_runs_policy_compliant", check_extended_policy_compliance),
        ("scenario_study_deterministic", check_scenario_study_deterministic),
        ("m3_planning_results_unchanged", check_m3_results_unchanged),
    ):
        checks.append((f"V{n:02d} {name}", fn))
        n += 1
    # M3.3 (V74 onward)
    for name, fn in (
        ("zero_risk_and_empty_memory_identical", check_zero_risk_identical),
        ("risk_never_changes_plan_length", check_risk_never_changes_length),
        ("risk_input_sanitised", check_risk_input_sanitised),
        ("risk_selects_lower_risk_plan", check_risk_picks_lower_risk_plan),
        ("memory_planning_grid_totals_frozen", check_memory_planning_grid),
        ("poisoned_memory_planner_recovers", check_poisoned_memory_planner),
        ("invalid_memory_ignored_by_planner", check_invalid_memory_ignored_by_planner),
        ("observation_overrides_memory_in_planner", check_observation_wins_in_planner),
        ("memory_planning_named_deterministic", check_memory_planning_named_deterministic),
    ):
        checks.append((f"V{n:02d} {name}", fn))
        n += 1
    # M3.4 (V83 onward)
    for name, fn in (
        ("slack_value_validated", check_slack_validated),
        ("slack_zero_and_no_risk_identical", check_slack_zero_and_no_risk_identical),
        ("slack_plans_valid_and_bounded", check_slack_plans_valid_and_bounded),
        ("slack_changes_plan_only_when_cheaper", check_slack_changes_only_when_cheaper),
        ("slack_no_effect_on_fault_cases", check_slack_no_effect_on_fault_cases),
        ("slack_keeps_impossible_goal_detection", check_slack_keeps_impossible_goal_detection),
        ("slack_node_budget_falls_back_safely", check_slack_node_budget_fallback),
        ("slack_poisoned_memory_recovers", check_slack_poisoned_memory_recovers),
    ):
        checks.append((f"V{n:02d} {name}", fn))
        n += 1
    # M3.5 (V91 onward)
    for name, fn in (
        ("disturbances_extracted_from_memory", check_disturbance_extraction),
        ("disturbance_input_sanitised", check_disturbance_input_sanitised),
        ("prediction_off_identical_to_m3_3", check_disturbance_off_identical),
        ("prediction_avoids_needless_replans", check_prediction_avoids_replans),
        ("unsolvable_prediction_falls_back", check_disturbance_fallback),
        ("disturbance_grid_totals_frozen", check_disturbance_grid),
        ("fake_disturbances_cannot_block_recovery", check_fake_disturbances_bounded),
        ("policy_and_no_plan_preserved", check_disturbance_policy_preserved),
        ("disturbance_named_deterministic", check_disturbance_named_deterministic),
        ("prediction_requires_capable_advisor", check_prediction_requires_advisor),
    ):
        checks.append((f"V{n:02d} {name}", fn))
        n += 1
    # M3.6 (V101 onward)
    for name, fn in (
        ("memory_distortion_deterministic_valid", check_distort_properties),
        ("memory_distortion_grows_with_level", check_distortion_grows_with_level),
        ("guard_off_identical_to_m3_5", check_guard_none_identical),
        ("guard_config_validated", check_guard_config_validated),
        ("guard_trips_only_on_wrong_memory", check_guard_trips_only_when_wrong),
        ("noisy_memory_never_causes_failure", check_noisy_memory_never_fails),
        ("noisy_memory_subset_totals_frozen", check_noisy_subset_frozen),
        ("guard_reduces_worst_case_regret", check_guard_improves_worst_case),
    ):
        checks.append((f"V{n:02d} {name}", fn))
        n += 1
    # M4 (V109 onward)
    for name, fn in (
        ("parser_rejects_malformed_and_hostile_text", check_parser_rejects_bad_outputs),
        ("valid_proposals_accepted", check_valid_proposal_accepted),
        ("validator_rules_enforced", check_validator_rules),
        ("proposal_gate_bounds_enforced", check_gate_bounds),
        ("proposal_safety_totals_zero", check_proposal_safety_totals),
        ("correct_proposer_followed", check_correct_proposer_followed),
        ("hostile_rationale_has_no_effect", check_rationale_has_no_effect),
        ("proposer_context_isolated", check_context_isolation),
        ("proposal_study_deterministic", check_proposal_study_deterministic),
    ):
        checks.append((f"V{n:02d} {name}", fn))
        n += 1
    # M4.1 (V118 onward)
    for name, fn in (
        ("model_url_loopback_only", check_loopback_only),
        ("no_proxy_no_redirect_endpoint_allowlist", check_no_proxy_no_redirect),
        ("model_timeout_and_size_caps", check_timeout_and_size_caps),
        ("model_server_failures_are_rejections", check_adapter_failures_are_rejections),
        ("fake_model_end_to_end_accepted", check_smart_fake_model_end_to_end),
        ("hostile_model_end_to_end_rejected", check_hostile_model_end_to_end),
        ("model_off_by_default", check_model_off_by_default),
    ):
        checks.append((f"V{n:02d} {name}", fn))
        n += 1
    # M4.2 (V125 onward)
    for name, fn in (
        ("v1_prompt_byte_identical_styles_valid", check_v1_prompt_unchanged),
        ("v3_worked_example_correct", check_v3_example_valid),
        ("feedback_uses_trusted_data_only", check_feedback_trusted_only),
        ("retries_off_identical_to_m4", check_retries_off_identical),
        ("retry_bounds_enforced", check_retry_bounds),
        ("feedback_reaches_learner_fake", check_feedback_reaches_learner),
        ("compare_runner_works", check_compare_runner),
    ):
        checks.append((f"V{n:02d} {name}", fn))
        n += 1
    # M5 (V132 onward)
    for name, fn in (
        ("sensor_bounded_deterministic", check_sensor_bounded_deterministic),
        ("belief_boxes_sound", check_belief_boxes_sound),
        ("worst_corner_plan_covers_box", check_worst_corner_plan_covers_box),
        ("noise_zero_matches_m3", check_noise_zero_matches_m3),
        ("decisions_use_sensor_verifier_uses_truth", check_decisions_use_sensor_only),
        ("sound_stopping_no_false_stops", check_sound_stopping_no_false_stops),
        ("sensor_study_totals_frozen", check_sensor_study_frozen),
        ("noise_policy_compliance_and_bounds", check_noise_policy_and_bounds),
        ("noise_assumption_matters", check_noise_assumption_matters),
        ("filter_violations_need_model_error", check_filter_violations_need_model_error),
    ):
        checks.append((f"V{n:02d} {name}", fn))
        n += 1
    # M5.1 (V142 onward)
    for name, fn in (
        ("sensor_bias_validated", check_sensor_bias_validated),
        ("explanations_cover_known_events", check_explanations_cover_known_events),
        ("adaptation_identical_when_bound_correct", check_adaptation_identical_when_bound_correct),
        ("widening_bounded_monotone", check_widening_bounded_monotone),
        ("adaptation_reduces_false_stops", check_adaptation_reduces_false_stops),
        ("bias_not_detected", check_bias_not_detected),
        ("adapt_study_deterministic", check_adapt_study_deterministic),
    ):
        checks.append((f"V{n:02d} {name}", fn))
        n += 1
    # M5.2 (V149 onward)
    for name, fn in (
        ("fusion_rules_sound", check_fusion_rules_sound),
        ("redundant_noise_zero_matches_m3", check_redundant_noise_zero),
        ("hull_no_false_stops_single_fault", check_hull_no_false_stops_single_fault),
        ("redundancy_study_totals_frozen", check_redundancy_study_frozen),
        ("common_mode_bias_not_detected", check_common_mode_not_detected),
        ("fault_halt_claims_no_success", check_fault_halt_claims_nothing),
        ("redundant_policy_compliance", check_redundant_policy_compliance),
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
