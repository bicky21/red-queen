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
