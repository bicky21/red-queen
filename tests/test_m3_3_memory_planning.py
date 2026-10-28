import json
import unittest

from red_queen.actions import ACTION_NAMES
from red_queen.environment import SimulatedEnvironment
from red_queen.goal import RecoveryGoal
from red_queen.memory import EpisodeMemory
from red_queen.memory_advice import MemoryAdvisor
from red_queen.memory_planning_study import (
    AGENTS,
    format_report,
    prior_memory,
    run_agent,
    run_case,
    run_named_part,
)
from red_queen.planner import BoundedPlanner, Plan
from red_queen.planning_agent import PlanningAgent
from red_queen.policy import POLICIES
from red_queen.scenario_study import REVERSED_ORDER
from red_queen.scenarios import ALL_SCENARIOS, get_scenario
from red_queen.scoring import MAX_MEMORY_DISCOUNT
from red_queen.state import SystemState

GOAL = RecoveryGoal()


class TestRiskPlanner(unittest.TestCase):
    def test_plan_has_default_risk(self):
        self.assertEqual(Plan((), (), True, 0, False).risk, 0.0)

    def test_zero_risk_matches_original(self):
        for order in (ACTION_NAMES, REVERSED_ORDER):
            planner = BoundedPlanner(GOAL, action_order=order)
            for s in ALL_SCENARIOS:
                with self.subTest(order=order[0], scenario=s.name):
                    self.assertEqual(planner.plan(s.initial, ACTION_NAMES).actions,
                                     planner.plan(s.initial, ACTION_NAMES, risk=lambda d: {}).actions)

    def test_risk_moves_risky_action_later(self):
        planner = BoundedPlanner(GOAL, action_order=REVERSED_ORDER)
        p = planner.plan(SystemState(10, 30, False), ACTION_NAMES,
                         risk=lambda d: {"restart_service": 0.9} if d < 3 else {})
        self.assertEqual(p.actions, ("charge", "charge", "charge", "restart_service"))
        self.assertEqual(p.risk, 0.0)

    def test_unavoidable_risk_reported(self):
        # Only restart can fix the service; risk on every step must be paid once.
        p = BoundedPlanner(GOAL).plan(SystemState(70, 30, False), ACTION_NAMES,
                                      risk=lambda d: {"restart_service": 0.9})
        self.assertEqual(p.actions, ("restart_service",))
        self.assertAlmostEqual(p.risk, 0.9)

    def test_risk_respects_allowed_and_cap(self):
        allowed = ("charge", "cool")
        p = BoundedPlanner(GOAL).plan(SystemState(20, 90, True), allowed,
                                      risk=lambda d: {"charge": 50, "restart_service": 1, "shell": 1})
        self.assertTrue(set(p.actions) <= set(allowed))
        self.assertLessEqual(p.risk, MAX_MEMORY_DISCOUNT * len(p.actions))

    def test_risk_with_garbage_values(self):
        for risk in (lambda d: None, lambda d: {"charge": float("nan")}, lambda d: {"cool": "x"}):
            with self.subTest():
                p = BoundedPlanner(GOAL).plan(SystemState(20, 30, True), ACTION_NAMES, risk=risk)
                self.assertTrue(p.reaches_goal)
                self.assertEqual(len(p.actions), 2)

    def test_risk_search_respects_node_limit(self):
        p = BoundedPlanner(GOAL, max_nodes=5).plan(get_scenario("compound_failure").initial,
                                                   ACTION_NAMES, risk=lambda d: {})
        self.assertTrue(p.hit_node_limit)
        self.assertLessEqual(p.nodes_expanded, 5)

    def test_risk_search_impossible_goal(self):
        p = BoundedPlanner(GOAL).plan(SystemState(70, 30, False), ("charge", "cool", "wait"),
                                      risk=lambda d: {})
        self.assertFalse(p.reaches_goal)


class _Flat:
    def discounts(self, step):
        return {a: 0.5 for a in ACTION_NAMES}


class TestPlanningAgentMemory(unittest.TestCase):
    def test_stats_mark_memory(self):
        r = PlanningAgent(GOAL, advisor=_Flat()).run(get_scenario("low_battery").make_env())
        self.assertTrue(r.stats["memory"])
        self.assertEqual(len(r.stats["plan_risks"]), r.stats["plans"])

    def test_no_advisor_stats_unchanged(self):
        r = PlanningAgent(GOAL).run(get_scenario("low_battery").make_env())
        self.assertNotIn("memory", r.stats)

    def test_memory_fixes_reversed_order(self):
        s = get_scenario("deep_battery_flaky_restart")
        with prior_memory(s.name, s.make_env) as memory:
            with_mem = run_agent("planner_reversed_memory", s.name, s.make_env, memory)
        without = run_agent("planner_reversed", s.name, s.make_env, None)
        self.assertEqual((without.action_count, with_mem.action_count), (7, 4))

    def test_policy_still_enforced_with_memory(self):
        s = get_scenario("service_down")
        with prior_memory(s.name, s.make_env) as memory:
            agent = PlanningAgent(GOAL, policy=POLICIES["no_restart"], advisor=MemoryAdvisor(memory, s.name))
            r = agent.run(s.make_env(), s.name)
        self.assertEqual(r.action_count, 0)
        self.assertFalse(r.success)

    def test_safety_poisoned_memory(self):
        for s in ALL_SCENARIOS:
            with self.subTest(scenario=s.name), prior_memory(s.name, s.make_env) as memory:
                memory._conn.execute("UPDATE events SET kind = 'blocked' WHERE kind = 'action'")
                self.assertTrue(run_agent("planner_memory", s.name, s.make_env, memory).success)

    def test_advisor_never_writes_memory(self):
        s = get_scenario("flaky_restart")
        with prior_memory(s.name, s.make_env) as memory:
            before = (memory.count(), memory.event_count())
            run_agent("planner_memory", s.name, s.make_env, memory)
            self.assertEqual((memory.count(), memory.event_count()), before)


class TestStudy(unittest.TestCase):
    def test_memory_agent_requires_memory(self):
        s = get_scenario("nominal")
        with self.assertRaises(ValueError):
            run_agent("planner_memory", s.name, s.make_env, None)
        with EpisodeMemory(":memory:") as memory, self.assertRaises(ValueError):
            run_agent("oracle", s.name, s.make_env, memory)

    def test_run_case_shape(self):
        s = get_scenario("flaky_restart_low_battery")
        case = run_case(s.name, s.make_env)
        self.assertEqual(set(case), set(AGENTS))
        self.assertTrue(all(c["success"] for c in case.values()))

    def test_grid_style_case(self):
        make_env = lambda: SimulatedEnvironment(SystemState(10, 60, False), {}, (("restart_service", 1, 3),))  # noqa: E731
        case = run_case("grid_test", make_env)
        self.assertLessEqual(case["planner_memory"]["action_count"], case["planner"]["action_count"])

    def test_named_report(self):
        named = run_named_part()
        self.assertEqual(len(named["cases"]), len(ALL_SCENARIOS))
        report = {"milestone": "M3.3", "version": "x", "memory_source": "m", "named": named,
                  "grid": {"cases": 0, "summary": {}, "head_to_head": {}}, "note": "n"}
        json.dumps(report)
        self.assertIn("deep_battery_flaky_restart", format_report(report))


if __name__ == "__main__":
    unittest.main()
