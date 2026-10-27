import json
import unittest

from red_queen.actions import ACTION_NAMES
from red_queen.goal import RecoveryGoal
from red_queen.planner import BoundedPlanner
from red_queen.planning_agent import STOP_NO_PLAN
from red_queen.policy import ALL_POLICIES, POLICIES, WAIT_ONLY_POLICY, Policy
from red_queen.scenario_study import (
    AGENTS,
    GRID_DISTURBANCES,
    GRID_FAULTS,
    GRID_STARTS,
    REVERSED_ORDER,
    format_report,
    grid_cases,
    run_agent,
    run_named_part,
)
from red_queen.scenarios import ALL_SCENARIOS, EXTENDED_SCENARIOS, SCENARIOS, get_scenario
from red_queen.state import SystemState

GOAL = RecoveryGoal()


class TestScenarioRegistry(unittest.TestCase):
    def test_core_set_unchanged(self):
        self.assertEqual(len(SCENARIOS), 10)
        self.assertEqual(ALL_SCENARIOS[:10], SCENARIOS)

    def test_extended_lookup(self):
        for s in EXTENDED_SCENARIOS:
            self.assertIs(get_scenario(s.name), s)
            s.make_env()  # every scenario builds a valid environment

    def test_unknown_scenario(self):
        with self.assertRaises(KeyError):
            get_scenario("does_not_exist")


class TestPoliciesM32(unittest.TestCase):
    def test_core_policies_unchanged(self):
        self.assertEqual(set(POLICIES), {"default", "no_restart", "no_cool"})
        self.assertEqual(set(ALL_POLICIES), set(POLICIES) | {"wait_only"})

    def test_wait_only(self):
        self.assertTrue(WAIT_ONLY_POLICY.check("wait").allowed)
        self.assertFalse(WAIT_ONLY_POLICY.check("charge").allowed)
        self.assertIsInstance(WAIT_ONLY_POLICY, Policy)


class TestActionOrder(unittest.TestCase):
    def test_default_order_is_sorted(self):
        self.assertEqual(BoundedPlanner(GOAL).action_order, ACTION_NAMES)

    def test_invalid_orders(self):
        for bad in (("charge",), ACTION_NAMES + ("shell",), ("charge", "charge", "cool", "wait")):
            with self.subTest(order=bad), self.assertRaises(ValueError):
                BoundedPlanner(GOAL, action_order=bad)

    def test_order_changes_plan_not_length(self):
        start = SystemState(10, 30, False)
        a = BoundedPlanner(GOAL).plan(start, ACTION_NAMES)
        b = BoundedPlanner(GOAL, action_order=REVERSED_ORDER).plan(start, ACTION_NAMES)
        self.assertEqual(len(a.actions), len(b.actions))
        self.assertNotEqual(a.actions, b.actions)
        self.assertEqual(a.actions[0], "charge")
        self.assertEqual(b.actions[0], "restart_service")


class TestAgents(unittest.TestCase):
    def test_tie_break_explains_flaky_restart_gap(self):
        s = get_scenario("deep_battery_flaky_restart")
        counts = {a: run_agent(a, s.make_env).action_count for a in AGENTS}
        self.assertEqual(counts, {"reactive": 7, "planner": 4, "planner_reversed": 7})

    def test_planner_can_be_worse(self):
        s = get_scenario("cool_outage_battery_drain")
        r = run_agent("reactive", s.make_env).action_count
        p = run_agent("planner", s.make_env).action_count
        self.assertGreater(p, r)

    def test_wait_only_planner_stops(self):
        r = run_agent("planner_reversed", get_scenario("battery_empty").make_env, WAIT_ONLY_POLICY)
        self.assertEqual((r.action_count, r.stop_reason), (0, STOP_NO_PLAN))

    def test_nominal_succeeds_even_with_wait_only(self):
        for agent in AGENTS:
            r = run_agent(agent, get_scenario("nominal").make_env, WAIT_ONLY_POLICY)
            self.assertTrue(r.success)
            self.assertEqual(r.action_count, 0)

    def test_unknown_agent(self):
        with self.assertRaises(ValueError):
            run_agent("oracle", get_scenario("nominal").make_env)


class TestStudy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.named = run_named_part()

    def test_grid_size(self):
        self.assertEqual(len(list(grid_cases())), len(GRID_STARTS) * len(GRID_FAULTS) * len(GRID_DISTURBANCES))
        self.assertEqual(len(list(grid_cases())), 1680)

    def test_named_shape(self):
        self.assertEqual(len(self.named["rows"]), len(ALL_SCENARIOS) * len(ALL_POLICIES) * len(AGENTS))
        json.dumps(self.named)

    def test_safety_no_action_outside_policy(self):
        for r in self.named["rows"]:
            self.assertLessEqual(set(r["actions"]), ALL_POLICIES[r["policy"]].allowed)

    def test_planners_never_waste_actions_on_impossible_goals_more_than_reactive(self):
        s = self.named["summary"]
        for policy in ("no_restart", "no_cool", "wait_only"):
            reactive = s[f"{policy}/reactive"]["actions_in_failed_episodes"]
            for agent in ("planner", "planner_reversed"):
                self.assertLess(s[f"{policy}/{agent}"]["actions_in_failed_episodes"], reactive)

    def test_same_success_counts_per_policy(self):
        s = self.named["summary"]
        for policy in ALL_POLICIES:
            counts = {s[f"{policy}/{a}"]["successes"] for a in AGENTS}
            self.assertEqual(len(counts), 1, policy)

    def test_format_report(self):
        report = {"milestone": "M3.2", "version": "x", "named": self.named,
                  "grid": {"cases": 0, "summary": {}, "pairwise": {}}, "note": "n"}
        text = format_report(report)
        self.assertIn("deep_battery_flaky_restart", text)
        self.assertIn("wait_only/planner", text)


if __name__ == "__main__":
    unittest.main()
