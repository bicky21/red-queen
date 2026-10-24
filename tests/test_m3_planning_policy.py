import json
import unittest

from red_queen.actions import ACTION_NAMES, ACTIONS
from red_queen.agent import STOP_GOAL, STOP_MAX_STEPS, STOP_NO_ALLOWED, STOP_POLICY, RecoveryAgent
from red_queen.affect import AffectConfig
from red_queen.environment import SimulatedEnvironment
from red_queen.goal import RecoveryGoal
from red_queen.planner import BoundedPlanner, Plan
from red_queen.planning_agent import STOP_MAX_REPLANS, STOP_NO_PLAN, PlanningAgent
from red_queen.planning_experiment import (
    AGENTS,
    format_report,
    run_agent,
    run_planner,
    run_planning_experiment,
)
from red_queen.policy import (
    DEFAULT_POLICY,
    POLICIES,
    Policy,
    PolicyConfigError,
    PolicyDenied,
    PolicyExecutor,
)
from red_queen.scenarios import SCENARIOS, get_scenario
from red_queen.state import SystemState

GOAL = RecoveryGoal()


class TestPolicy(unittest.TestCase):
    def test_default_allows_all(self):
        self.assertEqual(DEFAULT_POLICY.allowed, frozenset(ACTIONS))
        for a in ACTIONS:
            self.assertTrue(DEFAULT_POLICY.check(a).allowed)

    def test_restricted_and_unknown(self):
        self.assertFalse(POLICIES["no_restart"].check("restart_service").allowed)
        self.assertTrue(POLICIES["no_restart"].check("charge").allowed)
        self.assertFalse(DEFAULT_POLICY.check("shell").allowed)

    def test_safety_config_rejects_unknown_actions(self):
        with self.assertRaises(PolicyConfigError):
            Policy("bad", frozenset({"shell"}))
        with self.assertRaises(PolicyConfigError):
            Policy("bad", {"charge"})


class TestExecutor(unittest.TestCase):
    def test_safety_denied_action_never_reaches_env(self):
        env = SimulatedEnvironment(SystemState(50, 50, False))
        ex = PolicyExecutor(env, POLICIES["no_restart"], 5)
        with self.assertRaises(PolicyDenied):
            ex.execute("restart_service")
        self.assertEqual(env.step, 0)
        self.assertEqual(env.events, [])
        self.assertEqual(ex.denials[0]["action"], "restart_service")

    def test_budget(self):
        env = SimulatedEnvironment(SystemState(50, 50, True))
        ex = PolicyExecutor(env, DEFAULT_POLICY, 2)
        ex.execute("wait")
        ex.execute("wait")
        with self.assertRaises(PolicyDenied):
            ex.execute("wait")
        self.assertEqual(env.step, 2)

    def test_negative_budget(self):
        with self.assertRaises(ValueError):
            PolicyExecutor(SimulatedEnvironment(SystemState(1, 1, True)), DEFAULT_POLICY, -1)


class TestPlanner(unittest.TestCase):
    def test_goal_already_met(self):
        p = BoundedPlanner(GOAL).plan(SystemState(80, 30, True), ACTION_NAMES)
        self.assertTrue(p.reaches_goal)
        self.assertEqual(p.actions, ())

    def test_plan_predictions_follow_model(self):
        from red_queen.actions import apply_action

        start = get_scenario("compound_failure").initial
        p = BoundedPlanner(GOAL).plan(start, ACTION_NAMES)
        s = start
        for action, predicted in zip(p.actions, p.predicted):
            s = apply_action(s, action)
            self.assertEqual(s, predicted)
        self.assertTrue(GOAL.is_satisfied(s))

    def test_respects_allowed_actions(self):
        p = BoundedPlanner(GOAL).plan(SystemState(70, 90, True), [a for a in ACTION_NAMES if a != "cool"])
        self.assertFalse(p.reaches_goal)
        self.assertFalse(p.hit_node_limit)  # exhausted the reachable space, not the budget

    def test_node_limit(self):
        p = BoundedPlanner(GOAL, max_nodes=3).plan(get_scenario("compound_failure").initial, ACTION_NAMES)
        self.assertTrue(p.hit_node_limit)
        self.assertLessEqual(p.nodes_expanded, 3)

    def test_depth_zero(self):
        p = BoundedPlanner(GOAL, max_depth=0).plan(SystemState(20, 30, True), ACTION_NAMES)
        self.assertFalse(p.reaches_goal)

    def test_invalid_bounds(self):
        with self.assertRaises(ValueError):
            BoundedPlanner(GOAL, max_depth=-1)


class _FixedPlanner:
    def __init__(self, plan):
        self._plan = plan
        self.calls = 0

    def plan(self, start, allowed):
        self.calls += 1
        return self._plan


class TestPlanningAgent(unittest.TestCase):
    def test_recovers_all_scenarios(self):
        for s in SCENARIOS:
            with self.subTest(scenario=s.name):
                r = run_planner(s)
                self.assertTrue(r.success)
                self.assertEqual(r.stop_reason, STOP_GOAL)
                self.assertEqual(r.stats["agent"], "planner")

    def test_replans_on_blocked_action(self):
        r = run_planner(get_scenario("flaky_restart"))
        self.assertEqual(r.stats["replans"], 3)
        self.assertEqual(r.actions, ("restart_service",) * 4)

    def test_no_plan_stops_without_acting(self):
        r = run_planner(get_scenario("service_down"), POLICIES["no_restart"])
        self.assertEqual(r.action_count, 0)
        self.assertEqual(r.stop_reason, STOP_NO_PLAN)
        self.assertFalse(r.success)

    def test_safety_rogue_plan_rejected(self):
        bad = Plan(("restart_service",), (SystemState(60, 30, True),), True, 1, False)
        planner = _FixedPlanner(bad)
        agent = PlanningAgent(GOAL, planner=planner, policy=POLICIES["no_restart"], max_policy_denials=2)
        env = get_scenario("service_down").make_env()
        r = agent.run(env)
        self.assertEqual(r.stop_reason, STOP_POLICY)
        self.assertEqual(env.step, 0)
        self.assertEqual(planner.calls, 2)

    def test_safety_malformed_plan_rejected(self):
        bad = Plan(("charge", "charge"), (SystemState(40, 35, True),), True, 1, False)  # length mismatch
        agent = PlanningAgent(GOAL, planner=_FixedPlanner(bad))
        env = get_scenario("low_battery").make_env()
        r = agent.run(env)
        self.assertEqual(r.stop_reason, STOP_POLICY)
        self.assertEqual(env.step, 0)

    def test_max_replans(self):
        # Plan predicts a state that never happens, so every step forces a replan.
        wrong = Plan(("charge",), (SystemState(100, 0, True),), True, 1, False)
        agent = PlanningAgent(GOAL, planner=_FixedPlanner(wrong), max_replans=2)
        r = agent.run(get_scenario("compound_failure").make_env())
        self.assertEqual(r.stop_reason, STOP_MAX_REPLANS)
        self.assertEqual(r.action_count, 3)

    def test_max_steps(self):
        agent = PlanningAgent(GOAL, max_steps=2)
        r = agent.run(get_scenario("compound_failure").make_env())
        self.assertEqual(r.action_count, 2)
        self.assertEqual(r.stop_reason, STOP_MAX_STEPS)

    def test_invalid_bounds(self):
        with self.assertRaises(ValueError):
            PlanningAgent(GOAL, max_policy_denials=0)


class TestReactiveWithPolicy(unittest.TestCase):
    def test_default_policy_stats(self):
        r = run_agent("reactive", get_scenario("compound_failure"), DEFAULT_POLICY)
        self.assertEqual(r.stats, {"agent": "reactive", "policy": "default",
                                   "policy_filtered": 0, "policy_denials": 0})

    def test_restricted_policy_filters_and_never_executes(self):
        r = run_agent("reactive", get_scenario("service_down"), POLICIES["no_restart"])
        self.assertNotIn("restart_service", r.actions)
        self.assertGreater(r.stats["policy_filtered"], 0)
        self.assertEqual(r.stats["policy_denials"], 0)

    def test_empty_policy_stops(self):
        agent = RecoveryAgent(GOAL, AffectConfig(enabled=False), policy=Policy("none", frozenset()))
        r = agent.run(get_scenario("low_battery").make_env())
        self.assertEqual(r.stop_reason, STOP_NO_ALLOWED)
        self.assertEqual(r.action_count, 0)


class TestExperiment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_planning_experiment()

    def test_shape_and_json(self):
        self.assertEqual(len(self.report["rows"]), len(SCENARIOS) * len(POLICIES) * len(AGENTS))
        json.dumps(self.report)
        self.assertIn("no_restart/planner", format_report(self.report))

    def test_no_denied_action_executed(self):
        self.assertEqual(sum(r["denied_actions_executed"] for r in self.report["rows"]), 0)

    def test_same_successes_per_policy(self):
        for policy in POLICIES:
            self.assertEqual(self.report["summary"][f"{policy}/reactive"]["successes"],
                             self.report["summary"][f"{policy}/planner"]["successes"])

    def test_unknown_agent(self):
        with self.assertRaises(ValueError):
            run_agent("oracle", get_scenario("nominal"), DEFAULT_POLICY)


if __name__ == "__main__":
    unittest.main()
