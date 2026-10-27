import unittest

from red_queen.actions import ACTION_NAMES, apply_action
from red_queen.goal import RecoveryGoal
from red_queen.memory_planning_study import prior_memory
from red_queen.planner import MAX_SLACK, BoundedPlanner
from red_queen.scenarios import get_scenario
from red_queen.slack_study import AGENT_NAMES, case_type, format_report, run_agent, run_case
from red_queen.state import SystemState

GOAL = RecoveryGoal()

# The counterexample found by V86: every shortest plan pays risk twice (0.6 + 0.6),
# so one extra step (cost 1) is estimated to be cheaper.
STACKED_START = SystemState(70, 90, False)


def stacked_risk(depth):
    return {0: {"cool": 0.6}, 1: {"charge": 0.9, "restart_service": 0.6},
            2: {"charge": 0.9, "restart_service": 0.6}}.get(depth, {})


class TestSlackConfig(unittest.TestCase):
    def test_default_zero(self):
        self.assertEqual(BoundedPlanner(GOAL).slack, 0)

    def test_cap(self):
        BoundedPlanner(GOAL, slack=MAX_SLACK)
        for bad in (-1, MAX_SLACK + 1, True, 0.5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                BoundedPlanner(GOAL, slack=bad)


class TestSlackPlanning(unittest.TestCase):
    def test_slack_ignored_without_risk(self):
        start = get_scenario("compound_failure").initial
        self.assertEqual(BoundedPlanner(GOAL, slack=3).plan(start, ACTION_NAMES).actions,
                         BoundedPlanner(GOAL).plan(start, ACTION_NAMES).actions)

    def test_stacked_risk_prefers_longer_cheaper_plan(self):
        short = BoundedPlanner(GOAL).plan(STACKED_START, ACTION_NAMES, risk=stacked_risk)
        long = BoundedPlanner(GOAL, slack=1).plan(STACKED_START, ACTION_NAMES, risk=stacked_risk)
        self.assertEqual(len(short.actions), 3)
        self.assertEqual(len(long.actions), 4)
        self.assertLess(len(long.actions) + long.risk, len(short.actions) + short.risk)

    def test_longer_plan_is_model_consistent(self):
        p = BoundedPlanner(GOAL, slack=1).plan(STACKED_START, ACTION_NAMES, risk=stacked_risk)
        s = STACKED_START
        for i, (a, pred) in enumerate(zip(p.actions, p.predicted)):
            s = apply_action(s, a)
            self.assertEqual(s, pred)
            if i < len(p.actions) - 1:
                self.assertFalse(GOAL.is_satisfied(s))  # never passes through the goal
        self.assertTrue(GOAL.is_satisfied(s))

    def test_single_risky_step_never_worth_a_detour(self):
        p0 = BoundedPlanner(GOAL).plan(SystemState(70, 30, False), ACTION_NAMES,
                                       risk=lambda d: {"restart_service": 0.9} if d < 3 else {})
        p3 = BoundedPlanner(GOAL, slack=3).plan(SystemState(70, 30, False), ACTION_NAMES,
                                                risk=lambda d: {"restart_service": 0.9} if d < 3 else {})
        self.assertEqual(p0.actions, p3.actions)  # 1 + 0.9 < any detour of cost >= 2

    def test_impossible_goal_unchanged(self):
        p = BoundedPlanner(GOAL, slack=3).plan(SystemState(70, 30, False), ("charge", "cool", "wait"),
                                               risk=lambda d: {})
        self.assertFalse(p.reaches_goal)

    def test_respects_allowed_actions(self):
        allowed = ("cool", "restart_service")
        p = BoundedPlanner(GOAL, slack=3).plan(STACKED_START, allowed, risk=stacked_risk)
        self.assertTrue(set(p.actions) <= set(allowed))


class TestSlackStudy(unittest.TestCase):
    def test_case_type_labels(self):
        self.assertEqual(case_type((), ()), "no_fault+no_disturbance")
        self.assertEqual(case_type((("cool", 1, 1),), ((1, "heat_spike"),)), "fault+disturbance")

    def test_run_case_all_agents_succeed(self):
        s = get_scenario("deep_battery_flaky_restart")
        case = run_case(s.name, s.make_env)
        self.assertEqual(set(case), set(AGENT_NAMES))
        self.assertTrue(all(v["success"] for v in case.values()))

    def test_unknown_agent(self):
        s = get_scenario("nominal")
        with prior_memory(s.name, s.make_env) as memory, self.assertRaises(ValueError):
            run_agent("oracle", s.name, s.make_env, memory)

    def test_format_report(self):
        cell = {"episodes": 1, "successes": 1, "total_actions": 2}
        summary = {a: dict(cell) for a in AGENT_NAMES}
        report = {"milestone": "M3.4", "version": "x", "agents": list(AGENT_NAMES),
                  "grid": {"cases": 1, "summary": summary, "by_case_type": {"fault+disturbance": summary}},
                  "episodes_where_slack_changed_action_count": 0, "note": "n"}
        text = format_report(report)
        self.assertIn("F+D", text)
        self.assertIn("planner_memory_k3", text)


if __name__ == "__main__":
    unittest.main()
