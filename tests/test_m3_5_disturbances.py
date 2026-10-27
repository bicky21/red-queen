import unittest

from red_queen.actions import ACTION_NAMES, apply_action
from red_queen.disturbance_study import AGENTS, format_report, run_agent, run_case
from red_queen.environment import DISTURBANCES, SimulatedEnvironment
from red_queen.goal import RecoveryGoal
from red_queen.memory_advice import MemoryAdvisor
from red_queen.memory_planning_study import prior_memory
from red_queen.planner import BoundedPlanner
from red_queen.planning_agent import PlanningAgent
from red_queen.policy import POLICIES
from red_queen.scenarios import get_scenario
from red_queen.state import SystemState

GOAL = RecoveryGoal()


class TestAdvisorDisturbances(unittest.TestCase):
    def test_extracts_steps_and_names(self):
        s = get_scenario("double_disturbance")
        with prior_memory(s.name, s.make_env) as m:
            self.assertEqual(MemoryAdvisor(m, s.name).remembered_disturbances(),
                             {2: "heat_spike", 3: "battery_drain"})

    def test_no_disturbances(self):
        s = get_scenario("compound_failure")
        with prior_memory(s.name, s.make_env) as m:
            self.assertEqual(MemoryAdvisor(m, s.name).remembered_disturbances(), {})

    def test_returns_copy(self):
        s = get_scenario("double_disturbance")
        with prior_memory(s.name, s.make_env) as m:
            adv = MemoryAdvisor(m, s.name)
            adv.remembered_disturbances()[2] = "service_crash"
            self.assertEqual(adv.remembered_disturbances()[2], "heat_spike")

    def test_invalid_record_contributes_nothing(self):
        s = get_scenario("double_disturbance")
        with prior_memory(s.name, s.make_env) as m:
            m._conn.execute("UPDATE episodes SET action_count = 99")
            self.assertEqual(MemoryAdvisor(m, s.name).remembered_disturbances(), {})

    def test_majority_rule(self):
        # Three records of the same case, but the disturbances appear in only one:
        # 1 of 3 is below half, so nothing is remembered.
        s = get_scenario("double_disturbance")
        with prior_memory(s.name, s.make_env) as m:
            from red_queen.affect import AffectConfig
            from red_queen.experiments import run_scenario
            cfg = AffectConfig(enabled=False)
            clean = run_scenario(get_scenario("compound_failure"), False, affect=cfg)
            for _ in range(2):
                rid = m.record(clean, cfg)
                m._conn.execute("UPDATE episodes SET scenario = ? WHERE id = ?", (s.name, rid))
                m._conn.execute("UPDATE episodes SET summary = ? WHERE id = ?", (f"scenario {s.name}", rid))
            self.assertEqual(MemoryAdvisor(m, s.name).remembered_disturbances(), {})


class TestPlannerDisturbances(unittest.TestCase):
    def test_predictions_include_disturbance(self):
        start = SystemState(10, 60, False)
        p = BoundedPlanner(GOAL).plan(start, ACTION_NAMES,
                                      disturbances=lambda d: "heat_spike" if d == 1 else None)
        s = start
        for depth, (a, pred) in enumerate(zip(p.actions, p.predicted)):
            s = apply_action(s, a)
            if depth == 1:
                s = DISTURBANCES["heat_spike"](s)
            self.assertEqual(s, pred)
        self.assertTrue(GOAL.is_satisfied(s))

    def test_plans_around_remembered_crash(self):
        # Restarting before a remembered crash is wasted; prediction moves restart after it.
        start = SystemState(30, 30, False)
        p = BoundedPlanner(GOAL).plan(start, ACTION_NAMES,
                                      disturbances=lambda d: "service_crash" if d == 0 else None)
        self.assertNotEqual(p.actions[0], "restart_service")
        self.assertEqual(p.actions[-1], "restart_service")

    def test_node_limit_falls_back(self):
        start = get_scenario("compound_failure").initial
        tiny = BoundedPlanner(GOAL, max_nodes=50)
        p = tiny.plan(start, ACTION_NAMES, disturbances=lambda d: None)
        self.assertEqual(p.actions, tiny.plan(start, ACTION_NAMES).actions)

    def test_respects_allowed(self):
        allowed = ("charge", "restart_service", "wait")
        p = BoundedPlanner(GOAL).plan(SystemState(10, 30, False), allowed,
                                      disturbances=lambda d: "battery_drain" if d == 0 else None)
        self.assertTrue(set(p.actions) <= set(allowed))


class TestAgentDisturbances(unittest.TestCase):
    def test_stats_report_remembered(self):
        s = get_scenario("double_disturbance")
        with prior_memory(s.name, s.make_env) as m:
            r = run_agent("planner_dist_memory", s.name, s.make_env, m)
        self.assertEqual(r.stats["remembered_disturbances"], {"2": "heat_spike", "3": "battery_drain"})
        self.assertTrue(r.success)

    def test_wrong_memory_triggers_replan_and_recovers(self):
        # Memory says a crash comes at step 1, but the real episode has none.
        s = get_scenario("service_down")
        with prior_memory("x", lambda: SimulatedEnvironment(SystemState(70, 30, False), {1: "service_crash"})) as m:
            m._conn.execute("UPDATE episodes SET scenario = ?, summary = ?", (s.name, f"scenario {s.name}"))
            r = PlanningAgent(GOAL, advisor=MemoryAdvisor(m, s.name), predict_disturbances=True).run(
                s.make_env(), s.name)
        self.assertTrue(r.success)

    def test_policy_enforced(self):
        s = get_scenario("service_down")
        with prior_memory(s.name, s.make_env) as m:
            r = PlanningAgent(GOAL, policy=POLICIES["no_restart"], advisor=MemoryAdvisor(m, s.name),
                              predict_disturbances=True).run(s.make_env(), s.name)
        self.assertEqual(r.action_count, 0)

    def test_requires_capable_advisor(self):
        with self.assertRaises(ValueError):
            PlanningAgent(GOAL, predict_disturbances=True)


class TestStudy(unittest.TestCase):
    def test_run_case(self):
        s = get_scenario("double_disturbance")
        case = run_case(s.name, s.make_env)
        self.assertEqual(set(case), set(AGENTS))
        self.assertTrue(all(v["success"] for v in case.values()))

    def test_unknown_agent(self):
        s = get_scenario("nominal")
        with prior_memory(s.name, s.make_env) as m, self.assertRaises(ValueError):
            run_agent("oracle", s.name, s.make_env, m)

    def test_format_report(self):
        cell = {"episodes": 1, "successes": 1, "total_actions": 2, "replans": 0}
        summary = {a: dict(cell) for a in AGENTS}
        report = {"milestone": "M3.5", "version": "x", "agents": list(AGENTS),
                  "grid": {"cases": 1, "summary": summary, "by_case_type": {"no_fault+disturbance": summary},
                           "head_to_head": {"a_vs_b": {"a_fewer": 0, "b_fewer": 0, "equal": 1}}},
                  "note": "n"}
        text = format_report(report)
        self.assertIn("planner_reversed_dist_memory", text)
        self.assertIn(" D", text)


if __name__ == "__main__":
    unittest.main()
