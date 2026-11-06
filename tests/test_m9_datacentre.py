import os
import unittest

from red_queen.datacentre import (ACTION_KINDS, DCEnvironment, DCExecutor, DCState, DCVerifier, all_actions,
                                  bfs_plan, goal_distance, goal_met, parse_action, precondition, reference_plan, step)
from red_queen.datacentre_study import (AGENTS, DEV_SEEDS, EVAL_SEEDS, decide, format_report, greedy_action,
                                        make_scenario, run_episode, run_study)

GOOD = DCState((50, 50, 50), (40, 40, 40), (True, True, True), 120)


class TestState(unittest.TestCase):
    def test_goal(self):
        self.assertTrue(goal_met(GOOD))
        self.assertFalse(goal_met(DCState((50, 50, 71), (40, 40, 40), (True, True, True), 120)))
        self.assertFalse(goal_met(DCState((50, 50, 50), (60, 40, 20), (True, True, True), 120)))
        self.assertFalse(goal_met(DCState((50, 50, 50), (40, 40, 20), (True, True, True), 120)))
        self.assertEqual(goal_distance(GOOD), 0)

    def test_invalid_states(self):
        with self.assertRaises(ValueError):
            DCState((50, 50, 50), (40, 40, 40), (True, True, False), 120)
        with self.assertRaises(ValueError):
            DCState((50, True, 50), (40, 40, 40), (True, True, True), 120)

    def test_actions_list(self):
        acts = all_actions(3)
        self.assertEqual(len(acts), 3 + 6 + 3 + 3 + 3 + 1)
        self.assertEqual({a.split(":")[0] for a in acts}, set(ACTION_KINDS))


class TestActions(unittest.TestCase):
    def test_parse_rejects(self):
        for bad in ("cool:9", "migrate:0:0", "unknown", "cool:0:1", "", "cool: 0"):
            with self.assertRaises(ValueError):
                parse_action(bad, 3)

    def test_preconditions(self):
        s = DCState((50, 50, 80), (40, 40, 0), (True, True, False), 120)
        self.assertEqual(precondition(s, "restart:2"), "too_hot_to_restart")
        self.assertEqual(precondition(s, "cool:2"), "server_down")
        self.assertEqual(precondition(s, "migrate:0:2"), "target_down")
        self.assertIsNone(precondition(s, "unthrottle:0"))
        self.assertEqual(precondition(GOOD, "unthrottle:0"), "demand_already_served")
        with self.assertRaises(ValueError):
            step(s, "restart:2")

    def test_migrate_and_throttle(self):
        m = step(GOOD, "migrate:0:1")
        self.assertEqual(m.loads, (20, 60, 40))
        self.assertEqual(m.temps, (49, 51, 50))
        t = step(GOOD, "throttle:2")
        self.assertEqual(t.served, 100)


class TestExecutorAndPlanners(unittest.TestCase):
    def test_executor_budget_and_log(self):
        env = DCEnvironment(GOOD)
        ex = DCExecutor(env, max_actions=1)
        self.assertTrue(ex.execute("wait"))
        self.assertFalse(ex.execute("wait"))
        self.assertEqual(len(env.log), 1)
        with self.assertRaises(ValueError):
            DCExecutor(env, max_actions=-1)

    def test_bfs_and_reference(self):
        start = DCState((80, 50, 50), (40, 40, 40), (True, True, True), 120)
        b, r = bfs_plan(start), reference_plan(start)
        self.assertTrue(b.found and r.found)
        for plan in (b, r):
            s = start
            for a in plan.actions:
                s = step(s, a)
            self.assertTrue(goal_met(s))
        self.assertLessEqual(len(b.actions), len(r.actions))
        self.assertTrue(bfs_plan(GOOD).found and bfs_plan(GOOD).actions == ())
        with self.assertRaises(ValueError):
            bfs_plan(start, max_nodes=0)

    def test_node_limit(self):
        p = bfs_plan(make_scenario(1003), max_depth=14, max_nodes=10)
        self.assertTrue(p.found or p.hit_node_limit)


class TestStudy(unittest.TestCase):
    def test_scenarios_deterministic_and_not_solved(self):
        self.assertEqual(make_scenario(5), make_scenario(5))
        self.assertFalse(goal_met(make_scenario(5)))
        self.assertFalse(set(DEV_SEEDS) & set(EVAL_SEEDS))

    def test_episode_and_verifier(self):
        r = run_episode("reference", make_scenario(1000))
        self.assertTrue(r["success"])
        self.assertEqual(r["unchecked_actions"], 0)
        self.assertIsNotNone(greedy_action(make_scenario(1000)))
        with self.assertRaises(ValueError):
            run_episode("oracle", GOOD)

    def test_small_study(self):
        rep = run_study(seeds=(1000, 1001))
        self.assertIsNone(rep["decision"])
        self.assertEqual(set(rep["summary"]), set(AGENTS))
        self.assertIn("Pre-registered", format_report(dict(rep, decision={"x": True})))

    def test_decide(self):
        summ = {a: {"success": s, "unchecked_actions": 0, "max_actions": 25}
                for a, s in (("bfs", 80), ("reference", 100), ("greedy", 95))}
        self.assertTrue(decide({"summary": summ})["world_has_room"])

    def test_prereg_present(self):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "PREREG_M9.md")
        with open(path, encoding="utf-8") as f:
            self.assertIn("data centre", f.read())


if __name__ == "__main__":
    unittest.main()
