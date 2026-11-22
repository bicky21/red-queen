import math
import os
import unittest

from red_queen.datacentre import DCState, goal_distance, goal_met, reference_plan, step
from red_queen.datacentre_study import greedy_action, make_scenario
from red_queen.heuristic_study import AGENTS, EVAL_SEEDS, TRAIN_SEEDS, ScaledHand, decide, run_episode
from red_queen.learned_heuristic import (FEATURE_NAMES, LearnedHeuristic, features, fit_ridge, labelled_states,
                                         pearson, train, walk_states)

GOAL = DCState((50, 50, 50), (40, 40, 40), (True, True, True), 120)


class TestFeatures(unittest.TestCase):
    def test_values(self):
        s = DCState((80, 50, 75), (60, 40, 0), (True, True, False), 120)
        f = features(s)
        self.assertEqual(len(f), len(FEATURE_NAMES))
        self.assertEqual(f, [1.0, 1.0, 3.0, 2.0, 1.0, 1.0, 8.0, 1.0])

    def test_all_down(self):
        s = DCState((70, 70, 70), (0, 0, 0), (False, False, False), 120)
        self.assertEqual(features(s)[6], 0.0)


class TestFitting(unittest.TestCase):
    def test_ridge_recovers_line(self):
        xs = [[1.0, float(i)] for i in range(10)]
        w = fit_ridge(xs, [2.0 + 3.0 * i for i in range(10)], ridge=0.0)
        self.assertAlmostEqual(w[0], 2.0)
        self.assertAlmostEqual(w[1], 3.0)

    def test_bias_not_penalised(self):
        xs = [[1.0, 0.0] for _ in range(5)]
        self.assertAlmostEqual(fit_ridge(xs, [4.0] * 5, ridge=10.0)[0], 4.0)

    def test_pearson(self):
        self.assertAlmostEqual(pearson([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertIsNone(pearson([1, 1, 1], [1, 2, 3]))


class TestHeuristic(unittest.TestCase):
    def test_clip_and_fallback(self):
        self.assertEqual(LearnedHeuristic([-5.0] + [0.0] * 7)(GOAL), 0.0)
        h = LearnedHeuristic([math.inf] + [0.0] * 7)
        self.assertEqual(h(GOAL), goal_distance(GOAL))
        self.assertEqual(h.fallbacks, 1)
        with self.assertRaises(ValueError):
            LearnedHeuristic([1.0])

    def test_plugin_defaults(self):
        s = make_scenario(1002)
        self.assertEqual(reference_plan(s).actions, reference_plan(s, heuristic=goal_distance).actions)
        self.assertEqual(greedy_action(s), greedy_action(s, heuristic=goal_distance))

    def test_learned_plans_are_valid(self):
        h, n = train(range(2000, 2010), make_scenario)
        self.assertGreater(n, 0)
        for seed in range(1000, 1004):
            start = make_scenario(seed)
            plan = reference_plan(start, heuristic=h)
            if plan.found:
                st = start
                for a in plan.actions:
                    st = step(st, a)
                self.assertTrue(goal_met(st))


class TestData(unittest.TestCase):
    def test_walk_deterministic(self):
        s = make_scenario(2000)
        self.assertEqual(walk_states(s, 2000), walk_states(s, 2000))
        self.assertLessEqual(len(walk_states(s, 2000)), 6)

    def test_labels(self):
        data = labelled_states((2000, 2001), make_scenario)
        self.assertTrue(all(isinstance(y, int) and y >= 0 for _, y in data))


class TestStudy(unittest.TestCase):
    def test_seeds(self):
        self.assertFalse(set(TRAIN_SEEDS) & set(EVAL_SEEDS))

    def test_episode(self):
        h, _ = train(range(2000, 2005), make_scenario)
        r = run_episode("astar_learned", make_scenario(3000), h)
        self.assertEqual(r["unchecked_actions"], 0)
        with self.assertRaises(ValueError):
            run_episode("oracle", GOAL, h)
        self.assertEqual(ScaledHand(2.0)(GOAL), 0.0)

    def test_decide(self):
        summ = {"astar_hand": {"success": 100, "nodes": 1000, "actions": 500},
                "astar_learned": {"success": 100, "nodes": 700, "actions": 510}}
        self.assertTrue(decide(summ)["learned_heuristic_useful"])
        summ["astar_learned"]["nodes"] = 900
        self.assertFalse(decide(summ)["learned_heuristic_useful"])
        self.assertEqual(len(AGENTS), 4)

    def test_prereg_present(self):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "PREREG_M9_2.md")
        with open(path, encoding="utf-8") as f:
            self.assertIn("ridge regression", f.read())


if __name__ == "__main__":
    unittest.main()
