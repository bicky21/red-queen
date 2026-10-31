import os
import unittest

from red_queen.goal import RecoveryGoal
from red_queen.safe_agent import SafeAgent
from red_queen.scenarios import ALL_SCENARIOS, get_scenario
from red_queen.self_model import (ALERT_LEVELS, PLAN_STATUSES, WIDTH_BUCKETS, SelfModel, brier, ece, features,
                                  heuristic, separation)
from red_queen.self_model_study import TEST_SEEDS, TRAIN_SEEDS, collect, decide, format_report
from red_queen.sensors import NoisySensor

GOAL = RecoveryGoal()


def snap(**kw):
    base = {"stopping": False, "plan_len": 2, "steps_left": 10, "alert_counts": {}, "width": 10}
    base.update(kw)
    return base


class TestFeatures(unittest.TestCase):
    def test_plan_status(self):
        self.assertEqual(features(snap(stopping=True))[0], "stopping")
        self.assertEqual(features(snap(plan_len=None))[0], "none")
        self.assertEqual(features(snap(plan_len=5, steps_left=4))[0], "too_long")
        self.assertEqual(features(snap(plan_len=3, steps_left=5))[0], "tight")
        self.assertEqual(features(snap(plan_len=3, steps_left=6))[0], "fits")

    def test_alerts_most_severe_wins(self):
        self.assertEqual(features(snap(alert_counts={"sensors_disagree": 2}))[1], "disagree")
        self.assertEqual(features(snap(alert_counts={"sensors_disagree": 1, "bound_widened": 1}))[1], "widened")
        self.assertEqual(features(snap(alert_counts={"bound_widened": 1, "sensor_isolated": 1}))[1], "isolated")

    def test_width(self):
        self.assertEqual([features(snap(width=w))[2] for w in (20, 21, 40, 41)], ["<=20", "21-40", "21-40", ">40"])

    def test_feature_values_known(self):
        self.assertEqual(len(PLAN_STATUSES) * len(ALERT_LEVELS) * len(WIDTH_BUCKETS), 60)


class TestModelAndMetrics(unittest.TestCase):
    def test_fit_predict(self):
        k = ("fits", "none", "<=20")
        m = SelfModel().fit([(k, True), (k, True), (("none", "none", ">40"), False)])
        self.assertAlmostEqual(m.base_rate, 2 / 3)
        self.assertAlmostEqual(m.predict(k), (2 + 2 * (2 / 3)) / 4)
        self.assertEqual(m.predict(("too_long", "isolated", ">40")), m.base_rate)

    def test_errors(self):
        with self.assertRaises(ValueError):
            SelfModel().predict(("fits", "none", "<=20"))
        with self.assertRaises(ValueError):
            SelfModel().fit([])

    def test_metrics(self):
        self.assertEqual(brier([0.5, 0.5], [True, False]), 0.25)
        self.assertAlmostEqual(ece([0.9, 0.9], [True, False]), 0.4)
        self.assertEqual(separation([0.7], [True]), 0.0)
        self.assertEqual(heuristic(("fits", "none", "<=20")), 0.95)
        self.assertEqual(heuristic(("none", "none", "<=20")), 0.05)


class TestObserver(unittest.TestCase):
    def test_observer_called_and_ends_with_stop(self):
        snaps = []
        r = SafeAgent(GOAL, 5).run(get_scenario("low_battery").make_env(),
                                   [NoisySensor(5, 0), NoisySensor(5, 1000)], observer=snaps.append)
        self.assertTrue(r.success)
        self.assertEqual(len(snaps), len(r.actions) + 1)
        self.assertTrue(snaps[-1]["stopping"])
        self.assertEqual(r.stats["observer_errors"], 0)

    def test_no_observer_no_stat(self):
        r = SafeAgent(GOAL, 5).run(get_scenario("low_battery").make_env(), [NoisySensor(5, 0), NoisySensor(5, 1000)])
        self.assertNotIn("observer_errors", r.stats)


class TestStudy(unittest.TestCase):
    def test_collect_small(self):
        recs = collect((0,), conditions=("none", "A_dead_low"), scenarios=ALL_SCENARIOS[:3])
        self.assertTrue(recs)
        self.assertTrue(all(len(r["key"]) == 3 for r in recs))

    def test_seeds_and_decision(self):
        self.assertFalse(set(TRAIN_SEEDS) & set(TEST_SEEDS))
        scores = [{"predictor": "base_rate", "brier": 0.2, "ece": 0.0, "separation": 0},
                  {"predictor": "heuristic", "brier": 0.15, "ece": 0.1, "separation": 0.2},
                  {"predictor": "self_model", "brier": 0.1, "ece": 0.03, "separation": 0.5}]
        self.assertTrue(decide(scores)["indicator_present_measured_sense"])
        scores[2]["ece"] = 0.06
        self.assertFalse(decide(scores)["indicator_present_measured_sense"])

    def test_prereg_present(self):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "PREREG_M7.md")
        with open(path, encoding="utf-8") as f:
            self.assertIn("metacognitive monitoring", f.read())


if __name__ == "__main__":
    unittest.main()
