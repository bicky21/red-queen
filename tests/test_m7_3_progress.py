import os
import unittest

from red_queen.cross_isolation_study import sensors_for as m54_sensors
from red_queen.goal import RecoveryGoal
from red_queen.progress_study import FRESH_SEEDS, decide, scores, v2_keys, with_v2
from red_queen.safe_agent import SafeAgent
from red_queen.scenarios import ALL_SCENARIOS, get_scenario
from red_queen.self_model import (PROGRESS_LABELS, ConfidenceMonitor, SelfModel, features, features_v2,
                                  progress_labels)
from red_queen.self_model_study import TEST_SEEDS, TRAIN_SEEDS
from red_queen.sensors import NoisySensor
from red_queen.warning_study import collect_episodes

GOAL = RecoveryGoal()


class TestProgress(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(progress_labels([3, 2, 1], [False, False, False]), ["early", "early", "progressing"])
        self.assertEqual(progress_labels([3, 3, 3], [False] * 3), ["early", "early", "stalled"])
        self.assertEqual(progress_labels([None, None, None], [False] * 3)[-1], "stalled")
        self.assertEqual(progress_labels([2, 1, 0], [False, False, True])[-1], "progressing")
        self.assertEqual(set(PROGRESS_LABELS), {"early", "progressing", "stalled"})

    def test_features_v2(self):
        self.assertEqual(features_v2(("fits", "none", "<=20"), "early"), ("fits", "none", "<=20", "early"))
        with self.assertRaises(ValueError):
            features_v2(("fits", "none", "<=20"), "fast")


class TestMonitor(unittest.TestCase):
    def test_default_monitor_uses_v1_keys(self):
        k = ("fits", "none", "<=20")
        m = SelfModel().fit([(k, True)] * 4)
        mon = ConfidenceMonitor(m, 0.5)
        self.assertFalse(mon.use_progress)

    def test_progress_monitor_runs_live(self):
        eps = collect_episodes((0,), conditions=("none", "A_dead_low"), scenarios=ALL_SCENARIOS[:6])
        m = SelfModel().fit((k, e["success"]) for e in eps for k in v2_keys(e))
        mon = ConfidenceMonitor(m, 0.3, use_progress=True)
        r = SafeAgent(GOAL, 5).run(get_scenario("low_battery").make_env(), [NoisySensor(5, 0), NoisySensor(5, 1000)],
                                   observer=mon)
        self.assertEqual(r.stats["observer_errors"], 0)
        self.assertEqual(len(mon.confidences), len(r.actions) + 1)


class TestStudyParts(unittest.TestCase):
    def test_seeds_fresh(self):
        self.assertFalse(set(FRESH_SEEDS) & (set(TRAIN_SEEDS) | set(TEST_SEEDS)))

    def test_with_v2_and_scores(self):
        eps = collect_episodes((20,), conditions=("B_stuck_low",), scenarios=ALL_SCENARIOS[:3], sensors_fn=m54_sensors)
        v2 = with_v2(eps)
        self.assertTrue(all(len(k) == 4 for e in v2 for k in e["keys"]))
        self.assertTrue(all(len(k) == 3 for e in eps for k in e["keys"]))  # originals untouched
        m = SelfModel().fit((k, e["success"]) for e in v2 for k in e["keys"])
        self.assertIn("ece", scores(m, v2))

    def test_decide(self):
        rep = {"new_faults": {"v1": {"ece": 0.3}, "v2": {"ece": 0.09}, "v2_warning": {"recall": 0.95, "precision": 0.95}},
               "familiar": {"v1": {"brier": 0.1}, "v2": {"ece": 0.02, "brier": 0.09}}}
        self.assertTrue(decide(rep)["adopt_v2"])
        rep["new_faults"]["v2"]["ece"] = 0.2
        self.assertFalse(decide(rep)["adopt_v2"])

    def test_prereg_present(self):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "PREREG_M7_3.md")
        with open(path, encoding="utf-8") as f:
            self.assertIn("am i making progress", f.read().lower())


if __name__ == "__main__":
    unittest.main()
