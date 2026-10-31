import os
import unittest

from red_queen.cross_isolation_study import sensors_for as m54_sensors
from red_queen.generalisation_study import (M7_1_THRESHOLD, NEW_FAULTS, SEEN_TYPES, decide, format_report,
                                            snapshot_scores, summarise)
from red_queen.safety_layer_study import CONDITIONS as TRAIN_CONDITIONS
from red_queen.scenarios import ALL_SCENARIOS
from red_queen.self_model import SelfModel
from red_queen.warning_study import collect_episodes


def tiny_model():
    train = collect_episodes((0,), conditions=("none", "A_dead_low"), scenarios=ALL_SCENARIOS[:4])
    return SelfModel().fit((k, e["success"]) for e in train for k in e["keys"])


class TestDesign(unittest.TestCase):
    def test_new_faults_not_in_training(self):
        self.assertFalse(set(NEW_FAULTS) & set(TRAIN_CONDITIONS))
        self.assertTrue(set(SEEN_TYPES) <= set(TRAIN_CONDITIONS))

    def test_threshold_is_m7_1(self):
        self.assertEqual(M7_1_THRESHOLD, 0.3)

    def test_prereg_present(self):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "PREREG_M7_2.md")
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for name in NEW_FAULTS:
            self.assertIn(f"`{name}`", text)


class TestStudyParts(unittest.TestCase):
    def test_collect_with_m54_sensors(self):
        eps = collect_episodes((10,), conditions=("B_stuck_low",), scenarios=ALL_SCENARIOS[:3], sensors_fn=m54_sensors)
        self.assertEqual(len(eps), 3)
        self.assertEqual({e["condition"] for e in eps}, {"B_stuck_low"})

    def test_summarise_shapes(self):
        m = tiny_model()
        eps = collect_episodes((10,), conditions=("A_drift",), scenarios=ALL_SCENARIOS[:3], sensors_fn=m54_sensors)
        s = summarise(m, eps)
        self.assertEqual({r["predictor"] for r in s["snapshot_scores"]}, {"base_rate", "heuristic", "self_model"})
        self.assertIn("recall", s["warning"])
        self.assertEqual(len(snapshot_scores(m, eps)), 3)

    def test_decide(self):
        primary = {"snapshot_scores": [{"predictor": "base_rate", "brier": 0.4, "ece": 0.5},
                                       {"predictor": "self_model", "brier": 0.2, "ece": 0.08}],
                   "warning": {"recall": 0.9, "precision": 0.9}}
        self.assertTrue(decide(primary)["generalises"])
        primary["snapshot_scores"][1]["ece"] = 0.2
        self.assertFalse(decide(primary)["generalises"])

    def test_report_text(self):
        m = tiny_model()
        eps = collect_episodes((10,), conditions=tuple(SEEN_TYPES) + tuple(NEW_FAULTS), scenarios=ALL_SCENARIOS[:1],
                               sensors_fn=m54_sensors)
        block = summarise(m, eps)
        rep = {"train_base_rate": m.base_rate, "threshold": 0.3, "primary_new_faults": block, "seen_types": block,
               "per_condition": {"x": block}, "decision": {"generalises": False}}
        self.assertIn("NEW fault types", format_report(rep))


if __name__ == "__main__":
    unittest.main()
