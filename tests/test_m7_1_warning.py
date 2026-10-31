import unittest

from red_queen.goal import RecoveryGoal
from red_queen.safe_agent import SafeAgent
from red_queen.scenarios import ALL_SCENARIOS, get_scenario
from red_queen.self_model import ConfidenceMonitor, SelfModel
from red_queen.sensors import NoisySensor, StuckSensor
from red_queen.warning_study import (THRESHOLDS, baseline_fires, choose_threshold, collect_episodes, decide,
                                     evaluate, first_alert)

GOAL = RecoveryGoal()
GOOD, BAD = ("fits", "none", "<=20"), ("none", "isolated", ">40")


def model():
    return SelfModel().fit([(GOOD, True)] * 9 + [(BAD, False)] * 9)


def ep(keys, success, false_stop=False, actions=None):
    return {"keys": keys, "steps": list(range(len(keys))), "actions": len(keys) if actions is None else actions,
            "success": success, "false_stop": false_stop, "condition": "x"}


class TestMonitor(unittest.TestCase):
    def test_invalid(self):
        with self.assertRaises(ValueError):
            ConfidenceMonitor(SelfModel(), 0.5)
        for bad in (0, 1, 1.5, "0.3"):
            with self.assertRaises(ValueError):
                ConfidenceMonitor(model(), bad)

    def test_one_alert_and_report(self):
        mon = ConfidenceMonitor(model(), 0.5)
        good = {"step": 0, "stopping": False, "plan_len": 1, "steps_left": 10, "alert_counts": {}, "width": 10}
        bad = {"step": 1, "stopping": False, "plan_len": None, "steps_left": 9,
               "alert_counts": {"sensor_isolated": 1}, "width": 60}
        mon(good)
        self.assertEqual(mon.alerts, [])
        mon(bad)
        mon(dict(bad, step=2))
        self.assertEqual(len(mon.alerts), 1)
        self.assertEqual(mon.alerts[0]["step"], 1)
        self.assertEqual(mon.report()["threshold"], 0.5)

    def test_attached_to_agent_result(self):
        mon = ConfidenceMonitor(model(), 0.5)
        r = SafeAgent(GOAL, 5).run(get_scenario("low_battery").make_env(),
                                   [NoisySensor(5, 0), NoisySensor(5, 1000)], observer=mon)
        self.assertIn("observer_report", r.stats)

    def test_warns_on_stuck_low_sensor(self):
        train = collect_episodes(range(5), conditions=("none", "A_dead_low"), scenarios=ALL_SCENARIOS)
        real = SelfModel().fit((k, e["success"]) for e in train for k in e["keys"])
        mon = ConfidenceMonitor(real, 0.5)
        sensors = [StuckSensor(NoisySensor(5, 0), 2, battery=0, temperature=100), NoisySensor(5, 1000)]
        SafeAgent(GOAL, 5).run(get_scenario("compound_failure").make_env(), sensors, observer=mon)
        self.assertTrue(mon.alerts)


class TestEvaluation(unittest.TestCase):
    def test_first_alert(self):
        self.assertEqual(first_alert(ep([GOOD, BAD, BAD], False), lambda k: k == BAD), 1)
        self.assertIsNone(first_alert(ep([GOOD], True), lambda k: k == BAD))

    def test_counts_and_lead(self):
        eps = [ep([GOOD, BAD, BAD, BAD], False, actions=5), ep([GOOD], True), ep([GOOD], False, false_stop=True),
               ep([BAD], True)]
        r = evaluate(eps, lambda k: k == BAD)
        self.assertEqual((r["warned_failures"], r["false_alarms"], r["missed_failures"], r["quiet_successes"]),
                         (1, 1, 1, 1))
        self.assertEqual(r["median_lead_steps"], 4)
        self.assertEqual(r["recall_false_stop_failures"], 0.0)

    def test_baseline(self):
        self.assertTrue(baseline_fires(("too_long", "none", "<=20")))
        self.assertFalse(baseline_fires(("stopping", "none", "<=20")))

    def test_threshold_choice_ties_to_lower(self):
        m = model()
        eps = [ep([BAD], False), ep([GOOD], True)]
        choice = choose_threshold(m, eps)
        self.assertEqual(choice["chosen"], min(THRESHOLDS))

    def test_decide(self):
        m = {"recall": 0.9, "precision": 0.9, "median_lead_steps": 4, "f1": 0.9}
        self.assertTrue(decide(m, {"f1": 0.8})["warning_useful"])
        self.assertFalse(decide(dict(m, median_lead_steps=2), {"f1": 0.8})["warning_useful"])

    def test_collect_small(self):
        eps = collect_episodes((0,), conditions=("A_dead_low",), scenarios=ALL_SCENARIOS[:2])
        self.assertEqual(len(eps), 2)
        self.assertEqual(len(eps[0]["keys"]), len(eps[0]["steps"]))


if __name__ == "__main__":
    unittest.main()
