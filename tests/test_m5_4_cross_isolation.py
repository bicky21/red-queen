import os
import unittest

from red_queen.cross_isolation_study import (AGENTS, CONFIRMATION, DEVELOPMENT, decide, format_report, run_one,
                                             run_study, sensors_for)
from red_queen.goal import RecoveryGoal
from red_queen.safe_agent import SafeAgent, SafeEstimator
from red_queen.scenarios import ALL_SCENARIOS, get_scenario
from red_queen.sensors import DriftSensor, NoisySensor, StuckSensor
from red_queen.state import SystemState

GOAL = RecoveryGoal()


class TestDriftSensor(unittest.TestCase):
    def test_drift(self):
        d = DriftSensor(NoisySensor(0), 3, -2)
        self.assertEqual(d.read(SystemState(50, 50, True), 4), SystemState(62, 42, True))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            DriftSensor(NoisySensor(0), True, 0)


class TestCrossRule(unittest.TestCase):
    def _two_step(self, a1, b1):
        est = SafeEstimator(5, 2, True, "isolate", cross_isolate=True)
        first = [SystemState(50, 50, True), SystemState(50, 50, True)]
        est.update(first, 0, None)
        est.action_taken(first)
        return est, est.update([a1, b1], 1, "wait")

    def test_isolates_the_widened_sensor(self):
        est, e = self._two_step(SystemState(50, 50, True), SystemState(50, 66, True))
        self.assertEqual(est.trusted, [True, False])

    def test_no_isolation_when_boxes_overlap(self):
        est, e = self._two_step(SystemState(50, 50, True), SystemState(50, 58, True))
        self.assertEqual(est.trusted, [True, True])

    def test_no_isolation_when_both_widened(self):
        est, e = self._two_step(SystemState(66, 50, True), SystemState(34, 50, True))
        self.assertEqual(est.trusted, [True, True])

    def test_needs_two_sensors(self):
        with self.assertRaises(ValueError):
            SafeEstimator(5, 1, cross_isolate=True)

    def test_off_by_default(self):
        self.assertFalse(SafeAgent(GOAL, 5).cross_isolate)
        self.assertFalse(SafeEstimator(5).cross_isolate)


class TestAgentOnStuck(unittest.TestCase):
    def test_helps_with_pessimistic_stuck_sensor(self):
        better = 0
        for s in ALL_SCENARIOS:
            sensors = lambda: [NoisySensor(5, 3), StuckSensor(NoisySensor(5, 1003), 1, battery=10, temperature=90)]
            a = SafeAgent(GOAL, 5).run(s.make_env(), sensors())
            b = SafeAgent(GOAL, 5, cross_isolate=True).run(s.make_env(), sensors())
            better += b.success and not a.success
            self.assertFalse(b.stop_reason == "goal_believed_satisfied" and not b.success)
        self.assertGreater(better, 0)


class TestStudy(unittest.TestCase):
    def test_conditions_match_preregistration(self):
        self.assertEqual(set(CONFIRMATION), {"none", "wrong_bound", "A_drift", "A_stuck_mid", "B_stuck_low",
                                             "wrong_bound+B_stuck_low"})
        self.assertEqual(set(DEVELOPMENT), {"A_dead_low", "A_dead"})
        prereg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "PREREG_M5_4.md")
        with open(prereg, encoding="utf-8") as f:
            text = f.read()
        for name in list(CONFIRMATION) + list(DEVELOPMENT):
            self.assertIn(f"`{name}`", text)

    def test_small_study_and_decision(self):
        report = run_study(seeds=1, scenarios=ALL_SCENARIOS[:3])
        self.assertEqual(len(report["summary"]), 8 * len(AGENTS))
        self.assertIn("adopt_as_default", report["decision"])
        self.assertIn("Pre-registered decision", format_report(report))
        self.assertEqual(decide(report["summary"]), report["decision"])
        self.assertEqual(len(sensors_for("B_stuck_low", 0)), 2)
        with self.assertRaises(ValueError):
            run_one("oracle", ALL_SCENARIOS[0], "none", 0)


if __name__ == "__main__":
    unittest.main()
