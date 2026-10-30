import unittest

from red_queen.agent import STOP_GOAL
from red_queen.belief_agent import BeliefAgent, explanation_boxes, needed_noise, unexplained_gap
from red_queen.goal import RecoveryGoal
from red_queen.scenarios import ALL_SCENARIOS, get_scenario
from red_queen.sensor_adapt_study import format_report, run_row, run_study
from red_queen.sensors import MAX_NOISE, Box, NoisySensor, observation_box
from red_queen.state import SystemState

GOAL = RecoveryGoal()


class TestBias(unittest.TestCase):
    def test_bias_applied_and_clamped(self):
        self.assertEqual(NoisySensor(0, 0, 10, -10).read(SystemState(95, 5, True), 0), SystemState(100, 0, True))

    def test_bias_validated(self):
        for bad in (26, True, "1"):
            with self.assertRaises(ValueError):
                NoisySensor(5, 0, 0, bad)

    def test_zero_bias_matches_m5(self):
        truth = SystemState(40, 60, False)
        for k in range(30):
            self.assertEqual(NoisySensor(15, 2).read(truth, k), NoisySensor(15, 2, 0, 0).read(truth, k))


class TestExplanations(unittest.TestCase):
    def test_explanation_set(self):
        boxes = explanation_boxes(Box(50, 50, 50, 50, True), "charge")
        self.assertEqual(len(boxes), 6)  # (worked, blocked) x (none, battery_drain, heat_spike)
        self.assertIn(Box(70, 70, 55, 55, True), boxes)   # worked
        self.assertIn(Box(50, 50, 50, 50, True), boxes)   # blocked
        self.assertIn(Box(30, 30, 55, 55, True), boxes)   # worked + battery_drain
        self.assertIn(Box(50, 50, 95, 95, True), boxes)   # blocked + heat_spike

    def test_consistent_readings_explained(self):
        prev, new = observation_box(SystemState(50, 50, True), 5), observation_box(SystemState(72, 52, True), 5)
        self.assertEqual(unexplained_gap(prev, new, "charge"), 0)

    def test_inconsistent_readings_gap(self):
        prev, new = observation_box(SystemState(50, 50, True), 0), observation_box(SystemState(50, 60, True), 0)
        self.assertEqual(unexplained_gap(prev, new, "wait"), 10)

    def test_needed_noise(self):
        self.assertEqual(needed_noise(SystemState(50, 50, True), SystemState(50, 60, True), "wait", 0), 5)
        self.assertEqual(needed_noise(SystemState(50, 50, True), SystemState(50, 60, True), "wait", 7), 7)
        self.assertEqual(needed_noise(SystemState(0, 0, True), SystemState(100, 100, True), "wait", 0), MAX_NOISE)


class TestAdaptiveAgent(unittest.TestCase):
    def test_off_by_default(self):
        r = BeliefAgent(GOAL, 5).run(get_scenario("low_battery").make_env(), NoisySensor(5, 0))
        self.assertNotIn("adapt_noise", r.stats)

    def test_correct_bound_no_widening(self):
        for s in ALL_SCENARIOS:
            r = BeliefAgent(GOAL, 10, adapt_noise=True).run(s.make_env(), NoisySensor(10, 3))
            self.assertEqual(r.stats["widenings"], [], s.name)

    def test_widens_under_wrong_bound(self):
        s = get_scenario("compound_failure")
        r = BeliefAgent(GOAL, 0, adapt_noise=True).run(s.make_env(), NoisySensor(15, 1))
        self.assertGreater(r.stats["noise_final"], 0)
        self.assertTrue(r.success)
        steps = [w["to"] for w in r.stats["widenings"]]
        self.assertEqual(steps, sorted(set(steps)))

    def test_bias_can_fool_it(self):
        fooled = 0
        for s in ALL_SCENARIOS:
            for seed in range(5):
                r = BeliefAgent(GOAL, 5, adapt_noise=True).run(s.make_env(), NoisySensor(5, seed, 0, -10))
                fooled += r.stop_reason == STOP_GOAL and not r.success
        self.assertGreater(fooled, 0)


class TestStudy(unittest.TestCase):
    def test_row_and_report(self):
        row = run_row("interval", True, 15, 0, 2, ALL_SCENARIOS[:3])
        self.assertEqual(row["runs"], 6)
        report = run_study(seeds=1, scenarios=ALL_SCENARIOS[:2])
        self.assertEqual(len(report["wrong_bound"]), 16)
        self.assertEqual(len(report["correct_bound"]), 12)
        self.assertEqual(len(report["bias"]), 8)
        self.assertIn("A. Wrong bound", format_report(report))
        with self.assertRaises(ValueError):
            run_study(seeds=0)


if __name__ == "__main__":
    unittest.main()
