import unittest

from red_queen.agent import STOP_GOAL
from red_queen.belief_agent import BeliefAgent
from red_queen.goal import RecoveryGoal
from red_queen.redundant_agent import RedundantBeliefAgent
from red_queen.safe_agent import ALERT_KINDS, STOP_SAFE_HALT, SafeAgent, SafeEstimator
from red_queen.safety_layer_study import AGENTS, CONDITIONS, format_report, run_one, run_study, sensors_for
from red_queen.scenarios import ALL_SCENARIOS, get_scenario
from red_queen.sensors import NoisySensor, StuckSensor
from red_queen.state import SystemState

GOAL = RecoveryGoal()


class TestStuckSensor(unittest.TestCase):
    def test_stuck_after_step(self):
        s = StuckSensor(NoisySensor(0), 2, battery=100, temperature=0)
        truth = SystemState(20, 60, False)
        self.assertEqual(s.read(truth, 1), truth)
        self.assertEqual(s.read(truth, 2), SystemState(100, 0, False))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            StuckSensor(NoisySensor(0), -1)


class TestEstimator(unittest.TestCase):
    def test_invalid(self):
        for args in ((5, 3), (5, 2, True, "panic"), (40,)):
            with self.assertRaises(ValueError):
                SafeEstimator(*args)

    def test_hull_of_two(self):
        est = SafeEstimator(5, 2)
        e = est.update([SystemState(50, 50, True), SystemState(60, 40, True)], 0, None)
        self.assertEqual((e.belief.b_lo, e.belief.b_hi, e.belief.t_lo, e.belief.t_hi), (45, 65, 35, 55))

    def test_disagreement_alert(self):
        est = SafeEstimator(5, 2)
        est.update([SystemState(50, 50, True), SystemState(50, 80, True)], 0, None)
        self.assertEqual(est.report()["alert_counts"]["sensors_disagree"], 1)

    def test_unexplainable_single_halts(self):
        est = SafeEstimator(5, 1, True, "halt")
        est.update([SystemState(10, 50, True)], 0, None)
        est.action_taken([SystemState(10, 50, True)])
        e = est.update([SystemState(100, 50, True)], 1, "wait")
        self.assertTrue(e.halt)
        self.assertIsNone(e.belief)

    def test_unexplainable_isolated_with_two(self):
        est = SafeEstimator(5, 2, True, "isolate")
        first = [SystemState(10, 50, True), SystemState(10, 50, True)]
        est.update(first, 0, None)
        est.action_taken(first)
        e = est.update([SystemState(100, 50, True), SystemState(12, 50, True)], 1, "wait")
        self.assertFalse(e.halt)
        self.assertEqual(est.trusted, [False, True])
        self.assertEqual((e.belief.b_lo, e.belief.b_hi), (7, 17))

    def test_alert_kinds(self):
        self.assertEqual(set(ALERT_KINDS), {"bound_widened", "sensors_disagree", "reading_unexplainable",
                                            "sensor_isolated"})


class TestSafeAgent(unittest.TestCase):
    def test_reproduces_m5_and_m5_1(self):
        for s in ALL_SCENARIOS:
            a = BeliefAgent(GOAL, 5, "interval", adapt_noise=True).run(s.make_env(), NoisySensor(15, 2))
            b = SafeAgent(GOAL, 5, 1, True).run(s.make_env(), [NoisySensor(15, 2)])
            self.assertEqual(a.actions, b.actions, s.name)

    def test_reproduces_m5_2_hull(self):
        for s in ALL_SCENARIOS:
            sa, sb = NoisySensor(5, 1, 0, -10), NoisySensor(5, 1001)
            a = RedundantBeliefAgent(GOAL, 5, "hull").run(s.make_env(), sa, sb)
            b = SafeAgent(GOAL, 5, 2, False).run(s.make_env(), [sa, sb])
            self.assertEqual(a.actions, b.actions, s.name)

    def test_sensor_count_checked(self):
        with self.assertRaises(ValueError):
            SafeAgent(GOAL, 5, 2).run(get_scenario("low_battery").make_env(), [NoisySensor(5)])
        s = NoisySensor(5)
        with self.assertRaises(ValueError):
            SafeAgent(GOAL, 5, 2).run(get_scenario("low_battery").make_env(), [s, s])

    def test_dead_single_sensor_halts_not_false_stop(self):
        r = SafeAgent(GOAL, 5, 1, True, "halt").run(
            get_scenario("compound_failure").make_env(), [StuckSensor(NoisySensor(5, 0), 2)])
        self.assertEqual(r.stop_reason, STOP_SAFE_HALT)
        self.assertFalse(r.success)

    def test_common_mode_still_fools_it(self):
        fooled = sum(run_one("safe", s, "both_temp_-10", seed)["false_stop"]
                     for s in ALL_SCENARIOS for seed in range(5))
        self.assertGreater(fooled, 0)


class TestStudy(unittest.TestCase):
    def test_small_study(self):
        report = run_study(seeds=1, scenarios=ALL_SCENARIOS[:3], conditions=("none", "A_dead"))
        self.assertEqual(len(report["summary"]), 2 * len(AGENTS))
        self.assertIn("safe halt", format_report(report))
        self.assertEqual(len(CONDITIONS), 7)
        self.assertEqual(len(sensors_for("A_dead", 0)), 2)
        with self.assertRaises(ValueError):
            run_study(seeds=0)
        with self.assertRaises(ValueError):
            run_one("oracle", ALL_SCENARIOS[0], "none", 0)


if __name__ == "__main__":
    unittest.main()
