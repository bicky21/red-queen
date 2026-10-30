import unittest

from red_queen.agent import STOP_GOAL
from red_queen.goal import RecoveryGoal
from red_queen.policy import ALL_POLICIES
from red_queen.redundancy_study import CONDITIONS, format_report, run_one, run_study, sensors_for
from red_queen.redundant_agent import STOP_SENSOR_FAULT, RedundantBeliefAgent, hull
from red_queen.scenarios import ALL_SCENARIOS, get_scenario
from red_queen.sensors import Box, NoisySensor
from red_queen.state import SystemState

GOAL = RecoveryGoal()


class TestHull(unittest.TestCase):
    def test_hull(self):
        self.assertEqual(hull(Box(10, 20, 30, 40, True), Box(15, 50, 0, 5, True)), Box(10, 50, 0, 40, True))

    def test_hull_service_mismatch(self):
        with self.assertRaises(ValueError):
            hull(Box(0, 1, 0, 1, True), Box(0, 1, 0, 1, False))


class TestRedundantAgent(unittest.TestCase):
    def test_invalid(self):
        with self.assertRaises(ValueError):
            RedundantBeliefAgent(GOAL, 5, "vote")
        s = NoisySensor(5)
        with self.assertRaises(ValueError):
            RedundantBeliefAgent(GOAL, 5).run(get_scenario("low_battery").make_env(), s, s)

    def test_no_fault_all_fusions_recover(self):
        for fusion in ("intersect", "hull", "latch"):
            for s in ALL_SCENARIOS:
                with self.subTest(fusion=fusion, scenario=s.name):
                    a, b = sensors_for("none", 3)
                    self.assertTrue(RedundantBeliefAgent(GOAL, 5, fusion).run(s.make_env(), a, b).success)

    def test_hull_survives_large_bias(self):
        for seed in range(10):
            a, b = sensors_for("A_temp_-20", seed)
            r = RedundantBeliefAgent(GOAL, 5, "hull").run(get_scenario("overheating").make_env(), a, b)
            self.assertTrue(r.success)
            self.assertTrue(r.stats["disagreement_steps"])

    def test_intersect_halts_on_disagreement(self):
        a, b = sensors_for("A_temp_-20", 0)
        r = RedundantBeliefAgent(GOAL, 5, "intersect").run(get_scenario("low_battery").make_env(), a, b)
        self.assertEqual(r.stop_reason, STOP_SENSOR_FAULT)
        self.assertEqual(r.actions, ())
        self.assertFalse(r.success)

    def test_latch_switches_to_hull(self):
        a, b = sensors_for("A_temp_-20", 0)
        r = RedundantBeliefAgent(GOAL, 5, "latch").run(get_scenario("overheating").make_env(), a, b)
        self.assertTrue(r.stats["latched"])
        self.assertTrue(r.success)

    def test_common_mode_bias_fools_hull(self):
        fooled = 0
        for s in ALL_SCENARIOS:
            for seed in range(5):
                a, b = sensors_for("both_temp_-10", seed)
                r = RedundantBeliefAgent(GOAL, 5, "hull").run(s.make_env(), a, b)
                fooled += r.stop_reason == STOP_GOAL and not r.success
                self.assertEqual(r.stats["disagreement_steps"], [])
        self.assertGreater(fooled, 0)

    def test_wait_only_policy(self):
        a, b = sensors_for("none", 0)
        r = RedundantBeliefAgent(GOAL, 5, policy=ALL_POLICIES["wait_only"]).run(
            get_scenario("low_battery").make_env(), a, b)
        self.assertEqual(r.actions, ())
        self.assertEqual(r.stop_reason, "no_plan_found")

    def test_readings_recorded_from_both(self):
        a, b = sensors_for("A_batt_+10", 1)
        r = RedundantBeliefAgent(GOAL, 5, "hull").run(get_scenario("low_battery").make_env(), a, b)
        self.assertIn("a", r.steps[0].observed)
        self.assertIn("b", r.steps[0].observed)


class TestStudy(unittest.TestCase):
    def test_sensors_independent(self):
        a, b = sensors_for("none", 4)
        truth = SystemState(50, 50, True)
        self.assertNotEqual([a.read(truth, k) for k in range(10)], [b.read(truth, k) for k in range(10)])

    def test_small_study(self):
        report = run_study(seeds=1, scenarios=ALL_SCENARIOS[:3], conditions=("none", "A_temp_-10"))
        self.assertEqual(len(report["summary"]), 8)
        self.assertIn("fault halt", format_report(report))
        self.assertEqual(len(CONDITIONS), 6)
        with self.assertRaises(ValueError):
            run_study(seeds=0)
        with self.assertRaises(ValueError):
            run_one("oracle", ALL_SCENARIOS[0], "none", 0)


if __name__ == "__main__":
    unittest.main()
