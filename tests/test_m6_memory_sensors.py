import os
import unittest

from red_queen.goal import RecoveryGoal
from red_queen.memory_sensor_study import AGENTS, LEVELS, decide, format_report, run_study
from red_queen.safe_agent import MEMORY_GUARDS, SafeAgent
from red_queen.scenarios import ALL_SCENARIOS, get_scenario
from red_queen.sensors import NoisySensor

GOAL = RecoveryGoal()


class FixedMemory:
    def __init__(self, schedule):
        self.schedule = schedule

    def remembered_disturbances(self):
        return dict(self.schedule)


def sensors(seed=0, noise=5):
    return [NoisySensor(noise, seed), NoisySensor(noise, seed + 1000)]


class TestConfig(unittest.TestCase):
    def test_invalid(self):
        with self.assertRaises(ValueError):
            SafeAgent(GOAL, 5, predict_disturbances=True)
        with self.assertRaises(ValueError):
            SafeAgent(GOAL, 5, advisor=FixedMemory({}), disturbance_guard="on_mismatch")
        with self.assertRaises(ValueError):
            SafeAgent(GOAL, 5, advisor=FixedMemory({}), predict_disturbances=True, disturbance_guard="always")
        self.assertEqual(MEMORY_GUARDS, (None, "on_mismatch"))

    def test_off_by_default(self):
        r = SafeAgent(GOAL, 5).run(get_scenario("low_battery").make_env(), sensors())
        self.assertNotIn("remembered_disturbances", r.stats)


class TestMemoryBehaviour(unittest.TestCase):
    def test_exact_memory_recovers_all(self):
        for s in ALL_SCENARIOS:
            mem = FixedMemory(dict(s.disturbances))
            r = SafeAgent(GOAL, 5, advisor=mem, predict_disturbances=True,
                          disturbance_guard="on_mismatch").run(s.make_env(), sensors(2))
            self.assertTrue(r.success, s.name)

    def test_poisoned_memory_never_false_stops(self):
        poison = FixedMemory({k: "battery_drain" for k in range(1, 15)})
        for s in ALL_SCENARIOS:
            r = SafeAgent(GOAL, 5, advisor=poison, predict_disturbances=True).run(s.make_env(), sensors(1))
            self.assertFalse(r.stop_reason == "goal_believed_satisfied" and not r.success, s.name)

    def test_guard_trips_on_wrong_memory(self):
        s = get_scenario("low_battery")  # no disturbances in reality
        r = SafeAgent(GOAL, 5, advisor=FixedMemory({1: "heat_spike"}), predict_disturbances=True,
                      disturbance_guard="on_mismatch").run(s.make_env(), sensors(0))
        self.assertEqual(r.stats["guard_tripped_at"], 1)
        self.assertTrue(r.success)

    def test_guard_quiet_on_exact_memory_without_faults(self):
        s = get_scenario("disturbance_mid_recovery")
        r = SafeAgent(GOAL, 5, advisor=FixedMemory(dict(s.disturbances)), predict_disturbances=True,
                      disturbance_guard="on_mismatch").run(s.make_env(), sensors(0))
        self.assertIsNone(r.stats["guard_tripped_at"])


class TestStudy(unittest.TestCase):
    def test_small_study(self):
        report = run_study(stride=168, seeds=(0,))
        self.assertEqual(len(report["summary"]), len(LEVELS) * 2 * len(AGENTS))
        self.assertIsNone(report["decision"])  # decision only for the pre-registered configuration
        self.assertIn("regret", format_report(report))
        d = decide(report["summary"])
        self.assertIn("recommend_memory", d)
        with self.assertRaises(ValueError):
            run_study(stride=0)

    def test_prereg_mentions_agents(self):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "PREREG_M6.md")
        text = open(path, encoding="utf-8").read()
        for a in AGENTS:
            self.assertIn(f"`{a}`", text)


if __name__ == "__main__":
    unittest.main()
