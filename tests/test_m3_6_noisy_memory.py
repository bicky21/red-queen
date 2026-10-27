import unittest

from red_queen.environment import DISTURBANCES, SimulatedEnvironment
from red_queen.goal import RecoveryGoal
from red_queen.memory_advice import MemoryAdvisor
from red_queen.memory_noise import FAULTABLE, NOISE_LEVELS, distort
from red_queen.memory_planning_study import prior_memory
from red_queen.noisy_memory_study import (
    AGENTS,
    BASELINE_FOR,
    LEVELS,
    format_report,
    memory_for,
    run_agent,
    run_level,
    run_noisy_memory_study,
)
from red_queen.planning_agent import GUARDS, PlanningAgent
from red_queen.scenario_study import grid_cases
from red_queen.scenarios import get_scenario
from red_queen.state import SystemState

GOAL = RecoveryGoal()
FAULTS = (("restart_service", 1, 3),)
DIST = ((2, "heat_spike"),)


class TestDistort(unittest.TestCase):
    def test_level_zero_is_identity(self):
        self.assertEqual(distort(FAULTS, DIST, 0.0, "a"), (FAULTS, DIST))

    def test_deterministic_and_seed_dependent(self):
        self.assertEqual(distort(FAULTS, DIST, 1.0, "a"), distort(FAULTS, DIST, 1.0, "a"))
        outputs = {distort(FAULTS, DIST, 1.0, str(i)) for i in range(30)}
        self.assertGreater(len(outputs), 1)

    def test_outputs_always_valid(self):
        for i in range(200):
            for level in NOISE_LEVELS:
                faults, dist = distort(FAULTS, DIST, level, str(i))
                for action, first, last in faults:
                    self.assertIn(action, FAULTABLE)
                    self.assertTrue(1 <= first <= last)
                for step, name in dist:
                    self.assertGreaterEqual(step, 1)
                    self.assertIn(name, DISTURBANCES)
                SimulatedEnvironment(SystemState(10, 30, False), dict(dist), faults)  # accepted by env

    def test_invalid_level(self):
        for bad in (-0.1, 1.1):
            with self.subTest(level=bad), self.assertRaises(ValueError):
                distort(FAULTS, DIST, bad, "a")


class TestGuard(unittest.TestCase):
    def test_guard_values(self):
        self.assertEqual(GUARDS, (None, "on_contradiction", "on_any_mismatch"))

    def test_guard_requires_prediction(self):
        s = get_scenario("nominal")
        with prior_memory(s.name, s.make_env) as m, self.assertRaises(ValueError):
            PlanningAgent(GOAL, advisor=MemoryAdvisor(m, s.name), disturbance_guard="on_any_mismatch")

    def test_contradiction_guard_trips_when_remembered_disturbance_missing(self):
        s = get_scenario("service_down")
        remembered_env = lambda: SimulatedEnvironment(SystemState(70, 30, False), {1: "service_crash"})  # noqa: E731
        with prior_memory(s.name, remembered_env) as m:
            r = PlanningAgent(GOAL, advisor=MemoryAdvisor(m, s.name), predict_disturbances=True,
                              disturbance_guard="on_contradiction").run(s.make_env(), s.name)
        self.assertTrue(r.success)
        self.assertEqual(r.stats["remembered_disturbances"], {"1": "service_crash"})

    def test_guard_does_not_trip_with_exact_memory(self):
        s = get_scenario("double_disturbance")
        with prior_memory(s.name, s.make_env) as m:
            r = PlanningAgent(GOAL, advisor=MemoryAdvisor(m, s.name), predict_disturbances=True,
                              disturbance_guard="on_any_mismatch").run(s.make_env(), s.name)
        self.assertIsNone(r.stats["guard_tripped_at"])
        self.assertTrue(r.success)

    def test_tripped_guard_stops_prediction(self):
        s = get_scenario("disturbance_mid_recovery")
        wrong = lambda: SimulatedEnvironment(SystemState(30, 40, True), {2: "battery_drain"})  # noqa: E731
        with prior_memory(s.name, wrong) as m:
            r = PlanningAgent(GOAL, advisor=MemoryAdvisor(m, s.name), predict_disturbances=True,
                              disturbance_guard="on_any_mismatch").run(s.make_env(), s.name)
        self.assertEqual(r.stats["guard_tripped_at"], 1)
        self.assertTrue(r.success)


class TestStudy(unittest.TestCase):
    def test_levels_and_baselines(self):
        self.assertEqual(LEVELS[-1], "unrelated")
        for agent, base in BASELINE_FOR.items():
            self.assertIn(agent, AGENTS)
            self.assertIn(base, AGENTS)

    def test_memory_for_uses_case_name(self):
        cases = list(grid_cases())
        for level in ("unrelated", 0.5):
            with memory_for(level, 5, cases) as m:
                self.assertEqual(MemoryAdvisor(m, "grid_0005").summary()["records_used"], 1)

    def test_unknown_agent(self):
        with self.assertRaises(ValueError):
            run_agent("oracle", "x", get_scenario("nominal").make_env, None)

    def test_small_run_all_succeed(self):
        level = run_level(1.0, stride=120)
        self.assertEqual(level["cases"], 14)
        self.assertTrue(all(v == level["cases"] for v in level["successes"].values()))

    def test_report_and_stride_validation(self):
        with self.assertRaises(ValueError):
            run_noisy_memory_study(stride=0)
        report = run_noisy_memory_study(stride=400)
        text = format_report(report)
        self.assertIn("planner_dist_guard", text)
        self.assertIn("Failures: none", text)


if __name__ == "__main__":
    unittest.main()
