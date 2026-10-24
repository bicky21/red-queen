import json
import unittest

from red_queen.actions import UnknownActionError
from red_queen.affect import AffectConfig
from red_queen.environment import SimulatedEnvironment
from red_queen.experiments import run_scenario
from red_queen.scenarios import FAULT_SCENARIOS, M1_SCENARIO_NAMES, SCENARIOS, get_scenario
from red_queen.state import SystemState
from red_queen.sweeps import format_report, run_sweep, sweep_configs
from red_queen.validation import M1_FROZEN_SEQUENCES


class TestBlockedActions(unittest.TestCase):
    def test_blocked_action_has_no_effect_and_is_logged(self):
        env = SimulatedEnvironment(SystemState(30, 30, True), blocked=(("charge", 1, 1),))
        env.execute("charge")
        self.assertEqual(env.observe().battery, 30)
        self.assertEqual(env.step, 1)
        self.assertEqual(env.events[-1].kind, "blocked")

    def test_window_is_inclusive_and_ends(self):
        env = SimulatedEnvironment(SystemState(30, 30, True), blocked=(("charge", 2, 3),))
        results = [env.execute("charge").battery for _ in range(4)]
        self.assertEqual(results, [50, 50, 50, 70])

    def test_blocking_one_action_does_not_affect_others(self):
        env = SimulatedEnvironment(SystemState(30, 90, True), blocked=(("charge", 1, 5),))
        self.assertEqual(env.execute("cool").temperature, 70)

    def test_disturbance_still_applies_on_blocked_step(self):
        env = SimulatedEnvironment(SystemState(30, 30, True), {1: "service_crash"}, (("charge", 1, 1),))
        env.execute("charge")
        self.assertFalse(env.observe().service_up)
        self.assertEqual([e.kind for e in env.events], ["blocked", "disturbance"])

    def test_safety_cannot_block_unknown_action(self):
        with self.assertRaises(UnknownActionError):
            SimulatedEnvironment(SystemState(30, 30, True), blocked=(("shell", 1, 2),))

    def test_invalid_windows_rejected(self):
        for bad in (("charge", 0, 2), ("charge", 3, 2), ("charge", 1), ("charge", 1.0, 2)):
            with self.subTest(window=bad), self.assertRaises(ValueError):
                SimulatedEnvironment(SystemState(30, 30, True), blocked=(bad,))

    def test_rejected_action_still_rejected_with_faults(self):
        env = SimulatedEnvironment(SystemState(30, 30, True), blocked=(("charge", 1, 1),))
        with self.assertRaises(UnknownActionError):
            env.execute("rm")
        self.assertEqual(env.step, 0)


class TestFaultScenarios(unittest.TestCase):
    def test_m1_scenarios_listed_first_and_unchanged(self):
        self.assertEqual(tuple(s.name for s in SCENARIOS[: len(M1_SCENARIO_NAMES)]), M1_SCENARIO_NAMES)
        self.assertEqual(len(SCENARIOS), len(M1_SCENARIO_NAMES) + len(FAULT_SCENARIOS))

    def test_fault_scenarios_recover_in_both_modes(self):
        for s in FAULT_SCENARIOS:
            for enabled in (False, True):
                with self.subTest(scenario=s.name, affect=enabled):
                    self.assertTrue(run_scenario(s, enabled).success)

    def test_frustration_activates(self):
        r = run_scenario(get_scenario("flaky_restart"), True)
        self.assertGreater(max(s.affect["frustration"] for s in r.steps), 0)

    def test_baseline_repeats_blocked_restart(self):
        r = run_scenario(get_scenario("flaky_restart"), False)
        self.assertEqual(r.actions, ("restart_service",) * 4)

    def test_custom_affect_config(self):
        cfg = AffectConfig(enabled=True, concern_gain=0.0, frustration_penalty=2.0)
        r = run_scenario(get_scenario("flaky_restart_low_battery"), True, affect=cfg)
        self.assertTrue(r.success)
        self.assertIn("charge", r.actions[:3])

    def test_mismatched_affect_config_rejected(self):
        with self.assertRaises(ValueError):
            run_scenario(get_scenario("nominal"), False, affect=AffectConfig(enabled=True))


class TestRegression(unittest.TestCase):
    def test_m1_sequences_frozen(self):
        for name, expected in M1_FROZEN_SEQUENCES.items():
            for enabled in (False, True):
                with self.subTest(scenario=name, affect=enabled):
                    self.assertEqual(run_scenario(get_scenario(name), enabled).actions, expected)


class TestSweep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_sweep()

    def test_grid_size(self):
        self.assertEqual(len(sweep_configs()), 10)
        self.assertEqual(len(self.report["rows"]), 10 * len(SCENARIOS))

    def test_json_and_text(self):
        json.dumps(self.report)
        self.assertIn("Baseline", format_report(self.report))

    def test_outcome_counts_add_up(self):
        self.assertEqual(sum(self.report["outcome_counts"].values()), len(self.report["rows"]))

    def test_m1_scenarios_never_diverge(self):
        names = {r["scenario"] for r in self.report["divergences"]}
        self.assertFalse(names & set(M1_SCENARIO_NAMES))

    def test_deterministic(self):
        self.assertEqual(run_sweep(), self.report)


if __name__ == "__main__":
    unittest.main()
