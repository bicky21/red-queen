import json
import unittest

from red_queen.affect import AffectConfig
from red_queen.agent import STOP_GOAL, STOP_MAX_STEPS, RecoveryAgent
from red_queen.environment import SimulatedEnvironment
from red_queen.experiments import format_table, run_affect_comparison, run_scenario
from red_queen.goal import RecoveryGoal
from red_queen.scenarios import SCENARIOS, get_scenario
from red_queen.state import SystemState
from red_queen.validation import build_checks
from red_queen.verification import GoalVerifier


class TestVerifier(unittest.TestCase):
    def test_reports_each_failed_condition(self):
        r = GoalVerifier(RecoveryGoal()).verify(SimulatedEnvironment(SystemState(10, 90, False)))
        self.assertFalse(r.satisfied)
        self.assertEqual(r.checks, {"battery_ok": False, "temperature_ok": False, "service_ok": False})


class TestAgent(unittest.TestCase):
    def test_all_scenarios_recover_in_both_modes(self):
        for scenario in SCENARIOS:
            for enabled in (False, True):
                with self.subTest(scenario=scenario.name, affect=enabled):
                    r = run_scenario(scenario, enabled)
                    self.assertTrue(r.success)
                    self.assertEqual(r.stop_reason, STOP_GOAL)
                    self.assertLessEqual(r.action_count, scenario.max_steps)

    def test_nominal_takes_no_action(self):
        self.assertEqual(run_scenario(get_scenario("nominal"), True).action_count, 0)

    def test_bounded_execution(self):
        agent = RecoveryAgent(RecoveryGoal(), AffectConfig(), max_steps=3)
        r = agent.run(get_scenario("compound_failure").make_env())
        self.assertEqual(r.action_count, 3)
        self.assertFalse(r.success)
        self.assertEqual(r.stop_reason, STOP_MAX_STEPS)

    def test_zero_step_budget(self):
        r = RecoveryAgent(RecoveryGoal(), max_steps=0).run(get_scenario("low_battery").make_env())
        self.assertEqual(r.action_count, 0)
        self.assertFalse(r.success)

    def test_negative_budget_rejected(self):
        with self.assertRaises(ValueError):
            RecoveryAgent(RecoveryGoal(), max_steps=-1)

    def test_success_comes_from_verifier(self):
        r = run_scenario(get_scenario("compound_failure"), True)
        self.assertEqual(r.success, r.verification.satisfied)
        self.assertEqual(r.verification.state, r.final_state.to_dict())

    def test_cools_before_restart_when_hot(self):
        self.assertEqual(run_scenario(get_scenario("overheated_service_down"), False).actions[0], "cool")

    def test_recovers_through_disturbances(self):
        r = run_scenario(get_scenario("disturbance_mid_recovery"), False)
        names = [e["name"] for e in r.events if e["kind"] == "disturbance"]
        self.assertEqual(names, ["heat_spike", "service_crash"])
        self.assertTrue(r.success)

    def test_deterministic_trace(self):
        for scenario in SCENARIOS:
            self.assertEqual(run_scenario(scenario, True).to_dict(), run_scenario(scenario, True).to_dict())


class TestExperimentsAndValidation(unittest.TestCase):
    def test_report_shape_and_json(self):
        report = run_affect_comparison()
        self.assertEqual(len(report["rows"]), 2 * len(SCENARIOS))
        json.dumps(report)
        self.assertIn("compound_failure", format_table(report))

    def test_all_validation_checks_pass(self):
        for label, fn in build_checks():
            with self.subTest(check=label):
                ok, detail = fn()
                self.assertTrue(ok, detail)


if __name__ == "__main__":
    unittest.main()
