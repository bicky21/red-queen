import unittest

from red_queen.agent import STOP_GOAL
from red_queen.belief_agent import BELIEF_MODES, BeliefAgent, intersect, predict_box
from red_queen.goal import RecoveryGoal
from red_queen.planning_agent import STOP_NO_PLAN, PlanningAgent
from red_queen.policy import ALL_POLICIES
from red_queen.scenarios import ALL_SCENARIOS, get_scenario
from red_queen.sensor_study import AGENTS, format_report, run_one, run_study, true_states_by_step
from red_queen.sensors import Box, NoisySensor, SensedEnvironment, observation_box
from red_queen.state import SystemState

GOAL = RecoveryGoal()


class TestSensor(unittest.TestCase):
    def test_bounded_and_service_exact(self):
        truth = SystemState(50, 50, False)
        for step in range(100):
            r = NoisySensor(10, 1).read(truth, step)
            self.assertLessEqual(abs(r.battery - 50), 10)
            self.assertLessEqual(abs(r.temperature - 50), 10)
            self.assertFalse(r.service_up)

    def test_deterministic_and_seed_dependent(self):
        truth = SystemState(50, 50, True)
        a = [NoisySensor(10, 1).read(truth, k) for k in range(20)]
        self.assertEqual(a, [NoisySensor(10, 1).read(truth, k) for k in range(20)])
        self.assertNotEqual(a, [NoisySensor(10, 2).read(truth, k) for k in range(20)])

    def test_zero_noise_identity_and_invalid(self):
        truth = SystemState(13, 77, True)
        self.assertEqual(NoisySensor(0).read(truth, 5), truth)
        for bad in (-1, 26, True, 1.5):
            with self.assertRaises(ValueError):
                NoisySensor(bad)

    def test_sensed_environment_hides_truth(self):
        env = get_scenario("low_battery").make_env()
        sensed = SensedEnvironment(env, NoisySensor(15, 0))
        self.assertFalse(hasattr(sensed, "true_state"))
        sensed.execute("charge")
        self.assertEqual(env.step, 1)
        self.assertEqual(sensed.step, 1)


class TestBoxes(unittest.TestCase):
    def test_observation_box_clamped_and_contains_truth(self):
        box = observation_box(SystemState(3, 98, True), 10)
        self.assertEqual((box.b_lo, box.b_hi, box.t_lo, box.t_hi), (0, 13, 88, 100))
        self.assertTrue(box.contains(SystemState(0, 100, True)))

    def test_satisfies_means_every_state(self):
        self.assertTrue(Box(60, 70, 40, 50, True).satisfies(GOAL))
        self.assertFalse(Box(59, 90, 0, 10, True).satisfies(GOAL))
        self.assertFalse(Box(60, 90, 0, 51, True).satisfies(GOAL))

    def test_invalid_box(self):
        with self.assertRaises(ValueError):
            Box(50, 40, 0, 10, True)

    def test_predict_and_intersect(self):
        box = Box(20, 30, 40, 60, True)
        self.assertEqual(predict_box(box, "charge"), Box(40, 50, 45, 65, True))
        self.assertEqual(predict_box(box, "cool"), Box(20, 30, 20, 40, True))
        self.assertEqual(intersect(Box(0, 50, 0, 50, True), Box(40, 90, 45, 90, False)),
                         Box(40, 50, 45, 50, False))
        self.assertIsNone(intersect(Box(0, 10, 0, 10, True), Box(20, 30, 0, 10, True)))


class TestBeliefAgent(unittest.TestCase):
    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            BeliefAgent(GOAL, 5, mode="magic")
        with self.assertRaises(ValueError):
            BeliefAgent(GOAL, 99)

    def test_noise_zero_recovers_everything(self):
        for mode in BELIEF_MODES:
            for s in ALL_SCENARIOS:
                with self.subTest(mode=mode, scenario=s.name):
                    self.assertTrue(BeliefAgent(GOAL, 0, mode).run(s.make_env(), NoisySensor(0)).success)

    def test_sound_stop_under_noise(self):
        for mode in ("interval", "filter_verified"):
            for seed in range(10):
                r = BeliefAgent(GOAL, 15, mode).run(get_scenario("compound_failure").make_env(),
                                                    NoisySensor(15, seed))
                self.assertEqual(r.stop_reason, STOP_GOAL)
                self.assertTrue(r.success)

    def test_wait_only_stops_without_acting(self):
        r = BeliefAgent(GOAL, 10, policy=ALL_POLICIES["wait_only"]).run(
            get_scenario("low_battery").make_env(), NoisySensor(10, 0))
        self.assertEqual(r.stop_reason, STOP_NO_PLAN)
        self.assertEqual(r.actions, ())

    def test_filter_resets_on_disturbance(self):
        s = get_scenario("disturbance_mid_recovery")
        r = BeliefAgent(GOAL, 5, "filter").run(s.make_env(), NoisySensor(5, 0))
        self.assertGreater(r.stats["belief_resets"], 0)
        self.assertTrue(r.success)

    def test_lying_sensor_is_believed_but_verifier_uses_truth(self):
        class Liar:
            def read(self, truth, step):
                return SystemState(90, 10, True)

        r = BeliefAgent(GOAL, 0).run(get_scenario("low_battery").make_env(), Liar())
        self.assertEqual(r.actions, ())
        self.assertEqual(r.stop_reason, STOP_GOAL)
        self.assertFalse(r.success)


class TestStudy(unittest.TestCase):
    def test_naive_noise_zero_equals_planner(self):
        s = get_scenario("compound_failure")
        m3 = PlanningAgent(GOAL).run(s.make_env(), s.name)
        self.assertEqual(run_one("naive", s, 0, 0)["actions"], len(m3.actions))

    def test_true_states_by_step(self):
        s = get_scenario("disturbance_mid_recovery")
        env = s.make_env()
        env.execute("cool")
        truth = true_states_by_step(s.initial, env.events)
        self.assertEqual(truth[0], s.initial)
        self.assertEqual(truth[1].temperature, 20 + 45)

    def test_small_study_and_report(self):
        report = run_study(noise_levels=(0, 10), seeds=2, scenarios=ALL_SCENARIOS[:4])
        self.assertEqual(len(report["summary"]), 2 * len(AGENTS))
        self.assertIn("false stop", format_report(report))
        self.assertEqual(len(report["misspecified"]), 4)
        with self.assertRaises(ValueError):
            run_study(seeds=0)
        with self.assertRaises(ValueError):
            run_one("oracle", ALL_SCENARIOS[0], 0, 0)


if __name__ == "__main__":
    unittest.main()
