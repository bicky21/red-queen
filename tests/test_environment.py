import unittest

from red_queen.actions import UnknownActionError
from red_queen.environment import SimulatedEnvironment, UnknownDisturbanceError
from red_queen.state import SystemState


class TestEnvironment(unittest.TestCase):
    def test_execute_advances_step_and_logs(self):
        env = SimulatedEnvironment(SystemState(30, 30, True))
        env.execute("charge")
        self.assertEqual(env.step, 1)
        self.assertEqual(env.observe().battery, 50)
        self.assertEqual(len(env.events), 1)
        self.assertEqual(env.events[0].kind, "action")

    def test_disturbance_applied_after_action_at_scheduled_step(self):
        env = SimulatedEnvironment(SystemState(30, 30, True), {2: "service_crash"})
        env.execute("wait")
        self.assertTrue(env.observe().service_up)
        env.execute("wait")
        self.assertFalse(env.observe().service_up)
        self.assertEqual([e.kind for e in env.events], ["action", "action", "disturbance"])

    def test_safety_unknown_action_rejected_without_state_change(self):
        env = SimulatedEnvironment(SystemState(30, 30, True))
        before = env.observe()
        with self.assertRaises(UnknownActionError):
            env.execute("os.system")
        self.assertEqual(env.observe(), before)
        self.assertEqual(env.step, 0)
        self.assertEqual(env.events[-1].kind, "rejected")

    def test_unknown_disturbance_rejected_at_construction(self):
        with self.assertRaises(UnknownDisturbanceError):
            SimulatedEnvironment(SystemState(30, 30, True), {1: "meteor"})

    def test_invalid_disturbance_step(self):
        with self.assertRaises(ValueError):
            SimulatedEnvironment(SystemState(30, 30, True), {0: "heat_spike"})


if __name__ == "__main__":
    unittest.main()
