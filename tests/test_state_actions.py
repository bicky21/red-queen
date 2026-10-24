import unittest

from red_queen.actions import ACTION_NAMES, RESTART_MAX_TEMPERATURE, UnknownActionError, apply_action
from red_queen.state import SystemState


class TestSystemState(unittest.TestCase):
    def test_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            SystemState(101, 50, True)
        with self.assertRaises(ValueError):
            SystemState(50, -1, True)

    def test_rejects_wrong_types(self):
        with self.assertRaises(TypeError):
            SystemState(50.5, 50, True)
        with self.assertRaises(TypeError):
            SystemState(True, 50, True)
        with self.assertRaises(TypeError):
            SystemState(50, 50, 1)

    def test_with_changes_clamps(self):
        s = SystemState(95, 5, True).with_changes(battery=130, temperature=-40)
        self.assertEqual((s.battery, s.temperature), (100, 0))

    def test_state_is_immutable(self):
        s = SystemState(50, 50, True)
        with self.assertRaises(Exception):
            s.battery = 10


class TestActions(unittest.TestCase):
    def test_allowlist_is_sorted_and_fixed(self):
        self.assertEqual(ACTION_NAMES, ("charge", "cool", "restart_service", "wait"))

    def test_charge(self):
        s = apply_action(SystemState(30, 40, True), "charge")
        self.assertEqual((s.battery, s.temperature), (50, 45))

    def test_cool_clamps_at_zero(self):
        self.assertEqual(apply_action(SystemState(50, 10, True), "cool").temperature, 0)

    def test_restart_succeeds_at_limit(self):
        s = apply_action(SystemState(50, RESTART_MAX_TEMPERATURE, False), "restart_service")
        self.assertTrue(s.service_up)
        self.assertEqual(s.battery, 40)

    def test_restart_fails_when_hot_but_still_costs_battery(self):
        s = apply_action(SystemState(50, RESTART_MAX_TEMPERATURE + 1, False), "restart_service")
        self.assertFalse(s.service_up)
        self.assertEqual(s.battery, 40)

    def test_wait_is_noop(self):
        s = SystemState(50, 50, False)
        self.assertEqual(apply_action(s, "wait"), s)

    def test_unknown_action(self):
        with self.assertRaises(UnknownActionError):
            apply_action(SystemState(50, 50, True), "shell")


if __name__ == "__main__":
    unittest.main()
