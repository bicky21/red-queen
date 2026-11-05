import os
import unittest

from red_queen.goal import RecoveryGoal
from red_queen.safe_agent import SafeAgent
from red_queen.scenarios import get_scenario
from red_queen.self_model import SelfModel
from red_queen.sensors import NoisySensor
from red_queen.workspace import (MEMORY_OFF_KINDS, MemoryModule, Message, Module, OperatorModule, PlannerModule,
                                 SelfModule, SensorModule, Workspace, standard_workspace)
from red_queen.workspace_study import AGENTS, decide

GOAL = RecoveryGoal()
SNAP = {"step": 0, "stopping": False, "plan_len": 2, "steps_left": 10, "alert_counts": {}, "width": 10}


def toy_model():
    return SelfModel().fit([(("fits", "none", "<=20"), True)] * 5 + [(("too_long", "widened", ">40"), False)] * 5)


class Fixed(Module):
    def __init__(self, name, salience, kind="x"):
        self.name, self.salience, self.kind, self.got = name, salience, kind, []

    def propose(self, snapshot, context):
        return [Message(self.name, self.kind, self.salience)]

    def receive(self, message):
        self.got.append(message.kind)


class TestMessage(unittest.TestCase):
    def test_salience_range(self):
        with self.assertRaises(ValueError):
            Message("a", "b", 1.5)
        self.assertEqual(Message("a", "b", 0.25, {"x": 1}).to_dict()["content"], {"x": 1})


class TestWorkspace(unittest.TestCase):
    def test_selection_and_ties(self):
        ws = Workspace([Fixed("b", 0.5), Fixed("a", 0.5), Fixed("c", 0.2)], capacity=1)
        self.assertEqual([m.source for m in ws.cycle(SNAP, {})], ["a"])

    def test_broadcast_reaches_all(self):
        mods = [Fixed("a", 0.9), Fixed("b", 0.1)]
        Workspace(mods, capacity=1).cycle(SNAP, {})
        self.assertEqual([m.got for m in mods], [["x"], ["x"]])

    def test_full_capacity(self):
        self.assertEqual(len(Workspace([Fixed("a", 0.9), Fixed("b", 0.1)], None).cycle(SNAP, {})), 2)

    def test_invalid(self):
        for cap in (0, True, "1"):
            with self.assertRaises(ValueError):
                Workspace([Fixed("a", 0.1)], cap)


class TestModules(unittest.TestCase):
    def test_sensor_module_reports_only_new_alerts(self):
        m = SensorModule()
        first = m.propose({**SNAP, "alert_counts": {"sensors_disagree": 2}}, {})
        again = m.propose({**SNAP, "alert_counts": {"sensors_disagree": 2}}, {})
        self.assertEqual([x.kind for x in first], ["sensors_disagree"])
        self.assertEqual(again, [])

    def test_memory_module_switch(self):
        m = MemoryModule()
        self.assertTrue(m.propose(SNAP, {"next_disturbance": "heat_spike"}))
        m.receive(Message("sensor", "sensors_disagree", 0.6))
        self.assertFalse(m.memory_off)
        m.receive(Message("self", "expect_failure", 0.9))
        self.assertTrue(m.memory_off)
        self.assertEqual(m.propose(SNAP, {"next_disturbance": "heat_spike"}), [])
        self.assertEqual(MEMORY_OFF_KINDS, {"expect_failure", "sensor_isolated", "reading_unexplainable"})

    def test_self_planner_operator(self):
        sm = SelfModule(toy_model())
        self.assertEqual(sm.propose({**SNAP, "plan_len": 9, "steps_left": 3,
                                     "alert_counts": {"bound_widened": 1}, "width": 80}, {})[0].kind, "expect_failure")
        self.assertEqual(PlannerModule().propose(SNAP, {})[0].content["status"], "plan")
        op = OperatorModule()
        op.receive(Message("p", "status", 0.1))
        self.assertEqual(len(op.stream), 1)
        with self.assertRaises(ValueError):
            SelfModule(SelfModel())


class TestAgentIntegration(unittest.TestCase):
    def test_no_memory_no_change(self):
        s = get_scenario("compound_failure")
        sensors = lambda: [NoisySensor(5, 0), NoisySensor(5, 1000)]
        a = SafeAgent(GOAL, 5).run(s.make_env(), sensors())
        b = SafeAgent(GOAL, 5).run(s.make_env(), sensors(), workspace=standard_workspace(toy_model()))
        self.assertEqual(a.actions, b.actions)
        self.assertEqual(b.stats["workspace_errors"], 0)
        self.assertEqual(b.stats["workspace_report"]["cycles"], len(b.actions) + 1)

    def test_decide(self):
        rows = []
        for lv in ("0.0", "1.0"):
            for c in ("none", "B_stuck_low", "A_stuck_mid"):
                for a in AGENTS:
                    act = 98 if (a.startswith("safe_mem_guard_ws") and c == "B_stuck_low") else 100
                    rows.append({"level": lv, "sensors": c, "agent": a, "false_stops": 0, "actions": act,
                                 "stream_messages": 10 if a == "safe_mem_guard_ws" else 30})
        d = decide({"summary": rows, "capacity_runs_identical": 5, "capacity_runs_compared": 5})
        self.assertTrue(d["workspace_functionally_useful"])

    def test_prereg_present(self):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "PREREG_M8.md")
        with open(path, encoding="utf-8") as f:
            self.assertIn("Global Workspace Theory", f.read())


if __name__ == "__main__":
    unittest.main()
