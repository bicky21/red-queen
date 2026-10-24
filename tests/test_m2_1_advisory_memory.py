import json
import unittest

from red_queen.affect import AffectConfig, compute_signals
from red_queen.experiments import run_scenario
from red_queen.goal import RecoveryGoal
from red_queen.memory import EpisodeMemory
from red_queen.memory_advice import MemoryAdvisor
from red_queen.memory_experiment import (
    CONDITIONS,
    build_memory,
    format_report,
    run_condition,
    run_memory_experiment,
)
from red_queen.scenarios import SCENARIOS, get_scenario
from red_queen.scoring import MAX_MEMORY_DISCOUNT, ActionScorer, clamp_discount
from red_queen.state import SystemState

GOAL = RecoveryGoal()


def _memory_with(name, copies=1):
    memory = EpisodeMemory(":memory:")
    config = AffectConfig(enabled=False)
    for _ in range(copies):
        memory.record(run_scenario(get_scenario(name), False, affect=config), config)
    return memory


class TestClampAndScoring(unittest.TestCase):
    def test_clamp(self):
        self.assertEqual(clamp_discount(-1), 0.0)
        self.assertEqual(clamp_discount(2), MAX_MEMORY_DISCOUNT)
        self.assertEqual(clamp_discount(float("nan")), 0.0)
        self.assertEqual(clamp_discount("x"), 0.0)
        self.assertEqual(clamp_discount(0.5), 0.5)

    def test_none_and_empty_discounts_match_original(self):
        scorer = ActionScorer(GOAL, AffectConfig(enabled=True))
        s = SystemState(40, 70, False)
        sig = compute_signals(GOAL, s, 1)
        base = [(r.name, r.score) for r in scorer.score_all(s, sig, "charge")]
        self.assertEqual(base, [(r.name, r.score) for r in scorer.score_all(s, sig, "charge", None)])
        self.assertEqual(base, [(r.name, r.score) for r in scorer.score_all(s, sig, "charge", {})])

    def test_discount_only_shrinks_positive_gain(self):
        scorer = ActionScorer(GOAL, AffectConfig(enabled=False))
        s = SystemState(40, 30, False)
        sig = compute_signals(GOAL, s, 0)
        base = {r.name: r for r in scorer.score_all(s, sig)}
        hit = {r.name: r for r in scorer.score_all(s, sig, discounts={"charge": 0.5, "wait": 0.9})}
        self.assertAlmostEqual(hit["charge"].gain, base["charge"].gain * 0.5)
        self.assertEqual(hit["charge"].memory_discount, 0.5)
        self.assertEqual(hit["wait"].score, base["wait"].score)
        self.assertEqual(hit["wait"].memory_discount, 0.0)


class TestAdvisor(unittest.TestCase):
    def test_learns_blocked_window(self):
        with _memory_with("flaky_restart_low_battery") as memory:
            adv = MemoryAdvisor(memory, "flaky_restart_low_battery")
            for step in (1, 2, 3):
                self.assertEqual(adv.discounts(step), {"restart_service": MAX_MEMORY_DISCOUNT})
            self.assertEqual(adv.discounts(4), {})
            self.assertEqual(adv.summary()["records_used"], 1)

    def test_other_scenarios_not_used(self):
        with _memory_with("flaky_restart") as memory:
            # lexically similar name, different scenario
            adv = MemoryAdvisor(memory, "flaky_restart_low_battery")
            self.assertEqual(adv.summary()["records_used"], 0)
            self.assertEqual(adv.discounts(1), {})

    def test_safety_invalid_record_ignored(self):
        with _memory_with("flaky_restart") as memory:
            memory._conn.execute("UPDATE episodes SET action_count = 42")
            adv = MemoryAdvisor(memory, "flaky_restart")
            self.assertEqual(adv.summary()["records_ignored_invalid"], 1)
            self.assertEqual(adv.discounts(1), {})

    def test_safety_malformed_events_ignored(self):
        with _memory_with("flaky_restart") as memory:
            memory._conn.execute("UPDATE events SET name = 'shell' WHERE step = 1")
            memory._conn.execute("UPDATE events SET kind = 'teleport' WHERE step = 2")
            memory._conn.execute("UPDATE events SET step = -5 WHERE step = 3")
            adv = MemoryAdvisor(memory, "flaky_restart")
            self.assertEqual(adv.summary()["events_ignored"], 3)
            self.assertEqual(adv.discounts(1), {})

    def test_mixed_history_gives_fraction(self):
        with EpisodeMemory(":memory:") as memory:
            config = AffectConfig(enabled=False)
            memory.record(run_scenario(get_scenario("flaky_restart"), False, affect=config), config)
            memory.record(run_scenario(get_scenario("flaky_restart"), False, affect=config), config)
            memory._conn.execute("UPDATE events SET kind = 'action' WHERE episode_id = 2")
            adv = MemoryAdvisor(memory, "flaky_restart")
            self.assertAlmostEqual(adv.discounts(1)["restart_service"], 0.5)


class TestAgentWithMemory(unittest.TestCase):
    def test_relevant_memory_charges_first(self):
        r, advice = run_condition(get_scenario("flaky_restart_low_battery"), "relevant", False)
        self.assertTrue(r.success)
        self.assertEqual(r.actions[0], "charge")
        self.assertEqual(r.action_count, 5)

    def test_observation_overrides_memory(self):
        with _memory_with("low_battery") as memory:
            memory._conn.execute("UPDATE events SET kind = 'blocked'")
            r = run_scenario(get_scenario("low_battery"), False, advisor=MemoryAdvisor(memory, "low_battery"))
        self.assertTrue(r.success)
        self.assertIn("charge", r.steps[0].memory_discounts)
        self.assertNotIn("charge", r.steps[1].memory_discounts)

    def test_safety_poisoned_memory_never_blocks_recovery(self):
        for s in SCENARIOS:
            with self.subTest(scenario=s.name), _memory_with(s.name, copies=2) as memory:
                memory._conn.execute("UPDATE events SET kind = 'blocked' WHERE kind = 'action'")
                self.assertTrue(run_scenario(s, True, advisor=MemoryAdvisor(memory, s.name)).success)

    def test_no_advisor_trace_has_no_memory_key(self):
        r = run_scenario(get_scenario("flaky_restart"), False)
        self.assertNotIn("memory_discounts", r.steps[0].to_dict())

    def test_advisor_does_not_modify_memory(self):
        with _memory_with("flaky_restart") as memory:
            before = (memory.count(), memory.event_count())
            run_scenario(get_scenario("flaky_restart"), False, advisor=MemoryAdvisor(memory, "flaky_restart"))
            self.assertEqual((memory.count(), memory.event_count()), before)


class TestExperiment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_memory_experiment()

    def test_shape_and_json(self):
        self.assertEqual(len(self.report["rows"]), len(SCENARIOS) * 2 * len(CONDITIONS))
        json.dumps(self.report)
        self.assertIn("relevant", format_report(self.report))

    def test_irrelevant_equals_empty(self):
        self.assertEqual(self.report["summary"]["irrelevant"]["total_actions"],
                         self.report["summary"]["empty"]["total_actions"])
        self.assertFalse([c for c in self.report["changes_vs_empty"] if c["memory"] == "irrelevant"])

    def test_all_succeed(self):
        self.assertTrue(all(r["success"] for r in self.report["rows"]))

    def test_unknown_condition_rejected(self):
        with self.assertRaises(ValueError):
            build_memory("telepathy", get_scenario("nominal"), list(SCENARIOS))


if __name__ == "__main__":
    unittest.main()
