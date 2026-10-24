import unittest

from red_queen.affect import (
    NEUTRAL_WEIGHTS,
    AffectConfig,
    compute_signals,
    deficit_weights,
    update_frustration,
)
from red_queen.goal import RecoveryGoal
from red_queen.scoring import ActionScorer
from red_queen.state import SystemState

GOAL = RecoveryGoal()


class TestGoal(unittest.TestCase):
    def test_boundaries(self):
        self.assertTrue(GOAL.is_satisfied(SystemState(60, 50, True)))
        self.assertFalse(GOAL.is_satisfied(SystemState(59, 50, True)))
        self.assertFalse(GOAL.is_satisfied(SystemState(60, 51, True)))
        self.assertFalse(GOAL.is_satisfied(SystemState(60, 50, False)))

    def test_deficits_and_distance(self):
        d = GOAL.deficits(SystemState(40, 70, False))
        self.assertEqual(d, {"battery": 0.2, "temperature": 0.2, "service": 1.0})
        self.assertAlmostEqual(GOAL.distance(SystemState(40, 70, False)), 1.4)
        self.assertEqual(GOAL.distance(SystemState(80, 20, True)), 0.0)


class TestAffect(unittest.TestCase):
    def test_urgency_capped(self):
        self.assertEqual(compute_signals(GOAL, SystemState(0, 100, False), 0).urgency, 1.0)

    def test_frustration_update(self):
        self.assertEqual(update_frustration(3, 1.0, 0.5), 0)
        self.assertEqual(update_frustration(3, 1.0, 1.0), 4)

    def test_disabled_weights_are_neutral(self):
        s = compute_signals(GOAL, SystemState(0, 100, False), 2)
        self.assertEqual(deficit_weights(AffectConfig(enabled=False), s), NEUTRAL_WEIGHTS)

    def test_enabled_weights_amplify_worst_deficit(self):
        s = compute_signals(GOAL, SystemState(20, 50, True), 0)
        w = deficit_weights(AffectConfig(enabled=True), s)
        self.assertGreater(w["battery"], w["temperature"])


class TestScoring(unittest.TestCase):
    def test_ranking_is_deterministic_and_sorted(self):
        scorer = ActionScorer(GOAL, AffectConfig(enabled=True))
        s = SystemState(30, 30, True)
        sig = compute_signals(GOAL, s, 0)
        a = [(r.name, r.score) for r in scorer.score_all(s, sig)]
        b = [(r.name, r.score) for r in scorer.score_all(s, sig)]
        self.assertEqual(a, b)
        self.assertEqual([x[1] for x in a], sorted([x[1] for x in a], reverse=True))

    def test_cool_preferred_when_overheating(self):
        scorer = ActionScorer(GOAL, AffectConfig(enabled=False))
        s = SystemState(70, 90, True)
        self.assertEqual(scorer.score_all(s, compute_signals(GOAL, s, 0))[0].name, "cool")

    def test_restart_scores_negative_when_too_hot(self):
        scorer = ActionScorer(GOAL, AffectConfig(enabled=False))
        s = SystemState(70, 90, False)
        scores = {r.name: r.score for r in scorer.score_all(s, compute_signals(GOAL, s, 0))}
        self.assertLess(scores["restart_service"], 0)

    def test_frustration_penalty_only_when_enabled(self):
        s = SystemState(30, 30, True)
        sig = compute_signals(GOAL, s, 3)
        on = {r.name: r for r in ActionScorer(GOAL, AffectConfig(enabled=True)).score_all(s, sig, "charge")}
        off = {r.name: r for r in ActionScorer(GOAL, AffectConfig(enabled=False)).score_all(s, sig, "charge")}
        self.assertAlmostEqual(on["charge"].penalty, 0.15)
        self.assertEqual(off["charge"].penalty, 0.0)


if __name__ == "__main__":
    unittest.main()
