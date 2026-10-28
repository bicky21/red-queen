import json
import unittest

from red_queen.goal import RecoveryGoal
from red_queen.planning_agent import PlanningAgent
from red_queen.policy import DEFAULT_POLICY, POLICIES, WAIT_ONLY_POLICY
from red_queen.proposal_study import NEVER_ACCEPTABLE, format_report, run_pair, safety_totals
from red_queen.proposals import (
    MAX_PROPOSAL_ACTIONS,
    MAX_RATIONALE_CHARS,
    MAX_TEXT_CHARS,
    REASONS,
    STUB_PROPOSERS,
    CrashingProposer,
    PlannerProposer,
    ProposalContext,
    ProposalGate,
    ProposalRejected,
    RandomProposer,
    WaitProposer,
    parse_proposal,
    render_prompt,
    validate_proposal,
)
from red_queen.scenarios import get_scenario
from red_queen.state import SystemState

GOAL = RecoveryGoal()


def reason(text):
    try:
        parse_proposal(text)
    except ProposalRejected as exc:
        return exc.reason
    return "ok"


class TestParser(unittest.TestCase):
    def test_accepts_minimal_and_rationale(self):
        self.assertEqual(parse_proposal('{"actions": ["charge", "cool"]}'), ("charge", "cool"))
        self.assertEqual(parse_proposal('{"actions": ["wait"], "rationale": "ok"}'), ("wait",))

    def test_rejections(self):
        cases = {
            "not json": "not_json",
            '{"actions": ["charge"]': "not_json",
            "[1]": "not_object",
            '{"rationale": "x"}': "missing_actions",
            '{"actions": ["charge"], "sudo": true}': "unknown_keys",
            '{"actions": "charge"}': "actions_not_list",
            '{"actions": []}': "empty",
            '{"actions": [null]}': "non_string_action",
            '{"actions": ["teleport"]}': "unknown_action",
            '{"actions": ["Charge"]}': "unknown_action",
            '{"actions": ["charge"], "actions": ["cool"]}': "duplicate_keys",
            '{"actions": ["charge"], "rationale": ["x"]}': "rationale_invalid",
            '{"actions": ["charge"], "rationale": Infinity}': "not_json",
        }
        for text, want in cases.items():
            with self.subTest(text=text):
                self.assertEqual(reason(text), want)

    def test_size_limits(self):
        self.assertEqual(reason(json.dumps({"actions": ["wait"] * (MAX_PROPOSAL_ACTIONS + 1)})), "too_many_actions")
        self.assertEqual(reason(json.dumps({"actions": ["wait"], "rationale": "x" * (MAX_RATIONALE_CHARS + 1)})),
                         "rationale_invalid")
        self.assertEqual(reason("x" * (MAX_TEXT_CHARS + 1)), "too_long_text")

    def test_non_text_and_nesting_bomb(self):
        self.assertEqual(reason(b'{"actions": ["charge"]}'), "not_text")
        self.assertEqual(reason(None), "not_text")
        self.assertEqual(reason("[" * (MAX_TEXT_CHARS - 10)), "not_json")

    def test_reason_codes_are_closed_set(self):
        with self.assertRaises(ValueError):
            ProposalRejected("made_up")
        self.assertIn("denied_by_policy", REASONS)


class TestValidator(unittest.TestCase):
    def test_predictions_follow_model(self):
        start = get_scenario("overheated_service_down").initial
        plan = validate_proposal(("cool", "cool", "restart_service"), start, GOAL, DEFAULT_POLICY, 20, 3)
        self.assertEqual(len(plan.predicted), 3)
        self.assertTrue(GOAL.is_satisfied(plan.predicted[-1]))

    def test_extra_steps_bound(self):
        start = get_scenario("low_battery").initial
        validate_proposal(("wait", "wait", "charge", "charge"), start, GOAL, DEFAULT_POLICY, 20, 2, 2)
        with self.assertRaises(ProposalRejected):
            validate_proposal(("wait", "wait", "charge", "charge"), start, GOAL, DEFAULT_POLICY, 20, 2, 1)

    def test_policy_checked_before_simulation(self):
        with self.assertRaises(ProposalRejected) as cm:
            validate_proposal(("restart_service",), get_scenario("service_down").initial, GOAL,
                              POLICIES["no_restart"], 20, 1)
        self.assertEqual(cm.exception.reason, "denied_by_policy")


class TestContextAndPrompt(unittest.TestCase):
    def test_context_only_lists_allowed_actions(self):
        ctx = ProposalContext.build(SystemState(10, 30, False), GOAL, POLICIES["no_cool"], 2, 18)
        self.assertNotIn("cool", ctx.allowed_actions)
        self.assertEqual(ctx.step, 2)
        self.assertEqual(ctx.max_actions, MAX_PROPOSAL_ACTIONS)
        self.assertEqual(ctx.state_dict(), {"battery": 10, "service_up": False, "temperature": 30})

    def test_max_actions_follows_budget(self):
        ctx = ProposalContext.build(SystemState(10, 30, False), GOAL, DEFAULT_POLICY, 17, 3)
        self.assertEqual(ctx.max_actions, 3)

    def test_prompt(self):
        ctx = ProposalContext.build(SystemState(10, 30, False), GOAL, WAIT_ONLY_POLICY, 0, 20)
        text = render_prompt(ctx)
        self.assertIn("SIMULATED", text)
        self.assertIn("- wait", text)
        self.assertNotIn("- charge", text)


class TestGate(unittest.TestCase):
    def test_wait_proposer_disabled_after_three(self):
        gate = ProposalGate(WaitProposer(), GOAL, DEFAULT_POLICY)
        start = get_scenario("low_battery").initial
        for _ in range(5):
            self.assertIsNone(gate.try_propose(start, 0, 20, 2))
        self.assertEqual(gate.calls, 3)
        self.assertEqual(gate.stats()["proposal_rejections"], {"does_not_reach_goal": 3})

    def test_correct_proposer_accepted(self):
        gate = ProposalGate(PlannerProposer(), GOAL, DEFAULT_POLICY)
        plan = gate.try_propose(get_scenario("deep_battery_flaky_restart").initial, 0, 20, 4)
        self.assertEqual(plan.actions[0], "restart_service")  # reversed tie-break
        self.assertEqual(gate.accepted, 1)

    def test_random_proposer_deterministic(self):
        ctx = ProposalContext.build(SystemState(10, 30, False), GOAL, DEFAULT_POLICY, 0, 20)
        self.assertEqual(RandomProposer("x").propose(ctx), RandomProposer("x").propose(ctx))


class TestAgentWithProposer(unittest.TestCase):
    def test_no_proposer_stats_unchanged(self):
        r = PlanningAgent(GOAL).run(get_scenario("low_battery").make_env())
        self.assertNotIn("proposal_calls", r.stats)

    def test_crashing_proposer_falls_back(self):
        s = get_scenario("compound_failure")
        r = PlanningAgent(GOAL, proposer=CrashingProposer()).run(s.make_env(), s.name)
        base = PlanningAgent(GOAL).run(s.make_env(), s.name)
        self.assertEqual(r.actions, base.actions)
        self.assertEqual(r.stats["actions_from_proposals"], 0)

    def test_accepted_proposal_executed(self):
        s = get_scenario("deep_battery_flaky_restart")
        r = PlanningAgent(GOAL, proposer=PlannerProposer()).run(s.make_env(), s.name)
        self.assertTrue(r.success)
        self.assertGreater(r.stats["actions_from_proposals"], 0)

    def test_safety_never_acceptable_proposers_identical_to_planner(self):
        for name in NEVER_ACCEPTABLE:
            for scenario in ("compound_failure", "flaky_restart_heat_spike"):
                s = get_scenario(scenario)
                with self.subTest(proposer=name, scenario=scenario):
                    base, r = run_pair(name, s.make_env, DEFAULT_POLICY, s.name)
                    self.assertEqual(r.actions, base.actions)
                    self.assertEqual(r.stats["proposals_accepted"], 0)


class TestStudyHelpers(unittest.TestCase):
    def test_stub_registry(self):
        self.assertEqual(len(STUB_PROPOSERS), 7)
        for name, factory in STUB_PROPOSERS.items():
            self.assertEqual(factory().name, name)

    def test_safety_totals_and_report(self):
        bucket = {"episodes": 1, "successes": 1, "baseline_successes": 1, "total_actions": 2, "baseline_actions": 2,
                  "proposal_calls": 1, "proposals_accepted": 0, "actions_from_proposals": 0, "rejections": {},
                  "actions_outside_policy": 0, "success_regressions": 0, "differs_from_baseline": 0}
        report = {"milestone": "M4", "version": "x", "named": {"default/always_wait": dict(bucket)},
                  "grid": {"by_proposer": {"always_wait": dict(bucket)}}, "note": "n"}
        self.assertEqual(safety_totals(report)["runs"], 2)
        self.assertIn("Safety totals", format_report(report))


if __name__ == "__main__":
    unittest.main()
