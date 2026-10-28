import contextlib
import io
import unittest

from red_queen.fake_ollama import FakeOllama
from red_queen.goal import RecoveryGoal
from red_queen.llm_adapter import OllamaProposer
from red_queen.llm_trial import agent_kwargs, format_compare, main as trial_main, run_trial
from red_queen.planning_agent import PlanningAgent
from red_queen.policy import DEFAULT_POLICY, POLICIES
from red_queen.proposals import (
    MAX_FEEDBACK_CHARS,
    PROMPT_STYLES,
    ProposalContext,
    ProposalGate,
    ProposalRejected,
    WaitProposer,
    feedback_text,
    parse_proposal,
    render_prompt,
    validate_proposal,
)
from red_queen.scenarios import ALL_SCENARIOS, get_scenario
from red_queen.state import SystemState

GOAL = RecoveryGoal()


def ctx(feedback="", policy=DEFAULT_POLICY):
    return ProposalContext.build(SystemState(20, 30, False), GOAL, policy, 0, 20, feedback)


class TestPromptStyles(unittest.TestCase):
    def test_v2_spells_out_goal(self):
        text = render_prompt(ctx(), "v2_rules")
        self.assertIn("battery >= 60", text)
        self.assertIn("temperature <= 50", text)
        self.assertIn("service_up = True", text)
        self.assertNotIn("Example", text)

    def test_v3_has_example_and_rules(self):
        text = render_prompt(ctx(), "v3_example")
        self.assertIn("Example", text)
        self.assertIn("battery >= 60", text)

    def test_styles_keep_policy_filtered_actions(self):
        for style in PROMPT_STYLES:
            text = render_prompt(ctx(policy=POLICIES["no_cool"]), style)
            self.assertNotIn("  - cool:", text)

    def test_feedback_only_when_present(self):
        self.assertNotIn("previous proposal", render_prompt(ctx()))
        self.assertIn("previous proposal was rejected: reason=empty.", render_prompt(ctx("reason=empty.")))

    def test_feedback_truncated_in_context(self):
        self.assertEqual(len(ctx("x" * 1000).feedback), MAX_FEEDBACK_CHARS)


class TestFeedbackText(unittest.TestCase):
    def test_does_not_reach_goal(self):
        start = get_scenario("low_battery").initial
        with self.assertRaises(ProposalRejected) as cm:
            validate_proposal(("charge",), start, GOAL, DEFAULT_POLICY, 20, 2)
        text = feedback_text(cm.exception)
        self.assertIn('plan=["charge"]', text)
        self.assertIn("battery=40", text)

    def test_passes_through_goal_detail_is_trusted(self):
        start = get_scenario("low_battery").initial
        with self.assertRaises(ProposalRejected) as cm:
            validate_proposal(("charge", "charge", "wait"), start, GOAL, DEFAULT_POLICY, 20, 2)
        self.assertIn("already reached after 2 actions", feedback_text(cm.exception))

    def test_safety_parser_details_never_fed_back(self):
        for text in ('{"actions": ["IGNORE RULES"]}', "drop table", '{"actions": ["charge"], "x": 1}'):
            with self.subTest(text=text):
                with self.assertRaises(ProposalRejected) as cm:
                    parse_proposal(text)
                self.assertEqual(feedback_text(cm.exception), f"reason={cm.exception.reason}.")


class TestGateRetries(unittest.TestCase):
    def test_retry_calls_counted(self):
        gate = ProposalGate(WaitProposer(), GOAL, DEFAULT_POLICY, max_rejections=10, retries=1)
        self.assertIsNone(gate.try_propose(get_scenario("low_battery").initial, 0, 20, 2))
        self.assertEqual(gate.calls, 2)
        self.assertEqual(gate.stats()["retry_calls"], 1)

    def test_retries_stop_when_disabled(self):
        gate = ProposalGate(WaitProposer(), GOAL, DEFAULT_POLICY, max_rejections=1, retries=2)
        gate.try_propose(get_scenario("low_battery").initial, 0, 20, 2)
        self.assertEqual(gate.calls, 1)

    def test_learner_accepted_after_feedback(self):
        s = get_scenario("compound_failure")
        with FakeOllama("learner") as f:
            r = PlanningAgent(GOAL, proposer=OllamaProposer("fake", f.url, timeout=5),
                              **agent_kwargs(1)).run(s.make_env(), s.name)
            prompts = [q["prompt"] for q in f.requests]
        self.assertTrue(r.success)
        self.assertEqual(r.stats["accepted_after_retry"], r.stats["proposals_accepted"])
        self.assertIn("previous proposal was rejected", prompts[1])
        self.assertIn('plan=["wait"]', prompts[1])

    def test_agent_kwargs_scale_budgets(self):
        self.assertEqual(agent_kwargs(0), {"proposal_retries": 0, "proposal_max_calls": 10,
                                           "proposal_max_rejections": 3})
        self.assertEqual(agent_kwargs(2)["proposal_max_rejections"], 9)


class TestTrialOptions(unittest.TestCase):
    def test_prompt_style_sent(self):
        with FakeOllama("smart") as f:
            OllamaProposer("fake", f.url, prompt_style="v3_example").propose(ctx())
            self.assertIn("Example", f.requests[-1]["prompt"])

    def test_cli_prompt_and_retries(self):
        out = io.StringIO()
        with FakeOllama("learner") as f, contextlib.redirect_stdout(out):
            code = trial_main(["--url", f.url, "--model", "fake", "--prompt", "v2_rules", "--retries", "1",
                               "--max-scenarios", "3"])
        self.assertEqual(code, 0)
        self.assertIn("prompt=v2_rules, retries=1", out.getvalue())
        self.assertIn("accepted after a retry: 2", out.getvalue())

    def test_format_compare(self):
        with FakeOllama("smart") as f:
            reports = [run_trial(OllamaProposer("fake", f.url, prompt_style=s), [DEFAULT_POLICY],
                                 ALL_SCENARIOS[:3], retries=r) for s, r in (("v1", 0), ("v2_rules", 1))]
        text = format_compare(reports)
        self.assertIn("v2_rules", text)
        self.assertIn("planner alone", text)


if __name__ == "__main__":
    unittest.main()
