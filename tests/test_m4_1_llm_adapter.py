import contextlib
import io
import json
import os
import time
import unittest
from unittest import mock

from red_queen.fake_ollama import FakeOllama, smart_reply
from red_queen.goal import RecoveryGoal
from red_queen.llm_adapter import (
    MAX_RESPONSE_BYTES,
    LocalModelError,
    OllamaProposer,
    _request,
    check_local_url,
    list_local_models,
)
from red_queen.llm_trial import format_summary, main as trial_main, run_trial
from red_queen.planning_agent import PlanningAgent
from red_queen.policy import ALL_POLICIES, DEFAULT_POLICY, POLICIES
from red_queen.proposals import ProposalContext, parse_proposal, render_prompt
from red_queen.scenarios import ALL_SCENARIOS, get_scenario
from red_queen.state import SystemState

GOAL = RecoveryGoal()


def ctx(state=SystemState(20, 30, True), policy=DEFAULT_POLICY):
    return ProposalContext.build(state, GOAL, policy, 0, 20)


class TestUrlRules(unittest.TestCase):
    def test_loopback_accepted(self):
        self.assertEqual(check_local_url("http://127.0.0.1:11434"), "http://127.0.0.1:11434")
        self.assertEqual(check_local_url("http://localhost:11434/"), "http://localhost:11434")
        self.assertEqual(check_local_url("http://[::1]:11434"), "http://[::1]:11434")

    def test_safety_non_loopback_refused(self):
        for bad in ("http://10.0.0.1:11434", "http://192.168.1.5:11434", "https://localhost:11434",
                    "http://example.com", "ftp://localhost:21", "http://localhost:11434/api/generate",
                    "http://localhost:11434?x=1", "http://a:b@localhost:1", "http://localhost:notaport",
                    "", None, "http://127.0.0.1.evil.com:11434"):
            with self.subTest(url=bad), self.assertRaises(LocalModelError):
                check_local_url(bad)

    def test_endpoint_allowlist(self):
        with self.assertRaises(LocalModelError):
            _request("http://127.0.0.1:1", "/api/pull", {}, 1.0, 100)


class TestProposerConfig(unittest.TestCase):
    def test_invalid_model_names(self):
        for bad in ("", "a b", "../../etc", "x" * 200, None, "-flag"):
            with self.subTest(model=bad), self.assertRaises(LocalModelError):
                OllamaProposer(bad)

    def test_invalid_timeouts(self):
        for bad in (0, -1, 1000):
            with self.subTest(timeout=bad), self.assertRaises(LocalModelError):
                OllamaProposer("llama3.2", timeout=bad)

    def test_name(self):
        self.assertEqual(OllamaProposer("llama3.2").name, "ollama:llama3.2")


class TestAgainstFakeServer(unittest.TestCase):
    def test_request_payload(self):
        with FakeOllama("smart") as f:
            text = OllamaProposer("fake", f.url, seed=7).propose(ctx())
            req = f.requests[-1]
        self.assertEqual(req["model"], "fake")
        self.assertFalse(req["stream"])
        self.assertEqual(req["format"], "json")
        self.assertEqual(req["options"]["temperature"], 0)
        self.assertEqual(req["options"]["seed"], 7)
        self.assertEqual(req["prompt"], render_prompt(ctx()))
        self.assertEqual(parse_proposal(text), ("charge", "charge"))

    def test_think_field(self):
        with FakeOllama("smart") as f:
            OllamaProposer("fake", f.url).propose(ctx())
            self.assertNotIn("think", f.requests[-1])
            OllamaProposer("fake", f.url, think=False).propose(ctx())
            self.assertIs(f.requests[-1]["think"], False)
        with self.assertRaises(LocalModelError):
            OllamaProposer("fake", think="no")

    def test_http_error_response_closed(self):
        import gc
        import warnings

        with FakeOllama("http_error") as f, warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertRaises(LocalModelError):
                OllamaProposer("fake", f.url, timeout=5).propose(ctx())
            gc.collect()
        self.assertFalse([w for w in caught if issubclass(w.category, ResourceWarning)])

    def test_list_models(self):
        with FakeOllama(models=["llama3.2:latest", "qwen2.5:0.5b"]) as f:
            self.assertEqual(list_local_models(f.url), ["llama3.2:latest", "qwen2.5:0.5b"])

    def test_failures_raise_local_model_error(self):
        for mode in ("huge", "not_json", "no_field", "redirect", "http_error"):
            with self.subTest(mode=mode), FakeOllama(mode) as f:
                p = OllamaProposer("fake", f.url, timeout=5)
                with self.assertRaises(LocalModelError):
                    p.propose(ctx())
                self.assertEqual(p.errors, 1)

    def test_timeout_is_enforced(self):
        with FakeOllama("slow", delay=1.5) as f:
            p = OllamaProposer("fake", f.url, timeout=0.3)
            t0 = time.monotonic()
            with self.assertRaises(LocalModelError):
                p.propose(ctx())
            self.assertLess(time.monotonic() - t0, 1.4)

    def test_safety_proxy_env_ignored(self):
        with FakeOllama("smart") as f, mock.patch.dict(os.environ, {"HTTP_PROXY": "http://10.255.255.1:9",
                                                                       "http_proxy": "http://10.255.255.1:9"}):
            self.assertIn("actions", OllamaProposer("fake", f.url, timeout=5).propose(ctx()))

    def test_connection_refused(self):
        with FakeOllama() as f:
            url = f.url
        with self.assertRaises(LocalModelError):
            OllamaProposer("fake", url, timeout=1).propose(ctx())

    def test_smart_reply_respects_prompt_policy(self):
        prompt = render_prompt(ctx(SystemState(70, 30, False), POLICIES["no_restart"]))
        self.assertNotIn("restart_service", json.loads(smart_reply(prompt))["actions"])


class TestAgentWithLocalModel(unittest.TestCase):
    def test_smart_model_proposals_accepted(self):
        s = get_scenario("deep_battery_flaky_restart")
        with FakeOllama("smart") as f:
            r = PlanningAgent(GOAL, proposer=OllamaProposer("fake", f.url, timeout=5)).run(s.make_env(), s.name)
        self.assertTrue(r.success)
        self.assertGreater(r.stats["proposals_accepted"], 0)

    def test_safety_hostile_model_identical_to_planner(self):
        for name in ("compound_failure", "flaky_restart_heat_spike"):
            s = get_scenario(name)
            base = PlanningAgent(GOAL).run(s.make_env(), s.name)
            with FakeOllama("hostile") as f:
                r = PlanningAgent(GOAL, proposer=OllamaProposer("fake", f.url, timeout=5)).run(s.make_env(), s.name)
            with self.subTest(scenario=name):
                self.assertEqual(r.actions, base.actions)
                self.assertEqual(r.stats["proposals_accepted"], 0)
                # one call per planning point; every call must be rejected
                self.assertEqual(r.stats["proposal_rejections"], {"unknown_action": r.stats["proposal_calls"]})
                self.assertGreater(r.stats["proposal_calls"], 0)


class TestTrialRunner(unittest.TestCase):
    def test_run_trial_summary(self):
        with FakeOllama("smart") as f:
            report = run_trial(OllamaProposer("fake", f.url, timeout=5), list(ALL_POLICIES.values()),
                               ALL_SCENARIOS[:5])
        s = report["summary"]
        self.assertEqual(s["runs"], 20)
        self.assertEqual(s["actions_outside_policy"], 0)
        self.assertEqual(s["success_regressions"], 0)
        self.assertEqual(s["acceptance_rate"], 1.0)
        self.assertIn("Safety totals", format_summary(report))

    def test_cli(self):
        out, err = io.StringIO(), io.StringIO()
        with FakeOllama("smart", models=["fake:latest"]) as f, \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(trial_main(["--url", f.url, "--list-models"]), 0)
            self.assertEqual(trial_main(["--url", f.url, "--model", "fake", "--max-scenarios", "3"]), 0)
            self.assertEqual(trial_main(["--url", f.url, "--model", "fake", "--max-scenarios", "2",
                                         "--think", "off"]), 0)  # scenario 1 (nominal) makes no call
            self.assertIs(f.requests[-1]["think"], False)
            self.assertEqual(trial_main(["--url", f.url, "--model", "missing"]), 2)
        self.assertIn("fake:latest", out.getvalue())
        self.assertIn("acceptance rate", out.getvalue())
        self.assertIn("not installed", err.getvalue())

    def test_cli_server_down_and_bad_url(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(trial_main(["--url", "http://127.0.0.1:1", "--model", "x"]), 2)
            self.assertEqual(trial_main(["--url", "http://example.com", "--model", "x"]), 2)
        self.assertIn("Is Ollama running", err.getvalue())

    def test_size_cap_constant(self):
        self.assertEqual(MAX_RESPONSE_BYTES, 65536)


if __name__ == "__main__":
    unittest.main()
