# RED QUEEN AI — Engineering Record, Milestone M4.2

**Version:** 0.13.0 · **Date:** 2026-10-28 · **Status:** build-environment passed (fake server only); Linux acceptance pending; **real-model comparison pending on the user's laptop**
**Outcome so far:** two experimental levers (prompt style, trusted-feedback retries) are implemented and safety-tested. The decision rule below was written **before** any real-model data.

## 1. Review of M4.1

M4.1 (accepted on Linux) ran a real local model, `qwen3.5:4b` with `--think off`, as an untrusted proposer:
- 18/18 success, with 0 safety violations.
- Only **4/30 proposals accepted (13%)**; 25 rejections were `does_not_reach_goal`.
- 82 actions, against 79 for the planner alone.

The model mostly produced plans that do not reach the goal.

## 2. Limitation addressed

There was no way to test whether the low acceptance comes from how the
task is presented (the prompt), or from the model being unable to correct
itself without information. That needs to be tested rather than guessed.

## 3. Changes

| Change | File |
|---|---|
| `render_prompt(context, style)`, with 3 styles. **`v1`** is the M4 prompt, **byte-identical** (SHA-256 fingerprints frozen in V125). **`v2_rules`** spells out the goal conditions, says to track battery and temperature after every action, not to stop early, not to add actions after the goal, and to use as few actions as possible. **`v3_example`** is `v2_rules` plus one worked example (a different state), verified step by step against the transition model and shortest (V126). | `proposals.py` |
| Trusted feedback: `ProposalRejected` now carries the parsed (allowlisted) actions and the model-predicted end state, and `feedback_text()` builds a short message from them: reason, plan, predicted end state, and number-only details. Parser details are **never** fed back, because they can contain proposer text. Capped at 400 characters. | `proposals.py` |
| `ProposalContext.feedback`; the prompt adds "Your previous proposal was rejected: …" only when it is non-empty. | `proposals.py` |
| `ProposalGate(retries=0..2)`: after a rejection, ask again at the same planning point with the feedback. Every attempt counts as a call and a rejection. Stats report `retry_calls` and `accepted_after_retry`. | `proposals.py` |
| `PlanningAgent(proposal_retries=...)` and `OllamaProposer(prompt_style=...)`. | `planning_agent.py`, `llm_adapter.py` |
| `llm_trial --prompt v1/v2_rules/v3_example`, `--retries 0/1/2`, and `--compare`, which runs all 3 styles × retries 0 and 1 (6 configurations) and prints one table. With retries, the call and rejection budgets scale by (1 + retries), so a retry configuration is not switched off earlier than the no-retry one. | `llm_trial.py` |
| Fake server `learner` mode: wrong first, correct once feedback appears. Used to prove feedback actually reaches the model. | `fake_ollama.py` |

All defaults (`v1`, `retries=0`) reproduce M4/M4.1 exactly (V125, V128).

## 4. Acceptance criteria (build environment)

1. All earlier tests and V01–V124 pass with the same meaning.
2. `v1` is byte-identical to M4, the styles are distinct and deterministic, and invalid styles are refused (V125). The worked example is correct and shortest (V126).
3. Feedback uses trusted data only. Across 93 prompts to a hostile fake (62 carrying feedback), **0 contained proposer text** (V127).
4. `retries=0` is identical to M4 for all stub proposers (V128). Retry bounds hold: at most 1 + retries calls per planning point, and a hostile proposer is still 0 accepted at every retry level (V129).
5. Feedback demonstrably reaches the model: the learner fake goes from 0/57 accepted with no retries to 69/138 with 1 retry, all accepted after feedback (V130).
6. `--compare` runs all 6 configurations, with safety totals 0 (V131).

## 5. Results (build environment: Linux, Python 3.13.17, fake server only)

- `python3 -m unittest discover -s tests -v` → **Ran 256 tests, OK** (241 earlier + 15 new), 0 `ResourceWarning`s
- `python3 -m red_queen.validation` → **131/131 validation checks passed**, exit code 0, about 75 s

One new unit test failed on its first run because of a test bug. It read
an exception variable after its `except` block had ended, which Python
does not allow. It now uses `assertRaises`. No code was changed.

## 6. Pre-registered decision rule (written before real-model data)

The same 18 named scenarios and default policy as M4.1, with the same model (`qwen3.5:4b`, `--think off`).

- **Primary measure:** total actions compared with the planner alone (79). A configuration **adds value** only if it uses **fewer than 79 actions** with 0 actions outside policy and 0 success regressions.
- **Secondary measures:** acceptance rate, actions taken from proposals, and time.
- **Expectation, stated in advance:** higher acceptance is plausible, especially with feedback. Beating the planner is **not** expected. The validator only accepts plans at most 2 steps longer than the planner's shortest plan, so with an exact model a proposal can only tie or lose, except by tie-break luck against faults, as seen in M3.2. If acceptance rises but actions do not drop below 79, the conclusion is: **the model became more compliant, not more useful**.

## 7. Real-model comparison (user's laptop) — pending

```
cd ~/red-queen
git pull
python3 -m red_queen.llm_trial --model qwen3.5:4b --think off --compare --output results/m4_2_compare.json
```

That is 6 configurations × 18 scenarios. At about 4 s per model call this
is expected to take roughly 15–30 minutes. The results will be recorded
here as **user-environment results**.

## 8. Limitations

- Three hand-written prompt styles is a small, non-exhaustive search.
- The worked example is fixed. The model may copy its pattern instead of reasoning.
- Retries cost extra model calls and time. They are bounded, but slower.
- One model, one machine. Results will not generalise to other models.

## 9. Linux acceptance (user's laptop)

**Pending.** Run in the repository folder:

```
git pull
python3 -m unittest discover -s tests -v && python3 -m red_queen.validation
```

M4.2 is accepted when the output shows `Ran 256 tests ... OK` and
`131/131 validation checks passed`. These tests do not use Ollama.

## 10. Next milestone

Decided by the comparison. If no configuration adds value (the expected
outcome), the model's role should be reconsidered. For example, it could
**explain** the planner's plans to a human instead of proposing them, which
never affects execution. Alternatively the project could move to a
harder domain where the planner is *not* optimal (for example, partial
observability), where a model might have room to help.
