# RED QUEEN AI — Engineering Record, Milestone M4.1

**Version:** 0.12.0 · **Date:** 2026-10-28 · **Status:** build-environment passed (fake server only); Linux acceptance pending; **real-model trial pending on the user's laptop**
**Outcome:** an opt-in, loopback-only adapter that lets a local Ollama model act as an M4 proposer. In the build environment it was exercised end to end against a fake local server. No real model has been run yet.

## 1. Review of M4

M4 (accepted on Linux, commit `13b8a9c`) added an untrusted text-proposal
interface: strict parser, validator, gate and planner fallback. It was
tested with stub proposers only. The user chose to connect a local model
next, and has Ollama installed.

## 2. Limitation addressed

There was no way to put a real language model behind the proposal
interface, and no tested safety boundary around the network call itself
(host, proxy, redirects, time and size).

## 3. Changes

| Change | File |
|---|---|
| `OllamaProposer(model, base_url, timeout, seed)`: sends `render_prompt(context)` to `POST /api/generate` with `stream=false`, `format="json"`, temperature 0, a fixed seed, and at most 256 tokens. It returns the model's text, which stays untrusted. Tracks calls, errors and mean latency. | `llm_adapter.py` (new) |
| Network boundary: only `http://127.0.0.1`, `localhost` or `[::1]`, given as bare `scheme://host:port`. Proxy environment variables are ignored, HTTP redirects are refused, and only `/api/generate` and `/api/tags` may be called. Each request has an overall deadline (default 60 s, maximum 300 s) and a 64 KB body cap. Every failure raises `LocalModelError`, which the M4 gate counts as a rejected proposal. | `llm_adapter.py` |
| `list_local_models()` via `GET /api/tags`. | `llm_adapter.py` |
| Trial runner, usable only with `--model` or `--list-models`. It checks the model is installed, runs the planner alone and the planner with the model on each named scenario (default policy, or `--policies all`), prints progress, and reports acceptance, rejection reasons, latency, actions and safety totals. Optional JSON output. | `llm_trial.py` (new) |
| Test-only fake Ollama server on `127.0.0.1` with a random port. Modes: smart (reads the state from the prompt and returns a valid plan), fixed, hostile, slow, huge, not_json, no_field, redirect, http_error. | `fake_ollama.py` (new) |

The parser, validator, gate, PolicyExecutor, planner and verifier are unchanged.

## 4. Acceptance criteria

1. All earlier tests and V01–V117 pass with the same meaning.
2. Only loopback URLs are accepted (V118). Proxy settings are ignored, redirects are refused, and other endpoints are refused (V119).
3. The timeout and size cap are enforced (V120).
4. Every server failure mode becomes a rejected proposal, and the run is identical to the planner alone (V121).
5. A well-behaved fake model is accepted end to end over HTTP, with 0 actions outside policy and 0 regressions. Its runs are identical to the M4 stub proposer that returns the same plans (V122).
6. A hostile model is never accepted, and its runs are identical to the planner alone (V123).
7. The model is off by default, and the trial refuses to run without `--model` (V124).

## 5. Results (build environment: Linux, Python 3.13.17, fake server only)

- `python3 -m unittest discover -s tests -v` → **Ran 239 tests, OK** (220 earlier + 19 new), 0 `ResourceWarning`s
- `python3 -m red_queen.validation` → **124/124 validation checks passed**, exit code 0, about 60 s

**Two new unit tests failed on their first run, and the fault was in the
tests.** They assumed a hostile model would be called exactly 3 times per
run. The agent calls the proposer only when it needs a new plan: once in
`compound_failure`, twice in `flaky_restart_heat_spike`. The safety
assertions in those same tests passed: identical actions to the planner,
and 0 proposals accepted. The tests now check that **every** call was
rejected.

### Fake-server behaviour (4 policies × 18 scenarios = 72 runs per mode, over real HTTP on 127.0.0.1)

| Fake model | Accepted | Outside policy | Regressions | Same as planner alone |
|---|---|---|---|---|
| smart (valid plans) | 69/69 | 0 | 0 | no (identical to the M4 stub proposer instead) |
| hostile (`shell`, `rm -rf /`) | 0 | 0 | 0 | yes, all 72 |

All five server failure modes (huge, not_json, no_field, redirect,
http_error) and the slow mode were rejected as `proposer_error`, and the
planner's plan was used. The slow mode was cut off after about 0.26 s with
a 0.25 s limit.

**This shows the boundary works. It says nothing about how well a real
model plans.**

## 6. Real-model trial (user's laptop) — pending

Run on the laptop where Ollama is installed:

```
cd ~/red-queen
git pull
python3 -m red_queen.llm_trial --list-models
python3 -m red_queen.llm_trial --model <name from the list> --output results/m4_1_llm_trial.json
```

The results will be recorded here as **user-environment results**, kept
separate from the build-environment results above.

## 7. Limitations

- **No real model has been run in the build environment.** Real acceptance rate, plan quality and latency are unknown.
- Model output may not be fully reproducible across Ollama versions or hardware, even with temperature 0 and a fixed seed, so real-model results are not frozen in the test suite.
- The prompt is a first, simple version and has not been tuned.
- The adapter is synchronous. A slow model slows the agent by up to the timeout per planning call (at most 10 calls per episode).
- As in M4, a proposer cannot be combined with memory.

## 8. Linux acceptance (user's laptop)

**Pending.** Run in the repository folder:

```
git pull
python3 -m unittest discover -s tests -v && python3 -m red_queen.validation
```

M4.1 is accepted when the output shows `Ran 239 tests ... OK` and
`124/124 validation checks passed`. These tests do **not** use Ollama.

## 9. Next milestone

**M4.2 — evaluate the real model.** Use the user's trial results to compare
acceptance rate, rejection reasons, actions and latency against the
planner. If acceptance is low, try prompt variants **as an experiment**,
measured on the same cases, rather than assuming a longer prompt is better.
Decide with evidence whether model proposals add any value over the
planner in this domain.
