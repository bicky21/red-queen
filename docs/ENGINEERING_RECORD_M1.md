# RED QUEEN AI — Engineering Record, Milestone M1

**Version:** 0.1.0 · **Date:** 2026-10-24 · **Status:** build-environment passed; **accepted on Linux (user's laptop), 2026-10-24**

## 1. Starting point

M1 is the first milestone, so there is no previous implementation to
review. The limitation addressed is the absence of any tested control
loop: there was no way to observe a state, pick an action for an
explicit reason, act, and independently confirm the result.

## 2. Objective

Build the smallest deterministic agent that can recover a simulated
system to a goal state, using explicit action scoring, and measure
whether experimental affect signals change its behaviour.

## 3. Architecture

| Module | Responsibility |
|---|---|
| `state.py` | Immutable `SystemState` with range and type validation |
| `actions.py` | Fixed allowlist of 4 actions and the deterministic transition model |
| `environment.py` | Owns the true state; executes allowlisted actions only; applies scheduled disturbances; logs every event |
| `goal.py` | `RecoveryGoal`: explicit satisfaction test, per-condition deficits, distance |
| `affect.py` | Computational signals: urgency, concern, frustration |
| `scoring.py` | One-step scoring: weighted gain − cost − repeat penalty; name tie-break |
| `agent.py` | Bounded loop; records each step's ranking and signals |
| `verification.py` | Reads the environment's true state and checks each condition explicitly |
| `scenarios.py`, `experiments.py`, `validation.py` | Fixed scenarios, affect comparison, acceptance checks |

Control loop implemented in M1:

`Observe → (goal met? stop) → Compute signals → Score → Act → Observe → … → Verify`

Planning, memory, policy layers and preconditions are deliberately not
part of M1 (they arrive in M2–M3.6).

### Affect signals (experimental, not emotions)

- **urgency** = total deficit capped at 1.0 → reduces action cost weight by up to 50%.
- **concern** = per-condition deficit → weight `1 + concern` on that condition.
- **frustration** = consecutive non-improving actions → penalty `0.05 × frustration` for repeating the last action.

With affect disabled all weights are 1.0, cost is unchanged and no penalty applies.

## 4. Safety properties

- Only `charge`, `cool`, `restart_service`, `wait` can execute. Anything else is rejected, logged as a `rejected` event, and leaves state and step count unchanged.
- No shell, file, network or external-service access exists in the package.
- Execution is bounded by `max_steps` (default 20); negative budgets are refused.
- Success is decided only by `GoalVerifier`, which does not use the scorer or the distance function.
- State is immutable and validated on every construction.

## 5. Acceptance criteria

1. All 6 recovery scenarios succeed in both affect modes within the step bound.
2. Nominal scenario takes zero actions.
3. An insufficient step budget stops exactly at the bound and is reported as failure.
4. Repeated runs produce identical traces.
5. Non-allowlisted actions are rejected without changing state.
6. The agent cools before restarting when too hot to restart.
7. Verifier is correct at goal boundaries.
8. Scheduled disturbances are applied and logged.
9. Affect comparison report covers every scenario in both modes.

## 6. Results (build environment: Linux, Python 3.13.17)

- `python3 -m unittest discover -s tests -v` → **Ran 38 tests, OK**
- `python3 -m red_queen.validation` → **20/20 validation checks passed**, exit code 0

### Affect comparison

| Scenario | Off: actions | On: actions | Same sequence |
|---|---|---|---|
| nominal | 0 | 0 | yes |
| low_battery | 2 | 2 | yes |
| overheating | 2 | 2 | yes |
| service_down | 1 | 1 | yes |
| overheated_service_down | 3 | 3 | yes |
| compound_failure | 7 | 7 | yes |
| disturbance_mid_recovery | 6 | 6 | yes |

Both modes: 7/7 successes, 21 total actions.

**Finding:** in these scenarios affect signals had **no measurable effect**.
The action sequences were identical in every case. This is not evidence
that affect helps or hurts in general. The likely reason is that the
agent's transition model is exact, so the best action rarely changes
under modest re-weighting, and no action ever fails to improve, so the
frustration signal never activates.

## 7. Limitations

- The agent's model equals the environment's model (no uncertainty or model error).
- One-step greedy scoring; no multi-step planning.
- Scoring constants (costs, gains) are hand-set; edge cases near thresholds depend on them.
- No memory, no learning, no policy layer beyond the action allowlist, no formal preconditions.
- 7 hand-written scenarios; no randomised or stress testing.
- Disturbances only occur while the agent is still acting; disturbances after recovery are not exercised.

## 8. Linux acceptance (user's laptop)

**Accepted, 2026-10-24.** The target machine runs Linux (Ubuntu), so
acceptance is confirmed there instead of Windows.

Command run by the user in a fresh clone of `bicky21/red-queen` at commit `6f4bef5`:

```
python3 -m unittest discover -s tests -v && python3 -m red_queen.validation
```

User-reported output: `Ran 38 tests ... OK`, and validation lines V01–V20
all `[PASS]`. (The final summary line was cut off in the pasted output;
all 20 individual PASS lines were present.)

## 9. Next milestone

**M1.1** — make the affect comparison informative before adding memory:
add scenarios where action outcomes can fail (e.g. a restart that fails
intermittently on a fixed schedule) so frustration can activate, add
scoring-parameter sweeps, and record where affect-on and affect-off
diverge. Then M2 (persistent SQLite memory) can build on a loop whose
behaviour is measured.
