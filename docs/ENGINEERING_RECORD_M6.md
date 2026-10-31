# RED QUEEN AI — Engineering Record, Milestone M6

**Version:** 0.19.0 · **Date:** 2026-10-30 · **Status:** build-environment passed; **Linux acceptance passed 2026-10-31** (patch 0.19.1 fixes a test-hygiene warning)
**Outcome: pre-registered result, small and mixed.** Untrusted disturbance memory now works with the sensor-limited `SafeAgent`. It is **safe**: 0 false stops in 8,064 runs, because memory cannot reach the estimator or the stopping rule. But it is **barely useful**: exact memory saves 1.5–2.2% of actions, while wrong memory costs up to 2.1%. By the rule committed *before* any code, memory is **not recommended** as default. It stays available as an option.

## 1. Review of M5.4

M5.4 was accepted on Linux on 2026-10-30, with output identical by `diff`. The sensor layer (M5–M5.4) is now well characterised. The memory work (M2–M3.6) had only ever been tested with **exact** observations, and the two had never been combined.

## 2. Limitation addressed

Memory and noisy-sensor state estimation were separate. It was unknown whether disturbance memory still helps, and stays harmless, when the agent can only see the world through noisy sensors.

## 3. Method: pre-registration

`docs/PREREG_M6.md` was committed before any M6 code (commit `3592407`, 2026-10-30 12:58 UTC). It fixed the design, the agents, the conditions, the measures, the expectations, and decision criteria D1–D4. Nothing was changed after the run.

## 4. Changes

| Change | File |
|---|---|
| `SafeAgent(..., advisor=None, predict_disturbances=False, disturbance_guard=None)`. With memory on, each step's plan from the belief's worst corner also predicts **remembered disturbances** at future steps, using the M3.5 planner mode with its fallback. Remembered entries are **sanitised**: only known disturbance names at positive integer steps are kept (V171). | `safe_agent.py` |
| Guard `"on_mismatch"`: after each action, the agent predicts the belief box (previous belief → action worked → remembered disturbance for this step). If the new belief does not overlap that prediction, disturbance prediction is switched off for the rest of the episode. | `safe_agent.py` |
| Memory never reaches `SafeEstimator` or the stopping rule (V172). Defaults reproduce M5.3 (V170). | `safe_agent.py` |
| The pre-registered study, with the decision rule as code. | `memory_sensor_study.py` (new) |

**Why memory cannot cause a false stop.** The agent stops only when every state in the estimator's belief box meets the goal. The estimator uses only sensor readings, never memory. So memory changes *which* actions are taken, never *when* the agent believes it is done. A false stop can only come from the estimator's own assumptions (§7). This is checked structurally (V172) and empirically (V173).

## 5. Acceptance criteria (build environment)

1. All 321 earlier tests and V01–V169 pass with the same meaning.
2. Memory options off are identical to M5.3 (V170). Invalid configurations are refused, and junk memory is sanitised (V171).
3. Memory has no access to the estimator or the stopping rule (V172). There are 0 false stops across the study (V173).
4. The study totals are frozen and deterministic (V174). The pre-registered decision is computed by code (V175). The pre-registration file is present (V176).

## 6. Results (build environment: Linux, Python 3.13)

- `python3 -m unittest discover -s tests -v` → **Ran 329 tests, OK** (321 + 8 new)
- `python3 -m red_queen.validation` → **176/176 validation checks passed**

`python3 -m red_queen.memory_sensor_study`: 168 grid cases × 2 seeds = 336 runs per cell. **All 8,064 runs succeeded, with 0 false stops.** Each cell gives total actions, with regret / worst-case regret in brackets.

| memory | sensors | safe (no memory) | safe_mem | safe_mem_guard |
|---|---|---|---|---|
| exact (0.0) | noise 5 | 1945 | **1916** (12 / 2) | **1916** (10 / 1) |
| exact (0.0) | wrong bound | 2464 | **2411** (10 / 2) | **2413** (11 / 2) |
| 0.5 | noise 5 | 1945 | 1946 (29 / 2) | 1946 (25 / 2) |
| 0.5 | wrong bound | 2464 | 2444 (17 / 2) | 2447 (16 / 2) |
| 1.0 | noise 5 | 1945 | 1974 (43 / 4) | 1972 (37 / 4) |
| 1.0 | wrong bound | 2464 | 2451 (23 / 2) | 2451 (20 / 2) |
| unrelated | noise 5 | 1945 | **1985** (53 / 4) | 1976 (44 / 4) |
| unrelated | wrong bound | 2464 | 2463 (23 / 2) | 2462 (23 / 2) |

**Pre-registered decision** (computed by `decide`, V175):

| criterion | result |
|---|---|
| D1: memory agents have 0 false stops everywhere | **True** |
| D2: exact memory uses fewer actions, both sensor conditions | True for both agents |
| D3: wrong memory is within 2% of safe | safe_mem **False** (unrelated, noise 5: +2.06%); safe_mem_guard True (+1.6%) |
| D4: guard lowers worst-case regret | **False**: 4 vs 4 and 2 vs 2. It lowered *total* regret in most cells, but D4 was defined on the worst case. |
| candidate (from D4) | safe_mem |
| **recommend memory** | **No** |

**What this shows**
- **It is safe by construction, and confirmed in practice:** 0 false stops in 8,064 runs, including memory that was wrong or from the wrong case.
- **The benefit is small, as expected.** M3.5's gains came largely from avoiding *replans* after unexpected disturbances. `SafeAgent` already replans from its belief at every step, so only slightly better action choices remain. Exact memory saved 1.5% (noise 5) and 2.2% (wrong bound).
- **Wrong memory costs about as much as exact memory saves:** up to 2.1%, or 4 extra actions in the worst single run.
- **The guard trips often under noise.** Even with exact memory it switched memory off in 82 and 70 of 336 runs, mostly because blocked-action faults also break the prediction. It reduced total regret slightly, but not the worst case.
- **About the rule:** `safe_mem_guard` alone would have met D1–D3. The pre-registered rule chose the candidate by D4 (worst-case regret), so the outcome is "not recommended". I report that as written, rather than re-reading the rule after seeing the data.

**Expectations vs results:**
- "D1 holds": confirmed.
- "Exact memory helps less than in M3.5; D2 uncertain": D2 held, but the gain is small.
- "The guard may trip often": confirmed.

## 7. Limitations

- The safety argument rests on the estimator's assumptions from M5: a known noise bound or adaptation, and at most one faulty sensor. A common-mode bias is still undetected, with or without memory.
- Only disturbance prediction was combined. Memory-based risk tie-breaks (M3.3) were not.
- The guard compares belief boxes. Faults and noise trip it as well as wrong memory, so it cannot tell *why* a prediction failed.
- 168 of 1,680 grid cases, 2 seeds. The effects are small, so even modest sampling differences could change their sign.

## 8. Linux acceptance (user's laptop)

**Passed 2026-10-31.** On the user's Ubuntu laptop (Python 3.14):
`python3 -m unittest discover -s tests -v` → **Ran 329 tests in 246.413s, OK**;
`python3 -m red_queen.validation` → **176/176 validation checks passed**;
`python3 -m red_queen.memory_sensor_study`: the output, including the pre-registered decision, was compared with `diff` against the build environment and is **identical**.

**Issue found in the user's output, and fixed in patch 0.19.1.** Four `ResourceWarning: unclosed file` messages appeared, coming from `open(path).read()` in V169, V176 and two M5.4/M6 tests that read the pre-registration files. Results were not affected. My build-environment runs reported no warnings, because I only kept the last lines of the test output and these warnings are printed in the middle. That was a weakness in how I checked.

The fix:
- All four places now close the file (`with open(...)`).
- A new test, `tests/test_hygiene.py`, fails if any `open(...).read()` pattern appears in the code again.
- The full suite was re-run with `python3 -X dev`, which shows every ResourceWarning.

**The patch was accepted on Linux on 2026-10-31:** `Ran 330 tests in 246.172s, OK`, `176/176 validation checks passed`, and no ResourceWarning in the output.

## 9. Next milestone (candidates)

The project now has a safe, well-tested core: planning, policy, verification, memory, sensor-limited state estimation, and an untrusted language-model interface. Every add-on has been tested against a baseline. Natural next directions:

1. **M7: a harder domain.** The current domain is small and monotone, and an exact BFS planner is optimal there. That is why the language model (M4) and memory (M6) had little room to help. A domain with more state, non-monotone dynamics, or resource trade-offs would test whether these components ever add value.
2. **M6.1: stabilisation.** Validation now takes about 4–5 minutes. Split it into a fast suite (for every change) and a full suite (for acceptance), with no change to any check.
3. **Operator interface:** a read-only report of an episode (alerts, safe halts, why the agent stopped), as a first step towards a human in the loop.
