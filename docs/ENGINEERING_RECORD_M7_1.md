# RED QUEEN AI — Engineering Record, Milestone M7.1

**Version:** 0.21.0 · **Date:** 2026-10-31 · **Status:** build-environment passed; **Linux acceptance passed 2026-10-31**
**Outcome: pre-registered result, positive, with two honest qualifications.** RED QUEEN now warns an operator, "I expect to fail", when its self-model's confidence drops below a threshold chosen on training data. On 1,260 held-out episodes the warning was **always right** (precision 1.000, 0 false alarms). It caught **every failure it could perceive** (124/124), with a median of **10 steps** of warning before the end. All four pre-registered criteria passed.

The qualifications:
1. As predicted, it **missed all 7 common-mode false stops** (recall 0.947 overall).
2. A simple rule ("no plan fits") is **equally accurate** on this data. The self-model's advantage is only that it warns **earlier** (10 vs 7 steps).

## 1. Review of M7

M7 was accepted on Linux on 2026-10-31, with study output identical by `diff`. The self-model was calibrated (ECE 0.020) and informative (Brier 48% below the base rate), but it only *predicted*. Nothing used it.

## 2. Limitation addressed

Self-knowledge that never reaches anyone is useless. A human operator should be told early when RED QUEEN expects to fail, without being flooded with false alarms.

## 3. Method: pre-registration

`docs/PREREG_M7_1.md` was committed before any M7.1 code (commit `8dd7299`, 2026-10-31 07:23 UTC). It fixed:
- the monitor's behaviour
- that **the threshold is chosen on training seeds only**, by F1, from {0.3, 0.4, 0.5, 0.6, 0.7}, with ties going to the lower value
- the test set, the measures, the baseline alert, criteria W1–W4, and the expected blind spot

Nothing was changed after the run.

## 4. Changes

| Change | File |
|---|---|
| `ConfidenceMonitor(self_model, threshold)`: an observer that raises **one** `expect_failure` alert per episode (step, confidence, features). Information for a human only: it **never changes behaviour** (V185). | `self_model.py` |
| `SafeAgent`: if the observer offers `report()`, its result is attached as `stats["observer_report"]`. A failing report is recorded, never raised. | `safe_agent.py` |
| The pre-registered study: episode collection, threshold choice on training data, evaluation, and decision rule as code. | `warning_study.py` (new) |

## 5. Acceptance criteria (build environment)

1. All 342 earlier tests and V01–V183 pass with the same meaning.
2. The monitor validates its configuration and raises at most one alert (V184). It does not change behaviour, and its report is attached (V185). The live monitor's alert steps match the study's offline computation on 126 episodes (V186).
3. The threshold can be re-derived from the training seeds alone (V187). The test counts are frozen (V188). The pre-registered decision is applied, and the blind spot reported (V189). The pre-registration is present (V190).

## 6. Results (build environment: Linux, Python 3.13)

- `python3 -m unittest discover -s tests -v` → **Ran 352 tests, OK** (342 + 10 new); `python3 -X dev` run: **0 ResourceWarnings**
- `python3 -m red_queen.validation` → **190/190 validation checks passed**

One new unit test failed on its first run, a **test bug**. Its toy self-model knew only two hand-made feature combinations. Real snapshots fell back to the 0.5 base rate, which is never *below* the 0.5 threshold. The test now fits its model on real episodes.

**Threshold selection (training seeds 0–9):** F1 was 0.968 at both 0.3 and 0.4, falling to 0.56 at 0.5. Ties go to the lower value, so **τ = 0.3**.

**Test seeds 10–19** (1,260 episodes, 131 failures):

| alert | precision | recall | F1 | false-alarm rate | median lead |
|---|---|---|---|---|---|
| **self-model (τ = 0.3)** | **1.000** | **0.947** | **0.973** | **0.000** | **10 steps** |
| baseline ("no plan fits") | 1.000 | 0.947 | 0.973 | 0.000 | 7 steps |

| failure type | episodes | warned |
|---|---|---|
| perceivable failures (stuck pessimistic sensor) | 124 | **124 (100%)** |
| common-mode false stops (shared bias) | 7 | **0 (0%)** |

There were **0 false alarms** in all 1,129 successful episodes, including the wrong-bound conditions, where confidence is lower but still above 0.3.

**Pre-registered decision** (computed by `decide`, V189):
- **W1** (recall ≥ 0.80): True (0.947)
- **W2** (precision ≥ 0.80): True (1.000)
- **W3** (median lead ≥ 3 steps): True (10)
- **W4** (F1 ≥ baseline): True (tie)
- **Warning useful: yes**

**What this shows**
- **Self-knowledge can be turned into a useful, quiet warning.** Every warning was correct, and failures were flagged about 10 steps before the end. That is time a human could use.
- **It warns earlier than the simple rule, but is not more accurate here.** Both alerts catch the same 124 failures with no false alarms. On this data, the self-model's value is the **3 extra steps of warning**. That is honest but modest.
- **It cannot warn about what it cannot sense.** All 7 common-mode false stops passed silently, exactly as pre-registered. To RED QUEEN they look like successes.
- **The task is easy for a warning system.** All perceivable failures come from one condition (`A_dead_low`) and look very different from successes. A harder test with more varied, borderline failures would be more informative.

## 7. Limitations

- **This is not consciousness.** It is a measured functional use of one indicator property.
- The failures are concentrated in one fault type. Precision and recall may be much lower on varied or unseen faults (see the M7.2 candidate).
- One alert per episode, with no escalation and no action. Pausing or asking for help would need its own safety test.
- The threshold was chosen from 5 values on the same conditions it was tested on, just with different seeds.

## 8. Linux acceptance (user's laptop)

**Passed 2026-10-31.** On the user's Ubuntu laptop (Python 3.14):
`python3 -m unittest discover -s tests -v` → **Ran 352 tests in 277.805s, OK**;
`python3 -m red_queen.validation` → **190/190 validation checks passed**;
`python3 -m red_queen.warning_study`: compared with `diff` against the build environment, it is **identical**.

## 9. Next milestone (candidates) — M7.2 chosen

1. **M7.2: generalisation test (pre-registered).** Run the M7 self-model and the M7.1 warning, unchanged, on the M5.4 fault types they never saw (drift, stuck-mid, B stuck-low, wrong bound + stuck). This checks whether calibration and warning quality survive new situations. It is the honest next question.
2. **M8: global workspace**, the second indicator.
