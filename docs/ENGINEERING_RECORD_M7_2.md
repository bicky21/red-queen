# RED QUEEN AI — Engineering Record, Milestone M7.2

**Version:** 0.22.0 · **Date:** 2026-10-31 · **Status:** build-environment passed; Linux acceptance pending
**Outcome: pre-registered result, mixed. RED QUEEN's self-knowledge partly generalises.** On four fault types it had never seen, with nothing re-fitted:
- **The warning carried over perfectly:** 420/420 failures warned, 0 false alarms in 300 successes, median lead 9 steps.
- **The self-model still ranks situations well:** separation 0.573, and Brier 56% below the base rate.
- **Its calibration broke:** ECE 0.289, against a pre-registered limit of 0.10.

So by the decision rule committed *before* any code, self-knowledge does **not** fully generalise. In plain terms: RED QUEEN still knows *which* situations are bad, but no longer knows *how* bad.

## 1. Review of M7.1

M7.1 was accepted on Linux on 2026-10-31, with study output identical by `diff`. Its own record noted the test was easy: all perceivable failures came from one training fault type. M7.2 is the harder test it called for.

## 2. Limitation addressed

The self-model and warning had only been tested on new seeds of *known* fault types. A self-model that works only on familiar situations is of limited value.

## 3. Method: pre-registration

`docs/PREREG_M7_2.md` was committed before any M7.2 code (commit `6d8fe25`, 2026-10-31 07:56 UTC). It fixed the following:
- **Nothing is re-fitted.** The self-model, the base rate and the 0.3 warning threshold all come from M7 and M7.1.
- The test data: the M5.4 confirmation conditions, seeds 10–19.
- The primary set: the 4 never-seen fault types `A_drift`, `A_stuck_mid`, `B_stuck_low` and `wrong_bound+B_stuck_low`.
- The measures, criteria G1–G3, and the expectations.

Nothing was changed after the run.

## 4. Changes

| Change | File |
|---|---|
| `collect_episodes(..., sensors_fn=...)`: the sensor factory can be passed in. The default (M5.3 sensors) is unchanged (V196). | `warning_study.py` |
| The pre-registered study, with the decision rule as code. | `generalisation_study.py` (new) |

No agent, estimator or self-model code changed.

## 5. Acceptance criteria (build environment)

1. All 352 earlier tests and V01–V190 pass with the same meaning.
2. Nothing was re-fitted: the base rate and threshold are identical to M7 and M7.1 (V191). The new fault conditions are disjoint from the training conditions (V192).
3. The scores and warning counts are frozen, copied from the study output (V193). The pre-registered decision is applied (V194). The finding "ranking kept, calibration lost" is checked (V195). The default sensor factory is unchanged (V196). The pre-registration is present (V197).

## 6. Results (build environment: Linux, Python 3.13)

- `python3 -m unittest discover -s tests -v` → **Ran 359 tests, OK** (352 + 7 new); `python3 -X dev` run: **0 ResourceWarnings**
- `python3 -m red_queen.validation` → **197/197 validation checks passed**

`python3 -m red_queen.generalisation_study`. **New fault types** (primary): 720 episodes and 10,281 snapshots. True success 0.417, mean self-confidence 0.470.

| predictor | Brier ↓ | ECE ↓ | separation ↑ |
|---|---|---|---|
| base rate (M7) | 0.4527 | 0.5507 | 0.000 |
| heuristic | 0.5341 | 0.5633 | 0.249 |
| **self_model (M7, unchanged)** | **0.1994** | **0.2889** | **0.573** |

| alert (threshold 0.3, unchanged) | precision | recall | false alarms | median lead |
|---|---|---|---|---|
| **self-model warning** | **1.000** | **1.000** | **0 / 300** | **9 steps** |
| baseline ("no plan fits") | 1.000 | 1.000 | 0 / 300 | 6 steps |

| condition (test) | true success | mean self-confidence | ECE | warned / false alarms / missed |
|---|---|---|---|---|
| none (seen type) | 1.000 | 0.902 | 0.097 | 0 / 0 / 0 |
| wrong_bound (seen type) | 1.000 | 0.786 | 0.214 | 0 / 0 / 0 |
| A_drift (new) | 1.000 | 0.922 | 0.078 | 0 / 0 / 0 |
| A_stuck_mid (new) | 0.400 | 0.537 | 0.336 | 108 / 0 / 0 |
| B_stuck_low (new) | 0.156 | 0.355 | 0.329 | 152 / 0 / 0 |
| wrong_bound + B_stuck_low (new) | 0.111 | 0.359 | 0.347 | 160 / 0 / 0 |

**Pre-registered decision** (computed by `decide`, V194):
- **G1** (ECE ≤ 0.10): **False** (0.289)
- **G2** (Brier ≤ 0.8 × base rate): **True** (0.199 vs 0.453)
- **G3** (warning recall and precision ≥ 0.70): **True** (1.000 / 1.000)
- **Generalises: no**

**What this shows**
- **Ranking generalises; calibration does not.** On new faults, the self-model still puts failing situations below succeeding ones (separation 0.573, higher than in M7). That is why a fixed threshold warns perfectly. But its *numbers* are wrong. In the failing conditions it says about 0.36–0.54, while the true success rate is 0.11–0.40, so it is over-confident there. Its training never saw situations this bad.
- **Also under-confident where it succeeds.** Under a wrong bound, every run succeeds, but confidence averages 0.79 (ECE 0.214 for that condition alone). That is cautious, not dangerous.
- **The warning's advantage is again timing, not accuracy.** It is equally accurate as the "no plan fits" rule, but warns 3 steps earlier (9 vs 6).
- **The expectations held only partly.** `B_stuck_low` was warned perfectly, as expected. I expected `A_stuck_mid` to be the hardest, and it was the most over-confident condition in absolute terms (0.537 against 0.400), yet still fully warned. `A_drift` caused no false alarms. I named G1 as the most likely to fail, and it did.
- **This is a known, general property of learned confidence:** it is reliable inside its experience and miscalibrated outside it. RED QUEEN's self-model is no exception, and now that is measured, not assumed.

## 7. Limitations

- **This is not consciousness.** It tests whether one measured indicator survives new situations, and the answer is "partly".
- The new faults are still sensor faults in the same small domain. "Unseen" here means unseen fault *types*, not an unseen world.
- At snapshot level, long failing episodes weigh more. ECE is computed over snapshots, as pre-registered.
- **Possible fixes, not done here because they would need a new pre-registration:**
  - re-calibrate online from observed outcomes;
  - add an "out of experience" flag when a feature combination was rarely or never seen in training, and widen the uncertainty there.

## 8. Linux acceptance (user's laptop)

**Pending.** Run in the repository folder:

```
git pull
python3 -m unittest discover -s tests -v && python3 -m red_queen.validation
python3 -m red_queen.generalisation_study
```

M7.2 is accepted when the output shows `Ran 359 tests ... OK` and `197/197 validation checks passed`. The study output should match §6.

## 9. Next milestone (candidates)

1. **M7.3: knowing what it doesn't know (pre-registered).** The self-model reports *how familiar* the current situation is, from how often its feature combination appeared in training. In unfamiliar situations it pulls its confidence towards 0.5 and flags "unfamiliar". This would test whether calibration on new faults improves without hurting familiar ones. It is a second, deeper metacognitive indicator.
2. **M8: global workspace**, the second indicator family.
