# RED QUEEN AI — Engineering Record, Milestone M7

**Version:** 0.20.0 · **Date:** 2026-10-31 · **Status:** build-environment passed; **Linux acceptance passed 2026-10-31**
**Outcome: pre-registered result, positive, with its predicted blind spot.** RED QUEEN now has a measured **self-model**. Using only its own internal state, it predicts whether the current episode will succeed. On 10,253 held-out decision points, its predictions are **calibrated** (ECE 0.020) and **informative**: the Brier score is 48% lower than always guessing the average, and lower than a hand-written rule. All three pre-registered criteria passed. As predicted, it is **blind to what it cannot sense**: at the 7 moments where a shared sensor bias made it stop too early, it was 97.6% confident.

**This is one measured indicator property, not a claim of consciousness.**

## 1. Review of M6 and the new direction

M6 was accepted on Linux on 2026-10-31. The 0.19.1 hygiene patch was accepted the same day, with no warnings on Python 3.14.

After M6, the user asked to study consciousness within the project. Nobody knows how to build consciousness, so the project follows the honest research method of the 2023 report *Consciousness in Artificial Intelligence* (Butlin, Long et al.). It builds the **indicator properties** that scientific theories associate with consciousness, and *measures* each one, without claiming consciousness. The user chose this direction, starting with a self-model.

## 2. Limitation addressed

RED QUEEN monitored its sensors (M5.3 alerts), but had no representation of **how likely it is to succeed**. It could not say "I expect to fail here." This corresponds to the higher-order-theory indicator **metacognitive monitoring**.

## 3. Method: pre-registration

`docs/PREREG_M7.md` was committed before any M7 code (commit `c2554c1`, 2026-10-31 06:52 UTC). It fixed:
- the snapshot contents and the three features
- the fitting method (smoothed frequencies)
- the data (7 conditions × 18 scenarios), with a **train / test split by seed**: 0–9 to fit, 10–19 to test
- the baselines, the metrics, decision criteria S1–S3, and the expected blind spot

Nothing was changed after the run.

## 4. Changes

| Change | File |
|---|---|
| `SafeAgent.run(..., observer=None)`: a read-only introspection hook, called at each decision point (after planning, and at the stop decision). It receives a snapshot of the agent's **own** state only: belief box and width, alert counts, trusted sensors, noise bounds, worst-corner plan length, steps left, and whether it is stopping. The observer's return value is ignored. Its exceptions are caught and counted, so it cannot change behaviour (V177). The snapshot contains no true-state fields (V178). | `safe_agent.py` |
| `SelfModel`: maps a snapshot to 3 discrete features (plan status, most severe alert, belief width), and predicts P(success) as a smoothed success frequency, fitted from past episodes. Also `heuristic`, `brier`, `ece`, `separation`. It **only predicts and never acts.** | `self_model.py` (new) |
| The pre-registered study, with the decision rule as code. | `self_model_study.py` (new) |

## 5. Acceptance criteria (build environment)

1. All 330 earlier tests and V01–V176 pass with the same meaning.
2. A tampering, crashing observer does not change behaviour in 126 runs (V177). Snapshots contain exactly the pre-registered fields, with no true state (V178).
3. Features, smoothing and metrics are exact on constructed cases (V179).
4. The study scores are frozen (V180). The pre-registered decision is computed by code (V181). The blind spot is reported (V182). Train and test seeds are disjoint, and the pre-registration is present (V183).

## 6. Results (build environment: Linux, Python 3.13)

- `python3 -m unittest discover -s tests -v` → **Ran 342 tests, OK** (330 + 12 new); `python3 -X dev` run: **0 ResourceWarnings**
- `python3 -m red_queen.validation` → **183/183 validation checks passed**

Two of my new checks failed on their first run, both **errors in the checks, not in the code**:
- **V179:** my hand-calculated expected value assumed the wrong average. The smoothing pulls toward the training base rate, 3/8, so the correct value is (3 + 2·0.375)/6 = 0.625, which is what the code returned.
- **V180:** I had typed the frozen scores by hand instead of copying them from the study output. They now come from `results/m7_self_model_study.json`.

`python3 -m red_queen.self_model_study`. It was trained on 10,391 snapshots (seeds 0–9) and tested on 10,253 snapshots (seeds 10–19). The training success rate per snapshot was 0.734.

| predictor | Brier ↓ | ECE ↓ | separation ↑ |
|---|---|---|---|
| base rate | 0.1864 | 0.0193 | 0.000 |
| heuristic | 0.1574 | 0.1296 | 0.273 |
| **self_model** | **0.0962** | **0.0197** | **0.495** |

| condition (test) | true success | mean self-confidence | Brier |
|---|---|---|---|
| none | 1.000 | 0.902 | 0.011 |
| wrong_bound | 1.000 | 0.786 | 0.083 |
| A_temp_-10 | 1.000 | 0.943 | 0.023 |
| wrong_bound+A_temp | 1.000 | 0.748 | 0.101 |
| A_dead | 1.000 | 0.932 | 0.016 |
| A_dead_low | 0.096 | 0.418 | 0.225 |
| both_temp_-10 (common mode) | 0.954 | 0.906 | 0.047 |

**Blind spot:** there were 7 false-stop moments in the test set, all from the common-mode bias. The mean self-confidence there was **0.976**.

**Pre-registered decision** (computed by `decide`, V181):
- **S1** (Brier at least 20% below base rate): **True**, 48% below.
- **S2** (ECE ≤ 0.05): **True** (0.020).
- **S3** (not worse than the heuristic): **True**.
- **Indicator present, in this measured sense: yes.**

**What this shows**
- **RED QUEEN's confidence tracks reality.** When it says about 90%, it succeeds about that often (low ECE). Its confidence is much lower in the situations where it actually fails: 0.42 in `A_dead_low`, where only 9.6% succeed.
- **Learning from experience beats a hand-written rule.** The heuristic "a plan exists, so I'll succeed" is badly calibrated (ECE 0.13). The fitted self-model is not.
- **It is appropriately less sure under strain.** It is less confident under a wrong noise bound (0.75–0.79), even though those runs all succeeded. That is honest uncertainty, and it costs some Brier score.
- **It cannot know what it cannot sense.** A bias shared by both sensors leaves no trace in its internal state, so it was 97.6% sure exactly when it was wrong. Self-knowledge is limited by perception. That matches expectations and is reported, not hidden.

## 7. Limitations

- **This is one indicator property, measured functionally. It is not consciousness**, and passing S1–S3 says nothing about experience.
- The features are coarse (3 discrete features, 60 combinations), and the model is a lookup table.
- Snapshots from one episode are correlated, and long failing episodes contribute many snapshots. The per-snapshot base rate (0.734) differs from the per-episode success rate. The test seeds are new, but the scenarios and conditions are the same as in training. Generalisation to new *kinds* of faults is untested.
- The self-model predicts but does not act. Using it to change behaviour, for example raising an alert or asking a human when confidence is low, is a separate step that needs its own test.

## 8. Linux acceptance (user's laptop)

**Passed 2026-10-31.** On the user's Ubuntu laptop (Python 3.14):
`python3 -m unittest discover -s tests -v` → **Ran 342 tests in 258.055s, OK**;
`python3 -m red_queen.validation` → **183/183 validation checks passed**;
`python3 -m red_queen.self_model_study`: compared with `diff` against the build environment, it is **identical** (apart from trailing blank lines in the pasted text).

## 9. Next milestone (candidates)

1. **M7.1: use the self-model.** When confidence drops below a threshold, raise an "I expect to fail" alert to the operator. Measure how often the alert is right and how often it fires needlessly. Pre-registered.
2. **M8: global workspace**, the second indicator. A shared "stage" where modules (sensors, memory, self-model, planner) post information, the most important item is broadcast to all of them, and we measure whether this changes anything.
3. **Generalisation test:** does the self-model stay calibrated on *new* fault types it was never trained on, such as the M5.4 drift and stuck faults?
