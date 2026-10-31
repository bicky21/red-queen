# RED QUEEN AI — M7.3 pre-registration: "Am I making progress?"

**Written and committed before any M7.3 code was written or run.** Date: 2026-10-31.

## Background, and an honest change of plan

The M7.2 record proposed an "unfamiliarity" flag. **Before writing this file, I diagnosed the M7.2 miscalibration** on the already-published M7.2 test data (seeds 10–19). The diagnosis showed that the flag would not help:
- **95% of new-fault snapshots** (9,794 of 10,281) had feature combinations seen **200+ times** in training.
- The main error is one familiar combination, `("fits", "widened", ">40")`. It had a training success rate of about 0.66 (wrong-bound runs that recover) but a new-fault success rate of 0.01 (stuck-sensor runs that never finish).
- The difference: in recoverable runs, the worst-corner **plan gets shorter** at each step. With a stuck sensor, it does not.

So M7.3 tests a different metacognitive question: **is the agent making progress?**

Because the design was informed by looking at seeds 10–19 of the new faults, **M7.3 is evaluated on fresh seeds 20–29**, which nobody has run or looked at.

## What is built

- A **progress feature**, from the agent's own snapshot history in the current episode:
  - `early` for the first 2 decision points;
  - `progressing` if the current worst-corner plan length is shorter than it was 2 decision points ago;
  - `stalled` otherwise.

  A missing plan counts as infinitely long, and the stop moment counts as length 0.
- **`SelfModel` v2** uses the 3 M7 features plus progress, fitted exactly as in M7 (same training data: 7 safety-layer conditions, seeds 0–9; same smoothing).
- `ConfidenceMonitor(..., use_progress=True)` uses v2 live. **Defaults are unchanged.** The M7 model and monitor behave exactly as before.
- The warning threshold for v2 is chosen on the training seeds by the M7.1 procedure: F1 over {0.3, …, 0.7}, ties to the lower value.

## Evaluation

- **New faults, fresh seeds 20–29:** the M5.4 conditions `A_drift`, `A_stuck_mid`, `B_stuck_low` and `wrong_bound+B_stuck_low`, with sensors exactly as in M5.4.
- **Familiar faults:** the M7 test set (7 safety-layer conditions, seeds 10–19), to check that nothing gets worse.
- Both the **M7 model (v1)** and **v2** are scored on the same data.

## Decision rule (fixed now)

v2 is adopted as the **recommended self-model** only if all hold:

- **P1:** v2's ECE on new faults (seeds 20–29) is ≤ 0.10.
- **P2:** v2's ECE on new faults is at most half of v1's ECE on the same data.
- **P3:** on familiar faults, v2's ECE is ≤ 0.05, and v2's Brier is ≤ v1's Brier + 0.01.
- **P4:** the v2 warning on new faults has recall ≥ 0.90 and precision ≥ 0.90.

## Expectations (stated now)

- The `("fits", "widened", ">40")` confusion should be split by progress: stalled means failure, progressing means success.
- P1 is uncertain. Other combinations, such as `("fits", "disagree", ">40")` (train 0.41, new-fault 0.05), may or may not be separated by progress.
- P3 is likely to hold. Progress adds information and should not hurt familiar faults.
- **The common-mode blind spot will remain.** Progress cannot reveal a bias that both sensors share.
