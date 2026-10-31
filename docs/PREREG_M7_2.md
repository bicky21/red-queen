# RED QUEEN AI — M7.2 pre-registration: does self-knowledge generalise?

**Written and committed before any M7.2 code was written or run.** Date: 2026-10-31.

## Question

The M7 self-model and the M7.1 warning were tested on new *seeds*, but on the *same kinds* of faults they were trained on. Do they stay accurate on **fault types they have never seen**?

## What is tested (nothing is re-fitted)

- **Self-model:** fitted exactly as in M7, on the 7 safety-layer conditions, seeds 0–9.
- **Warning:** threshold **0.3**, as chosen in M7.1. It is **not** re-chosen.
- **Base-rate predictor:** the M7 training base rate. It is **not** re-estimated.
- **Agent:** `SafeAgent` defaults (M5.3 `safe`; cross-isolation off).

## Evaluation data

The M5.4 confirmation conditions, 18 scenarios × seeds 10–19, with sensors built exactly as in the M5.4 study:

- **New fault types (primary):** `A_drift`, `A_stuck_mid`, `B_stuck_low`, `wrong_bound+B_stuck_low`. None of these was in the self-model's training data.
- **Already-seen types (reported for reference only):** `none`, `wrong_bound`.

## Measures

Calculated as in M7 and M7.1:
- **Snapshot level:** Brier, ECE (10 bins), separation, for self_model, base_rate and heuristic.
- **Episode level:** warning precision, recall, F1, median lead time and false-alarm rate, for the self-model warning and the M7.1 baseline alert.

All are reported per condition and for the primary (new-fault) set combined.

## Decision rule (fixed now; primary set only)

Self-knowledge **generalises** to these new faults only if all of these hold:

- **G1:** self_model's ECE ≤ 0.10. This is looser than M7's 0.05, because some shift is expected.
- **G2:** self_model's Brier ≤ 0.8 × base_rate's Brier.
- **G3:** warning recall ≥ 0.70 **and** precision ≥ 0.70.

## Expectations (stated now)

- **`B_stuck_low`** resembles the training fault `A_dead_low`, with the other sensor stuck pessimistically. It is expected to be handled well.
- **`A_stuck_mid`** (stuck at plausible, healthy-looking values) is the hardest. Failures there come from running out of steps while the hull stays wide. The self-model may be over-confident there.
- **`A_drift`** episodes all succeeded in M5.4. Some false alarms are possible if drift triggers alerts.
- **Overall: uncertain.** G1 is the most likely to fail, because the failure rate on the new set is much higher than in training (in M5.4, `safe` recovered only 56/360 and 38/360 in two of these conditions).
