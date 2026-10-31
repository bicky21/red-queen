# RED QUEEN AI — M7 pre-registration: self-model (first consciousness indicator)

**Written and committed before any M7 code was written or run.** Date: 2026-10-31.

## Background

The 2023 report *Consciousness in Artificial Intelligence* (Butlin, Long et al.) lists **indicator properties** derived from scientific theories of consciousness. One family, from higher-order theories, is **metacognitive monitoring**: a system that represents how reliable its own internal states are.

M7 builds and *measures* a simple version: RED QUEEN predicts **whether it will succeed**, using only information about itself, and we score how accurate those predictions are.

**No claim of consciousness is made or implied.** Having an indicator is not the same as being conscious. We only measure whether the self-predictions are accurate (calibrated) and informative.

## What is built

1. **An introspection hook** in `SafeAgent`: an optional, read-only `observer`, called at each decision point. It receives only the agent's **own internal state**: belief box, operator-report alert counts, which sensors are trusted, noise bounds, the length of the worst-corner plan (or none), steps used and the step budget, and whether it is about to stop. It **never** receives the true state. With or without an observer, behaviour is identical.
2. **A `SelfModel`**: it maps an introspection snapshot to a few discrete **features**, then predicts **P(this episode will succeed)**. The features are:
   - plan status: none / doesn't fit the remaining budget / fits with ≤ 2 spare / fits with > 2 spare / stopping now
   - alerts so far: none / disagreement only / bound widened / sensor isolated (most severe wins)
   - belief width bucket: ≤ 20 / 21–40 / > 40 (battery width + temperature width)

   It is **fitted from experience**: success frequency per feature combination, smoothed toward the overall training success rate (add-2 smoothing). The self-model **only predicts. It never changes what the agent does.**

## Data

- The agent is `SafeAgent` with its defaults (M5.3 `safe`).
- The conditions are the 7 conditions of the M5.3 safety-layer study, × 18 named scenarios.
- **Training:** seeds 0–9 (1,260 episodes). **Test:** seeds 10–19 (1,260 episodes, never used for fitting).
- Every decision point of every test episode gives one prediction. The label is the verifier's true outcome for that episode.

## Compared predictors (all scored on the test set)

- **base rate:** the overall training success rate, the same for every snapshot.
- **heuristic:** 0.95 if the worst-corner plan exists and fits the remaining budget, or if the agent is stopping; otherwise 0.05.
- **self_model:** as described above.

## Measures

- **Brier score**: the mean squared error of the probability; lower is better.
- **ECE**: expected calibration error, with 10 equal-width bins, weighted by count.
- **Separation**: the mean predicted probability on snapshots from successful episodes minus the mean on failed ones.
- Also reported: mean confidence at the **stop moment** of episodes that ended in a **false stop**.

## Decision rule (fixed now)

The self-model counts as **"calibrated self-knowledge" (indicator present, in this measured sense)** only if **all** hold on the test set:

- **S1:** self_model's Brier is at least 20% lower than base rate's (relative).
- **S2:** self_model's ECE ≤ 0.05.
- **S3:** self_model's Brier ≤ heuristic's.

## Expectations (stated now)

- S1 and S2 are likely. Plan status and alerts should separate the `A_dead_low` failures well.
- **Known blind spot:** with a **common-mode bias** (both sensors read 10° too cool), the agent cannot sense the problem. At the stop moment of those false stops, its confidence is expected to be **high (≥ 0.8)**. A self-model can only know what its own senses can detect. This is reported as a limitation, not hidden.
- S3 is uncertain. The heuristic may already capture most of the information.
