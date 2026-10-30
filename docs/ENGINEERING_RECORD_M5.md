# RED QUEEN AI — Engineering Record, Milestone M5

**Version:** 0.14.0 · **Date:** 2026-10-30 · **Status:** build-environment passed; Linux acceptance pending
**Outcome:** noisy sensors added. Treating noisy readings as truth causes **false stops**: the agent stops believing it has recovered, and the verifier says it has not. Interval ("belief box") agents with a sound stopping rule had **0 false stops in 2,196 runs**. This costs up to 28% more actions at the highest noise. That guarantee holds **only if the agent knows the true noise bound**.

## 1. Review of M4.2

M4.2 was accepted on Linux on 2026-10-30. Its real-model comparison was a pre-registered negative result: no prompt or retry setting beat the planner. The settings used 82–84 actions against the planner's 79, with 0 safety violations. The user chose to address a real limitation of the system next instead of continuing with the proposer.

## 2. Limitation addressed

Every agent from M1 to M4.2 observed the **true** state exactly. Real systems see their state through sensors that have errors. With sensor errors, an agent can:
- decide the goal is met when it is not (a false stop);
- replan endlessly, because predictions never match readings exactly.

Its only protection was the independent verifier, which reads the truth. A deployed system has no such oracle.

## 3. Changes

| Change | File |
|---|---|
| `NoisySensor(noise, seed)` reports battery and temperature with bounded integer noise in [−noise, +noise], clamped to 0..100. The service status is reported exactly. The noise is deterministic, depending only on (seed, step). The noise bound is limited to 0..25. | `sensors.py` (new) |
| `SensedEnvironment`: what a sensor-limited agent receives. `observe()` returns a reading and `execute()` passes through. It does **not** expose the true state. | `sensors.py` |
| `Box`, the belief: battery and temperature intervals plus the service status. `worst()` gives the corner with the lowest battery and highest temperature. `satisfies(goal)` is true only if **every** state in the box meets the goal. `observation_box(reading, noise)` gives all true states consistent with one reading. | `sensors.py` |
| `BeliefAgent(goal, noise, mode)`. Each step it reads, updates the belief, stops if the stop box satisfies the goal, and otherwise plans with BFS from the **worst corner** and executes only the first action through the `PolicyExecutor`. It has three modes: **`interval`** uses the observation box only; **`filter`** intersects the model prediction with the observation and resets when they do not overlap; **`filter_verified`** plans on the filter belief but stops only on the observation box. | `belief_agent.py` (new) |
| Study: 4 agents × noise 0/5/10/15 × 18 named scenarios × 20 seeds (4,392 runs). It also runs a wrong-assumption test: the interval agent under true noise 15 while assuming 15, 10, 5 or 0. | `sensor_study.py` (new) |

The existing agents, environment and verifier are unchanged. The naive agent is the unmodified M3 `PlanningAgent`, given a `SensedEnvironment` instead of the environment.

**Why planning from the worst corner is enough.** Each action's effect is monotone: more battery stays more battery, and lower temperature stays lower or equal. A restart succeeds if the temperature is ≤ 80, so lower is better there too. The goal is monotone in the same directions. So a plan that works from the worst corner works from every state in the box. V134 checked this exhaustively: 968 boxes and 15,416 states.

**Why interval stopping is sound.** The noise is bounded, so the observation box always contains the truth (V133, 8,820 readings). If every state in that box meets the goal, the true state does too. This depends on the agent knowing the right noise bound (§7).

## 4. Acceptance criteria (build environment)

1. All 256 earlier tests and V01–V131 pass, with the same meaning.
2. The sensor is bounded, deterministic and the identity at noise 0, and invalid bounds are refused (V132).
3. Observation boxes contain the truth, and predicted boxes contain every successor state (V133). Worst-corner plans cover the whole box (V134).
4. At noise 0, every agent recovers 18/18 with 79 actions, and the naive agent is identical to the M3 planner (V135).
5. Decisions use the sensor only, and the verifier uses the truth. With a lying sensor that reports a healthy system, the agent takes 0 actions and the verifier reports failure (V136).
6. `interval` and `filter_verified` have 0 false stops and 0 failures across the full study (V137).
7. The study totals are frozen and deterministic, with 0 actions outside policy (V138). Under all 4 policies at noise 10: 0 actions outside policy, 0 over budget, and `wait_only` stops early (V139).
8. A wrong noise assumption is measured, not hidden (V140). Belief violations occur only in scenarios with faults or disturbances (V141).

## 5. Results (build environment: Linux, Python 3.13)

- `python3 -m unittest discover -s tests -v` → **Ran 273 tests, OK** (256 earlier + 17 new), 0 `ResourceWarning`s
- `python3 -m red_queen.validation` → **141/141 validation checks passed**

Study (`python3 -m red_queen.sensor_study`). There are 360 runs per noise level above 0, and 18 at noise 0. Success and false stops are judged by the verifier on the true state:

| agent | noise 0 | noise 5 | noise 10 | noise 15 |
|---|---|---|---|---|
| naive: success / false stops / actions | 18 / 0 / 79 | 360 / 0 / 1834 | 353 / **7** / 1883 | 344 / **16** / 1942 |
| interval | 18 / 0 / 79 | 360 / 0 / 2009 | 360 / 0 / 2204 | 360 / 0 / 2487 |
| filter | 18 / 0 / 79 | 360 / 0 / 1817 | 360 / 0 / 1901 | 359 / **1** / 1959 |
| filter_verified | 18 / 0 / 79 | 360 / 0 / 1990 | 360 / 0 / 2171 | 360 / 0 / 2382 |

Runs where the true state left the belief box at some step:
- filter and filter_verified: 31, 77 and 87 at noise 5, 10 and 15.
- interval: 0.

All of these happened in scenarios with faults or disturbances, which break the model's prediction.

Wrong noise assumption (interval agent, true noise 15, 360 runs each):

| assumed bound | 15 | 10 | 5 | 0 |
|---|---|---|---|---|
| success | 360 | 360 | 350 | 344 |
| false stops | 0 | 0 | **10** | **16** |
| actions | 2487 | 2300 | 2088 | 1942 |

**What this shows**
- **Naive is mostly fine, but not safe.** 95.6% (344/360) succeed at noise 15, because planned actions overshoot the thresholds: a charge adds +20 and a cool removes −20. The 16 failures are all false stops. The agent believed it was done when it was not.
- **Sound stopping removes false stops, at a cost.** The interval agent needs 28% more actions than naive at noise 15 (2487 vs 1942). filter_verified keeps the 0 false stops and needs 1–4% fewer actions than interval (4% at noise 15).
- **Filtering alone is unsafe when the model is wrong.** filter is the cheapest, but a fault or disturbance can leave a belief that excludes the truth (87 runs at noise 15). That caused 1 false stop. This is the same lesson as M3.6: model-based memory needs a guard.
- **The guarantee depends on the stated assumption.** If the agent underestimates the noise (assumes 5 or 0 when the truth is 15), false stops come back. Assuming 0 reproduces the naive agent's numbers exactly.

## 6. Pre-registered expectations vs results

I wrote these down before the full study. They are honest, but self-registered, not independently registered.
- The naive agent's false stops would grow with noise. **Confirmed:** 0, then 7, then 16. The rate was lower than I expected because of overshoot.
- interval and filter_verified would have 0 false stops. **Confirmed:** this follows from the construction, and the study is consistent with it.
- interval would use the most actions. **Confirmed.**
- filter could fail in fault or disturbance scenarios. **Confirmed,** with 1 false stop and 390 runs with belief violations.

## 7. Limitations

- **A known, bounded noise is assumed.** Real sensors can have unbounded outliers, drift or bias. §5 shows what happens when the bound is wrong.
- The noise is uniform and independent per step, and the service status is never noisy.
- The interval agent forgets everything between steps. The filter keeps memory, but is unsound under model errors. Neither does probabilistic (Bayesian) estimation.
- No information-gathering actions. The agent never chooses `wait` just to read again.
- One domain with monotone dynamics. The worst-corner argument does **not** carry over to non-monotone systems.
- The proposer (M4) and memory (M2–M3.6) are not combined with sensors in M5.

## 8. Linux acceptance (user's laptop)

**Pending.** Run in the repository folder:

```
git pull
python3 -m unittest discover -s tests -v && python3 -m red_queen.validation
python3 -m red_queen.sensor_study
```

M5 is accepted when the output shows `Ran 273 tests ... OK` and `141/141 validation checks passed`. The study should print the table above, because it is deterministic.

## 9. Next milestone (candidates)

1. **M5.1: robustness to bad sensor assumptions.** Detect when readings are inconsistent with the assumed bound (for example, a jump no action can explain) and widen the bound or stop safely. Add sensor outliers and bias to the study.
2. **M5.2: information gathering.** Let the agent re-read (`wait`) when the box is too wide to decide, and measure whether this lowers the extra actions that sound stopping costs.
