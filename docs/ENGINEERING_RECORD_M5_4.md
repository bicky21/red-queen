# RED QUEEN AI — Engineering Record, Milestone M5.4

**Version:** 0.18.0 · **Date:** 2026-10-30 · **Status:** build-environment passed; Linux acceptance pending
**Outcome: pre-registered result, mixed.** The cross-sensor isolation rule greatly improves recovery from stuck sensors: 142→318, 56→282 and 101→299 successes out of 360, with 0 false stops. But under a wrong noise bound it isolates **healthy** sensors (116/360 runs) and causes 3 false stops. By the decision rule committed *before* any code, the rule is **not adopted as default**. It stays available as an option (`cross_isolate=True`).

## 1. Review of M5.3

M5.3 was accepted on Linux on 2026-10-30, with study output identical by `diff`. Its open gap: a sensor stuck at pessimistic values was usually *explained away* by widening its noise bound, instead of being isolated. In `A_dead_low` the safe agent recovered only 101/360 runs. It failed safely, but uselessly.

## 2. Limitation addressed

The estimator used only each sensor's *own* history to judge it. It had no rule for deciding which of two disagreeing sensors to trust.

## 3. Method: pre-registration

Before writing any code, I committed `docs/PREREG_M5_4.md` (commit `844cb1f`, 2026-10-30 12:04 UTC). It fixed the rule to test, the six **new** confirmation conditions (never run before), the measures, the expectations, and the three decision criteria. The motivating M5.3 cases are reported separately as development conditions and do not count. The implementation follows the pre-registration exactly. Nothing was changed after the run.

## 4. Changes

| Change | File |
|---|---|
| `SafeEstimator(..., cross_isolate=False)` and `SafeAgent(..., cross_isolate=False)`. When both sensors are trusted, their boxes do not overlap, and **exactly one** of them has had its bound widened above the assumed value, that one is isolated. The alert is `sensor_isolated` with reason `cross_sensor`. Isolation alerts now always carry a `reason` field (`unexplainable` or `cross_sensor`). | `safe_agent.py` |
| `DriftSensor(base, battery_per_step, temperature_per_step)`: an offset that grows each step. | `sensors.py` |
| A study implementing the pre-registration, with the decision rule as code (`decide`). | `cross_isolation_study.py` (new) |

With `cross_isolate=False` (the default), behaviour is identical to M5.3, checked on 378 runs (V163).

## 5. Acceptance criteria (build environment)

These concern the engineering, not the experimental outcome:
1. All 311 earlier tests and V01–V162 pass with the same meaning.
2. The default is off and identical to M5.3 (V163). The drift sensor is exact, and invalid configurations are refused (V164). The rule works on a constructed case (V165).
3. The study totals are frozen and deterministic (V166). The pre-registered decision is computed by code and applied (V167). The gains on stuck faults are recorded (V168). The pre-registration file is present with all its terms (V169).

## 6. Results (build environment: Linux, Python 3.13)

- `python3 -m unittest discover -s tests -v` → **Ran 321 tests, OK** (311 + 10 new)
- `python3 -m red_queen.validation` → **169/169 validation checks passed**

`python3 -m red_queen.cross_isolation_study`: assumed bound 5, 360 runs per cell. Each cell is successes / false stops / wrongful isolations.

| condition | safe (M5.3) | safe_x (cross_isolate) |
|---|---|---|
| **confirmation** | | |
| none | 360 / 0 / 0 | 360 / 0 / 0 |
| wrong_bound (true 15) | 360 / 0 / 0 | 357 / **3** / **116** |
| A_drift (−2 per step) | 360 / 0 / 0 | 360 / 0 / 0 |
| A_stuck_mid (70 / 40) | 142 / 0 / 0 | **318** / 0 / 0 |
| B_stuck_low (10 / 90) | 56 / 0 / 0 | **282** / 0 / 0 |
| wrong_bound + B_stuck_low | 38 / 0 / 0 | 121 / **3** / **99** |
| **development** | | |
| A_dead_low (0 / 100) | 101 / 0 / 0 | 299 / 0 / 0 |
| A_dead (100 / 0) | 360 / 0 / 0 | 360 / 0 / 0 |

Actions on the stuck faults roughly halve. For example, B_stuck_low drops from 6188 to 2930, because the agent stops acting on a sensor that can never look healthy.

**Pre-registered decision** (computed by `decide`, V167):

| criterion | result |
|---|---|
| C1: safe_x has 0 false stops in every confirmation condition | **False** (3 in wrong_bound, 3 in wrong_bound + B_stuck_low) |
| C2: successes ≥ safe everywhere, and more on at least one fault | **False** (wrong_bound 357 < 360; the gains elsewhere are large) |
| C3: 0 wrongful isolations in none, ≤ 18 in wrong_bound | **False** (116) |
| **adopt as default** | **No** |

**What this shows**
- **The rule does what it was designed for when the noise bound is right.** It isolated the faulty sensor in the stuck and drift conditions, never isolated a healthy one there, and had no false stops.
- **It fails in exactly the way I named in advance.** Under a wrong bound, both sensors are noisier than assumed. Whichever one widens first looks guilty when the boxes next disagree, so a healthy sensor gets isolated (116/360). The agent then relies on the remaining sensor, whose bound is also wrong, and that led to 3 false stops. This is the same "single sensor with a wrong bound" weakness as in M5.1.
- **The rule's safety depends on the M5 assumption that the noise bound is known.** M5.3 `safe` did not need that assumption to avoid false stops; `safe_x` does. That is why it is not the default.
- **Expectations vs results:** `none` identical: confirmed. Stuck faults improved: confirmed. `A_drift` little difference: confirmed (360/360 both; it isolated A in 100 runs, always correctly). Wrongful isolations under wrong bound "possible, C3 uncertain": they occurred, and C3 failed.

## 7. Limitations

- One rule, with one threshold ("exactly one bound widened"). Variants were not tried, such as requiring several disagreeing steps or a minimum widening gap. Trying them now would be tuning on this data; any variant needs a new pre-registration.
- Stuck and drift faults only, on one sensor. Intermittent faults were not tested.
- Same domain and noise model as M5.

## 8. Linux acceptance (user's laptop)

**Pending.** Run in the repository folder:

```
git pull
python3 -m unittest discover -s tests -v && python3 -m red_queen.validation
python3 -m red_queen.cross_isolation_study
```

M5.4 is accepted when the output shows `Ran 321 tests ... OK` and `169/169 validation checks passed`. The study table and decision should match §6.

## 9. Next milestone (candidates)

1. **M5.5: pre-registered persistence variant.** Isolate only after the same sensor has been singled out on k consecutive disagreeing steps, and test whether this removes the wrongful isolations under a wrong bound while keeping most of the gain.
2. **M6: combine the sensor layer with memory and the planner**, meaning the disturbance memory (M3.5/M3.6) under noisy sensors. The sensor work is now well characterised; this would move the project forward instead of refining one component further.
