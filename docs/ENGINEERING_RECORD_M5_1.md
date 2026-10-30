# RED QUEEN AI — Engineering Record, Milestone M5.1

**Version:** 0.15.0 · **Date:** 2026-10-30 · **Status:** build-environment passed; **Linux acceptance passed 2026-10-30**
**Outcome:** the belief agent can now notice that its noise bound is too small and widen it. Under a wrong bound this cuts false stops by 55–81%, but **not to zero**. With a correct bound it changes nothing (2,160 paired runs identical). A constant **sensor bias is not detected at all**, and it causes up to 45 false stops in 360 runs. This is now the main open safety gap.

## 1. Review of M5

M5 was accepted on Linux on 2026-10-30, with a study table identical to the build environment. Its main weakness: the 0-false-stop guarantee of the interval and filter_verified agents assumes the agent knows the true noise bound. With true noise 15 and an assumed bound of 5 or 0, the interval agent had 10 and 16 false stops.

## 2. Limitation addressed

The agent trusted its noise assumption blindly, even when consecutive readings contradicted it.

## 3. Changes

| Change | File |
|---|---|
| `BeliefAgent(adapt_noise=True)`, off by default. After each action it asks whether some **known event** explains the previous and new readings with the current bound. A known event is: the action worked or was blocked, then no disturbance, `battery_drain` or `heat_spike`. If none fits, it widens the bound to the smallest value that does (`needed_noise`, capped at 25), resets the filter belief, and keeps the larger bound. The bound only grows. | `belief_agent.py` |
| `explanation_boxes`, `unexplained_gap`, `needed_noise`: pure interval arithmetic, unit-tested separately. | `belief_agent.py` |
| `NoisySensor(..., bias_battery=0, bias_temperature=0)`: an optional constant offset, a sensor fault. With bias 0, readings are identical to M5. | `sensors.py` |
| A study with three parts: A, wrong bound; B, correct bound; C, sensor bias. Each covers interval and filter_verified, with and without adaptation, over 18 scenarios × 20 seeds. | `sensor_adapt_study.py` (new) |

**Why a correct bound changes nothing.** Suppose the true noise is at most the bound, and only known events happen. Then the true previous state is in the previous observation box, and the true event applied to it gives the true new state. That state is in the matching explanation box (the prediction is monotone; V133) and in the new observation box. So the two boxes overlap, the gap is 0, and nothing widens. V143 checked this on 26,136 reading pairs, and V144 on 2,160 paired episodes.

**Stated assumption:** the agent knows the *kinds* of events that can happen (blocked actions and the two level disturbances), but not when they happen.

## 4. Acceptance criteria (build environment)

1. All 273 earlier tests and V01–V141 pass with the same meaning, so the M5 study totals are unchanged.
2. Bias is validated, exact, bounded, and identical to M5 when 0 (V142).
3. Every known event is explained under a correct bound (V143). Adaptation with a correct bound is identical to no adaptation (V144).
4. Widening only increases, is capped at 25, and the step budget holds even under true noise 25 (V145).
5. Under a wrong bound, adaptation lowers false stops for assumed 5 and 0 in both modes, with frozen totals (V146).
6. Temperature bias is **not** fixed by adaptation, with frozen totals (V147). The study is deterministic (V148).

## 5. Results (build environment: Linux, Python 3.13)

- `python3 -m unittest discover -s tests -v` → **Ran 285 tests, OK** (273 + 12 new)
- `python3 -m red_queen.validation` → **148/148 validation checks passed**

`python3 -m red_queen.sensor_adapt_study`: 360 runs per row. Success and false stops are judged by the verifier on the true state.

**A. Wrong bound (true noise 15).** Each cell is false stops / actions:

| assumed | interval | interval + adapt | filter_verified | filter_verified + adapt |
|---|---|---|---|---|
| 15 | 0 / 2487 | 0 / 2487 | 0 / 2382 | 0 / 2382 |
| 10 | 0 / 2300 | 0 / 2344 | 0 / 2246 | 0 / 2280 |
| 5 | 10 / 2088 | **4** / 2296 | 11 / 2079 | **5** / 2266 |
| 0 | 16 / 1942 | **3** / 2299 | 16 / 1942 | **4** / 2260 |

**B. Correct bound (5, 10, 15):** with and without adaptation the results are identical: 0 false stops, the same actions, and 0 runs widened.

**C. Sensor bias (noise 5, bound 5).** Each cell is false stops / actions:

| bias | interval | interval + adapt | filter_verified | filter_verified + adapt |
|---|---|---|---|---|
| battery reads +10 | 0 / 1710 | 0 / 1708 | 3 / 1686 | 3 / 1683 |
| temperature reads −10 | **27** / 1914 | **27** / 1915 | **45** / 1892 | **45** / 1893 |

**What this shows**
- **Adaptation helps, but cannot fully fix a wrong bound.** The detector only learns the smallest bound that explains what it has seen so far. In the remaining false stops it had widened to 5–8, not 15, or not at all. A short episode does not show the full noise range. Detection is evidence, not proof.
- **It is free when the assumption is right.** B is identical by construction, and the study confirms it.
- **It costs actions when it triggers.** For example, interval with assumed 0: 2299 vs 1942 actions. That is about what a correct bound costs (2487), and lower.
- **Bias is invisible to consistency checks.** A constant offset shifts every reading by the same amount, so consecutive readings stay consistent. A cool-reading temperature sensor caused 27 to 45 false stops (7.5–12.5%), whether or not adaptation was on. A battery bias in these scenarios caused 0 to 3. No conclusion about why is claimed; it was not analysed.

## 6. Pre-registered expectations vs results

Self-registered before the study was run:
- **"Adaptation reduces false stops under a wrong bound, but not to zero."** Confirmed.
- **"With a correct bound, adaptation is identical."** Confirmed.
- **"Bias is not detected."** Confirmed.

## 7. Limitations

- Detection is one-sided: it can only *widen*, and only after evidence appears. Early stops can happen before any evidence.
- The agent assumes it knows the set of possible events. An unknown event looks like noise and widens the bound. That is safe, but it costs actions.
- Bias cannot be detected without a second, independent source of information, such as a redundant sensor or a reference reading.
- This is still one simple, monotone domain with uniform noise.

## 8. Linux acceptance (user's laptop)

**Passed 2026-10-30.** On the user's Ubuntu laptop (Python 3.14):
`python3 -m unittest discover -s tests -v` → **Ran 285 tests in 107.434s, OK**;
`python3 -m red_queen.validation` → **148/148 validation checks passed**;
`python3 -m red_queen.sensor_adapt_study` printed tables A, B and C **identical** to §5.

## 9. Next milestone (candidates) — M5.2 chosen

1. **M5.2: redundant sensors against bias.** Use two independent sensors, and stop only when both agree that every state in their boxes meets the goal. Treat disagreement beyond the bounds as a detected sensor fault and stop safely. This is the natural answer to the gap found here.
2. **M5.3: information gathering.** Re-read (`wait`) before stopping, to catch early false stops and tighten the bound.
