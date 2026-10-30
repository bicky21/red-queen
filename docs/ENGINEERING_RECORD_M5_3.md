# RED QUEEN AI — Engineering Record, Milestone M5.3

**Version:** 0.17.0 · **Date:** 2026-10-30 · **Status:** build-environment passed; Linux acceptance pending
**Outcome:** the three sensor protections from M5 to M5.2 are now one state-estimation component, `SafeEstimator`, with operator alerts and a defined response to an unexplainable sensor. The combined agent had **0 false stops in all 2,160 runs without a common-mode fault**, including the combined fault "wrong noise bound + biased sensor", where every single protection alone still failed. It costs more actions, and two limitations are confirmed: a **common-mode bias** (16/360 false stops) and a **pessimistically stuck sensor** that the detector explains away instead of isolating (only 101/360 recovered, failing safely).

## 1. Review of M5.2

M5.2 was accepted on Linux on 2026-10-30, with a study table identical to the build environment. Each protection lived in its own agent: interval (M5), adaptive bound (M5.1), and hull of two sensors (M5.2). No agent combined them, and nothing told an operator what had gone wrong.

## 2. Limitation addressed

1. The protections could not be used together.
2. The state estimate was mixed into each agent's decision loop instead of being a separate, testable component, as the project principles ask.
3. An "untrustworthy estimate" had no defined outcome and no report.

## 3. Changes

| Change | File |
|---|---|
| **`SafeEstimator(noise, sensors=1/2, adapt, on_unexplainable)`**: readings in, belief box out. It never executes and never sees the environment (V157). It adapts each sensor's bound (M5.1 logic), fuses with the hull (M5.2), and records **alerts**: `bound_widened`, `sensors_disagree`, `reading_unexplainable`, `sensor_isolated`. Alerts carry numeric evidence and step numbers only. | `safe_agent.py` (new) |
| **Response to an unexplainable sensor**, meaning one whose readings no known event explains even at the maximum bound of 25. With **`"halt"`**, the agent stops with `safe_halt` and claims nothing. With **`"isolate"`** (the default), that sensor is excluded for the rest of the episode, and the agent halts only when no trusted sensor is left, which is always the case with one sensor. | `safe_agent.py` |
| **`SafeAgent`**: planning, execution and stopping exactly as in M5, with the estimate taken from `SafeEstimator`. The `operator_report` is returned in `stats`. | `safe_agent.py` |
| `StuckSensor(base, fail_from_step, battery, temperature)`: a dead or stuck sensor fault for testing. | `sensors.py` |
| A study of 5 agents × 7 conditions × 18 scenarios × 20 seeds. | `safety_layer_study.py` (new) |

**Regression by construction:** `SafeAgent` reproduces the earlier agents per run, with identical action sequences and stop reasons, checked on 540 episodes for each configuration (V156):
- `(1 sensor, fixed)` matches M5 interval.
- `(1 sensor, adapt)` matches M5.1.
- `(2 sensors, fixed)` matches M5.2 hull.

The earlier agents are unchanged.

### Design change made after seeing data (post-hoc, disclosed)

The first design was `"halt"` only. The first study run showed `safe_halt` stopping in 228/360 `A_dead` runs, even though sensor B was healthy and plain hull recovered 360/360. Halting a two-sensor system because one sensor died is overly conservative, so I added `"isolate"` and made it the default. Both variants are reported below.

I then added the symmetric condition `A_dead_low` (stuck at the *pessimistic* values), because `A_dead` turned out not to test isolation at all: optimistic stuck values never affect the hull's worst corner. Both additions were made after the first run, and they are labelled as such.

## 4. Acceptance criteria (build environment)

1. All 297 earlier tests and V01–V155 pass, with the same meaning.
2. `SafeAgent` reproduces the M5, M5.1 and M5.2 agents per run (V156). The estimator is separate from execution and validates its inputs (V157).
3. The safe agents have 0 false stops in every condition without a common-mode fault (V158). The study totals are frozen and deterministic (V159).
4. A dead single sensor leads to `safe_halt` without success claimed. A dead A plus a healthy B with isolate leads to recovery, with A marked untrusted (V160).
5. Operator alerts are of known kinds and contain only numbers and short labels (V161). Policy compliance holds under combined faults across 4 policies (V162).

## 5. Results (build environment: Linux, Python 3.13)

- `python3 -m unittest discover -s tests -v` → **Ran 311 tests, OK** (297 + 14 new)
- `python3 -m red_queen.validation` → **162/162 validation checks passed**

`python3 -m red_queen.safety_layer_study`: assumed bound 5, 360 runs per cell. Each cell is false stops / successes; safe halts are noted where they occur.

| condition | single (M5) | single_adapt (M5.1) | hull (M5.2) | safe_halt | **safe** |
|---|---|---|---|---|---|
| none | 0 / 360 | 0 / 360 | 0 / 360 | 0 / 360 | **0 / 360** |
| wrong bound (true 15) | 10 / 350 | 4 / 356 | 0 / 360 | 0 / 360 | **0 / 360** |
| A temperature −10 | 27 / 333 | 27 / 333 | 0 / 360 | 0 / 360 | **0 / 360** |
| wrong bound + A temperature −10 | 43 / 317 | 25 / 335 | **2** / 358 | 0 / 360 | **0 / 360** |
| A dead (stuck 100 / 0) | **210** / 120 | 0 / 120, 228 halts | 0 / 360 | 0 / 140, 228 halts | **0 / 360** |
| A dead low (stuck 0 / 100) | 0 / 20 | 0 / 40, 81 halts | 0 / 20 | 0 / 40, 81 halts | **0 / 101** |
| both temperature −10 (common mode) | 27 / 333 | 27 / 333 | 16 / 344 | 16 / 344 | **16 / 344** |

Actions for the safe agent compared with hull: 2036 vs 2036 with no fault, 2452 vs 2197 with the wrong bound (+12%), and 2404 vs 2146 with the combined fault (+12%).

**What this shows**
- **The combination matters only for combined faults.** With a single fault, hull alone already had 0 false stops. Under "wrong bound + biased sensor", each protection alone failed: hull 2, single_adapt 25, single 43. Only the combined agent had 0. The cost is about 12% more actions when the adaptation triggers, and nothing when nothing is wrong.
- **For a single sensor, halting turns false stops into honest stops.** A dead single sensor caused 210 false stops for M5's agent. With the M5.3 halt it caused 0 false stops and 228 safe halts. M5.1 already detected these readings, but only counted them and carried on. M5.3 gives that detection a defined outcome.
- **Isolation recovers what halting gives up.** In `A_dead`, safe recovered 360/360 against 140/360 for safe_halt.
- **Limitation found: the detector explains away a pessimistically stuck sensor.** In `A_dead_low`, a jump to battery 0 / temperature 100 can be "explained" as a known disturbance plus a wide noise bound. So A's bound widened to 25 in 198 runs, instead of A being isolated (81 runs). The hull then stays pessimistic, and the agent acts until its budget runs out. It fails **safely**: 0 false stops, and recovery in only 101/360. A permissive explanation model trades fault isolation for tolerance.
- **Common-mode bias is unchanged:** 16/360 false stops, as in M5.2.

## 6. Expectations vs results

Self-registered before the first run:
- **"safe has the fewest false stops under every non-common-mode fault."** Confirmed: 0 everywhere, tied with others in some rows.
- **"safe halts on the dead sensor."** Confirmed for safe_halt. It also turned out to be undesirable when a second sensor is healthy, which led to the post-hoc isolate variant.
- **"Common mode still fails."** Confirmed.

Before running `A_dead_low` (post-hoc condition), I expected hull to fail safely and isolate to recover. Hull: confirmed (20/360, 0 false stops). Isolate: **only partly**, 101/360, for the reason in §5.

## 7. Limitations

- A common-mode bias remains undetected without a different kind of evidence.
- The explanation model is permissive: a cap of 25 plus known disturbances can absorb a stuck sensor. Possible next steps are cross-sensor evidence (isolate the sensor whose bound had to widen while the other's did not) or a lower cap.
- A halt or isolation is permanent within an episode. There is no re-admission of a recovered sensor.
- `safe_halt` and the alerts are not yet connected to any real operator channel. They are data in the result.
- The post-hoc changes (isolate, `A_dead_low`) are evaluated on the same scenarios that motivated them. They need confirmation on new conditions.

## 8. Linux acceptance (user's laptop)

**Pending.** Run in the repository folder:

```
git pull
python3 -m unittest discover -s tests -v && python3 -m red_queen.validation
python3 -m red_queen.safety_layer_study
```

M5.3 is accepted when the output shows `Ran 311 tests ... OK` and `162/162 validation checks passed`. The study table should match §5.

## 9. Next milestone (candidates)

1. **M5.4: cross-sensor fault isolation.** When two sensors disagree and only one of them needed its bound widened, isolate that one. Test it on *new*, pre-registered stuck and drift faults, so it is not tuned on `A_dead_low`.
2. **M5.5: information gathering.** Re-read before stopping, to measure whether it reduces the extra actions the safe agent costs.
3. **M6: integrate `SafeEstimator` with memory and the planner.** That means the M3.5 disturbance memory and M3.6 guard, under noisy sensors. It would be the first milestone combining state estimation and memory.
