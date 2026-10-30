# RED QUEEN AI — Engineering Record, Milestone M5.2

**Version:** 0.16.0 · **Date:** 2026-10-30 · **Status:** build-environment passed; **Linux acceptance passed 2026-10-30**
**Outcome:** a second, independent sensor closes the bias gap found in M5.1 **when only one sensor is faulty**. The "hull" fusion had **0 false stops in 1,800 runs** across no fault and four single-sensor bias conditions. The single-sensor agent had up to 152 false stops in 360 runs. It costs 1.3% more actions than the single-sensor agent with no fault (2036 vs 2009), and 6–18% more under bias. There, the single sensor's lower count comes partly from stopping too early. A bias shared by **both** sensors (common mode) is still undetected: hull 16, single 27, and intersect 60 false stops in 360 runs.

## 1. Review of M5.1

M5.1 was accepted on Linux on 2026-10-30, with study tables identical to the build environment. Detecting a wrong noise bound helped, but a constant sensor bias was **invisible**: a temperature sensor reading 10 too low caused 27 to 45 false stops in 360 runs, with or without adaptation.

## 2. Limitation addressed

With one sensor there is no independent reference, so a bias cannot be detected or tolerated.

## 3. Changes

| Change | File |
|---|---|
| `RedundantBeliefAgent(goal, noise, fusion)` reads two sensors every step and fuses their observation boxes. **`intersect`** uses A ∩ B and halts with `sensor_fault_detected`, claiming nothing, when the boxes do not overlap. **`hull`** uses the smallest box containing both. **`latch`** intersects until the first disagreement, then uses the hull. Planning from the worst corner, executing one action through the `PolicyExecutor`, and stopping only when every state in the belief meets the goal are all as in M5. The same sensor object passed twice is refused. | `redundant_agent.py` (new) |
| `hull(a, b)` | `redundant_agent.py` |
| A study with 6 conditions (none; A temperature −10; B temperature −10; A temperature −20; A battery +10; both temperature −10) × 4 agents (single, intersect, hull, latch) × 18 scenarios × 20 seeds. Sensor B uses an independent noise stream. | `redundancy_study.py` (new) |

Nothing earlier was modified apart from the version string.

**Why hull cannot be fooled by one faulty sensor.** If at least one sensor is within its bound, the truth is inside that sensor's box, and therefore inside the hull. Stopping requires every state in the hull to meet the goal, so the truth meets it. V149 checked the fusion rules on 64,827 cases. This holds under two stated assumptions: **at most one sensor is faulty**, and **the noise bound is known**.

## 4. Acceptance criteria (build environment)

1. All 285 earlier tests and V01–V148 pass with the same meaning.
2. The fusion rules are sound: the hull contains every state in either box, and the intersection every state in both (V149).
3. At noise 0, with two exact sensors, every fusion recovers 18/18 with 79 actions, the same as M3 (V150).
4. Hull has 0 false stops and 100% success under no fault and every single-sensor bias (V151). The study totals are frozen and deterministic (V152).
5. Common-mode bias is measured and not hidden (V153). A fault halt never claims success or exceeds the budget, and invalid configurations are refused (V154).
6. Policy compliance under one biased sensor, across 4 policies and 3 fusions: 0 actions outside policy, 0 over budget (V155).

## 5. Results (build environment: Linux, Python 3.13)

- `python3 -m unittest discover -s tests -v` → **Ran 297 tests, OK** (285 + 12 new)
- `python3 -m red_queen.validation` → **155/155 validation checks passed**

`python3 -m red_queen.redundancy_study`: noise 5 per sensor, bound 5, 360 runs per row. Each cell is false stops / actions; for intersect, the number of fault halts is also given.

| condition | single (sensor A) | intersect | hull | latch |
|---|---|---|---|---|
| none | 0 / 2009 | 0 / 1954 | 0 / 2036 | 0 / 1954 |
| A temperature −10 | **27** / 1914 | 1 / 427, 325 halts | **0** / 2024 | 1 / 2019 |
| B temperature −10 | 0 / 2009 | 2 / 467, 326 halts | **0** / 2024 | 2 / 2012 |
| A temperature −20 | **152** / 1790 | 0 / 0, 360 halts | **0** / 2024 | 0 / 2024 |
| A battery +10 | 0 / 1710 | 0 / 441, 322 halts | **0** / 2017 | 0 / 1985 |
| both temperature −10 | 27 / 1914 | **60** / 1846 | **16** / 1939 | **60** / 1846 |

**What this shows**
- **Hull is the safe default against a single faulty sensor.** It had 0 false stops in every single-fault condition. It costs 1.3% more actions than the single sensor with no fault (2036 vs 2009). The B −10 row confirms the symmetry: the single agent reads only A, so it is unaffected there.
- **Intersect is safe but not useful under a fault.** It halted in about 90% of faulty runs, and only 17–19% recovered. It still had 1–2 false stops, where the boxes happened to overlap while one was wrong. With no fault it is the cheapest, using 2.7% fewer actions than single. The 20 "successes" in the A −20 row are the nominal scenario: the system was already healthy when the agent halted, and the agent claimed nothing.
- **Latch is almost hull, but not quite.** Before the first disagreement it trusts the intersection. That gave 1–2 false stops, the same weakness as M5's filter.
- **Common-mode bias is not solved.** The two sensors agree, so there is nothing to detect (0 disagreements). Tighter fusion makes it *worse*: intersect has 60 false stops against 27 for a single sensor. Hull is least bad at 16, but still unsafe.

## 6. Pre-registered expectations vs results

Self-registered before the study was run:
- **"Hull has 0 false stops with one biased sensor."** Confirmed.
- **"Intersect has fewer false stops than single, but more than 0."** Confirmed: 0–2 against 0–152. Its high halt rate, about 90%, was not anticipated.
- **"A common-mode bias defeats all of them."** Confirmed: every agent had false stops, and intersect and latch were worse than single.

## 7. Limitations

- **At most one sensor may be faulty.** With two sensors, a common-mode fault cannot be told apart from correct readings. Defending against it needs a different kind of evidence: a sensor of a different type, or a known reference state.
- The noise bound is assumed known. M5.1's adaptation is not combined with redundancy here.
- Both sensors have the same noise level. There is no voting with 3 or more sensors.
- When intersect halts, nothing is recovered. A real system would need a defined safe state and an operator alert.
- One monotone domain with uniform noise, as before.

## 8. Linux acceptance (user's laptop)

**Passed 2026-10-30.** On the user's Ubuntu laptop (Python 3.14):
`python3 -m unittest discover -s tests -v` → **Ran 297 tests in 114.075s, OK**;
`python3 -m red_queen.validation` → **155/155 validation checks passed**;
`python3 -m red_queen.redundancy_study` printed a table **identical** to §5.

## 9. Next milestone (candidates) — M5.3 chosen

1. **M5.3: consolidate the safety layer.** One `SafeEstimator` that combines adaptation (M5.1) and hull fusion (M5.2), with a defined **safe halt**: a named stop and an operator report when the evidence is contradictory. Re-run all sensor studies against it as a regression baseline.
2. **M5.4: information gathering.** Re-read before stopping, and measure whether it removes the remaining false stops of the latch and adaptive agents at a small action cost.
3. **Common-mode defence:** a reference check, such as a known state at the start, or a differently built sensor.
