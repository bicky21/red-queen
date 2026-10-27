# RED QUEEN AI — Engineering Record, Milestone M3.4

**Version:** 0.8.0 · **Date:** 2026-10-27 · **Status:** build-environment passed; Linux acceptance pending
**Outcome:** negative result. Slack planning is implemented and tested but **not adopted** (default `slack=0`).

## 1. Review of M3.3

M3.3 (accepted on Linux, commit `5c2e64b`) let memory break ties between
shortest plans. On the 1,680-case grid, the reactive agent with memory
(8,134 actions) remained more efficient than both memory planners (8,271
sorted, 8,219 reversed). The M3.3 hypothesis was that the planner was held
back by being restricted to *shortest* plans.

## 2. Hypothesis tested

Allowing plans up to *k* steps longer, chosen by lowest **expected cost =
length + summed remembered risk**, will reduce total actions and close the
gap to `reactive_memory`.

## 3. Changes

| Change | File |
|---|---|
| `BoundedPlanner(slack=k)`, k an int in [0, 3] (`MAX_SLACK`), validated. | `planner.py` |
| With a risk function and k > 0: find the shortest plan (M3.3), then search plans of length L..L+k by layers (states may repeat across depths) and return the lowest expected cost. Ties prefer the shorter plan, then discovery order. Goal states are not expanded. | `planner.py` |
| Safety: slack runs only **after** a shortest plan exists, so reachability is unchanged. It has its own node budget, and when that runs out it **falls back to the shortest plan** (V89). k = 0, or no risk function, gives exactly the M3.3/M3 behaviour (V84). | `planner.py` |
| Slack study: `reactive_memory` vs planner_memory k = 0..3 (sorted) and k = 0, 3 (reversed); the grid is split by case type. | `slack_study.py` (new) |

## 4. Acceptance criteria

1. All earlier tests and V01–V82 pass with the same meaning.
2. Slack is validated (V83); k = 0 and no risk are identical to before (V84).
3. Slack plans are model-consistent, end at the goal without passing through it, and stay within L + k (V85).
4. Slack changes a plan only when its expected cost is strictly lower (V86).
5. Impossible-goal detection, poisoned-memory recovery and node-budget fallback are preserved (V88–V90).
6. **Adoption criterion:** fewer total actions than `reactive_memory` on the grid. **Not met.**

## 5. Results (build environment: Linux, Python 3.13.17)

- `python3 -m unittest discover -s tests -v` → **Ran 170 tests, OK** (158 earlier + 12 new)
- `python3 -m red_queen.validation` → **90/90 validation checks passed**, exit code 0

**Two checks failed on their first run** and were corrected. Both failures are recorded honestly:

- **V86** originally asserted "slack never changes a plan under random risk". It found **1 counterexample in 1,800**, so the assertion was wrong. The check now tests the correct property (a change only happens when expected cost is strictly lower) and freezes the rate at 1/1,800. The case is kept as a unit test.
- **V89**'s node budget (400) was smaller than the 528 nodes the M3.3 search itself needs, so the test never reached the fallback. The budget is now measured, and the check confirms the fallback actually ran.

### Slack study (grid, 1,680 cases, all agents 100% success; total actions)

| Agent | All | F+D (1,296) | F (216) | D (144) | none (24) |
|---|---|---|---|---|---|
| **reactive_memory** | **8,134** | **6,552** | **812** | 690 | 80 |
| planner_memory_k0 (M3.3) | 8,271 | 6,677 | 852 | **662** | 80 |
| planner_memory_k1 | 8,271 | 6,677 | 852 | 662 | 80 |
| planner_memory_k2 | 8,271 | 6,677 | 852 | 662 | 80 |
| planner_memory_k3 | 8,271 | 6,677 | 852 | 662 | 80 |
| planner_reversed_memory_k0 | 8,219 | 6,632 | 819 | 688 | 80 |
| planner_reversed_memory_k3 | 8,219 | 6,632 | 819 | 688 | 80 |

F = action fault, D = disturbance. Across all grid and named episodes,
slack changed the action count **0 times**.

### Why slack does not help here

1. **A detour step costs exactly as much as a failed attempt.** In this simulation a blocked action wastes one step, and so does any extra plan step. Remembered risk per step is capped at 0.9 < 1. A longer plan only wins when every shortest plan must pass through **two or more** risky steps whose combined risk is above 1. That happened in 1 of 1,800 random synthetic risk patterns (V86) and **never** with the risk the memory actually produced.
2. **The remaining gap is from disturbances, not faults.** On fault-only cases, `reactive_memory` and the reversed memory planner are close (812 vs 819). Most of the gap is in cases with disturbances (F+D: 6,552 vs 6,632/6,677). The advisor ignores disturbance events, and the planner does not predict them. Which agent does better there depends on action order relative to unmodelled events. One example: cooling *after* a heat spike wastes less, because temperature is capped at 100.
3. On disturbance-only cases the sorted planner is actually best (662 vs 690), which confirms that order-vs-disturbance timing is luck rather than skill.

### Decision

Slack is **not adopted**: the default stays `slack=0`. The code is kept
because it is bounded and tested, it reproduces this negative result, and
it becomes relevant if a future fault model makes failed attempts cost more
than one step (for example, a blocked restart that still drains battery).

Data: `results/m3_4_slack_study.json`.

## 6. Limitations

- Expected cost counts a blocked attempt as `risk × 1 step`. Replanning overhead and knock-on effects are not modelled.
- Memory is one prior reactive episode per case, as in M3.3.
- The random-risk property test uses synthetic risk tables, not real memory.

## 7. Linux acceptance (user's laptop)

**Pending.** Run in the repository folder:

```
git pull
python3 -m unittest discover -s tests -v && python3 -m red_queen.validation
```

M3.4 is accepted when the output shows `Ran 170 tests ... OK` and
`90/90 validation checks passed`. The full run takes about 30–60 seconds.

Optional (about 35–60 seconds):

```
python3 -m red_queen.slack_study
```

## 8. Next milestone

**M3.5 — remembered disturbances.** The measured gap is in cases with
disturbances, so let the advisor also report remembered disturbances
(step and kind, from valid records of the same case). Let the planner
include them in its **predictions**, as advisory and bounded input that
observation overrides. Compare against `reactive_memory` on the same grid,
with D and F+D reported separately. Adopt only if it beats that baseline
without weakening any safety check.
