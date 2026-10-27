# RED QUEEN AI — Engineering Record, Milestone M3.3

**Version:** 0.7.0 · **Date:** 2026-10-27 · **Status:** build-environment passed; **accepted on Linux (user's laptop), 2026-10-27**

## 1. Review of M3.2

M3.2 (accepted on Linux, commit `737e0de`) showed that the planner's
efficiency gain over the reactive agent came mostly from its alphabetical
tie-break. On the 1,680-case grid, the planner in sorted order used 8,428
actions and in reversed order 8,713, against 8,741 for the reactive agent.

## 2. Limitation addressed

Equal-length plans were chosen by an arbitrary rule (action order).
RED QUEEN already has remembered fault data (M2.1 advisor) that could make
that choice on evidence instead.

## 3. Design

```
MemoryAdvisor.discounts(step) --(drop actions observed to work)--> risk(depth)
                                                                      |
BoundedPlanner._plan_min_risk: layer-by-layer BFS; per state keep lowest summed risk;
                               stop at the FIRST layer containing a goal -> shortest length
```

| Rule | Where |
|---|---|
| `plan(..., risk=None)` runs the original M3 search unchanged. | `planner.py` |
| With `risk`, the plan is always of the **shortest** length: risk only chooses among equal-length plans. Reachability and the allowed action set are unaffected (V75). | `planner.py` |
| Risk input is untrusted: non-allowed and unknown actions are dropped, values are clamped to [0, 0.9], and NaN, non-numbers and non-mappings are ignored (V76). | `planner.py` |
| Equal risk keeps discovery order, so zero risk gives exactly the M3 plan (V74). | `planner.py` |
| `PlanningAgent(advisor=...)`: risk for depth d = advisor discount at absolute step (actions so far + d + 1). An action whose observed result matched the prediction in this episode loses its risk (V81). | `planning_agent.py` |
| Plan validation, policy enforcement at the executor, and all step/replan/denial bounds are unchanged. | — |

## 4. Experiment

`memory_planning_study.py`: for each case (18 named scenarios and the
1,680-case M3.2 grid), a fresh in-RAM memory holds **one prior episode of
the same case, run by the reactive agent**. Six agents use it: reactive and
two planner orders, each with and without memory.

## 5. Acceptance criteria

1. All earlier tests and V01–V73 pass with the same meaning.
2. Zero risk and empty memory reproduce M3 plans and episodes exactly (V74).
3. Risk never changes plan length or reachability (V75), and malformed risk is handled (V76).
4. Risk selects the lower-risk plan when one exists (V77).
5. Grid totals are frozen, and memory improves **both** planner orders, with 100% success (V78).
6. Poisoned memory cannot prevent recovery (V79), and invalid records are ignored (V80).
7. Observation overrides memory inside the planner (V81).
8. Results are deterministic (V82).

## 6. Results (build environment: Linux, Python 3.13.17)

- `python3 -m unittest discover -s tests -v` → **Ran 158 tests, OK** (140 earlier + 18 new)
- `python3 -m red_queen.validation` → **82/82 validation checks passed**, exit code 0

### Fault grid (1,680 cases, default policy, all agents 100% success)

| Agent | Total actions | Blocked attempts |
|---|---|---|
| reactive | 8,741 | 1,224 |
| **reactive_memory** | **8,134** | **512** |
| planner | 8,428 | 1,100 |
| planner_reversed | 8,713 | 1,208 |
| planner_memory | 8,271 | 917 |
| planner_reversed_memory | 8,219 | 652 |

Head-to-head (cases where each side used fewer actions):

| Pair | First fewer | Second fewer | Equal |
|---|---|---|---|
| planner_memory vs planner | 140 | 5 | 1,535 |
| planner_reversed_memory vs planner_reversed | 367 | 14 | 1,299 |
| planner_memory vs planner_reversed_memory | 259 | 307 | 1,114 |
| planner_memory vs reactive_memory | 172 | 296 | 1,212 |
| reactive_memory vs reactive | 455 | 2 | 1,223 |

### Findings (limited to this simulation)

1. **Memory-informed tie-breaking works and beats the tie-break control.** Both planner orders improve: −157 actions (sorted) and −494 (reversed). The gap between the two orders shrinks from 285 actions to 52, so memory rather than order now decides most ties.
2. **It is not free of regressions.** The memory planners use more actions than their no-memory versions in 5 and 14 cases. The memory comes from one reactive run, so it only knows about actions the reactive agent tried at those steps, and fault timing shifts once the plan changes.
3. **The simplest memory user is the most efficient.** The reactive agent with memory (M2.1) uses the fewest actions (8,134) and the fewest blocked attempts (512), and beats planner_memory in 296 vs 172 cases. The cause is structural: by design, M3.3 memory may only choose among the **shortest model plans**. The reactive scorer can step off the shortest path to avoid a remembered fault, and the planner cannot.
4. The planner still keeps its unique advantages from M3/M3.2: impossible-goal detection and an explicit plan to verify against observations.

Data: `results/m3_3_memory_planning_study.json`.

## 7. Limitations

- Memory is one prior episode per case, from one agent type. More, or more varied, experience could change the ranking.
- Risk is indexed by absolute step, so it assumes faults recur at the same step numbers.
- The shortest-length constraint is a deliberate safety and simplicity choice, and it caps the benefit (finding 3).
- No sensor noise; the model is exact apart from injected faults and disturbances.

## 8. Linux acceptance (user's laptop)

**Accepted, 2026-10-27.** After `git pull` to commit `5c2e64b`, the user ran:

```
python3 -m unittest discover -s tests -v && python3 -m red_queen.validation
```

User-reported output: `Ran 158 tests in 8.999s ... OK` and `82/82 validation checks passed`,
including V78 with identical frozen grid totals.

## 9. Next milestone

**M3.4 — bounded expected-cost planning.** Allow the planner to accept a
plan up to *k* steps longer (k = 1 or 2) when its expected number of blocked
attempts, from memory, is lower. Keep it bounded: policy, validation and
step limits stay the same, and k = 0 must reproduce M3.3 exactly. Compare
it against `reactive_memory`, the current best, on the same grid. Adopt it
only if it beats that baseline without losing impossible-goal detection.
