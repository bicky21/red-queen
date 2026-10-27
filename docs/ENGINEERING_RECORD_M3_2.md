# RED QUEEN AI — Engineering Record, Milestone M3.2

**Version:** 0.6.0 · **Date:** 2026-10-27 · **Status:** build-environment passed; **accepted on Linux (user's laptop), 2026-10-27**

## 1. Review of M3

M3 (accepted on Linux, commit `d10af59`) reported that the planner used
33 actions vs 35 for the reactive agent under the default policy. The M3
record flagged this as possibly caused by the planner's tie-break (actions
expanded in alphabetical order), not by planning. It was measured on only
10 scenarios.

## 2. Limitation addressed

1. Too few scenarios to compare the agents.
2. No control to separate planning effects from tie-break effects.

## 3. Changes

| Change | File |
|---|---|
| `BoundedPlanner(action_order=...)`: the expansion order is configurable and validated (must be a permutation of the allowlist). The default (sorted) is unchanged, and order never changes plan length (V66). | `planner.py` |
| 8 extended scenarios (`EXTENDED_SCENARIOS`), kept out of `SCENARIOS` so all earlier experiments keep identical inputs and results. `ALL_SCENARIOS` = 18. Duplicate names are refused at import. | `scenarios.py` |
| `wait_only` policy (`ALL_POLICIES`), kept out of `POLICIES` for the same reason. | `policy.py` |
| Scenario study: 3 agents (reactive, planner, **planner_reversed** as the control). Part A: 18 scenarios × 4 policies = 216 runs. Part B: a systematic fault grid of 24 start states × 10 fault windows × 7 disturbance settings = 1,680 cases. | `scenario_study.py` (new) |
| Validation V64–V73, including frozen grid totals (V69) and frozen M3 results (V73). | `validation.py` |

The extended scenarios include cases picked from the Part B grid because
the agents disagree there. This is deliberate (they are stress cases), so
Part A alone over-represents disagreements. Part B is the unbiased measure.

## 4. Acceptance criteria

1. All earlier tests and V01–V63 pass with the same meaning, and M3 experiment totals are unchanged (V73).
2. All 3 agents recover all 18 scenarios under the default policy (V65).
3. Plan length is unaffected by action order (V66), and invalid orders are refused (V67).
4. No executed action falls outside the policy in any of the 216 runs (V71).
5. The planner detects the `wait_only` impossible goal (V70).
6. Study results are deterministic and grid totals are frozen (V69, V72).

## 5. Results (build environment: Linux, Python 3.13.17)

- `python3 -m unittest discover -s tests -v` → **Ran 140 tests, OK** (121 earlier + 19 new)
- `python3 -m red_queen.validation` → **73/73 validation checks passed**, exit code 0

### Finding 1 — Without faults, the reactive agent is already optimal

Over all 882 fault-free start states (battery and temperature in steps of
5, service up or down), the reactive agent's action count equals the
shortest possible plan in **every** case (V68). In this model, look-ahead
cannot help unless something unexpected happens.

### Finding 2 — The planner's advantage is mostly tie-breaking

Part B, 1,680 fault/disturbance cases, all agents 100% success:

| Agent | Total actions |
|---|---|
| reactive | 8,741 |
| planner (sorted order) | **8,428** |
| planner_reversed | 8,713 |

Head-to-head (cases where each side used fewer actions):

| Pair | First fewer | Second fewer | Equal |
|---|---|---|---|
| reactive vs planner | 325 | 499 | 856 |
| reactive vs planner_reversed | 108 | 122 | 1,450 |
| planner vs planner_reversed | 552 | 366 | 762 |

With reversed order, the planner is within 0.3% of the reactive agent
(8,713 vs 8,741). With sorted order it saves 3.6%. Both planners find
plans of equal length every time, so **the 3.6% comes from which shortest
plan is picked, not from planning.** Sorted order happens to try `charge`
first. That fills time usefully while restart faults are active, but it
is luck relative to these fault windows, not knowledge of them. Even the
sorted planner uses more actions than the reactive agent in 325 of 1,680
cases.

### Finding 3 — The planner's real benefit is impossible-goal detection

Part A, actions spent in episodes that failed:

| Policy | reactive | planner | planner_reversed |
|---|---|---|---|
| no_restart | 240 | 3 | 3 |
| no_cool | 200 | 4 | 4 |
| wait_only | 340 | 0 | 0 |

Success counts are identical across all three agents under every policy.

### Default policy, extended scenarios (actions: reactive / planner / reversed)

| Scenario | R | P | P-rev |
|---|---|---|---|
| battery_empty | 3 | 3 | 3 |
| max_heat_service_down | 5 | 5 | 5 |
| deep_battery_flaky_restart | 7 | **4** | 7 |
| flaky_restart_then_crash | 8 | **4** | 8 |
| flaky_restart_heat_spike | 9 | **6** | 9 |
| cool_outage_battery_drain | **6** | 9 | 8 |
| charger_outage_overheating | **5** | 7 | **5** |
| double_disturbance | 8 | 8 | 8 |

Data: `results/m3_2_scenario_study.json`.

## 6. Conclusions (limited to this simulation)

- The M3 claim "planner saves actions" should be read as "this tie-break saves actions on these fault windows". The engineering record for M3 already warned about this. M3.2 confirms it with a control.
- Planning is justified here for **impossible-goal detection** and for having an explicit plan to check against observations. It is not justified by efficiency.
- A principled tie-break is needed. The obvious candidate is the M2.1 memory advisor, which already estimates which actions are likely to be blocked.

## 7. Limitations

- The grid uses one fault and at most one disturbance per case, all starting at step 1 or 2.
- Only two expansion orders were tested (sorted and reversed), not all 24 permutations.
- Extended scenarios were partly chosen *because* the agents disagree there, so Part A is not a fair sample.
- The model is still exact apart from injected faults. There is no sensor noise.

## 8. Linux acceptance (user's laptop)

**Accepted, 2026-10-27.** After `git pull` to commit `737e0de`, the user ran:

```
python3 -m unittest discover -s tests -v && python3 -m red_queen.validation
```

User-reported output: `Ran 140 tests ... OK` and `73/73 validation checks passed`,
including V69 with the same frozen grid totals (8741 / 8428 / 8713).

## 9. Next milestone

**M3.3 — memory-informed planning.** Let the planner break ties between
equal-length plans using the M2.1 advisor (prefer plans whose early steps
avoid remembered faults). Keep it bounded and advisory, and keep
observation authoritative. Compare it against both fixed orders on the
same 1,680-case grid, so any gain must beat the tie-break control.
