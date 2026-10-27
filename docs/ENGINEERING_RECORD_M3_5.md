# RED QUEEN AI — Engineering Record, Milestone M3.5

**Version:** 0.9.0 · **Date:** 2026-10-27 · **Status:** build-environment passed; Linux acceptance pending
**Outcome:** adopted as the recommended memory-planning configuration (opt-in flag `predict_disturbances=True`).

## 1. Review of M3.4

M3.4 (accepted on Linux, commit `effd462`) showed that allowing longer
plans gave no benefit, and located the remaining gap to `reactive_memory`
in cases **with disturbances**. The memory advisor ignored disturbance
events and the planner did not predict them.

## 2. Limitation addressed

Remembered disturbances (step and kind) were stored in memory since M2 but
never used. The planner's predictions therefore disagreed with reality
whenever a known disturbance happened, which caused unnecessary replans and
poorly ordered actions.

## 3. Changes

| Change | File |
|---|---|
| `MemoryAdvisor.remembered_disturbances()`: step → disturbance name, from **valid records of the same case**. Only known names and integer steps ≥ 1 are kept, and malformed events are counted and ignored. A disturbance is kept only if it appears in **at least half** of the used records (ties by count, then name). `discounts()` and `summary()` are unchanged. | `memory_advice.py` |
| `BoundedPlanner.plan(..., disturbances=fn)`: predictions include the expected disturbance after each step. Because transitions now depend on time, the search keys states by (depth, state) and keeps the max_depth/max_nodes limits. **If it finds no plan or runs out of nodes, it falls back to the plan it would make without disturbances.** Disturbance input is sanitised (unknown names, non-strings and exceptions are ignored). Slack is not applied in this mode. | `planner.py` |
| `PlanningAgent(predict_disturbances=True)`: requires an advisor with `remembered_disturbances()`, and maps plan depth to absolute step. If a remembered disturbance does not happen, the observation will not match the prediction and the agent replans (observation wins). Default `False` keeps the M3.3 behaviour exactly. | `planning_agent.py` |
| Study: `reactive_memory`, the M3.3 planners and the M3.5 planners (both orders), on 18 named scenarios and the 1,680-case grid, split by case type. | `disturbance_study.py` (new) |

## 4. Acceptance criteria

1. All earlier tests and V01–V90 pass with the same meaning.
2. Disturbances are extracted correctly, and the advisor summary is unchanged (V91). Malformed input is ignored (V92).
3. With prediction off, behaviour is identical to M3.3 (V93).
4. Correct predictions remove needless replans (V94). Impossible predictions fall back safely (V95).
5. **Adoption criterion:** both planner orders use fewer total actions than `reactive_memory`, with 100% success (V96). **Met.**
6. Fake disturbances at every step cannot block recovery (V97).
7. Policy compliance holds, and failures stop early with `no_plan_found` under all 4 policies (V98).
8. Results are deterministic (V99), and invalid configuration is refused (V100).

## 5. Results (build environment: Linux, Python 3.13.17)

- `python3 -m unittest discover -s tests -v` → **Ran 186 tests, OK** (170 earlier + 16 new)
- `python3 -m red_queen.validation` → **100/100 validation checks passed**, exit code 0, about 31 s

### Grid (1,680 cases, default policy, all agents 100% success; total actions)

| Agent | All | F+D (1,296) | F (216) | D (144) | none (24) |
|---|---|---|---|---|---|
| reactive_memory (previous best) | 8,134 | 6,552 | **812** | 690 | 80 |
| planner_memory (M3.3) | 8,271 | 6,677 | 852 | 662 | 80 |
| planner_reversed_memory (M3.3) | 8,219 | 6,632 | 819 | 688 | 80 |
| **planner_dist_memory (M3.5)** | **8,062** | 6,497 | 852 | **633** | 80 |
| **planner_reversed_dist_memory (M3.5)** | **7,857** | **6,325** | 819 | **633** | 80 |

Head-to-head (cases where each side used fewer actions):

| Pair | First fewer | Second fewer | Equal |
|---|---|---|---|
| planner_dist_memory vs reactive_memory | 287 | 218 | 1,175 |
| planner_reversed_dist_memory vs reactive_memory | 320 | 109 | 1,251 |
| planner_dist_memory vs planner_memory | 224 | 43 | 1,413 |
| planner_reversed_dist_memory vs planner_reversed_memory | 319 | 46 | 1,315 |
| planner_dist_memory vs planner_reversed_dist_memory | 91 | 221 | 1,368 |

### Findings (limited to this simulation)

1. **Predicting remembered disturbances helps, and the gain is not a tie-break artefact.** On disturbance-only cases (D) both orders reach exactly the same total (633, down from 662/688). Both orders beat `reactive_memory` overall: −72 actions (sorted) and −277 (reversed), i.e. 0.9% and 3.4%.
2. **Fault-only cases are unchanged** (852/819), as expected: there are no disturbances to predict. `reactive_memory` is still best there (812).
3. **Action order still matters when faults and disturbances combine** (F+D: 6,497 vs 6,325), because fault ties are still partly decided by order where memory has no information.
4. **There are regressions.** M3.5 used more actions than M3.3 in 43 (sorted) and 46 (reversed) of 1,680 cases. A changed plan can run into faults the memory never saw, because the prior reactive run never tried that action at that step.
5. **Safety held.** Fake disturbances at every step recovered in all 54 episodes, needing at most 9 actions (V97).

**Fairness note:** the M3.5 planners use disturbance timing, which the
reactive scorer cannot use: one-step scoring cannot plan around a future
event. This result shows the value of model-based prediction, not a
like-for-like algorithm swap.

### Decision

Adopted as the **recommended** configuration for memory-assisted planning:
`PlanningAgent(..., advisor=MemoryAdvisor(...), predict_disturbances=True)`.
It stays **opt-in** (default `False`), so all earlier results and checks
are unchanged.

Data: `results/m3_5_disturbance_study.json`.

## 6. Limitations

- **Memory is perfectly predictive in this study.** The prior episode comes from the same deterministic case, so remembered disturbances always recur at the same step. Real experience is noisier. This is the biggest open question.
- One prior episode per case; the majority rule has only been exercised in unit tests.
- Disturbance prediction is by absolute step number only.
- Slack is not combined with disturbance prediction.

## 7. Linux acceptance (user's laptop)

**Pending.** Run in the repository folder:

```
git pull
python3 -m unittest discover -s tests -v && python3 -m red_queen.validation
```

M3.5 is accepted when the output shows `Ran 186 tests ... OK` and
`100/100 validation checks passed`. The full run takes about 1–2 minutes.

Optional (about 10–20 seconds):

```
python3 -m red_queen.disturbance_study
```

## 8. Next milestone

**M3.6 — imperfect memory.** Measure how M3.3/M3.5 behave when memory is
*wrong or noisy*. The prior episodes will have shifted fault windows and
shifted, missing or extra disturbances, at several noise levels. Measure
degradation against `reactive_memory` and against no memory, and confirm
every safety check still holds. The goal is to find the noise level where
memory stops helping, and to adopt a guard if needed (for example, ignore
memory after its first wrong prediction in an episode).
