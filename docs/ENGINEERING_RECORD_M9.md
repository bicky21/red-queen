# RED QUEEN AI — Engineering Record, Milestone M9

**Version:** 0.25.0 · **Date:** 2026-11-06 · **Status:** build-environment passed; Linux acceptance pending
**Outcome: pre-registered result. The new data-centre world does leave room for better decision-making.**

On 100 never-before-run scenarios:
- **M3-style bounded search** solved **80**, then gave up on the other 20 because it hit its search limit.
- A **heuristic reference search** solved **100**, using **95× less search effort**.
- **Surprise:** a simple **greedy** agent with no search at all solved **95**. My pre-registered expectation that greedy would do poorly (D3) was wrong.
- **Safety held:** 0 actions bypassed the executor.

## 1. Review of M8.1

M8.1 was accepted on Linux on 2026-11-06, and the laptop test time dropped from 339 s to 43 s. A pattern ran across M4, M6 and M8: the language model, memory and global workspace all showed little benefit, because in the old world an exact BFS planner already does almost everything right. The user chose a harder "data-centre" world for M9.

## 2. Limitation addressed

The old world was too easy to tell whether smarter components help. M9 builds a harder, **coupled** world, and first measures whether it leaves room for improvement, before trying any component in it.

## 3. The world (`datacentre.py`, new and separate; the old domain is unchanged)

Three servers share airflow and a fixed demand of 120. Each server has a temperature, a load (in steps of 20) and an up/down flag.

| action | effect |
|---|---|
| `cool:i` | temperature −15 on server i, **+5 on every other up server** (shared air) |
| `migrate:i:j` | move 20 load from i to j (heat follows load) |
| `throttle:i` / `unthrottle:i` | drop or restore 20 load |
| `restart:i` | bring a down server up, only if its temperature is ≤ 60 |
| `wait` | nothing |

**Physics after every action:**
- Each up server: temperature += load/20 − 2.
- Each down server: temperature −5.
- An up server above 90 **crashes** and loses its load.

**Goal:** all servers up, every temperature ≤ 70, every load ≤ 40, and served = demand.

**Safety boundary (`DCExecutor`):**
- action-kind allowlist;
- strict string parsing, so hostile strings like `rm -rf /` are rejected;
- preconditions;
- a hard budget of 25 actions.

Nothing reaches the environment without passing all four. An independent `DCVerifier` judges the true state at the end.

## 4. Method: pre-registration, with design work disclosed

`docs/PREREG_M9.md` was committed (commit `1420a4c`, 2026-11-06 08:54 UTC) **before the study was run on its evaluation seeds 0–99**. Design work done before that, on **development seeds 1000–1039 only**, is disclosed there:
- **Physics fix.** At first, down servers never cooled, which made some scenarios impossible for an uninteresting reason. They now cool by 5 per step.
- **Development measurements:** BFS found plans in 33 of 40 development scenarios, and the weighted A* reference in 40 of 40.

Nothing was changed after the evaluation run.

## 5. Changes

| Change | File |
|---|---|
| The world: `DCState`, `parse_action`, `precondition`, `step`, `goal_met`, `goal_distance` (a heuristic for greedy and A* only, never for verification), `DCEnvironment`, `DCExecutor`, `DCVerifier` | `datacentre.py` (new) |
| `bfs_plan` (bounded breadth-first, shortest) and `reference_plan` (weighted A*, f = g + 2h) | `datacentre.py` |
| The pre-registered study: scenario generator, three closed-loop agents, and the decision rule as code | `datacentre_study.py` (new) |
| 7 validation checks (V212–V218). The study checks V216 and V217 (about 50 s) are full-run only; the quick/full partition test is updated. | `validation.py`, `tests/` |

## 6. Acceptance criteria (build environment)

1. All 381 earlier tests and V01–V211 pass with the same meaning. The old domain is untouched.
2. Invalid states and 12 malformed or hostile action strings are refused (V212). The physics matches hand-computed transitions: cooling spill, crash, down-server cooling, restart (V213). The executor blocks policy-denied, invalid, precondition-failing and over-budget actions before the environment, and refuses an unknown action kind in the allowlist (V214). Every plan the planners find, replayed on the model, reaches the goal (V215).
3. The evaluation totals are frozen (V216). The pre-registered decision is applied (V217). The pre-registration is present, and development and evaluation seeds are disjoint (V218).

## 7. Results (build environment: Linux, Python 3.13)

- `python3 -m unittest discover -s tests -v` → **Ran 395 tests, OK** (381 + 14 new); `python3 -X dev` run: **0 ResourceWarnings**
- `python3 -m red_queen.validation` → **218/218 validation checks passed**; `--quick` → **161/161**

`python3 -m red_queen.datacentre_study` (seeds 0–99, about 50 s):

| agent | solved | actions | planner nodes | how runs ended |
|---|---|---|---|---|
| bfs (M3-style, 20,000 nodes) | 80 / 100 | 316 | 702,077 | 80 goal, **20 no plan** |
| **reference (weighted A*)** | **100 / 100** | 483 | **7,388** | 100 goal |
| greedy (no search) | 95 / 100 | 570 | 0 | 95 goal, 5 out of budget |

- **Plan quality on the 80 scenarios both planners solved:** bfs used 316 actions and the reference 318. They are essentially equal, so the faster search is not sloppier.
- **Safety:** 0 actions bypassed the executor, 0 executor rejections, and at most 25 actions in any episode.

**Pre-registered decision** (computed by `decide`, V217):
- **D1** (reference solves ≥ 90): **True** (100)
- **D2** (bfs solves at least 10 fewer than the reference): **True** (80)
- **D3** (greedy solves at least 10 fewer than the reference): **False** (95)
- **S** (safety): **True**
- **This world has room for better decision-making: yes.**

**What this shows**
- **This world breaks the old planner.** Blind breadth-first search, which was enough before, ran out of search budget in 20% of scenarios. A guided search solved all of them with about 1% of the effort. For the first time in the project, *how* RED QUEEN decides makes a large difference.
- **My expectation about greedy was wrong.** I predicted greedy would do poorly because cooling one server heats the others. It solved 95/100, with 5 failures where it ran out of actions. The heuristic captures most of the structure. I record this as a failed expectation, not something to explain away.
- **The useful knowledge here is the heuristic**, a hand-written estimate of "distance to goal". Both successful agents depend on it. That is the natural place for learning, memory or a language model to try to help next.
- **No claim that RED QUEEN is "smarter".** M9 only shows that this world, unlike the old one, leaves room to improve, and where that room is.

## 8. Limitations

- Exact observations only. The M5 sensor layer is not yet connected to this world.
- One fixed size: 3 servers, demand 120. Larger versions would be harder still.
- The heuristic is hand-written, and the reference planner is not guaranteed optimal. Here, though, its plans matched BFS in length.
- 100 evaluation scenarios from one generator.

## 9. Linux acceptance (user's laptop)

**Pending.** Run in the repository folder:

```
git pull
python3 -m unittest discover -s tests -v && python3 -m red_queen.validation
python3 -m red_queen.datacentre_study
```

M9 is accepted when the output shows `Ran 395 tests ... OK` and `218/218 validation checks passed`. The study output, which takes about 1 minute, should match §7.

## 10. Next milestone (candidates)

1. **M9.1: can the language model help here?** In this world the planner is no longer optimal or cheap. Re-run the M4 proposal gate (untrusted proposals, strictly validated) with the local model, or fake proposers first. Does it reduce search effort or failures?
2. **M9.2: learning the heuristic from experience** (memory). Fit "distance to goal" from past episodes, and compare it with the hand-written one on held-out scenarios.
3. **M9.3: noisy sensors in the data centre.** Connect the M5 safety layer and the M7 self-model to this world.
