# RED QUEEN AI — Engineering Record, Milestone M9.2

**Version:** 0.26.0 · **Date:** 2026-11-22 · **Status:** build-environment passed; Linux acceptance pending
**Outcome: pre-registered result, positive.** RED QUEEN learned its own "distance to goal" estimate from 1,200 labelled states drawn from past episodes, using linear ridge regression on 8 fixed features. On 100 **fresh** data-centre scenarios:
- planning with the learned estimate solved all 100, the same as the hand-written one;
- it needed **86% less search** (2,164 vs 15,110 nodes);
- it used slightly **fewer** actions (491 vs 500).

All three pre-registered criteria passed. An exploratory control shows the gain is **not just a matter of scale**: rescaling the hand-written estimate made it *worse* (26,010 nodes).

This is a fitted linear estimate. It is not "learning to think".

## 1. Review of M9

M9 was accepted on Linux on 2026-11-22, with study output identical by `diff`. It showed that the data-centre world leaves room for better decision-making. Both successful agents relied on one **hand-written** heuristic, `goal_distance`, which made that heuristic the natural place for learning to help.

## 2. Limitation addressed

RED QUEEN's guidance about "how far am I from the goal" was a fixed rule I wrote. It did not use any of the system's own experience.

## 3. Method: pre-registration

`docs/PREREG_M9_2.md` was committed before any M9.2 code (commit `e58b338`, 2026-11-22 10:47 UTC). It fixed:
- the 8 features, and the learner (ridge regression, λ = 0.01, exact solve)
- how labels are made: reference-plan length from states along deterministic random walks
- the **training seeds 2000–2199** and the **fresh evaluation seeds 3000–3099**
- the agents, criteria L1–L3, and the expectations

No pre-registered part was changed after the run. **One exploratory control was added afterwards.** It is labelled as such in the code and output and does not enter the decision (§6).

## 4. Changes

| Change | File |
|---|---|
| `reference_plan(..., heuristic=None)` and `greedy_action(..., heuristic=None)`: a pluggable heuristic. **The defaults are the hand-written `goal_distance`, so M9 is unchanged.** I re-ran the M9 study after the change and its output was identical by `diff` (V219). | `datacentre.py`, `datacentre_study.py` |
| `features`, `fit_ridge` (exact Gaussian elimination; the bias term is not penalised), and `LearnedHeuristic`, which clips predictions to ≥ 0 and falls back to the hand heuristic on non-finite values. Also `walk_states`, `labelled_states`, `train` and `pearson`. | `learned_heuristic.py` (new) |
| The pre-registered study, the decision rule as code, and the labelled exploratory control. | `heuristic_study.py` (new) |

**Safety:** the heuristic only changes the *order* in which the search looks at states. The executor (allowlist, parsing, preconditions, budget) and the verifier are unchanged. With deliberately corrupted weights (NaN, ±huge), every plan found still passed the executor and reached the goal (V221).

## 5. Acceptance criteria (build environment)

1. All 395 earlier tests and V01–V218 pass with the same meaning.
2. The plug-in defaults are identical to the explicit hand heuristic (V219). The ridge solver recovers exact weights and refuses bad input (V220). Garbage weights cannot break safety (V221).
3. The study is frozen, with values copied from the results file (V222). The pre-registered decision is applied (V223). The exploratory control is labelled and kept outside the decision (V224). The seed sets are pairwise disjoint across M9 and M9.2, and the pre-registration is present (V225).

## 6. Results (build environment: Linux, Python 3.13)

- `python3 -m unittest discover -s tests -v` → **Ran 409 tests, OK** (395 + 14 new); `python3 -X dev` run: **0 ResourceWarnings**
- `python3 -m red_queen.validation` → **225/225 validation checks passed**; `--quick` → **165/165**

`python3 -m red_queen.heuristic_study` (about 15 s):

**Learned weights** (from 1,200 labelled training states): bias 0.61; down servers 0.59; down heat over 60 0.56; up heat over 70 0.66; load over 40 0.91; unserved 0.75; max up temperature 0.03; heating servers 0.03.

**Accuracy on 600 held-out states:** mean absolute error 0.63 steps. Correlation with true distance: learned 0.930, hand 0.911.

| agent (fresh seeds 3000–3099) | solved | actions | planner nodes |
|---|---|---|---|
| A*, hand heuristic | 100 / 100 | 500 | 15,110 |
| **A*, learned heuristic** | **100 / 100** | **491** | **2,164** |
| greedy, hand | 95 / 100 | 595 | 0 |
| greedy, learned | 94 / 100 | 605 | 0 |

Safety: 0 actions bypassed the executor, and there were 0 heuristic fallbacks.

**Pre-registered decision** (A* only, computed by `decide`, V223):
- **L1** (successes not lower): **True**
- **L2** (nodes ≤ 80%): **True**, with a ratio of **0.143**
- **L3** (actions ≤ 105%): **True** (491 vs 500)
- **The learned heuristic is useful: yes.**

**Exploratory control** (added after the pre-registered run, not part of the decision). The hand heuristic multiplied by one constant fitted on the same training labels (0.722) solved 100/100 with 498 actions, but needed **26,010 nodes**, more than the original hand heuristic. So the learned gain does **not** come from merely putting the estimate on the right scale. It comes from **how the learned model weights the features**. For example, it weights excess load (0.91) more heavily than overheating (0.66).

**What this shows**
- **RED QUEEN's own experience made its planning much cheaper:** about 7× less search, with no loss in success and a small gain in plan length. It is the first clear, pre-registered case in this project of a learned component beating a hand-written one.
- **It helped search, not greedy.** Greedy with the learned estimate was slightly worse (94 vs 95). A better *ranking* across many states helps search, but does not necessarily improve a one-step choice. Greedy was not part of the decision, and the result is reported as it is.
- **Limits of the claim:** the labels come from a planner that uses the hand heuristic, so the model partly imitates it. It is a linear fit on 8 features I chose, in one small simulated world.

## 7. Limitations

- **This is not "learning to think".** It is a fitted linear estimate of distance to the goal.
- The training labels come from the hand-heuristic reference planner (stated in advance). The features were also chosen by hand.
- One world size (3 servers), exact observations, and 100 evaluation scenarios.
- Learning is offline (fit once, then frozen). There is no online updating yet.

## 8. Linux acceptance (user's laptop)

**Pending.** Run in the repository folder:

```
git pull
python3 -m unittest discover -s tests -v && python3 -m red_queen.validation
python3 -m red_queen.heuristic_study
```

M9.2 is accepted when the output shows `Ran 409 tests ... OK` and `225/225 validation checks passed`. The study output should match §6.

## 9. Next milestone (candidates)

1. **M9.3: noisy sensors in the data centre.** Connect the M5 safety layer and the M7 self-model, then check that the learned planner stays safe when it cannot see the exact state.
2. **M9.1: local language model as a plan proposer here.** Compare it against the learned planner, which is now a much stronger baseline.
3. **M9.4: a larger data centre** (4–5 servers), to test whether the learned heuristic still helps as the world grows.
