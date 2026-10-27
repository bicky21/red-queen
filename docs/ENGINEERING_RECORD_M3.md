# RED QUEEN AI — Engineering Record, Milestone M3

**Version:** 0.5.0 · **Date:** 2026-10-24 · **Status:** build-environment passed; **accepted on Linux (user's laptop), 2026-10-27**

## 1. Review of M2.1

M2.1 (accepted on Linux, commit `4d93c4a`) chooses one action at a time by
greedy scoring. It had no explicit policy layer and no way to recognise
that a goal is unreachable, so under a restricted action set it would keep
acting until the step limit.

## 2. Limitations addressed

1. No look-ahead: decisions are one step deep.
2. No policy enforcement beyond the environment's fixed allowlist.
3. No plan to compare against observations, so no explicit replanning.

## 3. Changes

| Change | File |
|---|---|
| `Policy`: explicit allowlist, validated against the environment's actions. Three named policies: `default`, `no_restart`, `no_cool`. | `policy.py` (new) |
| `PolicyExecutor`: the only path to the environment. Checks the policy **and** an action budget at the execution boundary; denials raise `PolicyDenied` and are logged. | `policy.py` |
| `BoundedPlanner`: breadth-first search over allowed actions using the transition model; shortest plan; depth limit 10, node limit 20,000; deterministic order; predicts only. | `planner.py` (new) |
| `PlanningAgent`: Observe → Plan → **Validate** (every action allowed, predictions consistent) → Act via executor → compare observation with prediction → replan on mismatch. Bounded by `max_steps`, `max_replans`, `max_policy_denials`; each bound has a named stop reason. | `planning_agent.py` (new) |
| Reactive agent: filters its ranking by policy and acts through `PolicyExecutor`. With the default policy its behaviour is identical to M2.1 (V51). Stops with `no_allowed_action` if the policy leaves nothing. | `agent.py` |
| `EpisodeResult.stats`: agent, policy, filtered/denied counts, plans, replans, mismatches, nodes expanded. | `agent.py` |
| Experiment: reactive vs planner × 3 policies × 10 scenarios (60 runs). | `planning_experiment.py` (new) |
| Fix: `EpisodeMemory` now closes its connection when it refuses a corrupt file (a `ResourceWarning` appeared in the test run). | `memory.py` |

## 4. Safety properties

- A denied action cannot reach the environment: it is filtered at decision time (reactive), rejected at plan validation (planner), and refused at the execution boundary (both). V60: 0 denied actions executed across 60 runs.
- A planner proposing a denied or malformed plan is rejected before any action runs, and the episode stops after the denial limit (V57).
- The action budget is enforced inside the executor, not only by the agent loop (V58).
- Policies naming unknown actions are refused at construction (V59).
- Planner search is bounded and reports when it hit its node limit (V56).

## 5. Acceptance criteria

1. All earlier tests and V01–V50 pass with the same meaning; reactive behaviour under the default policy is unchanged (V51), and M2.1 memory results are unchanged (V63).
2. The planner recovers all 10 scenarios under the default policy (V52).
3. Plans are shortest, checked against an independent exhaustive search (V53).
4. The planner replans when observations disagree with predictions (V54).
5. Planner runs are deterministic (V55), and the search is bounded (V56).
6. The policy safety properties in section 4 hold (V57–V60).
7. Impossible goals are detected by the planner (V61).
8. The experiment is complete and deterministic (V62).

## 6. Results (build environment: Linux, Python 3.13.17)

- `python3 -m unittest discover -s tests -v` → **Ran 121 tests, OK** (94 earlier + 27 new)
- `python3 -m red_queen.validation` → **63/63 validation checks passed**, exit code 0

Note: on the first run, V57 failed with a `NameError` (a missing import in
the check itself). The validator counted the crash as a failure, as
designed. The import was fixed and the check re-run.

### Planning experiment (60 runs)

| Policy / agent | Successes | Total actions | Actions in failed episodes | Replans | Nodes |
|---|---|---|---|---|---|
| default / reactive | 10/10 | 35 | 0 | 0 | 0 |
| default / planner | 10/10 | **33** | 0 | 8 | 551 |
| no_restart / reactive | 4/10 | 128 | 120 | 0 | 0 |
| no_restart / planner | 4/10 | **11** | **3** | 4 | 651 |
| no_cool / reactive | 6/10 | 97 | 80 | 0 | 0 |
| no_cool / planner | 6/10 | **16** | **1** | 7 | 379 |

**What this shows, limited to these scenarios:**

- **Same success rate everywhere.** The planner solves nothing the reactive agent cannot. Every failure is a goal that is impossible under the policy.
- **Main measured benefit: recognising impossible goals.** Under `no_restart`/`no_cool`, the planner stops with `no_plan_found` after 0–3 actions. The reactive agent spends all 20 steps first: 200 wasted actions in total vs 4.
- **Default policy: 33 vs 35 actions.** The whole difference is `flaky_restart_low_battery` (4 vs 6). This is **not** foresight about the fault. The planner doesn't model faults. Its shortest plan happens to put `charge` first (actions are expanded in name order), so the charges fall inside the restart fault window. A different tie-break could erase this.
- **Cost:** the planner did 1,581 state expansions in total; the reactive agent did none. Both are trivial at this scale.

Data: `results/m3_planning_experiment.json`.

## 7. Limitations

- The planner uses the same exact model as the scorer. It has no model of faults or disturbances, so replanning is its only defence.
- BFS treats all actions as equal cost and ignores action costs.
- Tie-breaking by action name affects which shortest plan is chosen, and that changed the result in one scenario.
- The planner ignores memory and affect. Combining them is left for later milestones.
- The state space is tiny. Node limits were never hit outside the dedicated test.

## 8. Linux acceptance (user's laptop)

**Accepted, 2026-10-27.** At commit `d10af59` (`git pull` reported "Already up to date"), the user ran:

```
python3 -m unittest discover -s tests -v && python3 -m red_queen.validation
python3 -m red_queen.planning_experiment
```

User-reported output: `Ran 121 tests ... OK`, `63/63 validation checks passed`,
and a planning-experiment table identical to section 6.

## 9. Next milestone

**M3.2 — expanded scenario validation.** Add more scenarios: different
start states, overlapping faults and disturbances, and cases where the
greedy and shortest choices disagree. Run both agents on all of them, and
add a tie-break sensitivity test (reverse action order) to separate real
planning effects from ordering luck.
