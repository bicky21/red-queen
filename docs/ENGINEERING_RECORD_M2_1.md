# RED QUEEN AI — Engineering Record, Milestone M2.1

**Version:** 0.4.0 · **Date:** 2026-10-24 · **Status:** build-environment passed; Linux acceptance pending

## 1. Review of M2

M2 (accepted on Linux, commit `d38586f`) stores episodes in SQLite and
retrieves them by keyword, but nothing used the memory. The agent still
repeated actions that past episodes show were blocked at those steps.

## 2. Limitation addressed

Stored experience had no path into decisions. M2.1 adds one narrow,
bounded, testable path, and keeps observation and safety authoritative.

## 3. Design

```
EpisodeMemory --search(same scenario)--> MemoryAdvisor --discounts(step)--> ActionScorer
   (untrusted)     valid records only      blocked / attempts per            gain x (1 - d)
                                           (step, action), capped 0.9        positive gain only
```

| Rule | Where |
|---|---|
| Only records of the **same scenario** are used (lexical hits on other scenarios are dropped). | `memory_advice.py` |
| Records with `valid=False` are ignored; events with unknown action, unknown kind, or bad step are ignored and counted. | `memory_advice.py` |
| `discount(step, action) = blocked attempts / all attempts` at that step in retrieved episodes. | `memory_advice.py` |
| Discount is clamped to [0, 0.9]; NaN, non-numbers and unknown action names are ignored. | `scoring.py` |
| Discount only shrinks a **positive** expected gain. It never changes costs or penalties, never adds or removes an action, and cannot make a harmful action look better. | `scoring.py` |
| **Observation wins:** once an action's observed result matches the model in this episode, its memory discount is dropped for the rest of the episode. | `agent.py` |
| With no advisor, scoring and traces are exactly as before; `memory_discounts` appears in a trace only when an advisor is used. | `agent.py`, `scoring.py` |
| The advisor only reads memory; it never writes to it. | test |

## 4. Other changes

- `memory_experiment.py`: empty vs irrelevant vs relevant memory, both affect modes, all 10 scenarios (60 runs).
- The M2 CLI test no longer prints into the test log; it captures and checks the output instead.

## 5. Acceptance criteria

1. All earlier tests and V01–V42 still pass with the same meaning.
2. Empty memory gives identical actions and events to no advisor (V43).
3. Irrelevant memory has no effect and is not used (V45).
4. Invalid records are ignored (V46).
5. Poisoned memory cannot prevent recovery in any scenario (V47).
6. Discount is capped, sanitised, and never touches non-positive gains (V48).
7. Observation overrides memory (V49).
8. Experiment covers 60 runs, is deterministic, and all runs succeed (V50).

## 6. Results (build environment: Linux, Python 3.13.17)

- `python3 -m unittest discover -s tests -v` → **Ran 94 tests, OK** (77 earlier + 17 new)
- `python3 -m red_queen.validation` → **50/50 validation checks passed**, exit code 0

### Memory experiment (60 runs)

| Memory | Successes | Total actions | Blocked attempts |
|---|---|---|---|
| empty | 20/20 | 70 | 16 |
| irrelevant | 20/20 | 70 | 16 |
| relevant | 20/20 | **68** | **13** |

Runs that changed with relevant memory:

| Scenario | Affect | Empty → relevant | Relevant sequence |
|---|---|---|---|
| flaky_restart_low_battery | off | 6 → **5** | charge > restart ×3 > charge |
| flaky_restart_low_battery | on | 6 → **5** | charge > restart ×3 > charge |
| charger_outage | on | 4 → 4 | charge > cool > charge > charge |

**Interpretation (limited to these scenarios):** memory helps only when
the agent has another useful action to do while the remembered fault is
active (charge instead of a blocked restart). In `flaky_restart` there is
no useful alternative, so relevant memory changes nothing. The saving is
2 actions out of 70 overall.

### Poisoned memory

With 3 copies of each scenario and every past event rewritten to
`blocked`, all 20 episodes still recovered (V47). Total actions: 68,
versus 70 with empty memory. Uniform poisoning discounts every action
equally, so the ranking barely changes. This test does **not** show
robustness to targeted poisoning of a single action.

Data: `results/m2_1_memory_experiment.json`.

## 7. Limitations

- Advice is indexed by step number, so it assumes faults recur at the same step. That holds in these fixed scenarios but not in general.
- Relevance is exact scenario name. The agent is effectively told which scenario it is in.
- Targeted poisoning (one action marked blocked) can still cost actions. It is bounded by the 0.9 cap and the observation override, but not measured here.
- "Worked" means observed state equals the model's prediction. A disturbance in the same step makes an action look like it failed, which keeps its discount (a conservative choice).
- The prior runs in the experiment come from the same deterministic scenarios, so memory is perfectly predictive. Real experience would be noisier.

## 8. Linux acceptance (user's laptop)

**Pending.** Run in the repository folder:

```
git pull
python3 -m unittest discover -s tests -v && python3 -m red_queen.validation
```

M2.1 is accepted when the output shows `Ran 94 tests ... OK` and
`50/50 validation checks passed`.

Optional:

```
python3 -m red_queen.memory_experiment
```

## 9. Next milestone

**M3 — goal-directed planning.** Replace one-step greedy scoring with a
bounded planner (fixed-depth search over the action allowlist using the
transition model), with replanning when an observation differs from the
plan's prediction. It adds a policy layer that rejects unauthorised
actions before execution, and keeps independent verification. It will be
compared experimentally against the M2.1 reactive scorer, not assumed better.
