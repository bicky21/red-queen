# RED QUEEN AI — Milestone M3.3

Deterministic simulated recovery, explicit action scoring, an
affect-enabled vs affect-disabled comparison, scheduled action faults,
an affect parameter sweep, persistent SQLite episodic memory, bounded
advisory memory, a bounded planner with policy enforcement, and an
expanded scenario study with a planner tie-break control, and
memory-informed planning.

Pure Python standard library. No dependencies, no network, no shell
execution, no real-world actions.

## What it does (in simple steps)

1. A **simulated system** has three things: `battery` (0–100),
   `temperature` (0–100) and `service_up` (true/false).
2. The **goal** is: battery ≥ 60, temperature ≤ 50, service up.
3. The **agent** looks at the state, scores every allowed action, and
   picks the best one.
4. The **environment** runs the action. Some scenarios add scheduled
   **disturbances** (heat spike, service crash) or **action faults**
   (an action silently does nothing for a few steps; the agent is not told).
5. The agent repeats until the goal is met or it hits `max_steps`.
6. An **independent verifier** reads the true state and decides success.
7. Finished episodes can be saved to a **SQLite memory** file and searched
   by keyword.
8. Optionally, an **advisor** reads past episodes of the same scenario and
   lowers the expected gain of actions that were blocked at that step
   (at most by 90%). What the agent currently observes always wins.
9. A **policy** lists which actions are permitted. Every action passes a
   policy check right before it runs, so a denied action can never execute.
10. The **planning agent** searches ahead for the shortest permitted plan,
    checks the plan against the policy, follows it, and replans when what
    it observes differs from what it predicted. If no plan exists, it stops.
11. With memory, the planner uses remembered faults to choose between
    equally short plans. Memory can never make a plan longer.

Allowed actions (the only ones that can ever run):

| Action | Effect |
|---|---|
| `charge` | battery +20, temperature +5 |
| `cool` | temperature −20 |
| `restart_service` | battery −10; starts service only if temperature ≤ 80 |
| `wait` | nothing |

## Requirements

Python 3.10 or newer and git. Check with:

```
python3 --version
git --version
```

## Get and test it (Linux, step by step)

1. Clone once:
   ```
   git clone https://github.com/bicky21/red-queen.git
   cd red-queen
   ```
2. For every new milestone, update and test:
   ```
   git pull
   python3 -m unittest discover -s tests -v && python3 -m red_queen.validation
   ```
   Expected: `Ran 158 tests ... OK` and `82/82 validation checks passed`
   (takes about 15-25 seconds).
3. Optional reports:
   ```
   python3 -m red_queen.experiments --output results/m1_affect_comparison.json
   python3 -m red_queen.sweeps --output results/m1_1_sweep.json
   ```
4. Optional memory demo (creates `data/red_queen_memory.sqlite`, ignored by git):
   ```
   python3 -m red_queen.memory_cli record-all
   python3 -m red_queen.memory_cli search "restart blocked"
   python3 -m red_queen.memory_cli stats
   ```
5. Optional memory experiment (empty vs irrelevant vs relevant memory):
   ```
   python3 -m red_queen.memory_experiment
   ```
6. Optional planning experiment (reactive vs planner under 3 policies):
   ```
   python3 -m red_queen.planning_experiment
   ```
7. Optional scenario study (18 scenarios x 4 policies, plus a 1,680-case fault grid):
   ```
   python3 -m red_queen.scenario_study
   ```
8. Optional memory-informed planning study (6 agents, about 7 seconds):
   ```
   python3 -m red_queen.memory_planning_study
   ```

Note: on Linux use `python3`, not `py` (`py` on Linux is an unrelated tool).

## Project layout

```
red_queen/
  state.py         immutable SystemState
  actions.py       action allowlist + deterministic transition model
  environment.py   simulated world, disturbances, action faults, event log
  goal.py          recovery goal, deficits, distance
  affect.py        experimental affect signals (urgency, concern, frustration)
  scoring.py       one-step action scoring
  agent.py         reactive agent: bounded observe-score-act loop
  verification.py  independent goal verifier
  scenarios.py     7 M1 + 3 fault scenarios + 8 extended (M3.2)
  experiments.py   affect on/off comparison
  sweeps.py        affect parameter sweep (M1.1)
  memory.py        SQLite episodic memory + lexical search (M2)
  memory_cli.py    record-all / search / stats (M2)
  memory_advice.py bounded advisory memory (M2.1)
  memory_experiment.py  empty/irrelevant/relevant memory comparison (M2.1)
  policy.py        policy allowlists + execution-boundary executor (M3)
  planner.py       bounded breadth-first planner (M3), risk tie-break (M3.3)
  planning_agent.py plan / validate / act / replan agent (M3)
  planning_experiment.py reactive vs planner x 3 policies (M3)
  scenario_study.py reactive vs planner vs reversed-order planner (M3.2)
  memory_planning_study.py six agents with/without memory (M3.3)
  validation.py    82 validation checks (V01-V20 M1, V21-V31 M1.1, V32-V42 M2,
                   V43-V50 M2.1, V51-V63 M3, V64-V73 M3.2, V74-V82 M3.3)
tests/             158 unittest tests
docs/ENGINEERING_RECORD_M1.md     (accepted on Linux)
docs/ENGINEERING_RECORD_M1_1.md   (accepted on Linux)
docs/ENGINEERING_RECORD_M2.md     (accepted on Linux)
docs/ENGINEERING_RECORD_M2_1.md   (accepted on Linux)
docs/ENGINEERING_RECORD_M3.md     (accepted on Linux)
docs/ENGINEERING_RECORD_M3_2.md   (accepted on Linux)
docs/ENGINEERING_RECORD_M3_3.md
results/                          build-environment reports
```

## Status

| Milestone | Build environment | Your laptop (Linux) |
|---|---|---|
| M1 | 38 tests OK, 20/20 checks | **Accepted** 2026-10-24 |
| M1.1 | 57 tests OK, 31/31 checks | **Accepted** 2026-10-24 |
| M2 | 77 tests OK, 42/42 checks | **Accepted** 2026-10-24 |
| M2.1 | 94 tests OK, 50/50 checks | **Accepted** 2026-10-24 |
| M3 | 121 tests OK, 63/63 checks | **Accepted** 2026-10-27 |
| M3.2 | 140 tests OK, 73/73 checks | **Accepted** 2026-10-27 |
| M3.3 | 158 tests OK, 82/82 checks | Pending — run step 2 |
