# RED QUEEN AI — Milestone M2

Deterministic simulated recovery, explicit action scoring, an
affect-enabled vs affect-disabled comparison, scheduled action faults,
an affect parameter sweep, and persistent SQLite episodic memory.

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
   by keyword. In M2 memory is stored only; it does not change any decision.

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
   Expected: `Ran 77 tests ... OK` and `42/42 validation checks passed`.
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
  agent.py         bounded observe-score-act loop
  verification.py  independent goal verifier
  scenarios.py     7 M1 + 3 fault scenarios
  experiments.py   affect on/off comparison
  sweeps.py        affect parameter sweep (M1.1)
  memory.py        SQLite episodic memory + lexical search (M2)
  memory_cli.py    record-all / search / stats (M2)
  validation.py    42 validation checks (V01-V20 M1, V21-V31 M1.1, V32-V42 M2)
tests/             77 unittest tests
docs/ENGINEERING_RECORD_M1.md     (accepted on Linux)
docs/ENGINEERING_RECORD_M1_1.md   (accepted on Linux)
docs/ENGINEERING_RECORD_M2.md     (accepted on Linux)
results/                          build-environment reports
```

## Status

| Milestone | Build environment | Your laptop (Linux) |
|---|---|---|
| M1 | 38 tests OK, 20/20 checks | **Accepted** 2026-10-24 |
| M1.1 | 57 tests OK, 31/31 checks | **Accepted** 2026-10-24 |
| M2 | 77 tests OK, 42/42 checks | **Accepted** 2026-10-24 |
