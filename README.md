# RED QUEEN AI — Milestone M1

Deterministic simulated recovery, explicit action scoring, and an
affect-enabled vs affect-disabled comparison.

Pure Python standard library. No dependencies, no network, no shell
execution, no real-world actions.

## What it does (in simple steps)

1. A **simulated system** has three things: `battery` (0–100),
   `temperature` (0–100) and `service_up` (true/false).
2. The **goal** is: battery ≥ 60, temperature ≤ 50, service up.
3. The **agent** looks at the state, scores every allowed action, and
   picks the best one.
4. The **environment** runs the action. Some scenarios add scheduled
   **disturbances** (heat spike, service crash).
5. The agent repeats until the goal is met or it hits `max_steps`.
6. An **independent verifier** reads the true state and decides success.

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
   Expected: `Ran 38 tests ... OK` and `20/20 validation checks passed`.
3. Optional affect comparison report:
   ```
   python3 -m red_queen.experiments --output results/m1_affect_comparison.json
   ```

Note: on Linux use `python3`, not `py` (`py` on Linux is an unrelated tool).

## Project layout

```
red_queen/
  state.py         immutable SystemState
  actions.py       action allowlist + deterministic transition model
  environment.py   simulated world, disturbances, event log
  goal.py          recovery goal, deficits, distance
  affect.py        experimental affect signals (urgency, concern, frustration)
  scoring.py       one-step action scoring
  agent.py         bounded observe-score-act loop
  verification.py  independent goal verifier
  scenarios.py     7 fixed validation scenarios
  experiments.py   affect on/off comparison
  validation.py    20 validation checks
tests/             38 unittest tests
docs/ENGINEERING_RECORD_M1.md
results/m1_affect_comparison.json   (build-environment run)
```

## Status

- Build environment (Linux, Python 3.13): 38/38 tests passed, 20/20 validation checks passed.
- **Linux acceptance (user's laptop): pending** — run step 2 and report the output.
