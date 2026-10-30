# RED QUEEN AI — Milestone M5.3

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
12. An optional `slack` setting (M3.4) lets the planner consider plans up to
    3 steps longer. It was tested and gave no benefit here, so it is off by default.
13. With `predict_disturbances=True` (M3.5), the planner also predicts
    disturbances remembered from earlier runs (for example, a heat spike
    after step 2). If one does not happen, the agent replans from what it sees.
    With `disturbance_guard="on_any_mismatch"` (M3.6), the planner stops
    predicting disturbances once memory has been proven wrong in the current run.
    This pair of settings is the recommended setup for memory-assisted planning.
14. An optional **proposer** (M4) can suggest plans as text, the way a language
    model would. Its text is strictly parsed and checked against the allowed
    actions, the policy and the model before anything runs; otherwise the
    planner's own plan is used.
15. Optionally (M4.1), a **local Ollama model** can be that proposer. It only
    talks to your own machine, has a time and size limit, and is off unless you
    pass `--model`. Its suggestions go through exactly the same checks.
16. Experimental options (M4.2): other prompt styles (`--prompt`) and a retry
    that tells the model why its plan was rejected (`--retries`), using only
    the system's own trusted checks. Both are off by default.
17. Noisy sensors (M5): an agent can be made to see battery and temperature
    only through a sensor with bounded noise. A **belief agent** keeps a range
    ("box") of possible states, plans for the worst case in that box, and only
    stops when every state in the box meets the goal.
18. Optionally (M5.1), the belief agent checks whether its readings can be
    explained with its assumed noise size. If not, it enlarges the assumption.
    A constant sensor offset (bias) cannot be detected this way.
19. Redundant sensors (M5.2): an agent can read two independent sensors and
    combine their ranges. The "hull" rule stays safe if one sensor is biased.
    A bias shared by both sensors is still not detected.
20. Safety layer (M5.3): one `SafeEstimator` combines the noise check (M5.1)
    and the two-sensor hull (M5.2). It writes an operator report of alerts, and
    it drops a sensor whose readings make no sense, or halts safely if none
    is left.

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
   Expected: `Ran 311 tests ... OK` and `162/162 validation checks passed`
   (takes about 5-8 minutes; does not need Ollama).
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
9. Optional slack study (about 35-60 seconds):
   ```
   python3 -m red_queen.slack_study
   ```
10. Optional disturbance-prediction study (about 10-20 seconds):
    ```
    python3 -m red_queen.disturbance_study
    ```
11. Optional imperfect-memory study (about 1.5-3 minutes; `--stride 8` for ~15 s):
    ```
    python3 -m red_queen.noisy_memory_study
    ```
12. Optional proposal study (stub proposers, a few seconds):
    ```
    python3 -m red_queen.proposal_study
    ```
13. Optional real local-model trial (needs Ollama running):
    ```
    python3 -m red_queen.llm_trial --list-models
    python3 -m red_queen.llm_trial --model llama3.2 --output results/m4_1_llm_trial.json
    ```
    For "thinking" models such as `qwen3.5`, add `--think off`.
    Compare all prompt styles and retry settings in one run (M4.2, slow):
    ```
    python3 -m red_queen.llm_trial --model qwen3.5:4b --think off --compare --output results/m4_2_compare.json
    ```
    If Ollama is not running, start it with `ollama serve` in another terminal.
    If no model is installed, get a small one with `ollama pull llama3.2`.
14. Noisy sensor study (M5, about 10 seconds):
    ```
    python3 -m red_queen.sensor_study --output results/m5_sensor_study.json
    ```
15. Noise-assumption study (M5.1, about 20 seconds):
    ```
    python3 -m red_queen.sensor_adapt_study --output results/m5_1_adapt_study.json
    ```
16. Redundant sensor study (M5.2, about 10 seconds):
    ```
    python3 -m red_queen.redundancy_study --output results/m5_2_redundancy_study.json
    ```
17. Safety layer study (M5.3, about 1 minute):
    ```
    python3 -m red_queen.safety_layer_study --output results/m5_3_safety_layer_study.json
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
  planner.py       bounded BFS planner (M3), risk tie-break (M3.3), slack option (M3.4),
                   disturbance prediction (M3.5)
  planning_agent.py plan / validate / act / replan agent (M3)
  planning_experiment.py reactive vs planner x 3 policies (M3)
  scenario_study.py reactive vs planner vs reversed-order planner (M3.2)
  memory_planning_study.py six agents with/without memory (M3.3)
  slack_study.py   slack k=0..3 vs reactive+memory (M3.4)
  disturbance_study.py disturbance prediction vs reactive+memory (M3.5)
  memory_noise.py  seeded distortion of remembered faults/disturbances (M3.6)
  noisy_memory_study.py 9 agents x 5 memory-noise levels (M3.6)
  proposals.py     untrusted proposal interface: context, prompt, parser, validator, gate (M4)
  proposal_study.py stub proposers vs planner alone (M4)
  llm_adapter.py   loopback-only Ollama proposer, timeout + size caps (M4.1)
  llm_trial.py     real local-model trial runner (M4.1, opt-in)
  fake_ollama.py   TEST ONLY: fake local Ollama server (M4.1)
  sensors.py       noisy sensor, sensed environment, belief box (M5)
  belief_agent.py  worst-case belief agent: interval / filter / filter_verified (M5)
  sensor_study.py  4 agents x 4 noise levels x 18 scenarios x 20 seeds (M5)
  sensor_adapt_study.py wrong noise bound, correct bound, sensor bias (M5.1)
  redundant_agent.py two-sensor belief agent: intersect / hull / latch (M5.2)
  redundancy_study.py 6 fault conditions x 4 agents (M5.2)
  safe_agent.py    SafeEstimator (state estimation + alerts) and SafeAgent (M5.3)
  safety_layer_study.py 7 fault conditions x 5 agents (M5.3)
  validation.py    162 validation checks (V01-V20 M1, V21-V31 M1.1, V32-V42 M2,
                   V43-V50 M2.1, V51-V63 M3, V64-V73 M3.2, V74-V82 M3.3,
                   V83-V90 M3.4, V91-V100 M3.5, V101-V108 M3.6, V109-V117 M4,
                   V118-V124 M4.1, V125-V131 M4.2, V132-V141 M5, V142-V148 M5.1, V149-V155 M5.2, V156-V162 M5.3)
tests/             311 unittest tests
docs/ENGINEERING_RECORD_M1.md     (accepted on Linux)
docs/ENGINEERING_RECORD_M1_1.md   (accepted on Linux)
docs/ENGINEERING_RECORD_M2.md     (accepted on Linux)
docs/ENGINEERING_RECORD_M2_1.md   (accepted on Linux)
docs/ENGINEERING_RECORD_M3.md     (accepted on Linux)
docs/ENGINEERING_RECORD_M3_2.md   (accepted on Linux)
docs/ENGINEERING_RECORD_M3_3.md   (accepted on Linux)
docs/ENGINEERING_RECORD_M3_4.md   (accepted on Linux)
docs/ENGINEERING_RECORD_M3_5.md   (accepted on Linux)
docs/ENGINEERING_RECORD_M3_6.md   (accepted on Linux)
docs/ENGINEERING_RECORD_M4.md     (accepted on Linux)
docs/ENGINEERING_RECORD_M4_1.md   (accepted on Linux; includes real-model trial)
docs/ENGINEERING_RECORD_M4_2.md   (accepted on Linux; real-model comparison, negative result)
docs/ENGINEERING_RECORD_M5.md     (accepted on Linux)
docs/ENGINEERING_RECORD_M5_1.md   (accepted on Linux)
docs/ENGINEERING_RECORD_M5_2.md   (accepted on Linux)
docs/ENGINEERING_RECORD_M5_3.md
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
| M3.3 | 158 tests OK, 82/82 checks | **Accepted** 2026-10-27 |
| M3.4 | 170 tests OK, 90/90 checks | **Accepted** 2026-10-27 |
| M3.5 | 186 tests OK, 100/100 checks | **Accepted** 2026-10-27 |
| M3.6 | 200 tests OK, 108/108 checks | **Accepted** 2026-10-28 |
| M4 | 220 tests OK, 117/117 checks | **Accepted** 2026-10-28 |
| M4.1 | 241 tests OK, 124/124 checks (fake server) | **Accepted** 2026-10-28; real model qwen3.5:4b: 18/18 success, 13% proposals accepted, 0 safety violations |
| M4.2 | 256 tests OK, 131/131 checks (fake server) | **Accepted** 2026-10-30; real-model `--compare`: no prompt/retry setting beat the planner (82–84 vs 79 actions), 0 safety violations — negative result |
| M5 | 273 tests OK, 141/141 checks | **Accepted** 2026-10-30; sensor study table identical to build environment |
| M5.1 | 285 tests OK, 148/148 checks | **Accepted** 2026-10-30; study tables identical to build environment |
| M5.2 | 297 tests OK, 155/155 checks | **Accepted** 2026-10-30; study table identical to build environment |
| M5.3 | 311 tests OK, 162/162 checks | Pending — run step 2 and step 17 |
