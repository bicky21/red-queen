# RED QUEEN AI — Engineering Record, Milestone M8

**Version:** 0.24.0 · **Date:** 2026-11-05 · **Status:** build-environment passed; **Linux acceptance passed 2026-11-05**
**Outcome: pre-registered result. The architecture indicator is present, and the functional effect is null, as predicted.** RED QUEEN now has a minimal **global workspace**:
- five modules propose messages in parallel;
- a limited-capacity workspace selects the most salient one;
- every module receives the broadcast.

Structural checks confirm all four indicator properties (GWT-1 to GWT-4). Functionally, letting the workspace switch off disturbance memory when the self-model or the sensors raise an alarm made **no real difference**: the largest action change was −0.04%. The bottleneck changed nothing except the operator stream, which became 53% shorter. **This is architecture, not consciousness, and here it did not improve performance.**

## 1. Review of M7.3

M7.3 was accepted on Linux on 2026-11-05, with study output identical by `diff`. That closed the self-model track (M7–M7.3): the self-model is calibrated on familiar faults, ranks unseen ones correctly, and its remaining errors are understood. M8 starts the second indicator family from the 2023 *Consciousness in AI* report.

## 2. Limitation addressed

RED QUEEN's components (sensor safety layer, self-model, memory, planner, operator report) were connected only by hand-written calls. There was no general mechanism for one module's information to reach all the others, and no selection of what matters most at a given moment.

## 3. Method: pre-registration

`docs/PREREG_M8.md` was committed before any M8 code (commit `ccd875e`, 2026-11-05 06:08 UTC). It fixed:
- the module set and the **saliences**
- the single behavioural link (memory can only be switched *off*)
- architecture checks A1–A4
- the functional study and criteria F1–F3
- the expectation that F2 would fail (a null result)

Nothing was changed after the run.

## 4. Changes

| Change | File |
|---|---|
| `Workspace(modules, capacity)`. Each cycle: every module proposes messages (source, kind, salience 0–1, numeric content). The top `capacity` by salience are selected, with ties broken by source and then kind. **Every module receives every selected message.** Each cycle is logged. | `workspace.py` (new) |
| Five modules, with the pre-registered saliences: **SelfModule** (`expect_failure`, salience 1 − confidence, below 0.3), **SensorModule** (new safety-layer alerts: isolated/unexplainable 0.9, disagree 0.6, widened 0.5), **MemoryModule** (`disturbance_expected` 0.4; switches memory off on an `expect_failure`, `sensor_isolated` or `reading_unexplainable` broadcast), **PlannerModule** (`status` 0.1, always), and **OperatorModule** (keeps the broadcast stream as the operator report). | `workspace.py` |
| `SafeAgent.run(..., workspace=None)`. The workspace runs at each decision point. Its **only** possible effect is switching disturbance memory off. Workspace errors are caught. The observer and the workspace each get their own fresh snapshot, so neither can alter what the other sees. | `safe_agent.py` |
| The pre-registered functional study, with the decision rule as code. | `workspace_study.py` (new) |

## 5. Acceptance criteria (build environment)

1. All 367 earlier tests and V01–V204 pass with the same meaning.
2. **A1/A2 (GWT-1/2):** every module is consulted each cycle, the top-capacity selection is correct, and invalid configurations are refused (V205). **A3 (GWT-3):** every module received every broadcast in a real episode (V206). **A4 (GWT-4):** selection depends on state. The same message wins when alone and loses to a more salient one, and the self module speaks only when its confidence is low (V207).
3. **Safety:** without memory, the standard workspace and a crashing workspace leave behaviour unchanged in 126 runs, and the crashing workspace's errors are caught (V208).
4. A reduced version of the study (stride 42, seed 0) is frozen (V209), and its decision is checked (V210). The full study takes about 100 seconds, so it runs only from the command line. The pre-registration is present (V211).

## 6. Results (build environment: Linux, Python 3.13)

- `python3 -m unittest discover -s tests -v` → **Ran 378 tests, OK** (367 + 11 new); `python3 -X dev` run: **0 ResourceWarnings**
- `python3 -m red_queen.validation` → **211/211 validation checks passed**

One new check (V207) failed on its first run. My constructed "hopeless" situation was one the self-model had never seen, so it returned the 0.73 base rate instead of low confidence. The check now uses a situation taken from the model's own training data, ("plan too long", "bound widened", "wide belief"), with confidence about 0.002. This was a check error, not a workspace bug.

`python3 -m red_queen.workspace_study`: 168 grid cases × 2 seeds = 336 runs per row. Each cell is successes / total actions.

| memory | sensors | safe (no memory) | safe_mem_guard (M6) | **+ workspace (capacity 1)** | + workspace (all) |
|---|---|---|---|---|---|
| exact | none | 336 / 1945 | 336 / 1916 | 336 / 1916 | 336 / 1916 |
| exact | B_stuck_low | 21 / 6328 | 21 / 6327 | 21 / 6327 | 21 / 6327 |
| exact | A_stuck_mid | 142 / 4673 | 152 / 4513 | 152 / 4513 | 152 / 4513 |
| wrong | none | 336 / 1945 | 336 / 1972 | 336 / 1972 | 336 / 1972 |
| wrong | B_stuck_low | 21 / 6328 | 21 / 6328 | 21 / 6328 | 21 / 6328 |
| wrong | A_stuck_mid | 142 / 4673 | 136 / 4764 | 136 / **4762** | 136 / 4762 |

There were 0 false stops in every cell. The workspace switched memory off in 182 and 214 runs (exact memory) and 128 and 140 runs (wrong memory) in the two faulty-sensor conditions.

Capacity 1 and full broadcast gave **identical action sequences in 2,016 of 2,016 paired runs**. The operator stream had **26,820 messages with capacity 1 against 56,832 with full broadcast** (47%).

**Pre-registered decision** (computed by `decide`):
- **F1** (no extra false stops): **True**
- **F2** (coordination useful: ≥ 1% fewer actions on a faulty condition): **False**. The largest change was −0.04% (4762 vs 4764).
- **F3** (capacity identical, stream at most half): **True**
- **Architecture indicator present (A1–A4): yes.**
- **Workspace functionally useful: no.**

**What this shows**
- **The four GWT properties can be built and verified** in a modular, safe way: parallel modules, a bottleneck, global broadcast, and state-dependent selection.
- **Broadcast reached memory, but there was nothing left to gain.** Memory was switched off in up to 64% of faulty runs (214 of 336), with essentially no effect. M6 already showed memory changes actions by about 2% at most, and the M6 guard usually caught the same problems. That matches the pre-registered expectation.
- **The bottleneck does no harm and no good here.** With these saliences, every coordination trigger outranks every non-trigger, so capacity 1 and full broadcast behave identically by construction. Its only effect is a **shorter, prioritised operator stream**: the most important message each cycle, instead of everything.
- **Consistent with the whole project:** in this small domain, an exact planner leaves little for extra architecture to improve. The workspace is infrastructure. Whether it matters needs a domain where modules have something to tell each other that changes decisions.

## 7. Limitations

- **This is not consciousness.** It is four architectural properties, verified structurally, plus one functional test with a null result.
- The saliences are hand-set, with no learned or adaptive attention. The capacity test is trivial by construction, as stated in advance.
- The only behavioural link is memory switch-off. The planner, policy and estimator deliberately ignore broadcasts, for safety.
- The full study is only in the CLI. Validation runs a reduced, frozen version.

## 8. Linux acceptance (user's laptop)

**Passed 2026-11-05.** On the user's Ubuntu laptop (Python 3.14):
`python3 -m unittest discover -s tests -v` → **Ran 378 tests in 338.945s, OK**;
`python3 -m red_queen.validation` → **211/211 validation checks passed**;
`python3 -m red_queen.workspace_study`: compared with `diff` against the build environment, it is **identical**.

## 9. Next milestone (candidates) — M8.1 (housekeeping) done next

1. **M8.1: faster validation (housekeeping).** Validation and tests now take about 12 minutes on the laptop. Split them into a quick suite (every change) and a full suite (acceptance), without changing any check.
2. **M9: a harder domain.** The planner, memory, language model and workspace have all shown little room to help in this small world. A richer simulation, with more resources, delayed effects, and trade-offs where the shortest plan is not obviously best, would show whether any of them can add real value.
3. **Status report:** a written summary for the user of all the consciousness-indicator work (M7–M8): what exists, what was measured, and what it does and does not mean.
