# RED QUEEN AI — M8 pre-registration: global workspace (second indicator family)

**Written and committed before any M8 code was written or run.** Date: 2026-11-05.

## Background

**Global Workspace Theory** (Baars; Dehaene) says that many specialised processes work in parallel, but only a little information at a time enters a limited-capacity "workspace", from which it is **broadcast** to all of them. The 2023 report *Consciousness in Artificial Intelligence* (Butlin, Long et al.) turns this into four indicator properties:

- **GWT-1:** multiple specialised modules working in parallel
- **GWT-2:** a limited-capacity workspace, i.e. a bottleneck
- **GWT-3:** global broadcast, so the workspace contents reach all modules
- **GWT-4:** state-dependent selection, so what enters depends on the system's current state

M8 builds a minimal version in RED QUEEN, checks the four properties **structurally**, and tests **functionally** whether broadcast-based coordination changes anything.

**No claim of consciousness is made or implied.** We measure architecture properties and their effects only.

## What is built

**`Workspace(modules, capacity)`.** Each cycle (one per decision point):
1. Every module **proposes** messages. A message has a source, a kind, a **salience** between 0 and 1, and numeric content.
2. The workspace **selects** the top `capacity` messages by salience. Ties are broken by source, then kind. `capacity=None` means everything is broadcast, with no bottleneck.
3. **Every** module **receives** every selected message.
4. The broadcasts are logged as the episode's **workspace stream**.

**Modules and their fixed saliences** (fixed now):

| module | proposes | salience |
|---|---|---|
| SelfModule (M7 self-model v1, threshold 0.3 from M7.1) | `expect_failure` when confidence < 0.3 | 1 − confidence (≥ 0.7) |
| SensorModule (new alerts from the safety layer this cycle) | `sensor_isolated` / `reading_unexplainable` | 0.9 |
| | `sensors_disagree` | 0.6 |
| | `bound_widened` | 0.5 |
| MemoryModule | `disturbance_expected`, when memory predicts a disturbance at the next step | 0.4 |
| PlannerModule | `status` (plan status), always | 0.1 |
| OperatorModule | proposes nothing; it keeps the received stream as the operator report | — |

**Coordination link (the only one that can affect behaviour).** When MemoryModule receives an `expect_failure`, `sensor_isolated` or `reading_unexplainable` broadcast, it switches **disturbance memory off** for the rest of the episode. The workspace can only **remove** memory's influence. It never adds actions, and never touches the estimator, the policy or the stopping rule.

**`SafeAgent(..., workspace=None)`.** The workspace runs once per decision point (after planning, and at the stop). A memory switch-off takes effect from the next step. With no workspace, behaviour is unchanged. Workspace errors are caught: they count as "no broadcast" and are recorded.

## Architecture checks (structural, A1–A4)

- **A1 (GWT-1):** at least 4 modules take part in every cycle (they are asked to propose).
- **A2 (GWT-2):** exactly min(capacity, number of proposals) messages are broadcast per cycle, and invalid capacities are refused.
- **A3 (GWT-3):** every module receives every broadcast message.
- **A4 (GWT-4):** selection depends on state. The same module's message is selected or not, depending on what the other modules propose (shown on constructed cases).

## Functional study

- **Cases:** every 10th case of the M3.2 grid (168). **Memory** as in M6, at levels 0.0 (exact) and 1.0 (wrong).
- **Sensors:** M5.4 conditions `none`, `B_stuck_low` and `A_stuck_mid` (sensors built as in M5.4), seeds 0 and 1.
- **Agents:**
  - `safe_mem_guard` (M6: memory + guard, no workspace)
  - `safe_mem_guard_ws` (the same plus a workspace with capacity 1)
  - `safe_mem_guard_ws_full` (the same with capacity None)
  - `safe` (no memory, for reference)
- **The self-model** for SelfModule is fitted exactly as in M7 (7 safety-layer conditions, seeds 0–9).

## Decision rule (fixed now)

- **F1 (safety):** `safe_mem_guard_ws` has no more false stops than `safe_mem_guard` in every condition.
- **F2 (coordination useful):** in at least one faulty-sensor condition (`B_stuck_low` or `A_stuck_mid`), at either memory level, `safe_mem_guard_ws` uses **≥ 1% fewer** total actions than `safe_mem_guard`, and it never uses more than 1% more in any condition.
- **F3 (bottleneck):** capacity 1 and capacity None give **identical actions**. This is expected by construction: every coordination trigger has higher salience than every non-trigger. With capacity 1, the operator stream holds **≤ 50%** of the messages of the full broadcast.

**Results:**
- **Architecture indicator present (measured sense)** if A1–A4 pass.
- **Workspace functionally useful** if F1 and F2 pass.
- F3 is reported. It shows the bottleneck's only effect here: a shorter operator stream.

## Expectations (stated now)

- A1–A4 pass. They are properties of the design, and the checks confirm the implementation has them.
- **F2 will probably fail (null result).** In M6, memory changed actions by only about 2% at most, so switching it off earlier can save very little.
- F3 holds by construction. The bottleneck does **not** improve performance here, only the readability of the stream. In this small system, I do not expect global broadcast to beat simple full sharing.
