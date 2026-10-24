# RED QUEEN AI — Engineering Record, Milestone M1.1

**Version:** 0.2.0 · **Date:** 2026-10-24 · **Status:** build-environment passed; Linux acceptance pending

## 1. Review of M1

M1 (accepted on Linux, commit `6f4bef5`) showed that affect-on and
affect-off produced identical action sequences in all 7 scenarios.

## 2. Limitation addressed

The M1 comparison could not show any effect, for two reasons:

1. The agent's model matched the environment exactly, so no action ever
   failed and the **frustration** signal never activated.
2. Only one affect configuration was tested.

## 3. Changes (smallest coherent improvement)

| Change | File |
|---|---|
| Scheduled action faults: `blocked=((action, first_step, last_step), ...)`. During the window the action has no effect and is logged as a `blocked` event. The agent is **not** told; it only sees that nothing changed. | `environment.py` |
| Fault windows are validated: unknown actions and bad step ranges are rejected. | `environment.py` |
| 3 fault scenarios: `flaky_restart`, `flaky_restart_low_battery`, `charger_outage`. M1 scenarios unchanged and listed first. | `scenarios.py` |
| `run_scenario(..., affect=AffectConfig(...))` accepts a custom config (existing calls unchanged). | `experiments.py` |
| Affect parameter sweep: concern_gain ∈ {0, 1} × frustration_penalty ∈ {0, 0.05, 0.5, 1, 2} = 10 configs × 10 scenarios = 100 runs, compared with affect-off. | `sweeps.py` (new) |
| Validation V21–V31 added; V01–V20 keep their M1 meaning. V31 freezes M1's exact action sequences as a regression check. | `validation.py` |

No new action types, no new capabilities, no external access.

## 4. Acceptance criteria

1. All 38 M1 tests and V01–V20 still pass with the same meaning.
2. M1 scenarios produce exactly the M1 action sequences (V31).
3. Blocked actions have no effect, are logged, and only inside their window (V21).
4. All fault scenarios recover in both modes (V22–V27).
5. Frustration activates under faults (V28).
6. Sweep covers the full grid and is deterministic (V29–V30).

## 5. Results (build environment: Linux, Python 3.13.17)

- `python3 -m unittest discover -s tests -v` → **Ran 57 tests, OK** (38 from M1 + 19 new)
- `python3 -m red_queen.validation` → **31/31 validation checks passed**, exit code 0

### Default affect config (M1 settings)

All 10 scenarios succeed in both modes with identical sequences
(35 total actions each). Frustration reaches 3 in `flaky_restart`, but
the default penalty (0.05) is too small to change any decision.

### Sweep (100 runs vs affect-off baseline)

Outcome counts: **same 92, affect_fewer_actions 5, affect_more_actions 3**.
No run failed. M1 scenarios never diverged.

| Scenario | Baseline actions | Affect actions | Configs (gain, penalty) |
|---|---|---|---|
| flaky_restart_low_battery | 6 | **5** (fewer) | (0, 0.5), (0, 1), (0, 2), (1, 1), (1, 2) |
| flaky_restart | 4 | **5** (more) | (0, 1), (0, 2), (1, 2) |
| charger_outage | 4 | 4 (different order, same count) | (0, 0.5), (0, 1), (0, 2), (1, 0.5), (1, 1), (1, 2) |

**Interpretation (limited to these scenarios):**

- *Fewer actions:* when restart is blocked and the battery is low, a large
  repeat penalty makes the agent charge instead of repeating restart. That
  charge is useful work, so total actions drop.
- *More actions:* when restart is blocked but the battery is already fine,
  the same penalty makes the agent charge anyway, which does not help, so
  total actions rise.
- *Same count:* in `charger_outage` the agent inserts a harmless `cool`,
  but the blocked window ends at the same time, so the count is unchanged.

So the frustration penalty helps only when there is another useful action
to switch to, and costs an action when there is not. There is **no
evidence of a general benefit**. The effect needs penalties ≥ 0.5, which is
10× the M1 default.

Full data: `results/m1_1_sweep.json`, `results/m1_affect_comparison.json`.

## 6. Limitations

- Faults are fixed windows, not stochastic; the agent never learns that an action is unreliable.
- Frustration resets on any progress and only penalises repeating the *last* action.
- One-step greedy scoring; the agent cannot reason "wait for the window to end".
- 3 fault scenarios, hand-written; sweep grid is small.
- Blocked actions cost nothing; a real fault might still consume battery.

## 7. Linux acceptance (user's laptop)

**Pending.** Run in the repository folder:

```
git pull
python3 -m unittest discover -s tests -v && python3 -m red_queen.validation
```

M1.1 counts as accepted only when the output shows `Ran 57 tests ... OK`
and `31/31 validation checks passed` (or all V01–V31 lines `[PASS]`).

## 8. Next milestone

**M2 — persistent SQLite memory.** Record each episode (scenario, config,
actions, outcome, events) in a local SQLite file, add lexical retrieval of
past episodes, and keep memory strictly advisory: retrieved memories may
inform scoring only through an explicit, tested interface, and current
observations always win. A natural first use: remember which actions were
recently blocked — the gap M1.1 exposed.
