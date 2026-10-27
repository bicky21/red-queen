# RED QUEEN AI — Engineering Record, Milestone M3.6

**Version:** 0.10.0 · **Date:** 2026-10-27 · **Status:** build-environment passed; Linux acceptance pending
**Outcome:** memory is safe under noise (no failures at any level). Disturbance prediction degrades badly when memory is wrong; a mismatch guard reduces, but does not remove, that cost. Guard adopted as an opt-in recommendation.

## 1. Review of M3.5

M3.5 (accepted on Linux, commit `30e4cb8`) showed gains from predicting
remembered disturbances. In that study, though, memory was **perfectly
predictive**: the prior episode came from the same deterministic case.
The M3.5 record named this as the largest open question.

## 2. Limitation addressed

There was no measurement of behaviour with wrong or noisy memory, and no
mechanism to stop trusting memory once it has been proven wrong in the
current episode.

## 3. Changes

| Change | File |
|---|---|
| `distort(faults, disturbances, level, seed)`: a seeded, deterministic distorted copy of a case. At level p, each fault window is shifted ±1 with probability p and dropped with p/2; each disturbance is moved ±1 with p and dropped with p/2; one spurious fault and one spurious disturbance are each added with p/2. Outputs are always valid (V101). | `memory_noise.py` (new) |
| `PlanningAgent(disturbance_guard=...)`: `"on_contradiction"` stops disturbance prediction after a mismatch at a step where memory expected a disturbance; `"on_any_mismatch"` stops it after any mismatch. `None` (default) is exactly M3.5 (V103). The guard requires `predict_disturbances=True` (V104). Stats report `guard_tripped_at`. | `planning_agent.py` |
| Study: 9 agents × 5 memory levels (0, 0.25, 0.5, 1.0, unrelated) × 1,680 grid cases. The prior episode is a reactive run on the distorted case, stored under the real case's name. | `noisy_memory_study.py` (new) |

Both guard variants were measured before choosing (section 5). Only
`on_any_mismatch` is used in the study and recommended.

## 4. Acceptance criteria

1. All earlier tests and V01–V100 pass with the same meaning.
2. Distortion is deterministic and valid, level 0 equals exact memory (V101), and distortion grows with level (V102).
3. With guard off, behaviour is identical to M3.5 (V103). Invalid guard configuration is refused (V104).
4. The guard stays off with exact memory and trips with wrong memory (V105).
5. **No failures at any noise level**: 9,450 runs on the stride-8 subset (V106), and 75,600 runs on the full grid.
6. Subset totals are frozen (V107). The guard reduces worst-case regret and costs ≤ 2% with exact memory (V108).

## 5. Results (build environment: Linux, Python 3.13.17)

- `python3 -m unittest discover -s tests -v` → **Ran 200 tests, OK** (186 earlier + 14 new)
- `python3 -m red_queen.validation` → **108/108 validation checks passed**, exit code 0, about 43 s

A new unit test failed on its first run. `run_agent("oracle", ...)` crashed
with `AttributeError` instead of raising `ValueError`, because unknown
names fell through to advisor construction. This was a real bug in
`noisy_memory_study.run_agent`. It now validates the agent name, and the
memory requirement, first.

### Choosing the guard (full grid, total actions)

| Configuration | 0.0 | 0.25 | 0.5 | 1.0 | unrelated | sum |
|---|---|---|---|---|---|---|
| sorted, no guard (M3.5) | 8,062 | 8,258 | 8,364 | 8,532 | 8,472 | 41,688 |
| sorted, on_contradiction | 8,062 | 8,248 | 8,359 | 8,512 | 8,472 | 41,653 |
| sorted, **on_any_mismatch** | 8,086 | 8,225 | 8,325 | 8,459 | 8,417 | **41,512** |
| reversed, no guard (M3.5) | 7,857 | 8,151 | 8,333 | 8,665 | 8,526 | 41,532 |
| reversed, on_contradiction | 7,857 | 8,150 | 8,331 | 8,642 | 8,526 | 41,506 |
| reversed, **on_any_mismatch** | 7,899 | 8,163 | 8,320 | 8,606 | 8,484 | **41,472** |

`on_any_mismatch` has the lowest sum and the lowest worst case for both
orders. `on_contradiction` is cost-free with exact memory but helps very little.

### Full study: memory agent minus the same agent without memory (negative = memory helps)

| Agent | 0.0 | 0.25 | 0.5 | 1.0 | unrelated |
|---|---|---|---|---|---|
| reactive_memory (M2.1) | −607 | −455 | −351 | −151 | −381 |
| planner_memory (M3.3) | −157 | −101 | −68 | −13 | −98 |
| planner_dist_memory (M3.5) | −366 | −170 | −64 | **+104** | **+44** |
| planner_dist_guard (M3.6) | −342 | −203 | −103 | **+31** | −11 |
| planner_reversed_dist_memory (M3.5) | −856 | −562 | −380 | −48 | −187 |
| planner_reversed_dist_guard (M3.6) | −814 | −550 | −393 | −107 | −229 |

Baselines without memory: reactive 8,741; planner 8,428; planner_reversed 8,713.
Data: `results/m3_6_noisy_memory_study.json`.

### Findings (limited to this simulation)

1. **Safety holds under wrong memory.** There were zero failures across all agents, levels and cases. Bounded, advisory memory with observation override never prevented recovery, even when memory came from an unrelated case.
2. **Benefit shrinks steadily with noise** for every memory agent.
3. **Disturbance prediction is the fragile part.** Unguarded, the sorted M3.5 planner is **worse than no memory** at noise 1.0 (+104) and with unrelated memory (+44).
4. **The guard reduces that cost but does not remove it.** Sorted worst case: +104 → +31 (still 0.4% worse than no memory at noise 1.0). The reversed planner with the guard stays better than no memory at every level. The cost with exact memory is +24 / +42 actions (0.3% / 0.5%).
5. **The risk-only M3.3 planner is the most robust memory planner.** It is better than no memory at every level, but its gains are small.
6. **Unrelated memory still helps the reactive agent (−381).** The grid shares a fault distribution across cases, so even another case's memory carries useful base-rate information, such as "restart is often blocked early". This is a property of the test grid, not evidence of general transfer.

### Decision

Recommended configuration for memory-assisted planning is now
`predict_disturbances=True, disturbance_guard="on_any_mismatch"`. It stays
opt-in (default `None`), so all earlier results are unchanged. When memory
quality is unknown and must never cost actions, the M3.3 risk-only planner
(`predict_disturbances=False`) is the conservative choice.

## 6. Limitations

- The noise model is synthetic: timing shifts, drops and spurious items, with one prior episode per case.
- The guard is binary and per-episode. It does not learn which memories are reliable over time.
- No guard is applied to risk (fault) memory. The data shows risk memory is already robust in totals.
- "Unrelated" memory comes from a fixed offset in the grid, which shares structure with the real case.

## 7. Linux acceptance (user's laptop)

**Pending.** Run in the repository folder:

```
git pull
python3 -m unittest discover -s tests -v && python3 -m red_queen.validation
```

M3.6 is accepted when the output shows `Ran 200 tests ... OK` and
`108/108 validation checks passed`. The full run takes about 1–3 minutes.

Optional (about 1.5–3 minutes; add `--stride 8` for a 15-second version):

```
python3 -m red_queen.noisy_memory_study
```

## 8. Next milestone

**M4 — untrusted proposal interface (LLM-ready, no LLM yet).** The project
goal is to integrate local LLMs later, while keeping them separate from
state, policy and execution. M4 adds a `Proposer` interface that returns
**text** proposals (a goal or an action sequence). A strict parser
validates each proposal against the action allowlist, the policy, and the
model's predictions before anything runs, and the planner remains the
fallback. M4 uses deterministic stub proposers only: correct, wrong,
malformed and adversarial. That way the whole safety path is tested before
any real model is connected.
