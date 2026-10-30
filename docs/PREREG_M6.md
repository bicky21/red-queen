# RED QUEEN AI — M6 pre-registration

**Written and committed before any M6 code was written or run.** Date: 2026-10-30.

## Question

Disturbance memory (M3.5) and its guard (M3.6) were only studied with **exact** state observations. Do they still help, and stay harmless, when the agent sees the world only through noisy sensors, using the M5.3 `SafeAgent` (two sensors, adaptive bounds, hull, isolate)?

## Design (fixed now)

- `SafeAgent` gets optional disturbance memory: `advisor` (the M2.1/M3.5 `MemoryAdvisor`, untrusted) and `predict_disturbances=True`. Each step, the planner plans from the belief's worst corner **including remembered disturbances at future steps**, using the existing M3.5 planner mode, which falls back to the plan without disturbances if no plan is found.
- **Memory never touches the estimator or the stopping rule.** Stopping still requires every state in the belief to meet the goal. So memory can change *which* actions are taken, but not *when* the agent believes it is done.
- Guard `disturbance_guard="on_mismatch"`: after each action, the agent predicts the belief box (previous belief → action worked → remembered disturbance for this step, if any). If the new belief box does **not** overlap that prediction, memory has been contradicted (or a fault happened), and disturbance prediction is switched off for the rest of the episode, as in M3.6.
- Risk discounts (the M3.3 tie-break) are **not** used. Only disturbance prediction is.

## Agents

- `safe`: no memory (M5.3).
- `safe_mem`: with disturbance memory.
- `safe_mem_guard`: with memory and the guard.

## Conditions

- **Cases:** every 10th case of the M3.2 fault/disturbance grid (168 cases).
- **Memory:** one prior episode per case, recorded by the reactive agent exactly as in M3.6, at memory-noise levels 0.0 (exact), 0.5, 1.0 and "unrelated" (a prior from a different case).
- **Sensors:** two, with an assumed bound of 5. `noise5` has true noise 5 and no fault; `wrong_bound` has true noise 15. Seeds 0 and 1 (sensor B seed + 1000).

## Measures

Per agent, memory level and sensor condition: successes, false stops, total actions. **Regret** compared with `safe` on the same run: extra actions, summed over runs where the memory agent used more actions, and the worst single-run regret.

## Decision rule (fixed now)

Memory (`safe_mem_guard`, or `safe_mem` if the guard does not help) becomes the **recommended configuration** only if all of these hold:

- **D1:** 0 false stops for the memory agents in every condition. Any false stop where `safe` has none counts as a failure.
- **D2:** with exact memory (level 0.0), total actions are lower than `safe` in **both** sensor conditions.
- **D3:** with wrong memory (levels 1.0 and unrelated), total actions are no more than 2% above `safe`.
- **D4:** the guard is "useful" if `safe_mem_guard` has a lower worst-case regret than `safe_mem` at level 1.0 or unrelated, at no more than 1% extra total actions at level 0.0.

## Expectations (stated now)

- **D1 holds,** because stopping does not use memory.
- **Exact memory helps less than in M3.5.** The agent already replans every step from its belief, so the M3.5 benefit of avoiding *replans* does not exist here. Only better first-action choices remain. D2 is uncertain.
- **Noise may trip the guard often,** whenever the boxes fail to overlap. That could make the guard switch memory off too early. Uncertain.
