# RED QUEEN AI — M5.4 pre-registration

**Written and committed before any M5.4 code was written or run.** Date: 2026-10-30.
The git commit containing this file, which comes before the M5.4 implementation commit, is the evidence of order.

## Question

In M5.3, a sensor stuck at pessimistic values (`A_dead_low`) was usually *explained away*: its noise bound widened to 25 instead of the sensor being isolated, so the safe agent recovered only 101 of 360 runs. Can a cross-sensor rule isolate such a sensor **without** isolating healthy sensors, and without creating false stops?

## Rule to be tested (`cross_isolate=True`)

At a step where both sensors are still trusted and their observation boxes do **not** overlap, the agent looks at how far each sensor's noise bound has grown above the assumed value. If **exactly one** sensor's bound has grown, that sensor is isolated, with an alert `sensor_isolated`, reason `cross_sensor`. Otherwise nothing changes. Everything else is as in M5.3 `safe` (adaptation, hull, isolate on unexplainable).

Rationale: with a correct bound, a healthy sensor's bound never grows (shown in M5.1). If only one sensor needed widening, and the two now disagree, that sensor is the likely faulty one.

Known risk, stated in advance: under a **wrong** bound, a healthy sensor can widen first and be isolated wrongly. The `wrong_bound` conditions test this.

## Agents compared

- `safe`: M5.3 default, unchanged.
- `safe_x`: `safe` with `cross_isolate=True`.

## Conditions

The assumed bound is 5 everywhere. Each condition covers 18 named scenarios × 20 seeds, with sensor B's seed equal to seed + 1000.

**Confirmation conditions (new; not seen before this file):**

| name | fault |
|---|---|
| `none` | no fault, true noise 5 |
| `wrong_bound` | true noise 15 on both, no other fault |
| `A_drift` | A's temperature reading drifts −2 per step (step k reads −2k), true noise 5 |
| `A_stuck_mid` | from step 1, A reports battery 70 / temperature 40 (plausible healthy values), true noise 5 |
| `B_stuck_low` | from step 1, **B** reports battery 10 / temperature 90, true noise 5 |
| `wrong_bound+B_stuck_low` | `B_stuck_low` with true noise 15 on both |

**Development conditions (the motivating cases; reported separately, not used for the decision):** `A_dead_low`, and `A_dead`, as defined in M5.3.

## Measures

For each agent and condition: successes, false stops, safe halts, actions. Also **wrongful isolations**: runs in which a sensor without an injected fault was isolated (in `none` and `wrong_bound`, any isolation is wrongful).

## Decision rule (fixed now)

`cross_isolate` becomes the default in `SafeAgent` **only if all three hold** on the confirmation conditions:

- **C1:** `safe_x` has 0 false stops in every confirmation condition.
- **C2:** `safe_x` has successes ≥ `safe` in every confirmation condition, and strictly more in at least one stuck or drift condition.
- **C3:** `safe_x` has 0 wrongful isolations in `none`, and no more than 5% of runs (18 of 360) in `wrong_bound`.

Otherwise the rule stays available, but off by default, and the result is reported as negative or mixed.

## Expectations (stated now)

- `none`: identical, no isolations.
- `A_stuck_mid` and `B_stuck_low`: `safe_x` succeeds more often than `safe`.
- `A_drift`: little difference. Drift is gradual, so the bound widens slowly, and the boxes may rarely be disjoint.
- `wrong_bound`: some wrongful isolations are possible. Whether C3 holds is uncertain.
