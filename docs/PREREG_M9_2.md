# RED QUEEN AI — M9.2 pre-registration: learning "distance to goal" from experience

**Written and committed before any M9.2 code was written or run.** Date: 2026-11-22.

## Question

In M9, both successful agents depended on a **hand-written** heuristic (`goal_distance`), an estimate of how far a state is from the goal. Can RED QUEEN **learn** a better estimate from its own past episodes, and does that make planning cheaper without making it worse?

## What is built

**Features** (fixed now), computed from a `DCState`:
1. a constant 1
2. the number of down servers
3. the sum over down servers of max(0, temperature − 60) / 5 (how long they must cool before a restart)
4. the sum over up servers of max(0, temperature − 70) / 5
5. the sum over all servers of max(0, load − 40) / 20
6. |demand − served| / 20
7. the maximum temperature among up servers / 10 (0 if none are up)
8. the number of up servers with load ≥ 60 (servers that are heating up)

**Learner:** linear ridge regression (λ = 0.01), solved exactly with the standard library. The prediction is clipped to ≥ 0, and any non-finite value falls back to the hand-written heuristic. **No randomness in fitting.**

**Training data (seeds 2000–2199, never used before):**
- For each seed, take `make_scenario(seed)` plus the states along a deterministic random walk of up to 5 valid actions (walk seed `walk:<seed>`).
- Label each state with the length of `reference_plan` (weighted A* with the hand heuristic) from that state.
- States with no plan found are dropped. Goal states are labelled 0.

**Plug-in:** `reference_plan(..., heuristic=...)` and `greedy_action(..., heuristic=...)`. **The defaults stay the hand-written `goal_distance`, so M9 results are unchanged.** The heuristic only changes the *order* in which the search looks at states. The executor, preconditions, budget and verifier are unchanged.

## Evaluation (seeds 3000–3099, never used before)

Closed loop exactly as in M9: exact observations, replan every step, at most 25 actions.

| agent | planner |
|---|---|
| `astar_hand` | weighted A* (weight 2, depth 25, 100,000 nodes), hand heuristic (= the M9 `reference`) |
| `astar_learned` | the same, with the learned heuristic |
| `greedy_hand` | M9 greedy, hand heuristic |
| `greedy_learned` | greedy, learned heuristic |

**Also reported:** on held-out states (from the same walk procedure on seeds 3000–3099), the learned heuristic's mean absolute error against the reference plan length, and the Pearson correlation with that length for both heuristics.

## Decision rule (fixed now; primary = A*)

The learned heuristic is **useful** only if all hold on seeds 3000–3099:

- **L1:** `astar_learned` successes ≥ `astar_hand` successes
- **L2:** `astar_learned` total planner nodes ≤ 0.8 × `astar_hand` total planner nodes
- **L3:** `astar_learned` total actions ≤ 1.05 × `astar_hand` total actions

Greedy results are reported, but they are not part of the decision.

## Expectations (stated now)

- The hand heuristic is already efficient: 7,388 nodes for 100 scenarios in M9. So a **20% node reduction is uncertain**.
- The learned heuristic should correlate with true distance at least as well as the hand one, because it is fitted to that target.
- **Labels come from the hand-heuristic planner**, so the learned model partly imitates it. That limits how much better it can get. This is a stated design weakness.
- Safety is unaffected by construction. No claim of "learning to think": this is a fitted linear estimate.
