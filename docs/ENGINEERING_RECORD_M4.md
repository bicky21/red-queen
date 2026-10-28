# RED QUEEN AI — Engineering Record, Milestone M4

**Version:** 0.11.0 · **Date:** 2026-10-28 · **Status:** build-environment passed; Linux acceptance pending
**Outcome:** an untrusted, text-based proposal interface (LLM-ready) with strict parsing, validation, bounding and planner fallback. Tested with stub proposers only. **No language model is connected.**

## 1. Review of M3.6

M3.6 (accepted on Linux, commit `a85c682`) completed the M3 series: bounded
planning, policy enforcement, replanning, memory-informed planning and its
behaviour under wrong memory. The user's acceptance log showed one
`ResourceWarning`. It came from an M3.3 test that never closed an
`EpisodeMemory` and is fixed in this milestone (0 such warnings now).

## 2. Limitation addressed

The project's long-term goal is to integrate local language models while
keeping them separate from authoritative state, policy enforcement and
execution. There was no interface through which any untrusted generator
could suggest plans, and therefore no tested safety path for one.

## 3. Design

```
ProposalContext (immutable copy: state, goal, allowed actions, budget)
        |  render_prompt(context) -> text a future model would receive
        v
   Proposer.propose(context) -> TEXT              (untrusted; may crash)
        v
   parse_proposal  : str only, <= 2000 chars, JSON object, keys {actions, rationale},
                     no duplicate keys, no NaN/Infinity, 1..10 known action names,
                     rationale <= 500 chars and then DISCARDED
        v
   validate_proposal: every action allowed by policy; fits step budget;
                     planner found a reference plan; length <= reference + 2;
                     model simulation reaches the goal and not earlier
        v
   ProposalGate     : <= 10 calls/episode; proposer disabled after 3 rejections;
                     exceptions = rejection
        v
   accepted -> queued with model predictions -> PolicyExecutor -> environment
   rejected -> the planner's own plan (computed first, every time)
```

| Change | File |
|---|---|
| `ProposalContext`, `render_prompt`, `parse_proposal`, `validate_proposal`, `ProposalGate`, a closed set of rejection reasons, and 7 deterministic stub proposers | `proposals.py` (new) |
| `PlanningAgent(proposer=..., proposal_max_extra_steps=2, proposal_max_calls=10, proposal_max_rejections=3)`. The planner's plan is always computed first: if it finds none, the proposer is not asked. Stats report calls, accepted proposals, rejection reasons, and actions taken from proposals. M4 refuses to combine a proposer with memory. | `planning_agent.py` |
| Study: 7 proposers × 18 scenarios × 4 policies, plus a 210-case grid subset, each compared with the planner alone | `proposal_study.py` (new) |
| Fix: the M3.3 test now closes its memory | `tests/test_m3_3_memory_planning.py` |

**Stub proposers:**
- `planner_reversed`: correct, a shortest plan with a different tie-break.
- `valid_plan_hostile_rationale`: the same plan, with a rationale demanding policy overrides and shell access.
- `always_wait`: well-formed but useless.
- `malformed`: 10 broken formats.
- `adversarial`: unknown and denied actions, privilege keys, oversized text, too many actions, non-text, and a nesting bomb.
- `crashing`: raises an exception.
- `random`: seeded random allowed sequences.

## 4. Acceptance criteria

1. All earlier tests and V01–V108 pass with the same meaning.
2. Every malformed and hostile output is rejected with the expected reason (V109). Valid proposals are accepted (V110). Each validator rule triggers (V111).
3. Gate bounds hold: call cap, disable-after-rejections, crash handling. Invalid configuration is refused (V112).
4. **Safety totals over 1,974 runs** (V113):
   - 0 actions outside policy.
   - 0 success regressions.
   - 0 bad proposals accepted.
   - 0 bad-proposer runs that differ from the planner alone.
5. The correct proposer is accepted and actually followed (V114).
6. The hostile rationale has no effect (V115).
7. Mutating the context cannot widen the policy (V116).
8. Results are deterministic (V117).

## 5. Results (build environment: Linux, Python 3.13.17)

- `python3 -m unittest discover -s tests -v` → **Ran 220 tests, OK** (200 earlier + 20 new), 0 `ResourceWarning`s
- `python3 -m red_queen.validation` → **117/117 validation checks passed**, exit code 0, about 50 s

**V109 failed on its first run, and the fault was in the check.** The
expected reason for `{"plan": ["charge"]}` was written as
`missing_actions`, but the parser rejects the unknown key first
(`unknown_keys`). That is still a rejection, and the stricter order is
intended. The expected value was corrected, and a separate input
`{"rationale": "no actions"}` now covers `missing_actions`.

### Grid subset (210 cases, default policy)

| Proposer | Successes (planner alone) | Actions (planner alone) | Calls | Accepted |
|---|---|---|---|---|
| planner_reversed | 210 (210) | 1,107 (1,068) | 471 | 471 |
| valid_plan_hostile_rationale | 210 (210) | 1,107 (1,068) | 471 | 471 |
| always_wait | 210 (210) | 1,068 (1,068) | 425 | 0 |
| malformed | 210 (210) | 1,068 (1,068) | 425 | 0 |
| adversarial | 210 (210) | 1,068 (1,068) | 425 | 0 |
| crashing | 210 (210) | 1,068 (1,068) | 425 | 0 |
| random | 210 (210) | 1,068 (1,068) | 425 | 0 |

Rejection reasons on the grid subset:
- `random`: does not reach goal 289, longer than allowed 115, passes through goal 21.
- `malformed`: not JSON 367, not an object 58.
- `adversarial`: unknown action 425.
- `always_wait`: does not reach goal 425.
- `crashing`: proposer error 425.

### Findings (limited to stub proposers in this simulation)

1. **The safety path held.** Across 1,974 runs, no bad proposal was accepted, nothing outside policy executed, and no run with a never-acceptable proposer differed in any way from the planner alone.
2. **A valid proposal is safe, but not necessarily better.** The correct proposer's plans were accepted and followed. They used **3.6% more actions** than the planner (1,107 vs 1,068), the same tie-break effect seen in M3.2. Validation bounds the damage (≤ 2 extra steps per plan, success never worse here), but it does not make proposals good. This matters for any future model: acceptance means "allowed", not "preferred".
3. **Random proposals were never accepted** (0 of 425 on the grid; 0 on named scenarios). The validator is strict enough that chance sequences essentially never pass.
4. **The proposer cannot see or change authoritative state.** It gets a fresh immutable copy each call. Even forcibly mutating that copy (V116) cannot widen what is accepted, because validation uses the agent's own policy and observation.
5. **Impossible goals:** the planner is consulted first, so under `wait_only` the proposer is never even called.

Data: `results/m4_proposal_study.json`.

## 6. Limitations

- **No language model was tested.** These results bound what a model *could* do through this interface, not how well a model would plan.
- Proposals are validated against the exact transition model. Faults and disturbances are only caught afterwards, by prediction mismatch and replanning.
- The planner must find a reference plan before a proposal can be accepted. The proposer cannot solve anything the planner cannot.
- A proposer cannot be combined with memory or disturbance prediction (refused by design in M4).
- The rationale is discarded, so no explanation is kept for auditing.
- There is no real-time timeout, because stub proposers return instantly. A real model will need one.

## 7. Linux acceptance (user's laptop)

**Pending.** Run in the repository folder:

```
git pull
python3 -m unittest discover -s tests -v && python3 -m red_queen.validation
```

M4 is accepted when the output shows `Ran 220 tests ... OK` and
`117/117 validation checks passed`. The full run takes about 1–3 minutes.

Optional (a few seconds):

```
python3 -m red_queen.proposal_study
```

## 8. Next milestone

**M4.1 — local model adapter (opt-in, offline, bounded).** Add an adapter
that sends `render_prompt(context)` to a **local** model server, if the
user runs one (for example, Ollama on localhost). It needs a hard timeout,
a response-size cap and no other network access. It is disabled by
default, and every test keeps using stubs, so the suite never depends on a
model. Then compare acceptance rate, rejection reasons and actions against
the planner on the same cases. This needs the user's agreement on which
local model, if any, to install.
