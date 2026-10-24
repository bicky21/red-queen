# RED QUEEN AI — Engineering Record, Milestone M2

**Version:** 0.3.0 · **Date:** 2026-10-24 · **Status:** build-environment passed; **accepted on Linux (user's laptop), 2026-10-24**

## 1. Review of M1.1

M1.1 (accepted on Linux, commit `5c50695`) added scheduled action faults
and an affect sweep. The agent kept repeating a blocked action because
nothing from earlier episodes was kept anywhere.

## 2. Limitation addressed

RED QUEEN had no persistent record of experience. Every run started from
nothing, and there was no way to look up what happened before.

## 3. Scope decision

M2 adds **storage and retrieval only**. Memory does not influence any
decision in M2. This keeps memory testable on its own and lets M2.1 (or
M3.7) compare relevant-memory vs empty-memory runs against a known baseline.

## 4. Changes

| Change | File |
|---|---|
| `EpisodeMemory`: SQLite store with `episodes` and `events` tables, a `meta` table holding the schema version, and foreign-key cascade on delete | `memory.py` (new) |
| `record(result, config)`: one transaction per episode; stores scenario, config, outcome, actions, final state, events, and a searchable summary built only from episode facts | `memory.py` |
| `search(query, limit)`: lexical retrieval, score = number of distinct query words in the summary; ties go to the newest; deterministic | `memory.py` |
| `MemoryRecord`: frozen, `source="memory"`, re-validated on read (`valid`, `problems`) | `memory.py` |
| CLI: `record-all`, `search`, `stats`; default file `data/red_queen_memory.sqlite`, ignored by git | `memory_cli.py` (new) |
| Validation V32–V42, including a frozen M1.1 regression (V42) | `validation.py` |

The agent, scorer, environment and verifier are unchanged.

## 5. Safeguards

- **Parameterised SQL only.** Injection-style queries are plain data (V38).
- **Caps.** At most 1,000 episodes by default (oldest pruned, events cascade) and 500 events per episode (truncation flagged). Queries are limited to 200 characters and 1–20 results (V36, V37).
- **Schema version.** A file with a different version, or a file that isn't a database, is refused with `MemoryStoreError` (V39).
- **Untrusted on read.** A record with non-allowlisted actions, invalid JSON or a mismatched count comes back `valid=False` with reasons (V40).
- **No behavioural coupling.** Runs before and after recording are identical (V41).

## 6. Acceptance criteria

1. All earlier tests and V01–V31 still pass with the same meaning.
2. M1 and M1.1 action sequences are unchanged (V31, V42).
3. Episodes and events round-trip exactly and persist across reopen (V32, V33).
4. Retrieval ranks relevant episodes first and is deterministic (V34, V35).
5. All safeguards in section 5 hold (V36–V40).
6. Memory has no effect on behaviour (V41).

## 7. Results (build environment: Linux, Python 3.13.17)

- `python3 -m unittest discover -s tests -v` → **Ran 77 tests, OK** (57 earlier + 20 new)
- `python3 -m red_queen.validation` → **42/42 validation checks passed**, exit code 0

Example: after `record-all` (20 episodes), the query `restart blocked`
returns the `flaky_restart*` episodes first, each with score 2.

No performance claim is made: memory is not yet used for decisions.

## 8. Limitations

- Lexical matching only: no stemming, synonyms or embeddings. A query word must appear exactly.
- Search scans every row in Python (fine at ≤1,000 episodes; not indexed full-text).
- No timestamps are stored, which keeps runs reproducible but means "recent" only means "higher id".
- Single process, single writer; no concurrency handling beyond SQLite's defaults.
- Validity checks cover structure and the action allowlist, not whether a record is *true*.

## 9. Linux acceptance (user's laptop)

**Accepted, 2026-10-24.** After `git pull` to commit `d38586f`, the user ran:

```
python3 -m unittest discover -s tests -v && python3 -m red_queen.validation
```

User-reported output: `Ran 77 tests ... OK` and `42/42 validation checks passed`.
The user also ran the memory demo: `record-all` stored 20 episodes,
`search "restart blocked"` ranked the four `flaky_restart*` episodes first
(score 2), and `stats` reported 20 episodes, 74 events, `integrity_ok=True`.

Note: `test_cli_record_search_stats` printed CLI output (including an
expected `error:` line for a rejected limit) into the test log. Harmless;
silenced in M2.1.

## 10. Next milestone

**M2.1 — advisory memory.** Let retrieved, *valid* memories add a small,
bounded adjustment to action scores (e.g. "this action was blocked
recently in a similar scenario"). The current observation must always win.
Invalid records are ignored, the adjustment has a hard cap, and the run
compares relevant-memory vs empty-memory on the fault scenarios.
