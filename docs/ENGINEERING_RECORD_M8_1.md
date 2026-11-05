# RED QUEEN AI — Engineering Record, Milestone M8.1 (housekeeping)

**Version:** 0.24.1 · **Date:** 2026-11-05 · **Status:** build-environment passed; Linux acceptance pending
**Outcome:** a quick validation mode for everyday use (**156 checks in about 22 s**). The full acceptance run no longer executes the whole validation twice. **No check was changed, added or removed:** there are still 211 checks, and the full run is identical.

## 1. Review of M8

M8 was accepted on Linux on 2026-11-05, with study output identical by `diff`. The acceptance command had grown to about 10 minutes on the user's laptop (tests 339 s plus validation). One unit test, `test_all_validation_checks_pass`, ran **all 211 checks** inside the test suite, which is about 95% of the test time, and then the acceptance command ran them all **again**.

## 2. Limitation addressed

Slow feedback discourages running the checks often, and running the full validation twice per acceptance added nothing.

## 3. Changes

| Change | File |
|---|---|
| `FULL_ONLY`: an explicit, visible set of 55 check IDs that are slow (more than about 1 s) **or** reuse a large cached study. The second condition matters because a cheap-looking check that reads a cache would recompute the whole study when run alone. The set was found by timing every check, then timing the remaining set on its own. | `validation.py` |
| `select_checks(quick)` and `python3 -m red_queen.validation --quick`, which runs the other 156 checks and says clearly that it is a quick run. Without `--quick`, the run is the full one, exactly as before. | `validation.py` |
| `test_all_validation_checks_pass` now runs the **quick** set. `RED_QUEEN_FULL_VALIDATION=1` makes it run the full set. The full set still runs in the acceptance command. | `tests/test_agent_integration.py` |
| New tests: every FULL_ONLY ID exists; quick and full partition the 211 checks in order; the quick set still contains at least one check from **every** milestone, M1 to M8. | `tests/test_m8_1_quick_validation.py` |

## 4. How to use it (simple)

- **Everyday, after any change (about 30–40 s):**
  ```
  python3 -m unittest discover -s tests && python3 -m red_queen.validation --quick
  ```
- **Acceptance, for each milestone (the full run):**
  ```
  python3 -m unittest discover -s tests -v && python3 -m red_queen.validation
  ```

## 5. Results (build environment: Linux, Python 3.13)

- `python3 -m unittest discover -s tests -v` → **Ran 381 tests, OK** (378 + 3 new); `python3 -X dev` run: **0 ResourceWarnings**
- `python3 -m red_queen.validation` (full) → **211/211 validation checks passed**
- `python3 -m red_queen.validation --quick` → **156/156 passed, 55 slow checks skipped**, in 22.3 s

Timing in the build environment (my machine; the laptop's times will differ):
- unit tests: about 270 s before, **37 s** now (measured)
- quick validation: **22 s**
- full validation: 253 s, as before (measured)

So the acceptance command (tests + full validation) drops from about 520 s to **about 290 s**, and the everyday check is now under a minute.

## 6. Limitations

- The quick run is a **subset**. A change can pass quick and fail full, so acceptance must always use the full run.
- The FULL_ONLY set is maintained by hand. New milestones must decide where their checks go, and the partition test catches stale IDs.

## 7. Linux acceptance (user's laptop)

**Pending.** Run in the repository folder:

```
git pull
python3 -m unittest discover -s tests -v && python3 -m red_queen.validation
python3 -m red_queen.validation --quick
```

M8.1 is accepted when the output shows `Ran 381 tests ... OK`, `211/211 validation checks passed`, and `QUICK run: 156/156 validation checks passed`.

## 8. Next milestone

**M9: a harder domain.** This is a change of direction, and the user decides. The language model (M4), memory (M6) and the global workspace (M8) all showed little room to help in the current world, because an exact planner already does almost everything right.
