# RED QUEEN AI — Engineering Record, Milestone M7.3

**Version:** 0.23.0 · **Date:** 2026-10-31 · **Status:** build-environment passed; Linux acceptance pending
**Outcome: pre-registered result. A large improvement, but below the bar I set in advance.** A new self-knowledge feature, **"am I making progress?"**, makes the self-model much better everywhere:
- On never-seen faults (fresh seeds): ECE from 0.292 to **0.188**, Brier from 0.203 to **0.120**.
- On familiar faults: ECE from 0.020 to **0.010**, Brier from 0.096 to **0.052**.

The pre-registered targets for new faults were ECE ≤ 0.10 and at least halved. Both were missed. By the rule committed before any code, the new model (**v2**) is **not adopted as the default**. It stays available as an option.

## 1. Review of M7.2

M7.2 was accepted on Linux on 2026-10-31, with study output identical by `diff`. Its finding was that ranking generalises to unseen faults but calibration does not (ECE 0.289). It proposed an "unfamiliarity" flag as the fix.

## 2. Diagnosis, and an honest change of plan

Before pre-registering, I checked *why* calibration broke, using the already-published M7.2 test data (seeds 10–19). This showed the unfamiliarity flag **would not have worked**:
- **95% of new-fault snapshots** (9,794 of 10,281) had feature combinations seen **200+ times** in training. The situations were familiar, not new.
- The main error was one familiar combination, ("plan fits", "bound widened", "wide belief"). Its success rate was about 0.66 in training (wrong-bound runs that recover), but 0.01 on the new faults (stuck-sensor runs that never finish).
- The difference was progress: in recoverable runs the worst-corner plan **shrinks** step by step; with a stuck sensor it does not.

So M7.3 tests a different question, "am I making progress?". Because the design was informed by seeds 10–19, it was **evaluated on fresh seeds 20–29**, which nobody had run before.

## 3. Method: pre-registration

`docs/PREREG_M7_3.md` was committed before any M7.3 code (commit `01f702d`, 2026-10-31 08:30 UTC). It records the diagnosis and the change of plan, and fixes:
- the progress feature
- that v2 is fitted on the same training data as M7
- the fresh evaluation seeds
- the threshold procedure
- criteria P1–P4, and the expectations

Nothing was changed after the run.

## 4. Changes

| Change | File |
|---|---|
| `progress_labels(plan_lens, stopping)`: for each decision point, `early` (first 2), `progressing` (worst-corner plan shorter than 2 decision points ago), or `stalled`. A missing plan counts as infinitely long, and the stop moment as 0. Uses only the agent's own history. `features_v2(key, progress)` adds it as a 4th feature. | `self_model.py` |
| `ConfidenceMonitor(..., use_progress=False)`: with `True`, it tracks the episode's plan lengths and uses v2 keys live. **The default is unchanged**, and the M7.1 results are identical (V199). | `self_model.py` |
| `collect_episodes` also records `plan_lens` and `stopping` for each snapshot (additive). | `warning_study.py` |
| The pre-registered study, with the decision rule as code. | `progress_study.py` (new) |

## 5. Acceptance criteria (build environment)

1. All 359 earlier tests and V01–V197 pass with the same meaning.
2. Progress labels are exact on a constructed case (V198). The M7.1 warning results are unchanged (V199). The live progress-aware monitor matches the offline computation on 72 fresh episodes (V200).
3. The scores are frozen, copied from the output, with fresh seeds 20–29 (V201). The pre-registered decision is applied (V202). The improvement is recorded (V203). The pre-registration is present (V204).

## 6. Results (build environment: Linux, Python 3.13)

- `python3 -m unittest discover -s tests -v` → **Ran 367 tests, OK** (359 + 8 new); `python3 -X dev` run: **0 ResourceWarnings**
- `python3 -m red_queen.validation` → **204/204 validation checks passed**

One new unit test failed on its first run, a **test bug**: it lowercased the file text but searched for a capital "I". It was fixed in the test.

`python3 -m red_queen.progress_study`:

| data | model | Brier ↓ | ECE ↓ | separation ↑ |
|---|---|---|---|---|
| **new faults, fresh seeds 20–29** | v1 (M7) | 0.2033 | 0.2916 | 0.561 |
| | **v2 (+ progress)** | **0.1202** | **0.1876** | **0.698** |
| **familiar faults (M7 test, 10–19)** | v1 (M7) | 0.0962 | 0.0197 | 0.495 |
| | **v2 (+ progress)** | **0.0520** | **0.0095** | **0.709** |

| new fault (seeds 20–29) | true success | v1 confidence | v2 confidence | v1 ECE | v2 ECE |
|---|---|---|---|---|---|
| A_drift | 1.000 | 0.909 | 0.936 | 0.091 | 0.064 |
| A_stuck_mid | 0.367 | 0.521 | 0.390 | 0.351 | 0.229 |
| B_stuck_low | 0.144 | 0.353 | 0.219 | 0.330 | 0.196 |
| wrong_bound + B_stuck_low | 0.144 | 0.367 | 0.256 | 0.346 | 0.237 |

**How progress splits the confused combinations** (training successes / count → v2 confidence):

| combination | early | progressing | stalled |
|---|---|---|---|
| plan fits, bound widened, wide belief | 164/164 → 0.997 | 1072/1072 → 0.9995 | **411/1267 → 0.325** |
| plan fits, sensors disagree, wide belief | 157/157 → 0.997 | 51/51 → 0.990 | **9/328 → 0.032** |

**Warning (v2, threshold 0.3, chosen on training data):** on new faults, precision 0.993, recall 1.000, 3 false alarms out of 298 successes, 0 missed, median lead 10 steps. For comparison, the v1 warning had precision 1.000, recall 1.000 and a median lead of 8 steps.

**Pre-registered decision** (computed by `decide`, V202):
- **P1** (new-fault ECE ≤ 0.10): **False** (0.188)
- **P2** (new-fault ECE halved): **False** (0.188 > 0.146)
- **P3** (familiar faults not worse): **True** (it got much better)
- **P4** (warning recall and precision ≥ 0.90): **True**
- **Adopt v2: no**

**What this shows**
- **"Am I making progress?" is real, useful self-knowledge.** Every measure improved, on new *and* familiar faults. The improvement on familiar faults is the largest: Brier fell 46%. That was not the target, but it is clear evidence that the feature carries information.
- **It fixes part of the confusion, not all of it.** "Plan fits, sensors disagree, wide belief, stalled" now correctly drops to 0.03. But the "bound widened … stalled" case is still mixed in training (411 of 1,267 succeeded), because wrong-bound runs also stall sometimes and then recover. v2 says 0.33 there, while the new faults succeed almost never. That is why calibration on new faults is still off.
- **The warning got earlier (10 vs 8 steps), but slightly noisier** (3 false alarms vs 0).
- **The decision rule is applied as written.** v2 is clearly better than v1, but it missed the targets I set, so it is not made the default. A reasonable next step would be to pre-register adopting v2 on the basis of its familiar-fault gains. That would be a new decision, not a re-reading of this one.

## 7. Limitations

- **This is not consciousness.** It measures one metacognitive indicator.
- The progress feature uses a fixed lag of 2 decision points, and no other lags were tried. Trying them now on this data would be tuning.
- The design was informed by seeds 10–19 of the new faults. The evaluation used fresh seeds, but the same fault *types*, so the new-fault result is not fully independent of the diagnosis.
- The common-mode blind spot is untouched. Progress cannot reveal a bias that both sensors share.

## 8. Linux acceptance (user's laptop)

**Pending.** Run in the repository folder:

```
git pull
python3 -m unittest discover -s tests -v && python3 -m red_queen.validation
python3 -m red_queen.progress_study
```

M7.3 is accepted when the output shows `Ran 367 tests ... OK` and `204/204 validation checks passed`. The study output should match §6.

## 9. Next milestone (candidates)

1. **M8: global workspace**, the second indicator family. The self-model work (M7–M7.3) has reached a natural stopping point: it is calibrated in familiar situations, ranks unseen ones correctly, and its remaining errors are understood.
2. **M7.4:** a pre-registered decision to adopt v2 as the default on the basis of its familiar-fault improvement, with a new and independent fault family to confirm it.
