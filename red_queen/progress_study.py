"""M7.3 study, exactly as pre-registered in docs/PREREG_M7_3.md.

Self-model v2 = M7 features + "am I making progress?". Both v1 (M7) and v2 are
fitted on the M7 training data (7 safety-layer conditions, seeds 0-9) and
scored on (a) NEW faults with FRESH seeds 20-29 and (b) the M7 familiar test
set (seeds 10-19). The v2 warning threshold is chosen on training data by the
M7.1 procedure.

    python3 -m red_queen.progress_study --output results/m7_3_progress_study.json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence

from .cross_isolation_study import sensors_for as m54_sensors
from .generalisation_study import NEW_FAULTS
from .self_model import SelfModel, brier, ece, features_v2, progress_labels, separation
from .self_model_study import TEST_SEEDS, TRAIN_SEEDS
from .warning_study import THRESHOLDS, collect_episodes, evaluate

FRESH_SEEDS = tuple(range(20, 30))


def v2_keys(ep: Dict[str, Any]) -> List[tuple]:
    labels = progress_labels(ep["plan_lens"], ep["stopping"])
    return [features_v2(k, p) for k, p in zip(ep["keys"], labels)]


def with_v2(episodes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Copies of the episodes whose `keys` are the v2 keys (for evaluate())."""
    return [dict(e, keys=v2_keys(e)) for e in episodes]


def scores(model: SelfModel, episodes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    preds = [model.predict(k) for e in episodes for k in e["keys"]]
    labels = [e["success"] for e in episodes for _ in e["keys"]]
    return {"n": len(preds), "brier": round(brier(preds, labels), 6), "ece": round(ece(preds, labels), 6),
            "separation": round(separation(preds, labels), 6), "mean_confidence": round(sum(preds) / len(preds), 4)}


def choose_threshold(model: SelfModel, train_v2: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    f1 = {t: evaluate(train_v2, lambda k, t=t: model.predict(k) < t)["f1"] for t in THRESHOLDS}
    best = max(f1.values())
    return {"chosen": min(t for t, v in f1.items() if v == best), "train_f1": f1}


def run_study() -> Dict[str, Any]:
    train = collect_episodes(TRAIN_SEEDS)
    train_v2 = with_v2(train)
    v1 = SelfModel().fit((k, e["success"]) for e in train for k in e["keys"])
    v2 = SelfModel().fit((k, e["success"]) for e in train_v2 for k in e["keys"])
    new = collect_episodes(FRESH_SEEDS, conditions=NEW_FAULTS, sensors_fn=m54_sensors)
    familiar = collect_episodes(TEST_SEEDS)
    new_v2, familiar_v2 = with_v2(new), with_v2(familiar)
    choice = choose_threshold(v2, train_v2)
    tau = choice["chosen"]
    per_condition = {}
    for c in NEW_FAULTS:
        sub1 = [e for e in new if e["condition"] == c]
        sub2 = [e for e in new_v2 if e["condition"] == c]
        per_condition[c] = {"success_rate": round(sum(e["success"] for e in sub1) / len(sub1), 4),
                            "v1": scores(v1, sub1), "v2": scores(v2, sub2)}
    report = {
        "milestone": "M7.3", "fresh_seeds": list(FRESH_SEEDS), "v2_threshold_selection": choice,
        "new_faults": {"v1": scores(v1, new), "v2": scores(v2, new_v2),
                       "v2_warning": evaluate(new_v2, lambda k: v2.predict(k) < tau),
                       "v1_warning_0.3": evaluate(new, lambda k: v1.predict(k) < 0.3)},
        "familiar": {"v1": scores(v1, familiar), "v2": scores(v2, familiar_v2)},
        "per_condition": per_condition,
        "progress_split": {str(k): [s, n, round(v2.predict(k), 4)] for k, (s, n) in sorted(v2.table.items())
                           if k[:3] in (("fits", "widened", ">40"), ("fits", "disagree", ">40"))},
    }
    report["decision"] = decide(report)
    return report


def decide(report: Dict[str, Any]) -> Dict[str, Any]:
    """The pre-registered decision rule (docs/PREREG_M7_3.md)."""
    n, f, w = report["new_faults"], report["familiar"], report["new_faults"]["v2_warning"]
    p1 = n["v2"]["ece"] <= 0.10
    p2 = n["v2"]["ece"] <= 0.5 * n["v1"]["ece"]
    p3 = f["v2"]["ece"] <= 0.05 and f["v2"]["brier"] <= f["v1"]["brier"] + 0.01
    p4 = w["recall"] >= 0.90 and w["precision"] >= 0.90
    return {"P1_new_fault_ece_at_most_0.10": p1, "P2_new_fault_ece_halved": p2,
            "P3_familiar_not_worse": p3, "P4_warning_recall_precision_0.90": p4, "adopt_v2": p1 and p2 and p3 and p4}


def format_report(report: Dict[str, Any]) -> str:
    sel = report["v2_threshold_selection"]
    lines = ["RED QUEEN M7.3 - 'am I making progress?' (pre-registered)",
             "v1 = M7 self-model; v2 = v1 + progress feature; both fitted on M7 training data (seeds 0-9)", "",
             f"  {'data':<34}{'model':<6}{'Brier':>9}{'ECE':>9}{'separation':>12}{'confidence':>12}", "  " + "-" * 82]
    for title, block in (("NEW faults, fresh seeds 20-29", report["new_faults"]),
                         ("familiar faults (M7 test, 10-19)", report["familiar"])):
        for m in ("v1", "v2"):
            r = block[m]
            lines.append(f"  {title:<34}{m:<6}{r['brier']:>9.4f}{r['ece']:>9.4f}{r['separation']:>12.4f}{r['mean_confidence']:>12.3f}")
    lines += ["", f"  {'new fault (seeds 20-29)':<26}{'success':>9}{'v1 conf':>9}{'v2 conf':>9}{'v1 ECE':>9}{'v2 ECE':>9}",
              "  " + "-" * 71]
    for c, r in report["per_condition"].items():
        lines.append(f"  {c:<26}{r['success_rate']:>9.3f}{r['v1']['mean_confidence']:>9.3f}{r['v2']['mean_confidence']:>9.3f}"
                     f"{r['v1']['ece']:>9.3f}{r['v2']['ece']:>9.3f}")
    w, w1 = report["new_faults"]["v2_warning"], report["new_faults"]["v1_warning_0.3"]
    lines += ["", f"v2 warning threshold (chosen on training data): {sel['chosen']}  train F1: {sel['train_f1']}",
              f"  v2 warning on new faults: precision {w['precision']:.3f} recall {w['recall']:.3f} "
              f"false alarms {w['false_alarms']} missed {w['missed_failures']} median lead {w['median_lead_steps']}",
              f"  v1 warning (0.3) for comparison: precision {w1['precision']:.3f} recall {w1['recall']:.3f} "
              f"median lead {w1['median_lead_steps']}",
              "", "progress split of the confused combinations (training successes, count, v2 confidence):"]
    for k, v in report["progress_split"].items():
        lines.append(f"  {k}: {v}")
    lines += ["", "Pre-registered decision:"] + [f"  {k}: {v}" for k, v in report["decision"].items()]
    lines += ["", "No claim of consciousness: this measures one metacognitive indicator only."]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="RED QUEEN M7.3 progress-aware self-model study")
    p.add_argument("--output", help="write the JSON report here")
    args = p.parse_args(argv)
    report = run_study()
    print(format_report(report))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        print(f"\nJSON report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
