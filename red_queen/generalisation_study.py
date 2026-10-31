"""M7.2 study, exactly as pre-registered in docs/PREREG_M7_2.md.

Does RED QUEEN's self-knowledge generalise to fault types it never saw?
The M7 self-model (fitted on the 7 safety-layer conditions, seeds 0-9), the
M7.1 warning threshold (0.3) and the M7 base rate are used UNCHANGED, and
evaluated on the M5.4 confirmation conditions, seeds 10-19.

    python3 -m red_queen.generalisation_study --output results/m7_2_generalisation_study.json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence

from .cross_isolation_study import CONFIRMATION, sensors_for as m54_sensors
from .self_model import SelfModel, brier, ece, heuristic, separation
from .self_model_study import TEST_SEEDS, TRAIN_SEEDS
from .warning_study import baseline_fires, collect_episodes, evaluate

M7_1_THRESHOLD = 0.3  # chosen in M7.1 on training data; NOT re-chosen here
NEW_FAULTS = ("A_drift", "A_stuck_mid", "B_stuck_low", "wrong_bound+B_stuck_low")
SEEN_TYPES = ("none", "wrong_bound")


def snapshot_scores(model: SelfModel, episodes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keys = [k for e in episodes for k in e["keys"]]
    labels = [e["success"] for e in episodes for _ in e["keys"]]
    preds = {"base_rate": [model.base_rate] * len(keys), "heuristic": [heuristic(k) for k in keys],
             "self_model": [model.predict(k) for k in keys]}
    return [{"predictor": n, "n": len(keys), "brier": round(brier(p, labels), 6), "ece": round(ece(p, labels), 6),
             "separation": round(separation(p, labels), 6)} for n, p in preds.items()]


def summarise(model: SelfModel, episodes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    success = sum(e["success"] for e in episodes)
    conf = [model.predict(k) for e in episodes for k in e["keys"]]
    return {"episodes": len(episodes), "success_rate": round(success / len(episodes), 4),
            "mean_self_confidence": round(sum(conf) / len(conf), 4),
            "snapshot_scores": snapshot_scores(model, episodes),
            "warning": evaluate(episodes, lambda k: model.predict(k) < M7_1_THRESHOLD),
            "baseline_alert": evaluate(episodes, baseline_fires)}


def run_study() -> Dict[str, Any]:
    train = collect_episodes(TRAIN_SEEDS)  # exactly the M7 / M7.1 training data
    model = SelfModel().fit((k, e["success"]) for e in train for k in e["keys"])
    test = collect_episodes(TEST_SEEDS, conditions=tuple(CONFIRMATION), sensors_fn=m54_sensors)
    per_condition = {c: summarise(model, [e for e in test if e["condition"] == c]) for c in CONFIRMATION}
    primary = summarise(model, [e for e in test if e["condition"] in NEW_FAULTS])
    seen = summarise(model, [e for e in test if e["condition"] in SEEN_TYPES])
    return {"milestone": "M7.2", "train_base_rate": round(model.base_rate, 6), "threshold": M7_1_THRESHOLD,
            "primary_new_faults": primary, "seen_types": seen, "per_condition": per_condition,
            "decision": decide(primary)}


def decide(primary: Dict[str, Any]) -> Dict[str, Any]:
    """The pre-registered decision rule (docs/PREREG_M7_2.md), primary set only."""
    s = {r["predictor"]: r for r in primary["snapshot_scores"]}
    w = primary["warning"]
    g1 = s["self_model"]["ece"] <= 0.10
    g2 = s["self_model"]["brier"] <= 0.8 * s["base_rate"]["brier"]
    g3 = w["recall"] >= 0.70 and w["precision"] >= 0.70
    return {"G1_ece_at_most_0.10": g1, "G2_brier_20pct_below_base_rate": g2,
            "G3_warning_recall_and_precision_at_least_0.70": g3, "generalises": g1 and g2 and g3}


def format_report(report: Dict[str, Any]) -> str:
    lines = ["RED QUEEN M7.2 - does self-knowledge generalise to unseen faults? (pre-registered)",
             f"self-model, base rate ({report['train_base_rate']:.3f}) and warning threshold ({report['threshold']}) "
             "from M7/M7.1, unchanged; test seeds 10-19", ""]
    for title, block in (("NEW fault types (primary): " + ", ".join(NEW_FAULTS), report["primary_new_faults"]),
                         ("already-seen types (reference): " + ", ".join(SEEN_TYPES), report["seen_types"])):
        lines += [title, f"  episodes {block['episodes']}, true success {block['success_rate']:.3f}, "
                  f"mean self-confidence {block['mean_self_confidence']:.3f}",
                  f"  {'predictor':<12}{'Brier':>9}{'ECE':>9}{'separation':>12}"]
        for r in block["snapshot_scores"]:
            lines.append(f"  {r['predictor']:<12}{r['brier']:>9.4f}{r['ece']:>9.4f}{r['separation']:>12.4f}")
        for name, w in (("warning", block["warning"]), ("baseline", block["baseline_alert"])):
            lines.append(f"  {name:<12} precision {w['precision']:.3f}  recall {w['recall']:.3f}  "
                         f"false alarms {w['false_alarms']}  missed {w['missed_failures']}  median lead {w['median_lead_steps']}")
        lines.append("")
    lines += [f"  {'condition':<25}{'success':>9}{'confidence':>12}{'ECE':>8}{'warned':>8}{'false al.':>11}{'missed':>8}",
              "  " + "-" * 81]
    for c, b in report["per_condition"].items():
        sm = {r["predictor"]: r for r in b["snapshot_scores"]}["self_model"]
        w = b["warning"]
        lines.append(f"  {c:<25}{b['success_rate']:>9.3f}{b['mean_self_confidence']:>12.3f}{sm['ece']:>8.3f}"
                     f"{w['warned_failures']:>8}{w['false_alarms']:>11}{w['missed_failures']:>8}")
    lines += ["", "Pre-registered decision (new fault types only):"] + [f"  {k}: {v}" for k, v in report["decision"].items()]
    lines += ["", "No claim of consciousness: this tests whether one measured indicator survives new situations."]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="RED QUEEN M7.2 self-knowledge generalisation study")
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
