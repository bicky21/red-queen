"""M7 study, exactly as pre-registered in docs/PREREG_M7.md.

SafeAgent (defaults) on the 7 M5.3 safety-layer conditions x 18 scenarios.
Train on seeds 0-9, test on seeds 10-19. Each decision point gives one
introspection snapshot; the label is the verifier's true episode outcome.

    python3 -m red_queen.self_model_study --output results/m7_self_model_study.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .agent import STOP_GOAL
from .goal import RecoveryGoal
from .safe_agent import SafeAgent
from .safety_layer_study import CONDITIONS, sensors_for
from .scenarios import ALL_SCENARIOS
from .self_model import SelfModel, brier, ece, features, heuristic, separation

GOAL = RecoveryGoal()
TRAIN_SEEDS = tuple(range(0, 10))
TEST_SEEDS = tuple(range(10, 20))


def collect(seeds: Sequence[int], conditions: Sequence[str] = tuple(CONDITIONS),
            scenarios=ALL_SCENARIOS) -> List[Dict[str, Any]]:
    """One record per decision point: condition, key, label, false-stop flag."""
    records = []
    for cond in conditions:
        for sc in scenarios:
            for seed in seeds:
                snaps: List[Dict[str, Any]] = []
                r = SafeAgent(GOAL, 5, max_steps=sc.max_steps).run(sc.make_env(), sensors_for(cond, seed),
                                                                   sc.name, observer=snaps.append)
                false_stop = r.stop_reason == STOP_GOAL and not r.success
                for snap in snaps:
                    records.append({"condition": cond, "key": features(snap), "success": r.success,
                                    "stopping": snap["stopping"], "false_stop": false_stop})
    return records


def score(name: str, preds: Sequence[float], labels: Sequence[bool]) -> Dict[str, Any]:
    return {"predictor": name, "n": len(preds), "brier": round(brier(preds, labels), 6),
            "ece": round(ece(preds, labels), 6), "separation": round(separation(preds, labels), 6)}


def run_study() -> Dict[str, Any]:
    train, test = collect(TRAIN_SEEDS), collect(TEST_SEEDS)
    model = SelfModel().fit((r["key"], r["success"]) for r in train)
    labels = [r["success"] for r in test]
    predictions = {
        "base_rate": [model.base_rate] * len(test),
        "heuristic": [heuristic(r["key"]) for r in test],
        "self_model": [model.predict(r["key"]) for r in test],
    }
    scores = [score(n, p, labels) for n, p in predictions.items()]
    per_condition = []
    for cond in CONDITIONS:
        idx = [i for i, r in enumerate(test) if r["condition"] == cond]
        sub_labels = [labels[i] for i in idx]
        per_condition.append({"condition": cond, "snapshots": len(idx),
                              "success_rate": round(sum(sub_labels) / len(idx), 4),
                              "mean_self_confidence": round(sum(predictions["self_model"][i] for i in idx) / len(idx), 4),
                              "self_model_brier": round(brier([predictions["self_model"][i] for i in idx], sub_labels), 6)})
    fs_idx = [i for i, r in enumerate(test) if r["stopping"] and r["false_stop"]]
    blind = {"false_stop_moments": len(fs_idx),
             "mean_self_confidence": round(sum(predictions["self_model"][i] for i in fs_idx) / len(fs_idx), 4)
             if fs_idx else None,
             "conditions": dict(Counter(test[i]["condition"] for i in fs_idx))}
    table = sorted(([list(k), s, n, round(model.predict(k), 4)] for k, (s, n) in model.table.items()),
                   key=lambda row: -row[2])
    return {"milestone": "M7", "train_seeds": list(TRAIN_SEEDS), "test_seeds": list(TEST_SEEDS),
            "train_snapshots": len(train), "test_snapshots": len(test), "train_base_rate": round(model.base_rate, 6),
            "scores": scores, "per_condition": per_condition, "blind_spot": blind,
            "self_model_table": table, "decision": decide(scores)}


def decide(scores: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The pre-registered decision rule (docs/PREREG_M7.md)."""
    s = {r["predictor"]: r for r in scores}
    sm, base, heur = s["self_model"], s["base_rate"], s["heuristic"]
    s1 = sm["brier"] <= 0.8 * base["brier"]
    s2 = sm["ece"] <= 0.05
    s3 = sm["brier"] <= heur["brier"]
    return {"S1_brier_20pct_below_base_rate": s1, "S2_ece_at_most_0.05": s2,
            "S3_not_worse_than_heuristic": s3, "indicator_present_measured_sense": s1 and s2 and s3}


def format_report(report: Dict[str, Any]) -> str:
    lines = ["RED QUEEN M7 - self-model: does RED QUEEN know when it will succeed? (pre-registered)",
             f"train seeds 0-9: {report['train_snapshots']} snapshots; test seeds 10-19: "
             f"{report['test_snapshots']} snapshots; training success rate {report['train_base_rate']:.3f}", "",
             f"  {'predictor':<12}{'Brier':>9}{'ECE':>9}{'separation':>12}", "  " + "-" * 42]
    for r in report["scores"]:
        lines.append(f"  {r['predictor']:<12}{r['brier']:>9.4f}{r['ece']:>9.4f}{r['separation']:>12.4f}")
    lines += ["", f"  {'condition':<20}{'true success':>13}{'self confidence':>17}{'Brier':>9}", "  " + "-" * 59]
    for r in report["per_condition"]:
        lines.append(f"  {r['condition']:<20}{r['success_rate']:>13.3f}{r['mean_self_confidence']:>17.3f}"
                     f"{r['self_model_brier']:>9.4f}")
    b = report["blind_spot"]
    lines += ["", f"Blind spot: {b['false_stop_moments']} false-stop moments in the test set; "
              f"mean self-confidence there = {b['mean_self_confidence']} ({b['conditions']})"]
    d = report["decision"]
    lines += ["", "Pre-registered decision (test set):"] + [f"  {k}: {v}" for k, v in d.items()]
    lines += ["", "Brier: lower is better. ECE: calibration error, lower is better.",
              "separation: mean confidence on successful minus failed episodes, higher is better.",
              "No claim of consciousness: this measures one indicator property only."]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="RED QUEEN M7 self-model study")
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
