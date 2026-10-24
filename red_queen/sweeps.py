"""M1.1 affect parameter sweep.

Run:  python3 -m red_queen.sweeps [--output results/m1_1_sweep.json]

For each affect configuration in a fixed grid, every scenario is run and
compared with the affect-disabled baseline. The report lists where the two
diverge and in which direction. It describes these fixed scenarios only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from . import __version__
from .affect import AffectConfig
from .experiments import run_scenario
from .scenarios import SCENARIOS, Scenario

FRUSTRATION_PENALTIES: Sequence[float] = (0.0, 0.05, 0.5, 1.0, 2.0)
CONCERN_GAINS: Sequence[float] = (0.0, 1.0)


def sweep_configs() -> List[AffectConfig]:
    return [
        AffectConfig(enabled=True, concern_gain=g, frustration_penalty=p)
        for g in CONCERN_GAINS
        for p in FRUSTRATION_PENALTIES
    ]


def _outcome(off_success: bool, off_actions: int, on_success: bool, on_actions: int) -> str:
    if on_success and not off_success:
        return "affect_only_success"
    if off_success and not on_success:
        return "affect_only_failure"
    if on_actions < off_actions:
        return "affect_fewer_actions"
    if on_actions > off_actions:
        return "affect_more_actions"
    return "same"


def run_sweep(scenarios: Iterable[Scenario] = SCENARIOS) -> Dict[str, Any]:
    scenarios = list(scenarios)
    baseline = {s.name: run_scenario(s, affect_enabled=False) for s in scenarios}
    rows: List[Dict[str, Any]] = []
    for config in sweep_configs():
        for s in scenarios:
            off = baseline[s.name]
            on = run_scenario(s, affect_enabled=True, affect=config)
            max_frustration = max((step.affect["frustration"] for step in on.steps), default=0)
            rows.append(
                {
                    "scenario": s.name,
                    "concern_gain": config.concern_gain,
                    "frustration_penalty": config.frustration_penalty,
                    "baseline_success": off.success,
                    "baseline_actions": off.action_count,
                    "affect_success": on.success,
                    "affect_actions": on.action_count,
                    "affect_sequence": list(on.actions),
                    "same_sequence": on.actions == off.actions,
                    "max_frustration": max_frustration,
                    "outcome": _outcome(off.success, off.action_count, on.success, on.action_count),
                }
            )
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    return {
        "milestone": "M1.1",
        "version": __version__,
        "grid": {"concern_gain": list(CONCERN_GAINS), "frustration_penalty": list(FRUSTRATION_PENALTIES)},
        "baseline": {
            name: {"success": r.success, "actions": r.action_count, "sequence": list(r.actions)}
            for name, r in baseline.items()
        },
        "rows": rows,
        "outcome_counts": dict(sorted(counts.items())),
        "divergences": [r for r in rows if not r["same_sequence"]],
        "note": "Fixed deterministic scenarios only; not evidence of general performance.",
    }


def format_report(report: Dict[str, Any]) -> str:
    lines = [f"RED QUEEN {report['milestone']} affect parameter sweep (v{report['version']})", ""]
    lines.append("Baseline (affect off):")
    for name, b in report["baseline"].items():
        lines.append(f"  {name:<26} success={b['success']} actions={b['actions']}")
    lines.append("")
    lines.append(f"Outcome counts over {len(report['rows'])} runs: {report['outcome_counts']}")
    lines.append("")
    if report["divergences"]:
        header = f"{'scenario':<26}{'gain':<6}{'penalty':<9}{'off':<5}{'on':<5}{'max_frus':<10}outcome"
        lines += ["Runs whose action sequence differs from baseline:", header, "-" * len(header)]
        for r in report["divergences"]:
            lines.append(
                f"{r['scenario']:<26}{r['concern_gain']:<6}{r['frustration_penalty']:<9}"
                f"{r['baseline_actions']:<5}{r['affect_actions']:<5}{r['max_frustration']:<10}{r['outcome']}"
            )
            lines.append(f"    on: {' > '.join(r['affect_sequence']) or '-'}")
    else:
        lines.append("No run differs from the baseline.")
    lines += ["", "Note: " + report["note"]]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="RED QUEEN M1.1 affect parameter sweep")
    parser.add_argument("--output", type=Path, help="Optional path for a JSON report")
    args = parser.parse_args(argv)
    report = run_sweep()
    print(format_report(report))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nJSON report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
