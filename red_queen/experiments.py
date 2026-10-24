"""M1 experiment: affect-enabled vs affect-disabled scoring.

Run:  python -m red_queen.experiments [--output results/m1_affect_comparison.json]

The report only describes what happened in these fixed scenarios. It is
not evidence that affect signals help in general.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import MILESTONE, __version__
from .affect import AffectConfig
from .agent import EpisodeResult, RecoveryAgent
from .goal import RecoveryGoal
from .scenarios import SCENARIOS, Scenario


def run_scenario(
    scenario: Scenario,
    affect_enabled: bool,
    goal: RecoveryGoal = RecoveryGoal(),
    affect: Optional[AffectConfig] = None,
) -> EpisodeResult:
    """Run one episode. `affect` (M1.1) overrides the default config; its
    `enabled` flag must agree with `affect_enabled`."""
    if affect is None:
        affect = AffectConfig(enabled=affect_enabled)
    elif affect.enabled != affect_enabled:
        raise ValueError("affect.enabled does not match affect_enabled")
    agent = RecoveryAgent(goal, affect, max_steps=scenario.max_steps)
    return agent.run(scenario.make_env(), scenario=scenario.name)


def _summary(rows: List[Dict[str, Any]], enabled: bool) -> Dict[str, Any]:
    mine = [r for r in rows if r["affect_enabled"] == enabled]
    ok = [r for r in mine if r["success"]]
    return {
        "episodes": len(mine),
        "successes": len(ok),
        "total_actions": sum(r["action_count"] for r in mine),
        "mean_actions_successful": round(sum(r["action_count"] for r in ok) / len(ok), 3) if ok else None,
        "total_events": sum(r["event_count"] for r in mine),
    }


def run_affect_comparison(scenarios: Iterable[Scenario] = SCENARIOS) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    per_scenario: List[Dict[str, Any]] = []
    for scenario in scenarios:
        off = run_scenario(scenario, affect_enabled=False)
        on = run_scenario(scenario, affect_enabled=True)
        for result in (off, on):
            rows.append(
                {
                    "scenario": scenario.name,
                    "affect_enabled": result.affect_enabled,
                    "success": result.success,
                    "stop_reason": result.stop_reason,
                    "action_count": result.action_count,
                    "actions": list(result.actions),
                    "event_count": result.event_count,
                    "final_state": result.final_state.to_dict(),
                }
            )
        if off.action_count == on.action_count:
            difference = "same"
        elif on.action_count < off.action_count:
            difference = "affect_fewer_actions"
        else:
            difference = "affect_more_actions"
        per_scenario.append(
            {
                "scenario": scenario.name,
                "same_action_sequence": off.actions == on.actions,
                "action_count_difference": difference,
                "both_succeeded": off.success and on.success,
            }
        )
    return {
        "milestone": MILESTONE,
        "version": __version__,
        "rows": rows,
        "per_scenario": per_scenario,
        "summary": {"affect_disabled": _summary(rows, False), "affect_enabled": _summary(rows, True)},
        "note": "Fixed deterministic scenarios only; not evidence of general performance.",
    }


def format_table(report: Dict[str, Any]) -> str:
    lines = [f"RED QUEEN {report['milestone']} affect comparison (v{report['version']})", ""]
    header = f"{'scenario':<26}{'affect':<8}{'success':<9}{'actions':<9}{'events':<8}sequence"
    lines += [header, "-" * len(header)]
    for r in report["rows"]:
        lines.append(
            f"{r['scenario']:<26}{('on' if r['affect_enabled'] else 'off'):<8}"
            f"{str(r['success']):<9}{r['action_count']:<9}{r['event_count']:<8}"
            f"{' > '.join(r['actions']) or '-'}"
        )
    lines.append("")
    for mode in ("affect_disabled", "affect_enabled"):
        s = report["summary"][mode]
        lines.append(
            f"{mode:<16} successes {s['successes']}/{s['episodes']}  total actions {s['total_actions']}"
            f"  mean actions (successful) {s['mean_actions_successful']}"
        )
    lines.append("")
    lines.append("Note: " + report["note"])
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="RED QUEEN M1 affect comparison")
    parser.add_argument("--output", type=Path, help="Optional path for a JSON report")
    args = parser.parse_args(argv)
    report = run_affect_comparison()
    print(format_table(report))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nJSON report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
