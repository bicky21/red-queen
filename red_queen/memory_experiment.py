"""M2.1 experiment: empty vs irrelevant vs relevant memory.

Run:  python3 -m red_queen.memory_experiment [--output results/m2_1_memory_experiment.json]

Memory conditions (built fresh, in RAM, for every scenario):
  empty      - no stored episodes
  irrelevant - one prior affect-off run of every *other* scenario
  relevant   - one prior affect-off run of every scenario, including this one

Each scenario is then run with affect off and on, with a MemoryAdvisor.
The report describes these fixed scenarios only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import MILESTONE, __version__
from .affect import AffectConfig
from .experiments import run_scenario
from .memory import EpisodeMemory
from .memory_advice import MemoryAdvisor
from .scenarios import SCENARIOS, Scenario

CONDITIONS = ("empty", "irrelevant", "relevant")


def build_memory(condition: str, target: Scenario, scenarios: List[Scenario]) -> EpisodeMemory:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown memory condition: {condition!r}")
    memory = EpisodeMemory(":memory:")
    if condition == "empty":
        return memory
    for s in scenarios:
        if condition == "irrelevant" and s.name == target.name:
            continue
        config = AffectConfig(enabled=False)
        memory.record(run_scenario(s, False, affect=config), config)
    return memory


def run_condition(scenario: Scenario, condition: str, affect_enabled: bool,
                  scenarios: Optional[List[Scenario]] = None):
    scenarios = list(scenarios or SCENARIOS)
    with build_memory(condition, scenario, scenarios) as memory:
        advisor = MemoryAdvisor(memory, scenario.name)
        result = run_scenario(scenario, affect_enabled, advisor=advisor)
        return result, advisor.summary()


def run_memory_experiment(scenarios: Iterable[Scenario] = SCENARIOS) -> Dict[str, Any]:
    scenarios = list(scenarios)
    rows: List[Dict[str, Any]] = []
    for s in scenarios:
        for enabled in (False, True):
            for condition in CONDITIONS:
                result, advice = run_condition(s, condition, enabled, scenarios)
                rows.append(
                    {
                        "scenario": s.name,
                        "affect_enabled": enabled,
                        "memory": condition,
                        "success": result.success,
                        "action_count": result.action_count,
                        "actions": list(result.actions),
                        "blocked_events": sum(1 for e in result.events if e["kind"] == "blocked"),
                        "advisor": advice,
                    }
                )
    summary: Dict[str, Any] = {}
    for condition in CONDITIONS:
        mine = [r for r in rows if r["memory"] == condition]
        summary[condition] = {
            "episodes": len(mine),
            "successes": sum(r["success"] for r in mine),
            "total_actions": sum(r["action_count"] for r in mine),
            "total_blocked_events": sum(r["blocked_events"] for r in mine),
        }
    changes = []
    for r in rows:
        if r["memory"] == "empty":
            continue
        base = next(b for b in rows if b["scenario"] == r["scenario"]
                    and b["affect_enabled"] == r["affect_enabled"] and b["memory"] == "empty")
        if r["actions"] != base["actions"]:
            changes.append({"scenario": r["scenario"], "affect_enabled": r["affect_enabled"],
                            "memory": r["memory"], "empty_actions": base["action_count"],
                            "actions": r["action_count"], "sequence": r["actions"]})
    return {
        "milestone": MILESTONE,
        "version": __version__,
        "rows": rows,
        "summary": summary,
        "changes_vs_empty": changes,
        "note": "Fixed deterministic scenarios only; not evidence of general performance.",
    }


def format_report(report: Dict[str, Any]) -> str:
    lines = [f"RED QUEEN {report['milestone']} memory experiment (v{report['version']})", ""]
    for condition, s in report["summary"].items():
        lines.append(f"{condition:<11} successes {s['successes']}/{s['episodes']}  total actions "
                     f"{s['total_actions']}  blocked events {s['total_blocked_events']}")
    lines.append("")
    if report["changes_vs_empty"]:
        lines.append("Runs whose actions differ from the empty-memory run:")
        for c in report["changes_vs_empty"]:
            lines.append(f"  {c['scenario']:<26} affect={'on ' if c['affect_enabled'] else 'off'} "
                         f"memory={c['memory']:<10} actions {c['empty_actions']} -> {c['actions']}: "
                         f"{' > '.join(c['sequence'])}")
    else:
        lines.append("No run differs from the empty-memory run.")
    lines += ["", "Note: " + report["note"]]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="RED QUEEN M2.1 memory experiment")
    parser.add_argument("--output", type=Path, help="Optional path for a JSON report")
    args = parser.parse_args(argv)
    report = run_memory_experiment()
    print(format_report(report))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nJSON report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
