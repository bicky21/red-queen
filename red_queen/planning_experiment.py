"""M3 experiment: reactive scorer vs bounded planner under three policies.

Run:  python3 -m red_queen.planning_experiment [--output results/m3_planning_experiment.json]

Both agents run with affect off and no memory, so the only difference is
the decision method. The report describes these fixed scenarios only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import MILESTONE, __version__
from .affect import AffectConfig
from .agent import EpisodeResult
from .experiments import run_scenario
from .goal import RecoveryGoal
from .planner import BoundedPlanner
from .planning_agent import PlanningAgent
from .policy import DEFAULT_POLICY, POLICIES, Policy
from .scenarios import SCENARIOS, Scenario

AGENTS = ("reactive", "planner")


def run_planner(scenario: Scenario, policy: Policy = DEFAULT_POLICY,
                planner: Optional[BoundedPlanner] = None) -> EpisodeResult:
    agent = PlanningAgent(RecoveryGoal(), planner=planner, policy=policy, max_steps=scenario.max_steps)
    return agent.run(scenario.make_env(), scenario=scenario.name)


def run_agent(agent: str, scenario: Scenario, policy: Policy) -> EpisodeResult:
    if agent == "reactive":
        return run_scenario(scenario, False, affect=AffectConfig(enabled=False), policy=policy)
    if agent == "planner":
        return run_planner(scenario, policy)
    raise ValueError(f"Unknown agent: {agent!r}")


def run_planning_experiment(scenarios: Iterable[Scenario] = SCENARIOS) -> Dict[str, Any]:
    scenarios = list(scenarios)
    rows: List[Dict[str, Any]] = []
    for policy in POLICIES.values():
        for s in scenarios:
            for agent in AGENTS:
                r = run_agent(agent, s, policy)
                rows.append(
                    {
                        "scenario": s.name,
                        "policy": policy.name,
                        "agent": agent,
                        "success": r.success,
                        "stop_reason": r.stop_reason,
                        "action_count": r.action_count,
                        "actions": list(r.actions),
                        "blocked_events": sum(1 for e in r.events if e["kind"] == "blocked"),
                        "denied_actions_executed": sum(
                            1 for e in r.events if e["kind"] in ("action", "blocked")
                            and e["name"] not in policy.allowed
                        ),
                        "stats": dict(r.stats),
                    }
                )
    summary: Dict[str, Any] = {}
    for policy in POLICIES:
        for agent in AGENTS:
            mine = [r for r in rows if r["policy"] == policy and r["agent"] == agent]
            summary[f"{policy}/{agent}"] = {
                "episodes": len(mine),
                "successes": sum(r["success"] for r in mine),
                "total_actions": sum(r["action_count"] for r in mine),
                "actions_in_failed_episodes": sum(r["action_count"] for r in mine if not r["success"]),
                "replans": sum(r["stats"].get("replans", 0) for r in mine),
                "nodes_expanded": sum(r["stats"].get("nodes_expanded", 0) for r in mine),
                "policy_denials": sum(r["stats"].get("policy_denials", 0) for r in mine),
                "denied_actions_executed": sum(r["denied_actions_executed"] for r in mine),
            }
    return {
        "milestone": MILESTONE,
        "version": __version__,
        "rows": rows,
        "summary": summary,
        "note": "Fixed deterministic scenarios only; not evidence of general performance.",
    }


def format_report(report: Dict[str, Any]) -> str:
    lines = [f"RED QUEEN {report['milestone']} planning experiment (v{report['version']})", ""]
    header = f"{'policy/agent':<22}{'success':<9}{'actions':<9}{'wasted':<8}{'replans':<9}{'nodes':<8}denied_run"
    lines += [header, "-" * len(header)]
    for key, s in report["summary"].items():
        lines.append(
            f"{key:<22}{str(s['successes']) + '/' + str(s['episodes']):<9}{s['total_actions']:<9}"
            f"{s['actions_in_failed_episodes']:<8}{s['replans']:<9}{s['nodes_expanded']:<8}{s['denied_actions_executed']}"
        )
    lines += ["", "wasted = actions spent in episodes that failed; denied_run = denied actions that reached the environment", ""]
    lines.append("Per-scenario differences (actions reactive -> planner):")
    for policy in POLICIES:
        for name in dict.fromkeys(r["scenario"] for r in report["rows"]):
            pair = {r["agent"]: r for r in report["rows"] if r["policy"] == policy and r["scenario"] == name}
            a, b = pair["reactive"], pair["planner"]
            if a["actions"] != b["actions"] or a["success"] != b["success"]:
                lines.append(
                    f"  {policy:<11}{name:<26}{a['action_count']:>3} ({a['stop_reason']}) -> "
                    f"{b['action_count']} ({b['stop_reason']})"
                )
    lines += ["", "Note: " + report["note"]]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="RED QUEEN M3 planning experiment")
    parser.add_argument("--output", type=Path, help="Optional path for a JSON report")
    args = parser.parse_args(argv)
    report = run_planning_experiment()
    print(format_report(report))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nJSON report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
