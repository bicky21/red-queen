"""M3.2 expanded scenario study with a planner tie-break control.

Run:  python3 -m red_queen.scenario_study [--output results/m3_2_scenario_study.json]

Agents (affect off, no memory):
  reactive          - one-step greedy scorer (M1-M2.1)
  planner           - bounded BFS planner, actions tried in sorted name order (M3)
  planner_reversed  - same planner, actions tried in REVERSED order (control)

Both planners always find plans of the same length. They differ only in
which equal-length plan they pick. So any difference between `planner`
and `planner_reversed` is caused by tie-breaking, not by planning.

Part A: all 18 named scenarios x 4 policies.
Part B: a systematic fault grid (start states x action-fault windows x
        disturbances) under the default policy.
The report describes these fixed, deterministic cases only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import MILESTONE, __version__
from .actions import ACTION_NAMES
from .affect import AffectConfig
from .agent import EpisodeResult, RecoveryAgent
from .environment import SimulatedEnvironment
from .goal import RecoveryGoal
from .planner import BoundedPlanner
from .planning_agent import PlanningAgent
from .policy import ALL_POLICIES, DEFAULT_POLICY, Policy
from .scenarios import ALL_SCENARIOS, Scenario
from .state import SystemState

AGENTS = ("reactive", "planner", "planner_reversed")
REVERSED_ORDER = tuple(reversed(ACTION_NAMES))
MAX_STEPS = 20

# Part B grid: 24 start states x 10 fault settings x 7 disturbance settings = 1680 cases.
GRID_STARTS: Tuple[SystemState, ...] = tuple(
    SystemState(b, t, up) for b in (10, 30, 50, 70) for t in (30, 60, 90) for up in (False, True)
)
GRID_FAULTS: Tuple[Tuple[Tuple[str, int, int], ...], ...] = ((),) + tuple(
    ((a, 1, w),) for a in ("charge", "cool", "restart_service") for w in (1, 2, 3)
)
GRID_DISTURBANCES: Tuple[Tuple[Tuple[int, str], ...], ...] = ((),) + tuple(
    ((step, d),) for step in (1, 2) for d in ("battery_drain", "heat_spike", "service_crash")
)


def run_agent(agent: str, make_env: Callable[[], SimulatedEnvironment], policy: Policy = DEFAULT_POLICY,
              max_steps: int = MAX_STEPS) -> EpisodeResult:
    goal = RecoveryGoal()
    if agent == "reactive":
        return RecoveryAgent(goal, AffectConfig(enabled=False), max_steps, policy=policy).run(make_env())
    if agent == "planner":
        return PlanningAgent(goal, policy=policy, max_steps=max_steps).run(make_env())
    if agent == "planner_reversed":
        planner = BoundedPlanner(goal, action_order=REVERSED_ORDER)
        return PlanningAgent(goal, planner=planner, policy=policy, max_steps=max_steps).run(make_env())
    raise ValueError(f"Unknown agent: {agent!r}")


def _summarise(rows: Sequence[Dict[str, Any]], group_keys: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for r in rows:
        key = "/".join(str(r[k]) for k in group_keys)
        s = out.setdefault(key, {"episodes": 0, "successes": 0, "total_actions": 0,
                                 "actions_in_failed_episodes": 0, "blocked_events": 0})
        s["episodes"] += 1
        s["successes"] += int(r["success"])
        s["total_actions"] += r["action_count"]
        s["blocked_events"] += r["blocked_events"]
        if not r["success"]:
            s["actions_in_failed_episodes"] += r["action_count"]
    return out


def _pairwise(cases: Sequence[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, int]]:
    """For each ordered agent pair, count cases where the first used fewer actions."""
    out: Dict[str, Dict[str, int]] = {}
    for i, a in enumerate(AGENTS):
        for b in AGENTS[i + 1:]:
            fewer_a = sum(1 for c in cases if c[a]["action_count"] < c[b]["action_count"])
            fewer_b = sum(1 for c in cases if c[b]["action_count"] < c[a]["action_count"])
            out[f"{a}_vs_{b}"] = {f"{a}_fewer": fewer_a, f"{b}_fewer": fewer_b,
                                  "equal": len(cases) - fewer_a - fewer_b}
    return out


def _row(result: EpisodeResult, **labels: Any) -> Dict[str, Any]:
    return {
        **labels,
        "success": result.success,
        "stop_reason": result.stop_reason,
        "action_count": result.action_count,
        "actions": list(result.actions),
        "blocked_events": sum(1 for e in result.events if e["kind"] == "blocked"),
    }


def run_named_part(scenarios: Sequence[Scenario] = ALL_SCENARIOS) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    cases: List[Dict[str, Dict[str, Any]]] = []
    for policy in ALL_POLICIES.values():
        for s in scenarios:
            case = {}
            for agent in AGENTS:
                row = _row(run_agent(agent, s.make_env, policy, s.max_steps),
                           scenario=s.name, policy=policy.name, agent=agent)
                rows.append(row)
                case[agent] = row
            cases.append(case)
    return {"rows": rows, "summary": _summarise(rows, ("policy", "agent")), "pairwise": _pairwise(cases)}


def grid_cases():
    for start in GRID_STARTS:
        for faults in GRID_FAULTS:
            for disturbances in GRID_DISTURBANCES:
                yield start, faults, disturbances


def run_grid_part() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    cases: List[Dict[str, Dict[str, Any]]] = []
    for start, faults, disturbances in grid_cases():
        make_env = lambda: SimulatedEnvironment(start, dict(disturbances), faults)  # noqa: E731
        case = {}
        for agent in AGENTS:
            row = _row(run_agent(agent, make_env), agent=agent)
            row.pop("actions")  # keep the JSON report small
            rows.append(row)
            case[agent] = row
        cases.append(case)
    return {
        "cases": len(cases),
        "summary": _summarise(rows, ("agent",)),
        "pairwise": _pairwise(cases),
    }


def run_scenario_study() -> Dict[str, Any]:
    return {
        "milestone": MILESTONE,
        "version": __version__,
        "agents": list(AGENTS),
        "named": run_named_part(),
        "grid": run_grid_part(),
        "note": "Fixed deterministic cases only; not evidence of general performance.",
    }


def format_report(report: Dict[str, Any]) -> str:
    lines = [f"RED QUEEN {report['milestone']} scenario study (v{report['version']})", ""]
    lines.append(f"Part A - {len(ALL_SCENARIOS)} named scenarios x {len(ALL_POLICIES)} policies")
    header = f"{'policy/agent':<28}{'success':<10}{'actions':<9}wasted"
    lines += [header, "-" * len(header)]
    for key, s in report["named"]["summary"].items():
        lines.append(f"{key:<28}{str(s['successes']) + '/' + str(s['episodes']):<10}"
                     f"{s['total_actions']:<9}{s['actions_in_failed_episodes']}")
    lines += ["", "Default policy, per scenario (actions: reactive / planner / planner_reversed):"]
    named = report["named"]["rows"]
    for s in ALL_SCENARIOS:
        r = {x["agent"]: x for x in named if x["scenario"] == s.name and x["policy"] == "default"}
        mark = "" if len({r[a]["action_count"] for a in AGENTS}) == 1 else "  *"
        lines.append(f"  {s.name:<28}{r['reactive']['action_count']:>3} /{r['planner']['action_count']:>3} /"
                     f"{r['planner_reversed']['action_count']:>3}{mark}")
    g = report["grid"]
    lines += ["", f"Part B - fault grid, default policy, {g['cases']} cases"]
    for key, s in g["summary"].items():
        lines.append(f"  {key:<18} successes {s['successes']}/{s['episodes']}  total actions {s['total_actions']}")
    lines.append("  Head-to-head (cases where each side used fewer actions):")
    for pair, counts in g["pairwise"].items():
        lines.append(f"    {pair:<36} " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    lines += ["", "* = agents disagree.", "Note: " + report["note"]]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="RED QUEEN M3.2 scenario study")
    parser.add_argument("--output", type=Path, help="Optional path for a JSON report")
    args = parser.parse_args(argv)
    report = run_scenario_study()
    print(format_report(report))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nJSON report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
