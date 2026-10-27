"""M3.4 study: does allowing longer plans (bounded slack) help?

Run:  python3 -m red_queen.slack_study [--output results/m3_4_slack_study.json]

Same setup as the M3.3 study: one prior reactive episode of the same case
in memory, default policy. Agents:

  reactive_memory            - M2.1, the most efficient agent so far (baseline)
  planner_memory_k0          - M3.3 (shortest plans only), sorted order
  planner_memory_k1..k3      - M3.4, plans up to 1..3 steps longer allowed
  planner_reversed_memory_k0 - M3.3, reversed order
  planner_reversed_memory_k3 - M3.4, reversed order, slack 3

The grid part is split by case type (fault / disturbance), because the
M3.4 analysis found the remaining gap comes from disturbances.
The report describes these fixed deterministic cases only. Runtime is about 40 s.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import MILESTONE, __version__
from .actions import ACTION_NAMES
from .environment import SimulatedEnvironment
from .goal import RecoveryGoal
from .memory import EpisodeMemory
from .memory_advice import MemoryAdvisor
from .memory_planning_study import prior_memory
from .memory_planning_study import run_agent as run_m33_agent
from .planner import BoundedPlanner
from .planning_agent import PlanningAgent
from .scenario_study import REVERSED_ORDER, grid_cases
from .scenarios import ALL_SCENARIOS

AGENTS: Tuple[Tuple[str, Optional[Tuple[str, ...]], int], ...] = (
    ("reactive_memory", None, 0),
    ("planner_memory_k0", ACTION_NAMES, 0),
    ("planner_memory_k1", ACTION_NAMES, 1),
    ("planner_memory_k2", ACTION_NAMES, 2),
    ("planner_memory_k3", ACTION_NAMES, 3),
    ("planner_reversed_memory_k0", REVERSED_ORDER, 0),
    ("planner_reversed_memory_k3", REVERSED_ORDER, 3),
)
AGENT_NAMES = tuple(a[0] for a in AGENTS)
EnvFactory = Callable[[], SimulatedEnvironment]


def run_agent(agent: str, name: str, make_env: EnvFactory, memory: EpisodeMemory, max_steps: int = 20):
    spec = {a: (order, k) for a, order, k in AGENTS}
    if agent not in spec:
        raise ValueError(f"Unknown agent: {agent!r}")
    order, slack = spec[agent]
    if order is None:
        return run_m33_agent("reactive_memory", name, make_env, memory, max_steps)
    goal = RecoveryGoal()
    planner = BoundedPlanner(goal, action_order=order, slack=slack)
    return PlanningAgent(goal, planner=planner, max_steps=max_steps,
                         advisor=MemoryAdvisor(memory, name)).run(make_env(), name)


def run_case(name: str, make_env: EnvFactory, max_steps: int = 20) -> Dict[str, Dict[str, Any]]:
    with prior_memory(name, make_env, max_steps) as memory:
        out = {}
        for agent in AGENT_NAMES:
            r = run_agent(agent, name, make_env, memory, max_steps)
            out[agent] = {"success": r.success, "action_count": r.action_count}
        return out


def _totals(cases: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    return {a: {"episodes": len(cases),
                "successes": sum(c[a]["success"] for c in cases),
                "total_actions": sum(c[a]["action_count"] for c in cases)} for a in AGENT_NAMES}


def case_type(faults, disturbances) -> str:
    return ("fault" if faults else "no_fault") + "+" + ("disturbance" if disturbances else "no_disturbance")


def run_slack_study() -> Dict[str, Any]:
    named = {s.name: run_case(s.name, s.make_env, s.max_steps) for s in ALL_SCENARIOS}
    by_type: Dict[str, List[Dict[str, Dict[str, Any]]]] = {}
    all_cases = []
    for i, (start, faults, disturbances) in enumerate(grid_cases()):
        make_env = lambda: SimulatedEnvironment(start, dict(disturbances), faults)  # noqa: E731
        case = run_case(f"grid_{i:04d}", make_env)
        all_cases.append(case)
        by_type.setdefault(case_type(faults, disturbances), []).append(case)
    slack_changed = sum(
        1 for c in all_cases + list(named.values())
        for k in (1, 2, 3)
        if c[f"planner_memory_k{k}"]["action_count"] != c["planner_memory_k0"]["action_count"]
    ) + sum(
        1 for c in all_cases + list(named.values())
        if c["planner_reversed_memory_k3"]["action_count"] != c["planner_reversed_memory_k0"]["action_count"]
    )
    return {
        "milestone": MILESTONE,
        "version": __version__,
        "agents": list(AGENT_NAMES),
        "named": {"cases": named, "summary": _totals(list(named.values()))},
        "grid": {
            "cases": len(all_cases),
            "summary": _totals(all_cases),
            "by_case_type": {t: _totals(cs) for t, cs in sorted(by_type.items())},
        },
        "episodes_where_slack_changed_action_count": slack_changed,
        "note": "Fixed deterministic cases only; not evidence of general performance.",
    }


def format_report(report: Dict[str, Any]) -> str:
    lines = [f"RED QUEEN {report['milestone']} slack study (v{report['version']})", ""]
    g = report["grid"]
    lines.append(f"Fault grid, default policy, {g['cases']} cases (total actions; all must succeed)")
    types = list(g["by_case_type"])
    short = {"fault+disturbance": "F+D", "fault+no_disturbance": "F", "no_fault+disturbance": "D",
             "no_fault+no_disturbance": "none"}
    header = f"  {'agent':<28}{'all':>7}" + "".join(f"{short.get(t, t):>7}" for t in types)
    lines += [header, "  " + "-" * (len(header) - 2)]
    for a in report["agents"]:
        row = f"  {a:<28}{g['summary'][a]['total_actions']:>7}"
        row += "".join(f"{g['by_case_type'][t][a]['total_actions']:>7}" for t in types)
        lines.append(row)
    lines.append("  F = action fault, D = disturbance; cases: " + ", ".join(
        f"{short.get(t, t)}={g['by_case_type'][t][report['agents'][0]]['episodes']}" for t in types))
    lines += ["", f"Episodes where slack changed the action count: "
                  f"{report['episodes_where_slack_changed_action_count']}", "", "Note: " + report["note"]]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="RED QUEEN M3.4 slack study")
    parser.add_argument("--output", type=Path, help="Optional path for a JSON report")
    args = parser.parse_args(argv)
    report = run_slack_study()
    print(format_report(report))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nJSON report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
