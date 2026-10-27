"""M3.5 study: does predicting remembered disturbances help?

Run:  python3 -m red_queen.disturbance_study [--output results/m3_5_disturbance_study.json]

Same setup as M3.3/M3.4: one prior reactive episode of the same case in
memory (its events include the disturbances that happened), default policy.

  reactive_memory               - M2.1 baseline (most efficient before M3.5)
  planner_memory                - M3.3, sorted order
  planner_reversed_memory       - M3.3, reversed order
  planner_dist_memory           - M3.5, sorted order, predicts remembered disturbances
  planner_reversed_dist_memory  - M3.5, reversed order, predicts remembered disturbances

Note on fairness: the M3.5 planners use information (disturbance timing)
that the reactive scorer has no way to use; one-step scoring cannot plan
around a future event. The comparison therefore measures the value of
model-based prediction, not a like-for-like algorithm swap.
The report describes these fixed deterministic cases only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import MILESTONE, __version__
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
from .slack_study import case_type

AGENTS = ("reactive_memory", "planner_memory", "planner_reversed_memory",
          "planner_dist_memory", "planner_reversed_dist_memory")
EnvFactory = Callable[[], SimulatedEnvironment]
PAIRS = (
    ("planner_dist_memory", "reactive_memory"),
    ("planner_reversed_dist_memory", "reactive_memory"),
    ("planner_dist_memory", "planner_memory"),
    ("planner_reversed_dist_memory", "planner_reversed_memory"),
    ("planner_dist_memory", "planner_reversed_dist_memory"),
)


def run_agent(agent: str, name: str, make_env: EnvFactory, memory: EpisodeMemory, max_steps: int = 20):
    if agent in ("reactive_memory", "planner_memory", "planner_reversed_memory"):
        return run_m33_agent(agent, name, make_env, memory, max_steps)
    goal = RecoveryGoal()
    if agent == "planner_dist_memory":
        planner = BoundedPlanner(goal)
    elif agent == "planner_reversed_dist_memory":
        planner = BoundedPlanner(goal, action_order=REVERSED_ORDER)
    else:
        raise ValueError(f"Unknown agent: {agent!r}")
    return PlanningAgent(goal, planner=planner, max_steps=max_steps, advisor=MemoryAdvisor(memory, name),
                         predict_disturbances=True).run(make_env(), name)


def run_case(name: str, make_env: EnvFactory, max_steps: int = 20) -> Dict[str, Dict[str, Any]]:
    with prior_memory(name, make_env, max_steps) as memory:
        out = {}
        for agent in AGENTS:
            r = run_agent(agent, name, make_env, memory, max_steps)
            out[agent] = {"success": r.success, "action_count": r.action_count,
                          "replans": r.stats.get("replans", 0)}
        return out


def _totals(cases: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    return {a: {"episodes": len(cases),
                "successes": sum(c[a]["success"] for c in cases),
                "total_actions": sum(c[a]["action_count"] for c in cases),
                "replans": sum(c[a]["replans"] for c in cases)} for a in AGENTS}


def _head_to_head(cases, a: str, b: str) -> Dict[str, int]:
    fa = sum(1 for c in cases if c[a]["action_count"] < c[b]["action_count"])
    fb = sum(1 for c in cases if c[b]["action_count"] < c[a]["action_count"])
    return {f"{a}_fewer": fa, f"{b}_fewer": fb, "equal": len(cases) - fa - fb}


def run_disturbance_study() -> Dict[str, Any]:
    named = {s.name: run_case(s.name, s.make_env, s.max_steps) for s in ALL_SCENARIOS}
    cases, by_type = [], {}
    for i, (start, faults, disturbances) in enumerate(grid_cases()):
        make_env = lambda: SimulatedEnvironment(start, dict(disturbances), faults)  # noqa: E731
        case = run_case(f"grid_{i:04d}", make_env)
        cases.append(case)
        by_type.setdefault(case_type(faults, disturbances), []).append(case)
    return {
        "milestone": MILESTONE,
        "version": __version__,
        "agents": list(AGENTS),
        "named": {"cases": named, "summary": _totals(list(named.values()))},
        "grid": {
            "cases": len(cases),
            "summary": _totals(cases),
            "by_case_type": {t: _totals(cs) for t, cs in sorted(by_type.items())},
            "head_to_head": {f"{a}_vs_{b}": _head_to_head(cases, a, b) for a, b in PAIRS},
        },
        "note": "Fixed deterministic cases only; not evidence of general performance.",
    }


SHORT = {"fault+disturbance": "F+D", "fault+no_disturbance": "F", "no_fault+disturbance": "D",
         "no_fault+no_disturbance": "none"}


def format_report(report: Dict[str, Any]) -> str:
    lines = [f"RED QUEEN {report['milestone']} disturbance-prediction study (v{report['version']})", ""]
    g = report["grid"]
    types = list(g["by_case_type"])
    lines.append(f"Fault grid, default policy, {g['cases']} cases (total actions)")
    header = f"  {'agent':<30}{'success':>10}{'all':>7}" + "".join(f"{SHORT.get(t, t):>7}" for t in types)
    lines += [header, "  " + "-" * (len(header) - 2)]
    for a in report["agents"]:
        s = g["summary"][a]
        row = f"  {a:<30}{str(s['successes']) + '/' + str(s['episodes']):>10}{s['total_actions']:>7}"
        row += "".join(f"{g['by_case_type'][t][a]['total_actions']:>7}" for t in types)
        lines.append(row)
    lines.append("  F = action fault, D = disturbance; cases: " + ", ".join(
        f"{SHORT.get(t, t)}={g['by_case_type'][t][report['agents'][0]]['episodes']}" for t in types))
    lines += ["", "  Head-to-head (cases where each side used fewer actions):"]
    for pair, counts in g["head_to_head"].items():
        lines.append(f"    {pair}")
        lines.append("      " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    lines += ["", "Note: " + report["note"]]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="RED QUEEN M3.5 disturbance-prediction study")
    parser.add_argument("--output", type=Path, help="Optional path for a JSON report")
    args = parser.parse_args(argv)
    report = run_disturbance_study()
    print(format_report(report))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nJSON report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
