"""M3.3 study: does memory-informed tie-breaking beat fixed action orders?

Run:  python3 -m red_queen.memory_planning_study [--output results/m3_3_memory_planning_study.json]

For every case (the 18 named scenarios and the 1,680-case M3.2 fault grid,
default policy), one prior episode of the SAME case is run by the reactive
agent (affect off) and stored in a fresh in-RAM EpisodeMemory. That memory
is the only experience available. Then six agents run:

  reactive                 - M1-M2.1 greedy scorer, no memory
  reactive_memory          - same, with the M2.1 memory advisor
  planner                  - M3 planner, sorted action order, no memory
  planner_reversed         - M3.2 control: reversed order, no memory
  planner_memory           - M3.3: sorted order, memory breaks ties
  planner_reversed_memory  - M3.3: reversed order, memory breaks ties

If memory really helps, both memory planners should improve, and by a
similar amount, because memory (not order) is then deciding the ties.
The report describes these fixed deterministic cases only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import MILESTONE, __version__
from .affect import AffectConfig
from .agent import EpisodeResult, RecoveryAgent
from .environment import SimulatedEnvironment
from .goal import RecoveryGoal
from .memory import EpisodeMemory
from .memory_advice import MemoryAdvisor
from .planner import BoundedPlanner
from .planning_agent import PlanningAgent
from .scenario_study import REVERSED_ORDER, grid_cases
from .scenarios import ALL_SCENARIOS

AGENTS = ("reactive", "reactive_memory", "planner", "planner_reversed",
          "planner_memory", "planner_reversed_memory")
MAX_STEPS = 20
EnvFactory = Callable[[], SimulatedEnvironment]


def prior_memory(name: str, make_env: EnvFactory, max_steps: int = MAX_STEPS) -> EpisodeMemory:
    """One reactive episode of this case, stored as the only experience."""
    memory = EpisodeMemory(":memory:")
    config = AffectConfig(enabled=False)
    result = RecoveryAgent(RecoveryGoal(), config, max_steps).run(make_env(), scenario=name)
    memory.record(result, config)
    return memory


def run_agent(agent: str, name: str, make_env: EnvFactory, memory: Optional[EpisodeMemory],
              max_steps: int = MAX_STEPS) -> EpisodeResult:
    goal = RecoveryGoal()
    advisor = MemoryAdvisor(memory, name) if (memory is not None and agent.endswith("_memory")) else None
    if agent.endswith("_memory") and advisor is None:
        raise ValueError(f"{agent} needs a memory")
    if agent in ("reactive", "reactive_memory"):
        return RecoveryAgent(goal, AffectConfig(enabled=False), max_steps, advisor=advisor).run(make_env(), name)
    if agent in ("planner", "planner_memory"):
        return PlanningAgent(goal, max_steps=max_steps, advisor=advisor).run(make_env(), name)
    if agent in ("planner_reversed", "planner_reversed_memory"):
        planner = BoundedPlanner(goal, action_order=REVERSED_ORDER)
        return PlanningAgent(goal, planner=planner, max_steps=max_steps, advisor=advisor).run(make_env(), name)
    raise ValueError(f"Unknown agent: {agent!r}")


def run_case(name: str, make_env: EnvFactory, max_steps: int = MAX_STEPS) -> Dict[str, Dict[str, Any]]:
    with prior_memory(name, make_env, max_steps) as memory:
        out = {}
        for agent in AGENTS:
            r = run_agent(agent, name, make_env, memory, max_steps)
            out[agent] = {
                "success": r.success,
                "action_count": r.action_count,
                "blocked_events": sum(1 for e in r.events if e["kind"] == "blocked"),
            }
        return out


def _summary(cases: Sequence[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    out = {}
    for agent in AGENTS:
        out[agent] = {
            "episodes": len(cases),
            "successes": sum(c[agent]["success"] for c in cases),
            "total_actions": sum(c[agent]["action_count"] for c in cases),
            "blocked_events": sum(c[agent]["blocked_events"] for c in cases),
        }
    return out


def _head_to_head(cases, a: str, b: str) -> Dict[str, int]:
    fewer_a = sum(1 for c in cases if c[a]["action_count"] < c[b]["action_count"])
    fewer_b = sum(1 for c in cases if c[b]["action_count"] < c[a]["action_count"])
    return {f"{a}_fewer": fewer_a, f"{b}_fewer": fewer_b, "equal": len(cases) - fewer_a - fewer_b}


PAIRS = (
    ("planner_memory", "planner"),
    ("planner_reversed_memory", "planner_reversed"),
    ("planner_memory", "planner_reversed_memory"),
    ("planner_memory", "reactive_memory"),
    ("reactive_memory", "reactive"),
)


def run_named_part() -> Dict[str, Any]:
    rows = {}
    for s in ALL_SCENARIOS:
        rows[s.name] = run_case(s.name, s.make_env, s.max_steps)
    cases = list(rows.values())
    return {"cases": rows, "summary": _summary(cases)}


def run_grid_part() -> Dict[str, Any]:
    cases = []
    for i, (start, faults, disturbances) in enumerate(grid_cases()):
        make_env = lambda: SimulatedEnvironment(start, dict(disturbances), faults)  # noqa: E731
        cases.append(run_case(f"grid_{i:04d}", make_env))
    return {
        "cases": len(cases),
        "summary": _summary(cases),
        "head_to_head": {f"{a}_vs_{b}": _head_to_head(cases, a, b) for a, b in PAIRS},
    }


def run_memory_planning_study() -> Dict[str, Any]:
    return {
        "milestone": MILESTONE,
        "version": __version__,
        "agents": list(AGENTS),
        "memory_source": "one prior reactive (affect off) episode of the same case",
        "named": run_named_part(),
        "grid": run_grid_part(),
        "note": "Fixed deterministic cases only; not evidence of general performance.",
    }


def format_report(report: Dict[str, Any]) -> str:
    lines = [f"RED QUEEN {report['milestone']} memory-informed planning study (v{report['version']})",
             f"Memory: {report['memory_source']}", ""]
    g = report["grid"]
    lines.append(f"Fault grid, default policy, {g['cases']} cases")
    header = f"  {'agent':<26}{'success':<12}{'actions':<9}blocked"
    lines += [header, "  " + "-" * (len(header) - 2)]
    for agent, s in g["summary"].items():
        lines.append(f"  {agent:<26}{str(s['successes']) + '/' + str(s['episodes']):<12}"
                     f"{s['total_actions']:<9}{s['blocked_events']}")
    lines += ["", "  Head-to-head (cases where each side used fewer actions):"]
    for pair, counts in g["head_to_head"].items():
        lines.append(f"    {pair:<46} " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    lines += ["", "Named scenarios (actions per agent: " + ", ".join(AGENTS) + "):"]
    for name, case in report["named"]["cases"].items():
        counts = [case[a]["action_count"] for a in AGENTS]
        mark = "" if len(set(counts)) == 1 else "  *"
        lines.append(f"  {name:<28}" + " ".join(f"{c:>3}" for c in counts) + mark)
    lines += ["", "* = agents disagree.", "Note: " + report["note"]]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="RED QUEEN M3.3 memory-informed planning study")
    parser.add_argument("--output", type=Path, help="Optional path for a JSON report")
    args = parser.parse_args(argv)
    report = run_memory_planning_study()
    print(format_report(report))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nJSON report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
