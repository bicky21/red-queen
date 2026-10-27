"""M3.6 study: how do memory-using agents behave when memory is wrong?

Run:  python3 -m red_queen.noisy_memory_study [--output results/m3_6_noisy_memory_study.json] [--stride N]

For each grid case, the single prior episode in memory is a reactive run
on a DISTORTED copy of the case (see memory_noise.py), recorded under the
real case's name. Levels: 0.0 (exact, = M3.3/M3.5 setup), 0.25, 0.5, 1.0,
and "unrelated" (the prior comes from a different grid case).

Agents (default policy):
  reactive, planner, planner_reversed                 - no memory (baselines)
  reactive_memory                                     - M2.1
  planner_memory                                      - M3.3 (risk tie-break only)
  planner_dist_memory, planner_reversed_dist_memory   - M3.5 (+ disturbance prediction)
  planner_dist_guard, planner_reversed_dist_guard     - M3.6 (+ guard "on_any_mismatch")

`--stride N` uses every N-th grid case (default 1 = all 1,680). The full
run takes about 2 minutes. The report describes these fixed cases only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import MILESTONE, __version__
from .actions import ACTION_NAMES
from .affect import AffectConfig
from .agent import EpisodeResult, RecoveryAgent
from .environment import SimulatedEnvironment
from .goal import RecoveryGoal
from .memory import EpisodeMemory
from .memory_advice import MemoryAdvisor
from .memory_noise import NOISE_LEVELS, distort
from .planner import BoundedPlanner
from .planning_agent import PlanningAgent
from .scenario_study import REVERSED_ORDER, grid_cases

LEVELS: Sequence[Any] = tuple(NOISE_LEVELS) + ("unrelated",)
UNRELATED_OFFSET = 837  # prior for case i comes from case (i + 837) mod N
AGENTS = (
    "reactive", "planner", "planner_reversed",
    "reactive_memory", "planner_memory",
    "planner_dist_memory", "planner_reversed_dist_memory",
    "planner_dist_guard", "planner_reversed_dist_guard",
)
BASELINE_FOR = {  # memory agent -> same agent without memory
    "reactive_memory": "reactive",
    "planner_memory": "planner",
    "planner_dist_memory": "planner",
    "planner_dist_guard": "planner",
    "planner_reversed_dist_memory": "planner_reversed",
    "planner_reversed_dist_guard": "planner_reversed",
}
RECOMMENDED_GUARD = "on_any_mismatch"


def _prior(name: str, start, faults, disturbances) -> EpisodeMemory:
    memory = EpisodeMemory(":memory:")
    config = AffectConfig(enabled=False)
    env = SimulatedEnvironment(start, dict(disturbances), faults)
    memory.record(RecoveryAgent(RecoveryGoal(), config, 20).run(env, scenario=name), config)
    return memory


def memory_for(level, index: int, cases: Sequence) -> EpisodeMemory:
    name = f"grid_{index:04d}"
    start, faults, disturbances = cases[index]
    if level == "unrelated":
        return _prior(name, *cases[(index + UNRELATED_OFFSET) % len(cases)])
    noisy_faults, noisy_dist = distort(faults, disturbances, float(level), str(index))
    return _prior(name, start, noisy_faults, noisy_dist)


def run_agent(agent: str, name: str, make_env, memory: Optional[EpisodeMemory]) -> EpisodeResult:
    if agent not in AGENTS:
        raise ValueError(f"Unknown agent: {agent!r}")
    if agent in BASELINE_FOR and memory is None:
        raise ValueError(f"{agent} needs a memory")
    goal = RecoveryGoal()
    if agent == "reactive":
        return RecoveryAgent(goal, AffectConfig(enabled=False), 20).run(make_env(), name)
    if agent == "reactive_memory":
        return RecoveryAgent(goal, AffectConfig(enabled=False), 20,
                             advisor=MemoryAdvisor(memory, name)).run(make_env(), name)
    order = REVERSED_ORDER if "reversed" in agent else ACTION_NAMES
    planner = BoundedPlanner(goal, action_order=order)
    if agent in ("planner", "planner_reversed"):
        return PlanningAgent(goal, planner=planner).run(make_env(), name)
    advisor = MemoryAdvisor(memory, name)
    if agent == "planner_memory":
        return PlanningAgent(goal, planner=planner, advisor=advisor).run(make_env(), name)
    if agent in ("planner_dist_memory", "planner_reversed_dist_memory"):
        return PlanningAgent(goal, planner=planner, advisor=advisor,
                             predict_disturbances=True).run(make_env(), name)
    if agent in ("planner_dist_guard", "planner_reversed_dist_guard"):
        return PlanningAgent(goal, planner=planner, advisor=advisor, predict_disturbances=True,
                             disturbance_guard=RECOMMENDED_GUARD).run(make_env(), name)
    raise ValueError(f"Unknown agent: {agent!r}")


def run_level(level, stride: int = 1) -> Dict[str, Any]:
    cases = list(grid_cases())
    totals = {a: 0 for a in AGENTS}
    successes = {a: 0 for a in AGENTS}
    worse_than_baseline = {a: 0 for a in BASELINE_FOR}
    n = 0
    for i in range(0, len(cases), stride):
        start, faults, disturbances = cases[i]
        name = f"grid_{i:04d}"
        make_env = lambda: SimulatedEnvironment(start, dict(disturbances), faults)  # noqa: E731
        counts = {}
        with memory_for(level, i, cases) as memory:
            for agent in AGENTS:
                r = run_agent(agent, name, make_env, memory)
                counts[agent] = r.action_count
                totals[agent] += r.action_count
                successes[agent] += int(r.success)
        for agent, base in BASELINE_FOR.items():
            worse_than_baseline[agent] += counts[agent] > counts[base]
        n += 1
    return {"level": level, "cases": n, "total_actions": totals, "successes": successes,
            "cases_worse_than_no_memory": worse_than_baseline}


def run_noisy_memory_study(stride: int = 1) -> Dict[str, Any]:
    if stride < 1:
        raise ValueError("stride must be >= 1")
    levels = [run_level(level, stride) for level in LEVELS]
    return {
        "milestone": MILESTONE,
        "version": __version__,
        "stride": stride,
        "agents": list(AGENTS),
        "levels": levels,
        "note": "Fixed deterministic cases only; not evidence of general performance.",
    }


def format_report(report: Dict[str, Any]) -> str:
    levels = report["levels"]
    lines = [f"RED QUEEN {report['milestone']} imperfect-memory study (v{report['version']})",
             f"Grid cases per level: {levels[0]['cases']} (stride {report['stride']}); all runs must succeed", "",
             "Total actions by memory noise level:"]
    header = f"  {'agent':<30}" + "".join(f"{str(l['level']):>11}" for l in levels)
    lines += [header, "  " + "-" * (len(header) - 2)]
    for a in report["agents"]:
        lines.append(f"  {a:<30}" + "".join(f"{l['total_actions'][a]:>11}" for l in levels))
    lines += ["", "Memory agent minus the same agent without memory (negative = memory helps):"]
    lines += [header, "  " + "-" * (len(header) - 2)]
    for a, base in BASELINE_FOR.items():
        lines.append(f"  {a:<30}" + "".join(
            f"{l['total_actions'][a] - l['total_actions'][base]:>+11}" for l in levels))
    failed = {str(l["level"]): {a: l["cases"] - s for a, s in l["successes"].items() if s != l["cases"]}
              for l in levels}
    lines += ["", "Failures: " + (json.dumps({k: v for k, v in failed.items() if v}) or "none")
              if any(failed.values()) else "", "Note: " + report["note"]]
    if not any(failed.values()):
        lines[-2] = "Failures: none"
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="RED QUEEN M3.6 imperfect-memory study")
    parser.add_argument("--output", type=Path, help="Optional path for a JSON report")
    parser.add_argument("--stride", type=int, default=1, help="Use every N-th grid case (default 1 = all)")
    args = parser.parse_args(argv)
    report = run_noisy_memory_study(args.stride)
    print(format_report(report))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nJSON report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
