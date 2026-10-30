"""M5.2 study: do two sensors protect against sensor bias?

Noise 5 on each sensor, bound 5 known. Sensor B uses an independent noise
stream (seed + 1000). Fault conditions (bias makes the system look healthier):

  none             no bias
  A_temp_-10       sensor A reads temperature 10 too low
  B_temp_-10       sensor B reads temperature 10 too low (symmetry check)
  A_temp_-20       larger bias on A
  A_batt_+10       sensor A reads battery 10 too high
  both_temp_-10    BOTH sensors read 10 too low (common-mode fault)

Agents: single (M5 interval agent on sensor A only), intersect, hull, latch.
18 named scenarios x 20 seeds per row; success and false stops are judged by
the GoalVerifier on the TRUE state.

    python3 -m red_queen.redundancy_study --output results/m5_2_redundancy_study.json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence

from .agent import STOP_GOAL
from .belief_agent import BeliefAgent
from .goal import RecoveryGoal
from .redundant_agent import FUSIONS, STOP_SENSOR_FAULT, RedundantBeliefAgent
from .scenarios import ALL_SCENARIOS, Scenario
from .sensors import NoisySensor

GOAL = RecoveryGoal()
NOISE = 5
SEED_OFFSET_B = 1000
DEFAULT_SEEDS = 20
AGENTS = ("single",) + FUSIONS
# name: ((A battery bias, A temperature bias), (B battery bias, B temperature bias))
CONDITIONS = {
    "none": ((0, 0), (0, 0)),
    "A_temp_-10": ((0, -10), (0, 0)),
    "B_temp_-10": ((0, 0), (0, -10)),
    "A_temp_-20": ((0, -20), (0, 0)),
    "A_batt_+10": ((10, 0), (0, 0)),
    "both_temp_-10": ((0, -10), (0, -10)),
}


def sensors_for(condition: str, seed: int):
    (ab, at), (bb, bt) = CONDITIONS[condition]
    return NoisySensor(NOISE, seed, ab, at), NoisySensor(NOISE, seed + SEED_OFFSET_B, bb, bt)


def run_one(agent: str, scenario: Scenario, condition: str, seed: int) -> Dict[str, Any]:
    sa, sb = sensors_for(condition, seed)
    env = scenario.make_env()
    if agent == "single":
        r = BeliefAgent(GOAL, NOISE, "interval", max_steps=scenario.max_steps).run(env, sa, scenario.name)
    elif agent in FUSIONS:
        r = RedundantBeliefAgent(GOAL, NOISE, agent, max_steps=scenario.max_steps).run(env, sa, sb, scenario.name)
    else:
        raise ValueError(f"unknown agent {agent!r}")
    return {"agent": agent, "scenario": scenario.name, "condition": condition, "seed": seed,
            "success": r.success, "false_stop": r.stop_reason == STOP_GOAL and not r.success,
            "fault_halt": r.stop_reason == STOP_SENSOR_FAULT, "stop_reason": r.stop_reason,
            "actions": len(r.actions), "disagreed": bool(r.stats.get("disagreement_steps"))}


def run_study(seeds: int = DEFAULT_SEEDS, scenarios: Sequence[Scenario] = ALL_SCENARIOS,
              conditions: Sequence[str] = tuple(CONDITIONS)) -> Dict[str, Any]:
    if seeds < 1:
        raise ValueError("seeds must be >= 1")
    summary = []
    for condition in conditions:
        for agent in AGENTS:
            rows = [run_one(agent, s, condition, seed) for s in scenarios for seed in range(seeds)]
            summary.append({"condition": condition, "agent": agent, "runs": len(rows),
                            "success": sum(r["success"] for r in rows),
                            "false_stops": sum(r["false_stop"] for r in rows),
                            "fault_halts": sum(r["fault_halt"] for r in rows),
                            "runs_disagreed": sum(r["disagreed"] for r in rows),
                            "actions": sum(r["actions"] for r in rows)})
    return {"milestone": "M5.2", "noise": NOISE, "seeds": seeds,
            "scenarios": [s.name for s in scenarios], "summary": summary}


def format_report(report: Dict[str, Any]) -> str:
    lines = ["RED QUEEN M5.2 - redundant sensors against bias",
             f"noise {report['noise']} on each sensor (bound known), {len(report['scenarios'])} scenarios x "
             f"{report['seeds']} seeds per row", "",
             f"  {'condition':<16}{'agent':<11}{'success':>11}{'false stop':>12}{'fault halt':>12}"
             f"{'disagreed':>11}{'actions':>9}", "  " + "-" * 82]
    last = None
    for r in report["summary"]:
        if last is not None and r["condition"] != last:
            lines.append("")
        last = r["condition"]
        lines.append(f"  {r['condition']:<16}{r['agent']:<11}{r['success']:>7}/{r['runs']:<3}{r['false_stops']:>12}"
                     f"{r['fault_halts']:>12}{r['runs_disagreed']:>11}{r['actions']:>9}")
    lines += ["", "false stop = stopped believing the goal was met, verifier (true state) says not.",
              "fault halt = stopped because the two sensors cannot both be right (no success claimed).",
              "disagreed = runs where the two sensor boxes did not overlap at some step."]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="RED QUEEN M5.2 redundant sensor study")
    p.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--output", help="write the JSON report here")
    args = p.parse_args(argv)
    report = run_study(seeds=args.seeds)
    print(format_report(report))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        print(f"\nJSON report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
