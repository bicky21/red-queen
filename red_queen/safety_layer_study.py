"""M5.3 study: the consolidated safety layer against single and combined faults.

Agents (all SafeAgent configurations; each reproduces an earlier agent exactly):
  single        1 sensor, fixed bound            (= M5 interval)
  single_adapt  1 sensor, adaptive bound         (= M5.1)
  hull          2 sensors, fixed bound, hull     (= M5.2 hull)
  safe_halt     2 sensors, adaptive bounds, hull; halt on an unexplainable
                sensor (first design)
  safe          2 sensors, adaptive bounds, hull; isolate an unexplainable
                sensor, halt only if none is left (added after the first run)

Conditions (assumed noise bound 5 for every agent and sensor):
  none                 true noise 5
  wrong_bound          true noise 15 on both sensors
  A_temp_-10           true noise 5; sensor A reads temperature 10 too low
  wrong_bound+A_temp   true noise 15 on both AND sensor A temperature -10
  A_dead               true noise 5; from step 2 sensor A reports battery 100,
                       temperature 0 (dead / stuck sensor, optimistic values)
  A_dead_low           same, but stuck at battery 0, temperature 100
                       (pessimistic values; added as the symmetric check)
  both_temp_-10        true noise 5; BOTH sensors read temperature 10 too low

18 named scenarios x 20 seeds per row. Success and false stops are judged by
the GoalVerifier on the TRUE state. A safe halt claims nothing.

    python3 -m red_queen.safety_layer_study --output results/m5_3_safety_layer_study.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from .agent import STOP_GOAL
from .goal import RecoveryGoal
from .safe_agent import STOP_SAFE_HALT, SafeAgent
from .scenarios import ALL_SCENARIOS, Scenario
from .sensors import NoisySensor, StuckSensor

GOAL = RecoveryGoal()
ASSUMED = 5
SEED_OFFSET_B = 1000
DEFAULT_SEEDS = 20
AGENTS = {  # name: (sensors, adapt, on_unexplainable)
    "single": (1, False, "halt"),
    "single_adapt": (1, True, "halt"),
    "hull": (2, False, "halt"),
    "safe_halt": (2, True, "halt"),
    "safe": (2, True, "isolate"),
}
# name: (true noise, A temperature bias, B temperature bias, (A dead from step, battery, temperature) or None)
CONDITIONS = {
    "none": (5, 0, 0, None),
    "wrong_bound": (15, 0, 0, None),
    "A_temp_-10": (5, -10, 0, None),
    "wrong_bound+A_temp": (15, -10, 0, None),
    "A_dead": (5, 0, 0, (2, 100, 0)),
    "A_dead_low": (5, 0, 0, (2, 0, 100)),
    "both_temp_-10": (5, -10, -10, None),
}


def sensors_for(condition: str, seed: int):
    noise, a_bias, b_bias, dead = CONDITIONS[condition]
    a = NoisySensor(noise, seed, 0, a_bias)
    if dead is not None:
        a = StuckSensor(a, dead[0], battery=dead[1], temperature=dead[2])
    return [a, NoisySensor(noise, seed + SEED_OFFSET_B, 0, b_bias)]


def run_one(agent: str, scenario: Scenario, condition: str, seed: int) -> Dict[str, Any]:
    if agent not in AGENTS:
        raise ValueError(f"unknown agent {agent!r}")
    n_sensors, adapt, on_unexplainable = AGENTS[agent]
    sensors = sensors_for(condition, seed)[:n_sensors]
    r = SafeAgent(GOAL, ASSUMED, n_sensors, adapt, on_unexplainable, max_steps=scenario.max_steps).run(
        scenario.make_env(), sensors, scenario.name)
    counts = r.stats["operator_report"]["alert_counts"]
    return {"agent": agent, "scenario": scenario.name, "condition": condition, "seed": seed,
            "success": r.success, "false_stop": r.stop_reason == STOP_GOAL and not r.success,
            "safe_halt": r.stop_reason == STOP_SAFE_HALT, "stop_reason": r.stop_reason,
            "actions": len(r.actions), "alerts": counts}


def run_study(seeds: int = DEFAULT_SEEDS, scenarios: Sequence[Scenario] = ALL_SCENARIOS,
              conditions: Sequence[str] = tuple(CONDITIONS)) -> Dict[str, Any]:
    if seeds < 1:
        raise ValueError("seeds must be >= 1")
    summary = []
    for condition in conditions:
        for agent in AGENTS:
            rows = [run_one(agent, s, condition, seed) for s in scenarios for seed in range(seeds)]
            alerts = Counter()
            for r in rows:
                alerts.update({k: v for k, v in r["alerts"].items() if v})
            summary.append({"condition": condition, "agent": agent, "runs": len(rows),
                            "success": sum(r["success"] for r in rows),
                            "false_stops": sum(r["false_stop"] for r in rows),
                            "safe_halts": sum(r["safe_halt"] for r in rows),
                            "actions": sum(r["actions"] for r in rows),
                            "runs_with_alerts": sum(any(r["alerts"].values()) for r in rows),
                            "alert_totals": dict(sorted(alerts.items()))})
    return {"milestone": "M5.3", "assumed_noise": ASSUMED, "seeds": seeds,
            "scenarios": [s.name for s in scenarios], "summary": summary}


def format_report(report: Dict[str, Any]) -> str:
    lines = ["RED QUEEN M5.3 - consolidated safety layer",
             f"assumed noise bound {report['assumed_noise']}, {len(report['scenarios'])} scenarios x "
             f"{report['seeds']} seeds per row", "",
             f"  {'condition':<20}{'agent':<14}{'success':>11}{'false stop':>12}{'safe halt':>11}"
             f"{'actions':>9}{'alerted':>9}", "  " + "-" * 84]
    last = None
    for r in report["summary"]:
        if last is not None and r["condition"] != last:
            lines.append("")
        last = r["condition"]
        lines.append(f"  {r['condition']:<20}{r['agent']:<14}{r['success']:>7}/{r['runs']:<3}{r['false_stops']:>12}"
                     f"{r['safe_halts']:>11}{r['actions']:>9}{r['runs_with_alerts']:>9}")
    lines += ["", "false stop = stopped believing the goal was met; verifier (true state) says not.",
              "safe halt = estimate declared untrustworthy; agent stopped claiming nothing.",
              "alerted = runs whose operator report contains at least one alert."]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="RED QUEEN M5.3 safety layer study")
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
