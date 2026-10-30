"""M5.4 study, exactly as pre-registered in docs/PREREG_M5_4.md.

Agents: safe (M5.3 default) and safe_x (safe + cross_isolate).
Confirmation conditions decide; development conditions (the motivating
M5.3 cases) are reported separately and do not count for the decision.

    python3 -m red_queen.cross_isolation_study --output results/m5_4_cross_isolation_study.json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence

from .agent import STOP_GOAL
from .goal import RecoveryGoal
from .safe_agent import STOP_SAFE_HALT, SafeAgent
from .scenarios import ALL_SCENARIOS, Scenario
from .sensors import DriftSensor, NoisySensor, StuckSensor

GOAL = RecoveryGoal()
ASSUMED = 5
SEED_OFFSET_B = 1000
DEFAULT_SEEDS = 20
AGENTS = {"safe": False, "safe_x": True}  # name: cross_isolate
# name: (true noise, faulty sensor or None, fault spec)
CONFIRMATION = {
    "none": (5, None, None),
    "wrong_bound": (15, None, None),
    "A_drift": (5, "A", ("drift", 0, -2)),
    "A_stuck_mid": (5, "A", ("stuck", 1, 70, 40)),
    "B_stuck_low": (5, "B", ("stuck", 1, 10, 90)),
    "wrong_bound+B_stuck_low": (15, "B", ("stuck", 1, 10, 90)),
}
DEVELOPMENT = {
    "A_dead_low": (5, "A", ("stuck", 2, 0, 100)),
    "A_dead": (5, "A", ("stuck", 2, 100, 0)),
}
CONDITIONS = {**CONFIRMATION, **DEVELOPMENT}


def _faulty(base: NoisySensor, spec):
    if spec[0] == "drift":
        return DriftSensor(base, spec[1], spec[2])
    return StuckSensor(base, spec[1], battery=spec[2], temperature=spec[3])


def sensors_for(condition: str, seed: int):
    noise, which, spec = CONDITIONS[condition]
    a, b = NoisySensor(noise, seed), NoisySensor(noise, seed + SEED_OFFSET_B)
    if which == "A":
        a = _faulty(a, spec)
    elif which == "B":
        b = _faulty(b, spec)
    return [a, b]


def run_one(agent: str, scenario: Scenario, condition: str, seed: int) -> Dict[str, Any]:
    if agent not in AGENTS:
        raise ValueError(f"unknown agent {agent!r}")
    r = SafeAgent(GOAL, ASSUMED, 2, True, "isolate", cross_isolate=AGENTS[agent],
                  max_steps=scenario.max_steps).run(scenario.make_env(), sensors_for(condition, seed), scenario.name)
    rep = r.stats["operator_report"]
    faulty = CONDITIONS[condition][1]
    isolated = [n for n, ok in zip("AB", rep["trusted"]) if not ok]
    return {"agent": agent, "scenario": scenario.name, "condition": condition, "seed": seed,
            "success": r.success, "false_stop": r.stop_reason == STOP_GOAL and not r.success,
            "safe_halt": r.stop_reason == STOP_SAFE_HALT, "actions": len(r.actions),
            "isolated": isolated, "wrongful_isolation": any(n != faulty for n in isolated),
            "cross_isolations": sum(a["kind"] == "sensor_isolated" and a["detail"].get("reason") == "cross_sensor"
                                    for a in rep["alerts"])}


def run_study(seeds: int = DEFAULT_SEEDS, scenarios: Sequence[Scenario] = ALL_SCENARIOS,
              conditions: Sequence[str] = tuple(CONDITIONS)) -> Dict[str, Any]:
    if seeds < 1:
        raise ValueError("seeds must be >= 1")
    summary = []
    for condition in conditions:
        for agent in AGENTS:
            rows = [run_one(agent, s, condition, seed) for s in scenarios for seed in range(seeds)]
            summary.append({"condition": condition, "set": "confirmation" if condition in CONFIRMATION else "development",
                            "agent": agent, "runs": len(rows),
                            "success": sum(r["success"] for r in rows),
                            "false_stops": sum(r["false_stop"] for r in rows),
                            "safe_halts": sum(r["safe_halt"] for r in rows),
                            "actions": sum(r["actions"] for r in rows),
                            "wrongful_isolations": sum(r["wrongful_isolation"] for r in rows),
                            "runs_cross_isolated": sum(r["cross_isolations"] > 0 for r in rows)})
    return {"milestone": "M5.4", "assumed_noise": ASSUMED, "seeds": seeds,
            "scenarios": [s.name for s in scenarios], "summary": summary,
            "decision": decide(summary) if set(CONFIRMATION) <= set(conditions) else None}


def decide(summary: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The pre-registered decision rule (docs/PREREG_M5_4.md)."""
    get = {(r["condition"], r["agent"]): r for r in summary}
    conf = list(CONFIRMATION)
    c1 = all(get[(c, "safe_x")]["false_stops"] == 0 for c in conf)
    c2 = (all(get[(c, "safe_x")]["success"] >= get[(c, "safe")]["success"] for c in conf)
          and any(get[(c, "safe_x")]["success"] > get[(c, "safe")]["success"]
                  for c in ("A_drift", "A_stuck_mid", "B_stuck_low", "wrong_bound+B_stuck_low")))
    c3 = (get[("none", "safe_x")]["wrongful_isolations"] == 0
          and get[("wrong_bound", "safe_x")]["wrongful_isolations"] <= 18)
    return {"C1_no_false_stops": c1, "C2_no_regression_and_gain": c2, "C3_few_wrongful_isolations": c3,
            "adopt_as_default": c1 and c2 and c3}


def format_report(report: Dict[str, Any]) -> str:
    lines = ["RED QUEEN M5.4 - cross-sensor isolation (pre-registered)",
             f"assumed noise bound {report['assumed_noise']}, {len(report['scenarios'])} scenarios x "
             f"{report['seeds']} seeds per row", "",
             f"  {'condition':<25}{'agent':<8}{'success':>11}{'false stop':>12}{'safe halt':>11}"
             f"{'actions':>9}{'wrong iso':>11}{'x-iso':>7}", "  " + "-" * 92]
    last_set = last = None
    for r in report["summary"]:
        if r["set"] != last_set:
            lines.append(f"  [{r['set']}]")
            last_set = r["set"]
        elif last is not None and r["condition"] != last:
            lines.append("")
        last = r["condition"]
        lines.append(f"  {r['condition']:<25}{r['agent']:<8}{r['success']:>7}/{r['runs']:<3}{r['false_stops']:>12}"
                     f"{r['safe_halts']:>11}{r['actions']:>9}{r['wrongful_isolations']:>11}{r['runs_cross_isolated']:>7}")
    d = report.get("decision")
    if d:
        lines += ["", "Pre-registered decision (confirmation conditions only):",
                  f"  C1 safe_x 0 false stops everywhere:            {d['C1_no_false_stops']}",
                  f"  C2 successes >= safe, and > on a fault:        {d['C2_no_regression_and_gain']}",
                  f"  C3 wrongful isolations: 0 in none, <=18 wrong: {d['C3_few_wrongful_isolations']}",
                  f"  => adopt cross_isolate as default:             {d['adopt_as_default']}"]
    lines += ["", "wrong iso = runs where a sensor without an injected fault was isolated.",
              "x-iso = runs where the cross-sensor rule isolated a sensor."]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="RED QUEEN M5.4 cross-sensor isolation study")
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
