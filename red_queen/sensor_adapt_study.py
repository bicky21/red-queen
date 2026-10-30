"""M5.1 study: can the belief agent notice that its noise assumption is wrong?

Part A - wrong bound: true noise 15, assumed 15/10/5/0.
Part B - correct bound: true = assumed in 5/10/15 (adaptation must change nothing).
Part C - sensor bias: true noise 5 (assumed 5) plus a constant offset that
         makes the system look healthier than it is (battery reads +10, or
         temperature reads -10).

Agents: interval and filter_verified, each without and with `adapt_noise`.
18 named scenarios x 20 seeds per row. Success and false stops are judged by
the GoalVerifier on the TRUE state.

    python3 -m red_queen.sensor_adapt_study --output results/m5_1_adapt_study.json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence

from .agent import STOP_GOAL
from .belief_agent import BeliefAgent
from .goal import RecoveryGoal
from .scenarios import ALL_SCENARIOS, Scenario
from .sensors import NoisySensor

GOAL = RecoveryGoal()
MODES = ("interval", "filter_verified")
DEFAULT_SEEDS = 20
WRONG_TRUE_NOISE = 15
WRONG_ASSUMED = (15, 10, 5, 0)
CORRECT_NOISE = (5, 10, 15)
BIASES = (("battery_+10", 10, 0), ("temperature_-10", 0, -10))
BIAS_NOISE = 5


def run_row(mode: str, adapt: bool, true_noise: int, assumed: int, seeds: int,
            scenarios: Sequence[Scenario], bias_battery: int = 0, bias_temperature: int = 0) -> Dict[str, Any]:
    success = false_stops = actions = widened = at_cap = 0
    for s in scenarios:
        for seed in range(seeds):
            r = BeliefAgent(GOAL, assumed, mode, max_steps=s.max_steps, adapt_noise=adapt).run(
                s.make_env(), NoisySensor(true_noise, seed, bias_battery, bias_temperature), s.name)
            success += r.success
            false_stops += r.stop_reason == STOP_GOAL and not r.success
            actions += len(r.actions)
            widened += bool(r.stats.get("widenings"))
            at_cap += r.stats.get("unexplained_at_cap", 0) > 0
    return {"mode": mode, "adapt": adapt, "true_noise": true_noise, "assumed_noise": assumed,
            "bias_battery": bias_battery, "bias_temperature": bias_temperature,
            "runs": len(scenarios) * seeds, "success": success, "false_stops": false_stops,
            "actions": actions, "runs_widened": widened, "runs_unexplained_at_cap": at_cap}


def run_study(seeds: int = DEFAULT_SEEDS, scenarios: Sequence[Scenario] = ALL_SCENARIOS) -> Dict[str, Any]:
    if seeds < 1:
        raise ValueError("seeds must be >= 1")
    wrong = [run_row(m, a, WRONG_TRUE_NOISE, n, seeds, scenarios)
             for m in MODES for n in WRONG_ASSUMED for a in (False, True)]
    correct = [run_row(m, a, n, n, seeds, scenarios) for m in MODES for n in CORRECT_NOISE for a in (False, True)]
    bias = [dict(run_row(m, a, BIAS_NOISE, BIAS_NOISE, seeds, scenarios, bb, bt), bias=name)
            for m in MODES for name, bb, bt in BIASES for a in (False, True)]
    return {"milestone": "M5.1", "seeds": seeds, "scenarios": [s.name for s in scenarios],
            "wrong_bound": wrong, "correct_bound": correct, "bias": bias}


def _line(label: str, r: Dict[str, Any]) -> str:
    return (f"  {label:<26}{r['mode']:<16}{'yes' if r['adapt'] else 'no':>6}{r['success']:>6}/{r['runs']:<4}"
            f"{r['false_stops']:>8}{r['actions']:>9}{r['runs_widened']:>9}")


def format_report(report: Dict[str, Any]) -> str:
    head = (f"  {'case':<26}{'agent':<16}{'adapt':>6}{'success':>11}{'false':>8}{'actions':>9}{'widened':>9}")
    lines = ["RED QUEEN M5.1 - detecting a wrong noise assumption",
             f"{len(report['scenarios'])} scenarios x {report['seeds']} seeds per row", "",
             f"A. Wrong bound (true noise {WRONG_TRUE_NOISE})", head, "  " + "-" * 84]
    lines += [_line(f"assumes {r['assumed_noise']}", r) for r in report["wrong_bound"]]
    lines += ["", "B. Correct bound (adaptation should change nothing)", head, "  " + "-" * 84]
    lines += [_line(f"noise {r['true_noise']}", r) for r in report["correct_bound"]]
    lines += ["", f"C. Sensor bias (noise {BIAS_NOISE}, assumed {BIAS_NOISE})", head, "  " + "-" * 84]
    lines += [_line(r["bias"], r) for r in report["bias"]]
    lines += ["", "false = false stops: the agent stopped believing the goal was met; the verifier",
              "(true state) says it was not. widened = runs where the agent enlarged its bound."]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="RED QUEEN M5.1 noise-assumption study")
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
