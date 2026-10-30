"""M6 study, exactly as pre-registered in docs/PREREG_M6.md.

Disturbance memory (M3.5) and its guard (M3.6) under noisy sensors, using
the M5.3 SafeAgent (two sensors, adaptive bounds, hull, isolate).

Agents: safe (no memory), safe_mem (disturbance prediction),
        safe_mem_guard (prediction + guard "on_mismatch").
Cases:  every 10th M3.2 grid case (168). Memory: one reactive prior episode
        per case, at memory-noise levels 0.0, 0.5, 1.0 and "unrelated"
        (built exactly as in the M3.6 study).
Sensors: assumed bound 5; "noise5" (true 5) and "wrong_bound" (true 15);
        seeds 0 and 1 (sensor B: seed + 1000).

    python3 -m red_queen.memory_sensor_study --output results/m6_memory_sensor_study.json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence

from .agent import STOP_GOAL
from .environment import SimulatedEnvironment
from .goal import RecoveryGoal
from .memory_advice import MemoryAdvisor
from .noisy_memory_study import memory_for
from .safe_agent import SafeAgent
from .scenario_study import grid_cases
from .sensors import NoisySensor

GOAL = RecoveryGoal()
ASSUMED = 5
STRIDE = 10
LEVELS = (0.0, 0.5, 1.0, "unrelated")
SENSOR_CONDITIONS = {"noise5": 5, "wrong_bound": 15}
SEEDS = (0, 1)
AGENTS = ("safe", "safe_mem", "safe_mem_guard")


def _agent(name: str, advisor) -> SafeAgent:
    if name == "safe":
        return SafeAgent(GOAL, ASSUMED)
    if name == "safe_mem":
        return SafeAgent(GOAL, ASSUMED, advisor=advisor, predict_disturbances=True)
    if name == "safe_mem_guard":
        return SafeAgent(GOAL, ASSUMED, advisor=advisor, predict_disturbances=True, disturbance_guard="on_mismatch")
    raise ValueError(f"unknown agent {name!r}")


def run_study(stride: int = STRIDE, seeds: Sequence[int] = SEEDS) -> Dict[str, Any]:
    if stride < 1:
        raise ValueError("stride must be >= 1")
    cases = list(grid_cases())
    picked = list(range(0, len(cases), stride))
    totals: Dict[tuple, Dict[str, Any]] = {}
    for level in LEVELS:
        for cond, true_noise in SENSOR_CONDITIONS.items():
            acc = {a: {"runs": 0, "success": 0, "false_stops": 0, "actions": 0, "regret": 0,
                       "worst_regret": 0, "guard_trips": 0} for a in AGENTS}
            for i in picked:
                start, faults, disturbances = cases[i]
                name = f"grid_{i:04d}"
                memory = memory_for(level, i, cases)
                try:
                    advisor = MemoryAdvisor(memory, name)
                    for seed in seeds:
                        base = None
                        for a in AGENTS:
                            env = SimulatedEnvironment(start, dict(disturbances), faults)
                            sensors = [NoisySensor(true_noise, seed), NoisySensor(true_noise, seed + 1000)]
                            r = _agent(a, advisor).run(env, sensors, name)
                            n = len(r.actions)
                            t = acc[a]
                            t["runs"] += 1
                            t["success"] += r.success
                            t["false_stops"] += r.stop_reason == STOP_GOAL and not r.success
                            t["actions"] += n
                            t["guard_trips"] += r.stats.get("guard_tripped_at") is not None
                            if a == "safe":
                                base = n
                            else:
                                t["regret"] += max(0, n - base)
                                t["worst_regret"] = max(t["worst_regret"], n - base)
                finally:
                    memory.close()
            for a in AGENTS:
                totals[(str(level), cond, a)] = acc[a]
    summary = [{"level": lv, "sensors": c, "agent": a, **v} for (lv, c, a), v in totals.items()]
    return {"milestone": "M6", "stride": stride, "cases": len(picked), "seeds": list(seeds),
            "summary": summary, "decision": decide(summary) if stride == STRIDE and tuple(seeds) == SEEDS else None}


def decide(summary: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The pre-registered decision rule (docs/PREREG_M6.md)."""
    g = {(r["level"], r["sensors"], r["agent"]): r for r in summary}
    conds = tuple(SENSOR_CONDITIONS)
    mem_agents = ("safe_mem", "safe_mem_guard")
    d1 = all(g[(str(lv), c, a)]["false_stops"] == 0 for lv in LEVELS for c in conds for a in mem_agents)
    out = {"D1_no_false_stops": d1}
    for a in mem_agents:
        d2 = all(g[("0.0", c, a)]["actions"] < g[("0.0", c, "safe")]["actions"] for c in conds)
        d3 = all(g[(lv, c, a)]["actions"] <= 1.02 * g[(lv, c, "safe")]["actions"]
                 for lv in ("1.0", "unrelated") for c in conds)
        out[f"D2_exact_memory_helps[{a}]"] = d2
        out[f"D3_wrong_memory_within_2pct[{a}]"] = d3
    worst = lambda a, lv: max(g[(lv, c, a)]["worst_regret"] for c in conds)  # noqa: E731
    lower = any(worst("safe_mem_guard", lv) < worst("safe_mem", lv) for lv in ("1.0", "unrelated"))
    cost = all(g[("0.0", c, "safe_mem_guard")]["actions"] <= 1.01 * g[("0.0", c, "safe_mem")]["actions"] for c in conds)
    out["D4_guard_useful"] = lower and cost
    pick = "safe_mem_guard" if out["D4_guard_useful"] else "safe_mem"
    out["candidate"] = pick
    out["recommend_memory"] = d1 and out[f"D2_exact_memory_helps[{pick}]"] and out[f"D3_wrong_memory_within_2pct[{pick}]"]
    return out


def format_report(report: Dict[str, Any]) -> str:
    lines = ["RED QUEEN M6 - disturbance memory under noisy sensors (pre-registered)",
             f"{report['cases']} grid cases x seeds {report['seeds']}; assumed noise bound {ASSUMED}", "",
             f"  {'memory':<10}{'sensors':<13}{'agent':<16}{'success':>11}{'false':>7}{'actions':>9}"
             f"{'regret':>8}{'worst':>7}{'guard':>7}", "  " + "-" * 86]
    last = None
    for r in report["summary"]:
        key = (r["level"], r["sensors"])
        if last is not None and key != last:
            lines.append("")
        last = key
        lines.append(f"  {r['level']:<10}{r['sensors']:<13}{r['agent']:<16}{r['success']:>6}/{r['runs']:<4}"
                     f"{r['false_stops']:>7}{r['actions']:>9}{r['regret']:>8}{r['worst_regret']:>7}{r['guard_trips']:>7}")
    d = report.get("decision")
    if d:
        lines += ["", "Pre-registered decision:"] + [f"  {k}: {v}" for k, v in d.items()]
    lines += ["", "regret = extra actions vs safe (no memory) on the same run, summed where positive;",
              "worst = largest single-run difference; guard = runs where the guard switched memory off."]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="RED QUEEN M6 memory-under-noisy-sensors study")
    p.add_argument("--output", help="write the JSON report here")
    args = p.parse_args(argv)
    report = run_study()
    print(format_report(report))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        print(f"\nJSON report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
