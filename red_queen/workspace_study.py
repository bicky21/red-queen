"""M8 functional study, exactly as pre-registered in docs/PREREG_M8.md.

Agents (all SafeAgent with two sensors, adaptive bounds, hull, isolate):
  safe                    no memory (reference)
  safe_mem_guard          M6 disturbance memory + guard, no workspace
  safe_mem_guard_ws       + global workspace, capacity 1
  safe_mem_guard_ws_full  + global workspace, capacity None (broadcast everything)

Cases: every 10th M3.2 grid case (168); memory levels 0.0 and 1.0 (M6
pipeline); sensors: M5.4 conditions none, B_stuck_low, A_stuck_mid; seeds 0, 1.
The SelfModule's self-model is fitted exactly as in M7.

    python3 -m red_queen.workspace_study --output results/m8_workspace_study.json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from .agent import STOP_GOAL
from .cross_isolation_study import sensors_for as m54_sensors
from .environment import SimulatedEnvironment
from .goal import RecoveryGoal
from .memory_advice import MemoryAdvisor
from .noisy_memory_study import memory_for
from .safe_agent import SafeAgent
from .scenario_study import grid_cases
from .self_model import SelfModel
from .self_model_study import TRAIN_SEEDS
from .warning_study import collect_episodes
from .workspace import standard_workspace

GOAL = RecoveryGoal()
STRIDE = 10
LEVELS = (0.0, 1.0)
SENSOR_CONDITIONS = ("none", "B_stuck_low", "A_stuck_mid")
FAULTY = ("B_stuck_low", "A_stuck_mid")
SEEDS = (0, 1)
AGENTS = ("safe", "safe_mem_guard", "safe_mem_guard_ws", "safe_mem_guard_ws_full")


def fit_self_model() -> SelfModel:
    train = collect_episodes(TRAIN_SEEDS)
    return SelfModel().fit((k, e["success"]) for e in train for k in e["keys"])


def run_agent(agent: str, env, sensors, name: str, advisor, self_model: SelfModel):
    if agent == "safe":
        return SafeAgent(GOAL, 5).run(env, sensors, name)
    mem = SafeAgent(GOAL, 5, advisor=advisor, predict_disturbances=True, disturbance_guard="on_mismatch")
    if agent == "safe_mem_guard":
        return mem.run(env, sensors, name)
    if agent == "safe_mem_guard_ws":
        return mem.run(env, sensors, name, workspace=standard_workspace(self_model, 1))
    if agent == "safe_mem_guard_ws_full":
        return mem.run(env, sensors, name, workspace=standard_workspace(self_model, None))
    raise ValueError(f"unknown agent {agent!r}")


def run_study(stride: int = STRIDE, seeds=SEEDS, self_model: Optional[SelfModel] = None) -> Dict[str, Any]:
    if stride < 1:
        raise ValueError("stride must be >= 1")
    model = self_model if self_model is not None else fit_self_model()
    cases = list(grid_cases())
    picked = list(range(0, len(cases), stride))
    summary = []
    identical_cap = 0
    compared_cap = 0
    for level in LEVELS:
        for cond in SENSOR_CONDITIONS:
            acc = {a: {"runs": 0, "success": 0, "false_stops": 0, "actions": 0, "memory_off_by_workspace": 0,
                       "stream_messages": 0, "proposed_messages": 0} for a in AGENTS}
            for i in picked:
                start, faults, disturbances = cases[i]
                name = f"grid_{i:04d}"
                memory = memory_for(level, i, cases)
                try:
                    advisor = MemoryAdvisor(memory, name)
                    for seed in seeds:
                        actions_by = {}
                        for a in AGENTS:
                            env = SimulatedEnvironment(start, dict(disturbances), faults)
                            r = run_agent(a, env, m54_sensors(cond, seed), name, advisor, model)
                            t = acc[a]
                            t["runs"] += 1
                            t["success"] += r.success
                            t["false_stops"] += r.stop_reason == STOP_GOAL and not r.success
                            t["actions"] += len(r.actions)
                            actions_by[a] = r.actions
                            if "workspace_report" in r.stats:
                                t["memory_off_by_workspace"] += r.stats["workspace_memory_off_at"] is not None
                                t["stream_messages"] += r.stats["workspace_report"]["broadcast"]
                                t["proposed_messages"] += r.stats["workspace_report"]["proposed"]
                        compared_cap += 1
                        identical_cap += actions_by["safe_mem_guard_ws"] == actions_by["safe_mem_guard_ws_full"]
                finally:
                    memory.close()
            for a in AGENTS:
                summary.append({"level": str(level), "sensors": cond, "agent": a, **acc[a]})
    report = {"milestone": "M8", "stride": stride, "cases": len(picked), "seeds": list(seeds),
              "summary": summary, "capacity_runs_identical": identical_cap, "capacity_runs_compared": compared_cap}
    report["decision"] = decide(report)
    return report


def decide(report: Dict[str, Any]) -> Dict[str, Any]:
    """The pre-registered functional decision rule (docs/PREREG_M8.md)."""
    g = {(r["level"], r["sensors"], r["agent"]): r for r in report["summary"]}
    keys = [(str(lv), c) for lv in LEVELS for c in SENSOR_CONDITIONS]
    f1 = all(g[k + ("safe_mem_guard_ws",)]["false_stops"] <= g[k + ("safe_mem_guard",)]["false_stops"] for k in keys)
    ratio = {f"{lv}/{c}": g[(lv, c, "safe_mem_guard_ws")]["actions"] / g[(lv, c, "safe_mem_guard")]["actions"]
             for lv, c in keys}
    gain = any(ratio[f"{str(lv)}/{c}"] <= 0.99 for lv in LEVELS for c in FAULTY)
    never_worse = all(r <= 1.01 for r in ratio.values())
    f2 = gain and never_worse
    full_msgs = sum(g[k + ("safe_mem_guard_ws_full",)]["stream_messages"] for k in keys)
    cap1_msgs = sum(g[k + ("safe_mem_guard_ws",)]["stream_messages"] for k in keys)
    f3 = report["capacity_runs_identical"] == report["capacity_runs_compared"] and cap1_msgs <= 0.5 * full_msgs
    return {"F1_no_extra_false_stops": f1, "F2_coordination_useful": f2, "F3_capacity_identical_and_stream_halved": f3,
            "action_ratio_ws_vs_guard": {k: round(v, 4) for k, v in ratio.items()},
            "stream_messages_cap1_vs_full": [cap1_msgs, full_msgs],
            "workspace_functionally_useful": f1 and f2}


def format_report(report: Dict[str, Any]) -> str:
    lines = ["RED QUEEN M8 - global workspace, functional test (pre-registered)",
             f"{report['cases']} grid cases x seeds {report['seeds']}; memory levels {list(LEVELS)}; "
             f"sensors {list(SENSOR_CONDITIONS)}", "",
             f"  {'memory':<7}{'sensors':<13}{'agent':<24}{'success':>11}{'false':>7}{'actions':>9}"
             f"{'mem off':>9}{'stream':>8}{'proposed':>10}", "  " + "-" * 96]
    last = None
    for r in report["summary"]:
        key = (r["level"], r["sensors"])
        if last is not None and key != last:
            lines.append("")
        last = key
        lines.append(f"  {r['level']:<7}{r['sensors']:<13}{r['agent']:<24}{r['success']:>6}/{r['runs']:<4}"
                     f"{r['false_stops']:>7}{r['actions']:>9}{r['memory_off_by_workspace']:>9}"
                     f"{r['stream_messages']:>8}{r['proposed_messages']:>10}")
    d = report["decision"]
    lines += ["", f"capacity 1 vs capacity None: identical action sequences in {report['capacity_runs_identical']}"
              f"/{report['capacity_runs_compared']} paired runs", "", "Pre-registered decision:"]
    lines += [f"  {k}: {v}" for k, v in d.items()]
    lines += ["", "mem off = runs where a workspace broadcast switched disturbance memory off.",
              "stream = messages broadcast to all modules; proposed = messages offered by modules.",
              "No claim of consciousness: this measures architecture properties and their effect only."]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="RED QUEEN M8 global workspace study")
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
