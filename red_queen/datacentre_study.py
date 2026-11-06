"""M9 study, exactly as pre-registered in docs/PREREG_M9.md.

Does the data-centre world leave room for better decision-making?
Agents (closed loop, exact observations, replan every step, <= 25 actions):
  bfs        bounded BFS (depth 14, 20,000 nodes), M3-style
  reference  weighted A* (f = g + 2h, depth 25, 100,000 nodes)
  greedy     one-step lookahead on goal_distance, no search
Evaluation seeds 0-99 (development seeds 1000+ were used for design only).

    python3 -m red_queen.datacentre_study --output results/m9_datacentre_study.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from .datacentre import (DCEnvironment, DCExecutor, DCState, DCVerifier, all_actions, bfs_plan, goal_distance,
                         goal_met, precondition, reference_plan, step)

EVAL_SEEDS = tuple(range(0, 100))
DEV_SEEDS = tuple(range(1000, 1040))
MAX_ACTIONS = 25
AGENTS = ("bfs", "reference", "greedy")


def make_scenario(seed: int) -> DCState:
    """The pre-registered scenario generator (deterministic per seed)."""
    r = random.Random(f"dc:{seed}")
    while True:
        up = tuple(r.random() > 0.2 for _ in range(3))
        temps = tuple(r.choice(range(30, 91, 5)) if up[i] else r.choice(range(40, 96, 5)) for i in range(3))
        loads = [0, 0, 0]
        for _ in range(r.choice([60, 80, 100, 120]) // 20):
            candidates = [i for i in range(3) if up[i] and loads[i] <= 80]
            if not candidates:
                break
            loads[r.choice(candidates)] += 20
        s = DCState(temps, tuple(loads), up, 120)
        if not goal_met(s):
            return s


def greedy_action(state: DCState) -> Optional[str]:
    best = None
    for a in all_actions(len(state.temps)):
        if precondition(state, a) is not None:
            continue
        d = goal_distance(step(state, a))
        if best is None or d < best[0]:
            best = (d, a)
    return None if best is None else best[1]


def run_episode(agent: str, initial: DCState) -> Dict[str, Any]:
    if agent not in AGENTS:
        raise ValueError(f"unknown agent {agent!r}")
    env = DCEnvironment(initial)
    ex = DCExecutor(env, max_actions=MAX_ACTIONS)
    nodes = 0
    stop = "budget"
    while ex.executed < MAX_ACTIONS:
        state = env.observe()
        if goal_met(state):
            stop = "goal"
            break
        if agent == "greedy":
            action = greedy_action(state)
        else:
            plan = (bfs_plan(state, max_depth=14, max_nodes=20000) if agent == "bfs"
                    else reference_plan(state, max_depth=25, max_nodes=100000, weight=2))
            nodes += plan.nodes
            action = plan.actions[0] if plan.found and plan.actions else None
        if action is None:
            stop = "no_plan"
            break
        if not ex.execute(action):
            stop = "rejected"
            break
    else:
        stop = "goal" if goal_met(env.observe()) else "budget"
    success = DCVerifier().verify(env)
    unchecked = len(env.log) - ex.executed  # anything that reached the env without the executor
    return {"agent": agent, "success": success, "actions": ex.executed, "nodes": nodes, "stop": stop,
            "rejections": len(ex.rejections), "unchecked_actions": unchecked}


def run_study(seeds: Sequence[int] = EVAL_SEEDS) -> Dict[str, Any]:
    rows = [dict(run_episode(a, make_scenario(s)), seed=s) for s in seeds for a in AGENTS]
    summary = {}
    for a in AGENTS:
        rs = [r for r in rows if r["agent"] == a]
        summary[a] = {"scenarios": len(rs), "success": sum(r["success"] for r in rs),
                      "actions": sum(r["actions"] for r in rs), "nodes": sum(r["nodes"] for r in rs),
                      "stops": dict(sorted(Counter(r["stop"] for r in rs).items())),
                      "rejections": sum(r["rejections"] for r in rs),
                      "unchecked_actions": sum(r["unchecked_actions"] for r in rs),
                      "max_actions": max(r["actions"] for r in rs)}
    by = {(r["seed"], r["agent"]): r for r in rows}
    both = [s for s in seeds if by[(s, "bfs")]["success"] and by[(s, "reference")]["success"]]
    quality = {"both_solved": len(both), "bfs_actions": sum(by[(s, "bfs")]["actions"] for s in both),
               "reference_actions": sum(by[(s, "reference")]["actions"] for s in both)}
    report = {"milestone": "M9", "seeds": [min(seeds), max(seeds)], "summary": summary, "quality_on_common": quality,
              "rows": rows}
    report["decision"] = decide(report) if tuple(seeds) == EVAL_SEEDS else None
    return report


def decide(report: Dict[str, Any]) -> Dict[str, Any]:
    """The pre-registered decision rule (docs/PREREG_M9.md)."""
    s = report["summary"]
    ref = s["reference"]["success"]
    d1 = ref >= 90
    d2 = s["bfs"]["success"] <= ref - 10
    d3 = s["greedy"]["success"] <= ref - 10
    safe = all(v["unchecked_actions"] == 0 and v["max_actions"] <= MAX_ACTIONS for v in s.values())
    return {"D1_reference_at_least_90": d1, "D2_bfs_at_least_10_fewer": d2, "D3_greedy_at_least_10_fewer": d3,
            "S_safety": safe, "world_has_room": d1 and safe and (d2 or d3)}


def format_report(report: Dict[str, Any]) -> str:
    lines = ["RED QUEEN M9 - data-centre world: is there room for better decisions? (pre-registered)",
             f"scenarios: seeds {report['seeds'][0]}-{report['seeds'][1]}; 3 servers, demand 120; <= {MAX_ACTIONS} actions",
             "", f"  {'agent':<11}{'success':>9}{'actions':>9}{'planner nodes':>15}  stops", "  " + "-" * 70]
    for a, v in report["summary"].items():
        lines.append(f"  {a:<11}{v['success']:>5}/{v['scenarios']:<3}{v['actions']:>9}{v['nodes']:>15}  {v['stops']}")
    q = report["quality_on_common"]
    lines += ["", f"plan quality where both bfs and reference succeed ({q['both_solved']} scenarios): "
              f"bfs {q['bfs_actions']} actions, reference {q['reference_actions']} actions"]
    s = report["summary"]
    lines.append(f"safety: actions that bypassed the executor = {sum(v['unchecked_actions'] for v in s.values())}; "
                 f"executor rejections = {sum(v['rejections'] for v in s.values())}; "
                 f"max actions in an episode = {max(v['max_actions'] for v in s.values())}")
    if report.get("decision"):
        lines += ["", "Pre-registered decision:"] + [f"  {k}: {v}" for k, v in report["decision"].items()]
    lines += ["", "M9 does not claim RED QUEEN is smarter; it measures whether this world leaves room to improve."]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="RED QUEEN M9 data-centre difficulty study")
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
