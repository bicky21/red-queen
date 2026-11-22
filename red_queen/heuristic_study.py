"""M9.2 study, exactly as pre-registered in docs/PREREG_M9_2.md.

Train a "distance to goal" heuristic on seeds 2000-2199, evaluate on fresh
seeds 3000-3099 in the closed loop used in M9 (exact observations, replan
every step, <= 25 actions):
  astar_hand / astar_learned    weighted A* (w=2, depth 25, 100,000 nodes)
  greedy_hand / greedy_learned  one-step lookahead

    python3 -m red_queen.heuristic_study --output results/m9_2_heuristic_study.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from .datacentre import DCEnvironment, DCExecutor, DCState, DCVerifier, goal_distance, goal_met, reference_plan
from .datacentre_study import MAX_ACTIONS, greedy_action, make_scenario
from .learned_heuristic import LearnedHeuristic, labelled_states, pearson, train

TRAIN_SEEDS = tuple(range(2000, 2200))
EVAL_SEEDS = tuple(range(3000, 3100))
AGENTS = ("astar_hand", "astar_learned", "greedy_hand", "greedy_learned")


def run_episode(agent: str, initial: DCState, learned: LearnedHeuristic) -> Dict[str, Any]:
    if agent not in AGENTS:
        raise ValueError(f"unknown agent {agent!r}")
    h = learned if agent.endswith("learned") else None
    env = DCEnvironment(initial)
    ex = DCExecutor(env, max_actions=MAX_ACTIONS)
    nodes = 0
    stop = "budget"
    while ex.executed < MAX_ACTIONS:
        state = env.observe()
        if goal_met(state):
            stop = "goal"
            break
        if agent.startswith("greedy"):
            action = greedy_action(state, heuristic=h)
        else:
            plan = reference_plan(state, max_depth=25, max_nodes=100000, weight=2, heuristic=h)
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
    return {"agent": agent, "success": DCVerifier().verify(env), "actions": ex.executed, "nodes": nodes,
            "stop": stop, "unchecked_actions": len(env.log) - ex.executed}


class ScaledHand:
    """EXPLORATORY control (added after the pre-registered run, not part of the
    decision): the hand heuristic times one constant fitted on the training labels.
    Tests whether the learned gain is only a matter of scale."""

    def __init__(self, scale: float) -> None:
        self.scale = float(scale)
        self.fallbacks = 0

    def __call__(self, s: DCState) -> float:
        return self.scale * goal_distance(s)


def run_study(train_seeds: Sequence[int] = TRAIN_SEEDS, eval_seeds: Sequence[int] = EVAL_SEEDS) -> Dict[str, Any]:
    learned, n_train = train(train_seeds, make_scenario)
    held = labelled_states(eval_seeds, make_scenario)
    ys = [float(y) for _, y in held]
    lp = [learned(s) for s, _ in held]
    hp = [float(goal_distance(s)) for s, _ in held]
    accuracy = {"held_out_states": len(held), "learned_mae": round(sum(abs(a - b) for a, b in zip(lp, ys)) / len(ys), 4),
                "learned_pearson": round(pearson(lp, ys), 4), "hand_pearson": round(pearson(hp, ys), 4)}
    rows = [dict(run_episode(a, make_scenario(s), learned), seed=s) for s in eval_seeds for a in AGENTS]
    summary = {}
    for a in AGENTS:
        rs = [r for r in rows if r["agent"] == a]
        summary[a] = {"scenarios": len(rs), "success": sum(r["success"] for r in rs),
                      "actions": sum(r["actions"] for r in rs), "nodes": sum(r["nodes"] for r in rs),
                      "stops": dict(sorted(Counter(r["stop"] for r in rs).items())),
                      "unchecked_actions": sum(r["unchecked_actions"] for r in rs)}
    train_data = labelled_states(train_seeds, make_scenario)
    scale = (sum(goal_distance(s) * y for s, y in train_data) / sum(goal_distance(s) ** 2 for s, _ in train_data))
    control = ScaledHand(scale)
    ctl = [run_episode("astar_learned", make_scenario(sd), control) for sd in eval_seeds]
    exploratory = {"note": "added after the pre-registered run; not part of the decision",
                   "hand_scale": round(scale, 4), "success": sum(r["success"] for r in ctl),
                   "actions": sum(r["actions"] for r in ctl), "nodes": sum(r["nodes"] for r in ctl)}
    report = {"milestone": "M9.2", "train_seeds": [min(train_seeds), max(train_seeds)],
              "eval_seeds": [min(eval_seeds), max(eval_seeds)], "training_states": n_train,
              "weights": learned.to_dict(), "heuristic_fallbacks": learned.fallbacks,
              "accuracy": accuracy, "summary": summary, "exploratory_scaled_hand": exploratory, "rows": rows}
    report["decision"] = (decide(summary) if tuple(train_seeds) == TRAIN_SEEDS and tuple(eval_seeds) == EVAL_SEEDS
                          else None)
    return report


def decide(summary: Dict[str, Any]) -> Dict[str, Any]:
    """The pre-registered decision rule (docs/PREREG_M9_2.md)."""
    h, l = summary["astar_hand"], summary["astar_learned"]
    l1 = l["success"] >= h["success"]
    l2 = l["nodes"] <= 0.8 * h["nodes"]
    l3 = l["actions"] <= 1.05 * h["actions"]
    return {"L1_success_not_lower": l1, "L2_nodes_at_most_80pct": l2, "L3_actions_at_most_105pct": l3,
            "node_ratio": round(l["nodes"] / h["nodes"], 4) if h["nodes"] else None,
            "learned_heuristic_useful": l1 and l2 and l3}


def format_report(report: Dict[str, Any]) -> str:
    acc = report["accuracy"]
    lines = ["RED QUEEN M9.2 - learning 'distance to goal' from experience (pre-registered)",
             f"trained on seeds {report['train_seeds'][0]}-{report['train_seeds'][1]} ({report['training_states']} labelled states); "
             f"evaluated on fresh seeds {report['eval_seeds'][0]}-{report['eval_seeds'][1]}", "",
             "learned weights: " + ", ".join(f"{k}={v}" for k, v in report["weights"].items()),
             f"held-out states {acc['held_out_states']}: learned MAE {acc['learned_mae']} steps; "
             f"correlation with true distance: learned {acc['learned_pearson']}, hand {acc['hand_pearson']}", "",
             f"  {'agent':<16}{'success':>9}{'actions':>9}{'planner nodes':>15}  stops", "  " + "-" * 66]
    for a, v in report["summary"].items():
        lines.append(f"  {a:<16}{v['success']:>5}/{v['scenarios']:<3}{v['actions']:>9}{v['nodes']:>15}  {v['stops']}")
    lines.append(f"safety: actions that bypassed the executor = "
                 f"{sum(v['unchecked_actions'] for v in report['summary'].values())}; heuristic fallbacks = {report['heuristic_fallbacks']}")
    e = report["exploratory_scaled_hand"]
    lines += ["", f"EXPLORATORY control ({e['note']}):",
              f"  A* with hand heuristic x {e['hand_scale']}: success {e['success']}, actions {e['actions']}, nodes {e['nodes']}"]
    if report.get("decision"):
        lines += ["", "Pre-registered decision (A* only):"] + [f"  {k}: {v}" for k, v in report["decision"].items()]
    lines += ["", "No claim of 'learning to think': this is a fitted linear estimate."]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="RED QUEEN M9.2 learned heuristic study")
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
