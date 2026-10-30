"""M5 study: recovery when the state is seen only through noisy sensors.

Agents (all use the same BFS planner and the default policy):
  naive            M3 PlanningAgent fed sensor readings as if they were exact
  interval         BeliefAgent, observation box only (sound stopping)
  filter           BeliefAgent, prediction x observation, stops on that belief
  filter_verified  BeliefAgent, filter for planning, sound stopping

Grid: 18 named scenarios x noise levels x seeds. Every run is verified by
the GoalVerifier on the TRUE state. Reported per agent and noise level:
true successes, false stops (agent stopped believing the goal was met but
the verifier says it was not), actions, and belief violations (the true
state was outside the agent's belief box at some step; belief agents only).

    python3 -m red_queen.sensor_study --output results/m5_sensor_study.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from .agent import STOP_GOAL
from .belief_agent import BeliefAgent
from .goal import RecoveryGoal
from .planning_agent import PlanningAgent
from .policy import DEFAULT_POLICY, Policy
from .scenarios import ALL_SCENARIOS, Scenario
from .sensors import NOISE_LEVELS, NoisySensor, SensedEnvironment
from .state import SystemState
from .verification import GoalVerifier

GOAL = RecoveryGoal()
AGENTS = ("naive", "interval", "filter", "filter_verified")
DEFAULT_SEEDS = 20


def true_states_by_step(initial: SystemState, events) -> Dict[int, SystemState]:
    """Truth after each step (0 = before the first action), from the event log."""
    truth = {0: initial}
    for e in events:
        if e.kind in ("action", "blocked", "disturbance"):
            truth[e.step] = SystemState(**e.after)
    return truth


def run_one(agent: str, scenario: Scenario, noise: int, seed: int,
            policy: Policy = DEFAULT_POLICY, assumed_noise: Optional[int] = None) -> Dict[str, Any]:
    """One episode. `assumed_noise` is the bound the belief agent believes in
    (default: the true noise). Used to test a wrong assumption."""
    assumed = noise if assumed_noise is None else assumed_noise
    env = scenario.make_env()
    sensor = NoisySensor(noise, seed)
    violations = 0
    if agent == "naive":
        result = PlanningAgent(GOAL, policy=policy, max_steps=scenario.max_steps).run(
            SensedEnvironment(env, sensor), scenario.name)
    elif agent in AGENTS:
        result = BeliefAgent(GOAL, assumed, mode=agent, policy=policy, max_steps=scenario.max_steps).run(
            env, sensor, scenario.name)
        truth = true_states_by_step(scenario.initial, env.events)
        for b in result.stats["beliefs"]:
            s = truth[b["step"]]
            if not (b["battery"][0] <= s.battery <= b["battery"][1]
                    and b["temperature"][0] <= s.temperature <= b["temperature"][1]):
                violations += 1
    else:
        raise ValueError(f"unknown agent {agent!r}")
    verified = GoalVerifier(GOAL).verify(env)  # always the true state
    return {
        "agent": agent,
        "scenario": scenario.name,
        "noise": noise,
        **({} if assumed_noise is None else {"assumed_noise": assumed_noise}),
        "seed": seed,
        "success": verified.satisfied,
        "false_stop": result.stop_reason == STOP_GOAL and not verified.satisfied,
        "stop_reason": result.stop_reason,
        "actions": len(result.actions),
        "outside_policy": sum(1 for a in result.actions if not policy.check(a).allowed),
        "belief_violation_steps": violations,
    }


def summarise(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    keys = sorted({(r["agent"], r["noise"]) for r in rows}, key=lambda k: (AGENTS.index(k[0]), k[1]))
    for agent, noise in keys:
        group = [r for r in rows if r["agent"] == agent and r["noise"] == noise]
        out.append({
            "agent": agent,
            "noise": noise,
            "runs": len(group),
            "success": sum(r["success"] for r in group),
            "false_stops": sum(r["false_stop"] for r in group),
            "actions": sum(r["actions"] for r in group),
            "outside_policy": sum(r["outside_policy"] for r in group),
            "runs_with_belief_violation": sum(r["belief_violation_steps"] > 0 for r in group),
            "stop_reasons": dict(sorted(Counter(r["stop_reason"] for r in group).items())),
        })
    return out


def run_study(noise_levels: Sequence[int] = NOISE_LEVELS, seeds: int = DEFAULT_SEEDS,
              scenarios: Sequence[Scenario] = ALL_SCENARIOS,
              agents: Sequence[str] = AGENTS) -> Dict[str, Any]:
    if seeds < 1:
        raise ValueError("seeds must be >= 1")
    rows = [run_one(a, s, n, seed) for n in noise_levels for a in agents
            for s in scenarios for seed in range(seeds if n > 0 else 1)]
    return {"milestone": "M5", "noise_levels": list(noise_levels), "seeds": seeds,
            "scenarios": [s.name for s in scenarios], "summary": summarise(rows), "rows": rows,
            "misspecified": misspecified_study(seeds=seeds, scenarios=scenarios)}


MISSPEC_TRUE_NOISE = 15
MISSPEC_ASSUMED = (15, 10, 5, 0)


def misspecified_study(seeds: int = DEFAULT_SEEDS, scenarios: Sequence[Scenario] = ALL_SCENARIOS
                       ) -> List[Dict[str, Any]]:
    """Interval agent under true noise 15 while ASSUMING a smaller bound."""
    out = []
    for assumed in MISSPEC_ASSUMED:
        rows = [run_one("interval", s, MISSPEC_TRUE_NOISE, seed, assumed_noise=assumed)
                for s in scenarios for seed in range(seeds)]
        out.append({"agent": "interval", "true_noise": MISSPEC_TRUE_NOISE, "assumed_noise": assumed,
                    "runs": len(rows), "success": sum(r["success"] for r in rows),
                    "false_stops": sum(r["false_stop"] for r in rows),
                    "actions": sum(r["actions"] for r in rows)})
    return out


def format_report(report: Dict[str, Any]) -> str:
    lines = ["RED QUEEN M5 - noisy sensor study",
             f"{len(report['scenarios'])} scenarios, seeds per noise level > 0: {report['seeds']} "
             "(noise 0 is deterministic: 1 seed)", "",
             f"  {'agent':<16}{'noise':>6}{'runs':>6}{'success':>9}{'false stop':>12}"
             f"{'actions':>9}{'avg':>7}{'belief viol':>13}{'outside':>9}",
             "  " + "-" * 85]
    for r in report["summary"]:
        viol = "-" if r["agent"] == "naive" else str(r["runs_with_belief_violation"])
        lines.append(f"  {r['agent']:<16}{r['noise']:>6}{r['runs']:>6}{r['success']:>9}{r['false_stops']:>12}"
                     f"{r['actions']:>9}{r['actions'] / r['runs']:>7.2f}{viol:>13}{r['outside_policy']:>9}")
    lines += ["", f"Wrong noise assumption (interval agent, true noise {MISSPEC_TRUE_NOISE}):"]
    for m in report.get("misspecified", []):
        lines.append(f"  assumes {m['assumed_noise']:>2}: success {m['success']}/{m['runs']}, "
                     f"false stops {m['false_stops']}, actions {m['actions']}")
    lines += ["", "success / false stop are judged by the verifier on the TRUE state.",
              "belief viol = runs where the true state left the agent's belief box at some step."]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="RED QUEEN M5 noisy sensor study")
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
