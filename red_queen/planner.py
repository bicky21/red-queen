"""M3 bounded planner.

Breadth-first search over the policy-allowed actions using the agent's
transition model. It returns the shortest action sequence that the model
predicts reaches the goal. Search is bounded by `max_depth` (plan length)
and `max_nodes` (successor states generated). Actions are expanded in a
fixed order (sorted names by default; `action_order` can change it, M3.2)
and visited states are skipped, so the result is fully deterministic. The
order only chooses *which* shortest plan is returned, never its length.

M3.3: an optional `risk` function gives, for each plan depth, a per-action
risk in [0, MAX_MEMORY_DISCOUNT] (e.g. remembered chance of being blocked).
It is used ONLY to choose among plans of the shortest length: plan length,
reachability and the allowed action set are unaffected. With `risk=None`
the original M3 search runs unchanged.

The planner only predicts. It never executes anything.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .actions import ACTION_NAMES, apply_action
from .goal import RecoveryGoal
from .scoring import clamp_discount
from .state import SystemState


@dataclass(frozen=True)
class Plan:
    actions: Tuple[str, ...]
    predicted: Tuple[SystemState, ...]  # model-predicted state after each action
    reaches_goal: bool
    nodes_expanded: int
    hit_node_limit: bool
    risk: float = 0.0  # M3.3: summed tie-break risk of the chosen plan


RiskFn = Callable[[int], Mapping[str, float]]  # plan depth (0-based) -> action -> risk


class BoundedPlanner:
    def __init__(
        self,
        goal: RecoveryGoal,
        max_depth: int = 10,
        max_nodes: int = 20000,
        action_order: Sequence[str] = ACTION_NAMES,
    ) -> None:
        if max_depth < 0 or max_nodes < 0:
            raise ValueError("max_depth and max_nodes must be >= 0")
        order = tuple(action_order)
        if sorted(order) != sorted(ACTION_NAMES):
            raise ValueError("action_order must be a permutation of the action allowlist")
        self.goal = goal
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.action_order = order

    def plan(self, start: SystemState, allowed: Iterable[str], risk: Optional[RiskFn] = None) -> Plan:
        allowed_set = set(allowed)
        order = [a for a in self.action_order if a in allowed_set]
        if self.goal.is_satisfied(start):
            return Plan((), (), True, 0, False)
        if risk is not None:
            return self._plan_min_risk(start, order, risk)

        parent: Dict[SystemState, Optional[Tuple[SystemState, str]]] = {start: None}
        depth = {start: 0}
        frontier = deque([start])
        nodes = 0
        while frontier:
            state = frontier.popleft()
            if depth[state] >= self.max_depth:
                continue
            for action in order:
                if nodes >= self.max_nodes:
                    return Plan((), (), False, nodes, True)
                nxt = apply_action(state, action)
                nodes += 1
                if nxt in parent:
                    continue
                parent[nxt] = (state, action)
                depth[nxt] = depth[state] + 1
                if self.goal.is_satisfied(nxt):
                    return self._reconstruct(parent, nxt, nodes)
                frontier.append(nxt)
        return Plan((), (), False, nodes, False)

    def _plan_min_risk(self, start: SystemState, order: Sequence[str], risk: RiskFn) -> Plan:
        """Layer-by-layer BFS. Within a layer each state keeps its lowest-risk
        parent; the first layer containing a goal state ends the search, so the
        plan is always of the shortest length. Ties keep discovery order."""
        parent: Dict[SystemState, Optional[Tuple[SystemState, str]]] = {start: None}
        cost: Dict[SystemState, float] = {start: 0.0}
        layer = [start]
        nodes = 0
        for depth in range(self.max_depth):
            step_risk = _sanitise(risk(depth), order)
            nxt_layer: Dict[SystemState, None] = {}
            for state in layer:
                for action in order:
                    if nodes >= self.max_nodes:
                        return Plan((), (), False, nodes, True)
                    nxt = apply_action(state, action)
                    nodes += 1
                    if nxt in parent and nxt not in nxt_layer:
                        continue  # reached in an earlier layer, or is the start
                    c = round(cost[state] + step_risk.get(action, 0.0), 9)
                    if nxt not in nxt_layer or c < cost[nxt]:
                        parent[nxt] = (state, action)
                        cost[nxt] = c
                        nxt_layer[nxt] = None
            goals = [s for s in nxt_layer if self.goal.is_satisfied(s)]
            if goals:
                best = min(goals, key=lambda s: cost[s])  # min() keeps the first of equal costs
                plan = self._reconstruct(parent, best, nodes)
                return Plan(plan.actions, plan.predicted, True, nodes, False, cost[best])
            if not nxt_layer:
                break
            layer = list(nxt_layer)
        return Plan((), (), False, nodes, False)

    @staticmethod
    def _reconstruct(parent, end: SystemState, nodes: int) -> Plan:
        actions, states = [], []
        node = end
        while parent[node] is not None:
            prev, action = parent[node]
            actions.append(action)
            states.append(node)
            node = prev
        return Plan(tuple(reversed(actions)), tuple(reversed(states)), True, nodes, False)


def _sanitise(raw: Mapping[str, float], order: Sequence[str]) -> Dict[str, float]:
    """Untrusted risk input: keep only allowed actions, clamp to [0, cap], drop NaN/junk."""
    out: Dict[str, float] = {}
    try:
        items = list(raw.items())
    except AttributeError:
        return out
    for action, value in items:
        if action in order:
            v = clamp_discount(value)
            if v > 0:
                out[action] = v
    return out
