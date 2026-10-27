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

M3.4: with `slack` k > 0 (and a risk function), the planner may return a
plan up to k steps longer than the shortest one when its expected cost,
length + summed risk, is lower. k is capped at MAX_SLACK. k = 0, or no risk
function, gives exactly the M3.3 / M3 behaviour. Reachability never changes:
a slack search only runs after a shortest plan has been found, and falls
back to that plan if its own node budget runs out.

The planner only predicts. It never executes anything.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
MAX_SLACK = 3  # M3.4: hard cap on extra plan length


class BoundedPlanner:
    def __init__(
        self,
        goal: RecoveryGoal,
        max_depth: int = 10,
        max_nodes: int = 20000,
        action_order: Sequence[str] = ACTION_NAMES,
        slack: int = 0,
    ) -> None:
        if max_depth < 0 or max_nodes < 0:
            raise ValueError("max_depth and max_nodes must be >= 0")
        if isinstance(slack, bool) or not isinstance(slack, int) or not 0 <= slack <= MAX_SLACK:
            raise ValueError(f"slack must be an int in [0, {MAX_SLACK}]")
        order = tuple(action_order)
        if sorted(order) != sorted(ACTION_NAMES):
            raise ValueError("action_order must be a permutation of the action allowlist")
        self.goal = goal
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.action_order = order
        self.slack = slack

    def plan(self, start: SystemState, allowed: Iterable[str], risk: Optional[RiskFn] = None) -> Plan:
        allowed_set = set(allowed)
        order = [a for a in self.action_order if a in allowed_set]
        if self.goal.is_satisfied(start):
            return Plan((), (), True, 0, False)
        if risk is not None:
            shortest = self._plan_min_risk(start, order, risk)
            if self.slack == 0 or not shortest.reaches_goal:
                return shortest
            return self._plan_expected_cost(start, order, risk, shortest)

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

    def _plan_expected_cost(self, start: SystemState, order: Sequence[str], risk: RiskFn,
                            shortest: Plan) -> Plan:
        """Search plans of length L..L+slack (L = shortest) and return the one with
        the lowest expected cost = length + summed risk. Ties prefer the shorter
        plan, then discovery order. Goal states are not expanded further, because
        the agent stops as soon as the goal is reached."""
        shortest_len = len(shortest.actions)
        limit = min(self.max_depth, shortest_len + self.slack)
        nodes = shortest.nodes_expanded
        budget = nodes + self.max_nodes
        # layer[d]: state -> (summed risk, previous state); states may repeat across depths
        layers: List[Dict[SystemState, Tuple[float, Optional[SystemState], Optional[str]]]] = [
            {start: (0.0, None, None)}
        ]
        best = (shortest_len + shortest.risk, shortest_len)
        best_end: Optional[Tuple[int, SystemState]] = None
        for depth in range(limit):
            step_risk = _sanitise(risk(depth), order)
            nxt_layer: Dict[SystemState, Tuple[float, Optional[SystemState], Optional[str]]] = {}
            for state, (cost, _, _) in layers[depth].items():
                if depth > 0 and self.goal.is_satisfied(state):
                    continue
                for action in order:
                    if nodes >= budget:
                        return shortest  # out of budget: keep the safe shortest plan
                    nxt = apply_action(state, action)
                    nodes += 1
                    c = round(cost + step_risk.get(action, 0.0), 9)
                    if nxt not in nxt_layer or c < nxt_layer[nxt][0]:
                        nxt_layer[nxt] = (c, state, action)
            layers.append(nxt_layer)
            length = depth + 1
            if length < shortest_len:
                continue
            for state, (c, _, _) in nxt_layer.items():
                if self.goal.is_satisfied(state):
                    key = (round(length + c, 9), length)
                    if key < best:
                        best, best_end = key, (length, state)
        if best_end is None:
            return Plan(shortest.actions, shortest.predicted, True, nodes, False, shortest.risk)
        length, state = best_end
        actions: List[str] = []
        states: List[SystemState] = []
        for d in range(length, 0, -1):
            cost, prev, action = layers[d][state]
            actions.append(action)
            states.append(state)
            state = prev
        return Plan(tuple(reversed(actions)), tuple(reversed(states)), True, nodes, False,
                    round(best[0] - length, 9))

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
