"""M3 bounded planner.

Breadth-first search over the policy-allowed actions using the agent's
transition model. It returns the shortest action sequence that the model
predicts reaches the goal. Search is bounded by `max_depth` (plan length)
and `max_nodes` (successor states generated). Actions are expanded in
sorted name order and visited states are skipped, so the result is fully
deterministic.

The planner only predicts. It never executes anything.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

from .actions import ACTION_NAMES, apply_action
from .goal import RecoveryGoal
from .state import SystemState


@dataclass(frozen=True)
class Plan:
    actions: Tuple[str, ...]
    predicted: Tuple[SystemState, ...]  # model-predicted state after each action
    reaches_goal: bool
    nodes_expanded: int
    hit_node_limit: bool


class BoundedPlanner:
    def __init__(self, goal: RecoveryGoal, max_depth: int = 10, max_nodes: int = 20000) -> None:
        if max_depth < 0 or max_nodes < 0:
            raise ValueError("max_depth and max_nodes must be >= 0")
        self.goal = goal
        self.max_depth = max_depth
        self.max_nodes = max_nodes

    def plan(self, start: SystemState, allowed: Iterable[str]) -> Plan:
        allowed_set = set(allowed)
        order = [a for a in ACTION_NAMES if a in allowed_set]
        if self.goal.is_satisfied(start):
            return Plan((), (), True, 0, False)

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
