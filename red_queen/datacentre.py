"""M9 data-centre world: a harder, coupled simulation (separate from the M1-M8 domain).

Three servers share airflow and a fixed workload (demand). Each server has a
temperature (0-100), a load (0-100, in steps of 20) and an up/down flag.
Fixing one server can make another worse:
  - cool:i         server i temperature -15, every OTHER up server +5 (shared air)
  - migrate:i:j    move 20 load from i to j (heat follows load next step)
  - throttle:i     drop 20 load on i (it becomes unserved demand)
  - unthrottle:i   add 20 load back on i (only while served < demand)
  - restart:i      bring a DOWN server back up with load 0 (only if temp <= 60)
  - wait

After every action, physics runs once:
  every up server: temp += load // 20 - 2  (load 40 is heat-neutral)
  every down server: temp -= 5  (it cools while switched off)
  an up server whose temp > 90 crashes: it goes down and its load is lost.

Goal (checked by an independent verifier): all servers up, every temp <= 70,
every load <= 40 (sustainable: no server heats up) and served == demand.

Everything is deterministic and simulated; actions are strings checked by an
allowlist and preconditions before anything changes (DCExecutor).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

N_SERVERS = 3
LOAD_STEP = 20
MAX_LOAD = 100
CRASH_TEMP = 90
RESTART_MAX_TEMP = 60
GOAL_MAX_TEMP = 70
GOAL_MAX_LOAD = 40
COOL_DROP = 15
COOL_SPILL = 5
DOWN_COOLING = 5
ACTION_KINDS = ("cool", "migrate", "throttle", "unthrottle", "restart", "wait")


def _clamp(v: int) -> int:
    return max(0, min(100, v))


@dataclass(frozen=True)
class DCState:
    temps: Tuple[int, ...]
    loads: Tuple[int, ...]
    up: Tuple[bool, ...]
    demand: int

    def __post_init__(self) -> None:
        n = len(self.temps)
        if not (n == len(self.loads) == len(self.up) and n >= 1):
            raise ValueError("temps, loads and up must have the same length")
        for t in self.temps:
            if isinstance(t, bool) or not isinstance(t, int) or not 0 <= t <= 100:
                raise ValueError(f"invalid temperature {t!r}")
        for ld, u in zip(self.loads, self.up):
            if isinstance(ld, bool) or not isinstance(ld, int) or not 0 <= ld <= MAX_LOAD or ld % LOAD_STEP:
                raise ValueError(f"invalid load {ld!r}")
            if not u and ld:
                raise ValueError("a down server cannot carry load")
        if isinstance(self.demand, bool) or not isinstance(self.demand, int) or self.demand < 0 or self.demand % LOAD_STEP:
            raise ValueError("invalid demand")

    @property
    def served(self) -> int:
        return sum(self.loads)

    def to_dict(self) -> Dict:
        return {"temps": list(self.temps), "loads": list(self.loads), "up": list(self.up), "demand": self.demand}


def all_actions(n: int = N_SERVERS) -> List[str]:
    acts = [f"cool:{i}" for i in range(n)]
    acts += [f"migrate:{i}:{j}" for i in range(n) for j in range(n) if i != j]
    acts += [f"throttle:{i}" for i in range(n)] + [f"unthrottle:{i}" for i in range(n)]
    acts += [f"restart:{i}" for i in range(n)] + ["wait"]
    return acts


def parse_action(action: str, n: int) -> Tuple[str, Tuple[int, ...]]:
    """Strictly parse an action string; raises ValueError on anything unexpected."""
    if not isinstance(action, str) or len(action) > 20:
        raise ValueError("action must be a short string")
    parts = action.split(":")
    kind, args = parts[0], parts[1:]
    if kind not in ACTION_KINDS:
        raise ValueError(f"unknown action kind {kind!r}")
    want = {"wait": 0, "migrate": 2}.get(kind, 1)
    if len(args) != want or not all(a.isdigit() and len(a) == 1 for a in args):
        raise ValueError(f"bad arguments for {kind!r}")
    idx = tuple(int(a) for a in args)
    if any(i >= n for i in idx) or (kind == "migrate" and idx[0] == idx[1]):
        raise ValueError("server index out of range")
    return kind, idx


def precondition(state: DCState, action: str) -> Optional[str]:
    """None if the action may run in this state, otherwise the reason it may not."""
    kind, idx = parse_action(action, len(state.temps))
    if kind == "wait":
        return None
    i = idx[0]
    if kind == "restart":
        if state.up[i]:
            return "already_up"
        return None if state.temps[i] <= RESTART_MAX_TEMP else "too_hot_to_restart"
    if not state.up[i]:
        return "server_down"
    if kind == "cool":
        return None
    if kind == "throttle":
        return None if state.loads[i] >= LOAD_STEP else "no_load"
    if kind == "unthrottle":
        if state.loads[i] > MAX_LOAD - LOAD_STEP:
            return "server_full"
        return None if state.served < state.demand else "demand_already_served"
    j = idx[1]  # migrate
    if not state.up[j]:
        return "target_down"
    if state.loads[i] < LOAD_STEP:
        return "no_load"
    return None if state.loads[j] <= MAX_LOAD - LOAD_STEP else "target_full"


def step(state: DCState, action: str) -> DCState:
    """Apply one action (which must satisfy its precondition) and then the physics."""
    reason = precondition(state, action)
    if reason is not None:
        raise ValueError(f"precondition failed for {action}: {reason}")
    kind, idx = parse_action(action, len(state.temps))
    temps, loads, up = list(state.temps), list(state.loads), list(state.up)
    if kind == "cool":
        i = idx[0]
        temps[i] = _clamp(temps[i] - COOL_DROP)
        for k in range(len(temps)):
            if k != i and up[k]:
                temps[k] = _clamp(temps[k] + COOL_SPILL)
    elif kind == "migrate":
        i, j = idx
        loads[i] -= LOAD_STEP
        loads[j] += LOAD_STEP
    elif kind == "throttle":
        loads[idx[0]] -= LOAD_STEP
    elif kind == "unthrottle":
        loads[idx[0]] += LOAD_STEP
    elif kind == "restart":
        up[idx[0]] = True
        loads[idx[0]] = 0
    # physics
    for k in range(len(temps)):
        if up[k]:
            temps[k] = _clamp(temps[k] + loads[k] // LOAD_STEP - 2)
            if temps[k] > CRASH_TEMP:
                up[k] = False
                loads[k] = 0
        else:
            temps[k] = _clamp(temps[k] - DOWN_COOLING)
    return DCState(tuple(temps), tuple(loads), tuple(up), state.demand)


def goal_met(state: DCState) -> bool:
    return (all(state.up) and all(t <= GOAL_MAX_TEMP for t in state.temps)
            and all(ld <= GOAL_MAX_LOAD for ld in state.loads) and state.served == state.demand)


def goal_distance(state: DCState) -> int:
    """Heuristic distance used only by the greedy baseline (never by the verifier)."""
    d = 3 * sum(not u for u in state.up)
    d += sum(max(0, t - GOAL_MAX_TEMP) for t in state.temps) // 5
    d += sum(max(0, ld - GOAL_MAX_LOAD) for ld in state.loads) // LOAD_STEP
    d += abs(state.demand - state.served) // LOAD_STEP
    return d


class DCVerifier:
    """Independent goal check on the TRUE state, after the episode."""

    def verify(self, env: "DCEnvironment") -> bool:
        return goal_met(env.state)


class DCEnvironment:
    def __init__(self, initial: DCState) -> None:
        self.state = initial
        self.steps = 0
        self.log: List[Dict] = []

    def observe(self) -> DCState:
        return self.state

    def execute(self, action: str) -> DCState:
        before = self.state
        self.state = step(self.state, action)  # raises before any change if invalid
        self.steps += 1
        self.log.append({"step": self.steps, "action": action, "before": before.to_dict(),
                         "after": self.state.to_dict()})
        return self.state


class DCExecutor:
    """Execution boundary: allowlist of action kinds, strict parsing, preconditions,
    and a hard action budget. Nothing reaches the environment unless all pass."""

    def __init__(self, env: DCEnvironment, allowed_kinds: Sequence[str] = ACTION_KINDS, max_actions: int = 25) -> None:
        bad = set(allowed_kinds) - set(ACTION_KINDS)
        if bad:
            raise ValueError(f"unknown action kinds in allowlist: {sorted(bad)}")
        if max_actions < 0:
            raise ValueError("max_actions must be >= 0")
        self.env = env
        self.allowed = frozenset(allowed_kinds)
        self.max_actions = max_actions
        self.executed = 0
        self.rejections: List[Tuple[str, str]] = []

    def execute(self, action: str) -> bool:
        try:
            kind, _ = parse_action(action, len(self.env.state.temps))
        except ValueError as exc:
            self.rejections.append((str(action)[:20], f"invalid: {exc}"))
            return False
        if kind not in self.allowed:
            self.rejections.append((action, "denied_by_policy"))
            return False
        if self.executed >= self.max_actions:
            self.rejections.append((action, "budget_exhausted"))
            return False
        reason = precondition(self.env.state, action)
        if reason is not None:
            self.rejections.append((action, reason))
            return False
        self.env.execute(action)
        self.executed += 1
        return True


@dataclass
class DCPlan:
    actions: Tuple[str, ...]
    found: bool
    nodes: int
    hit_node_limit: bool


def bfs_plan(start: DCState, allowed_kinds: Sequence[str] = ACTION_KINDS, max_depth: int = 12,
             max_nodes: int = 20000) -> DCPlan:
    """Bounded breadth-first search for a shortest plan (deterministic action order)."""
    if max_depth < 0 or max_nodes < 1:
        raise ValueError("invalid bounds")
    if goal_met(start):
        return DCPlan((), True, 0, False)
    acts = [a for a in all_actions(len(start.temps)) if a.split(":")[0] in set(allowed_kinds)]
    frontier = deque([(start, ())])
    seen = {start}
    nodes = 0
    while frontier:
        state, path = frontier.popleft()
        if len(path) >= max_depth:
            continue
        nodes += 1
        if nodes > max_nodes:
            return DCPlan((), False, nodes - 1, True)
        for a in acts:
            if precondition(state, a) is not None:
                continue
            nxt = step(state, a)
            if nxt in seen:
                continue
            if goal_met(nxt):
                return DCPlan(path + (a,), True, nodes, False)
            seen.add(nxt)
            frontier.append((nxt, path + (a,)))
    return DCPlan((), False, nodes, False)


def reference_plan(start: DCState, allowed_kinds: Sequence[str] = ACTION_KINDS, max_depth: int = 25,
                   max_nodes: int = 100000, weight: int = 2) -> DCPlan:
    """Weighted A* (f = g + weight*h) with the goal_distance heuristic. Not
    guaranteed shortest; used as a stronger REFERENCE solver to tell
    'unsolvable' apart from 'too hard for bounded BFS'."""
    import heapq
    if goal_met(start):
        return DCPlan((), True, 0, False)
    acts = [a for a in all_actions(len(start.temps)) if a.split(":")[0] in set(allowed_kinds)]
    counter = 0
    heap = [(weight * goal_distance(start), 0, counter, start, ())]
    best_g = {start: 0}
    nodes = 0
    while heap:
        _, g, _, state, path = heapq.heappop(heap)
        if g > best_g.get(state, g):
            continue
        if goal_met(state):
            return DCPlan(path, True, nodes, False)
        if len(path) >= max_depth:
            continue
        nodes += 1
        if nodes > max_nodes:
            return DCPlan((), False, nodes - 1, True)
        for a in acts:
            if precondition(state, a) is not None:
                continue
            nxt = step(state, a)
            ng = g + 1
            if ng < best_g.get(nxt, 1 << 30):
                best_g[nxt] = ng
                counter += 1
                heapq.heappush(heap, (ng + weight * goal_distance(nxt), ng, counter, nxt, path + (a,)))
    return DCPlan((), False, nodes, False)
