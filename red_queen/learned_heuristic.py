"""M9.2 learned heuristic (pre-registered in docs/PREREG_M9_2.md).

RED QUEEN fits a "distance to goal" estimate for the data-centre world from
its own past episodes: linear ridge regression (lambda 0.01) on 8 fixed
features, solved exactly with the standard library (no randomness).

Safety: the heuristic only changes the ORDER in which search looks at states.
The executor, preconditions, budget and verifier are unchanged. Predictions
are clipped to >= 0 and any non-finite value falls back to the hand-written
goal_distance, so a corrupt model cannot crash or steer the system unsafely.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

from .datacentre import DCState, all_actions, goal_distance, goal_met, precondition, reference_plan, step

FEATURE_NAMES = ("bias", "down_servers", "down_heat_over_60", "up_heat_over_70", "load_over_40",
                 "unserved", "max_up_temp", "heating_servers")
RIDGE = 0.01
WALK_STEPS = 5


def features(s: DCState) -> List[float]:
    up_temps = [t for t, u in zip(s.temps, s.up) if u]
    return [
        1.0,
        float(sum(not u for u in s.up)),
        sum(max(0, t - 60) for t, u in zip(s.temps, s.up) if not u) / 5.0,
        sum(max(0, t - 70) for t in up_temps) / 5.0,
        sum(max(0, ld - 40) for ld in s.loads) / 20.0,
        abs(s.demand - s.served) / 20.0,
        (max(up_temps) / 10.0) if up_temps else 0.0,
        float(sum(1 for ld, u in zip(s.loads, s.up) if u and ld >= 60)),
    ]


def _solve(a: List[List[float]], b: List[float]) -> List[float]:
    """Gaussian elimination with partial pivoting (deterministic)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) < 1e-12:
            raise ValueError("singular system")
        m[c], m[p] = m[p], m[c]
        for r in range(c + 1, n):
            f = m[r][c] / m[c][c]
            for k in range(c, n + 1):
                m[r][k] -= f * m[c][k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (m[r][n] - sum(m[r][k] * x[k] for k in range(r + 1, n))) / m[r][r]
    return x


def fit_ridge(xs: Sequence[Sequence[float]], ys: Sequence[float], ridge: float = RIDGE) -> List[float]:
    if not xs or len(xs) != len(ys):
        raise ValueError("need matching, non-empty training data")
    d = len(xs[0])
    a = [[sum(x[i] * x[j] for x in xs) + (ridge if i == j and i > 0 else 0.0) for j in range(d)] for i in range(d)]
    b = [sum(x[i] * y for x, y in zip(xs, ys)) for i in range(d)]
    return _solve(a, b)


class LearnedHeuristic:
    """Callable heuristic: max(0, w . features(state)); non-finite -> hand heuristic."""

    def __init__(self, weights: Sequence[float]) -> None:
        if len(weights) != len(FEATURE_NAMES):
            raise ValueError(f"expected {len(FEATURE_NAMES)} weights")
        self.weights = [float(w) for w in weights]
        self.fallbacks = 0

    def __call__(self, s: DCState) -> float:
        try:
            v = sum(w * f for w, f in zip(self.weights, features(s)))
        except (TypeError, OverflowError):
            v = float("nan")
        if not math.isfinite(v):
            self.fallbacks += 1
            return float(goal_distance(s))
        return max(0.0, v)

    def to_dict(self) -> Dict[str, float]:
        return {n: round(w, 6) for n, w in zip(FEATURE_NAMES, self.weights)}


def walk_states(start: DCState, seed: int, steps: int = WALK_STEPS) -> List[DCState]:
    """The start plus states along a deterministic random walk of valid actions."""
    r = random.Random(f"walk:{seed}")
    out = [start]
    s = start
    for _ in range(steps):
        valid = [a for a in all_actions(len(s.temps)) if precondition(s, a) is None]
        if not valid:
            break
        s = step(s, r.choice(valid))
        out.append(s)
    return out


def labelled_states(seeds: Sequence[int], make_scenario) -> List[Tuple[DCState, int]]:
    """(state, reference-plan length) pairs; states with no plan found are dropped."""
    data = []
    for seed in seeds:
        for s in walk_states(make_scenario(seed), seed):
            if goal_met(s):
                data.append((s, 0))
                continue
            plan = reference_plan(s)
            if plan.found:
                data.append((s, len(plan.actions)))
    return data


def train(seeds: Sequence[int], make_scenario) -> Tuple[LearnedHeuristic, int]:
    data = labelled_states(seeds, make_scenario)
    weights = fit_ridge([features(s) for s, _ in data], [float(y) for _, y in data])
    return LearnedHeuristic(weights), len(data)


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)
