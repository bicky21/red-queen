"""M7 self-model: RED QUEEN predicts its own success (metacognitive monitoring).

Pre-registered in docs/PREREG_M7.md. This is a measured *indicator property*
from the 2023 "Consciousness in Artificial Intelligence" report (higher-order
theories: metacognitive monitoring). Having it is NOT a claim of
consciousness. We only measure whether the self-predictions are accurate.

Input: an introspection snapshot from SafeAgent's observer hook. It holds
the agent's own internal state only, never the true state.
Output: P(this episode will succeed).

The self-model only predicts. It never changes what the agent does.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

PLAN_STATUSES = ("none", "too_long", "tight", "fits", "stopping")
ALERT_LEVELS = ("none", "disagree", "widened", "isolated")
WIDTH_BUCKETS = ("<=20", "21-40", ">40")
SMOOTHING = 2.0  # pseudo-counts toward the overall training success rate

Key = Tuple[str, str, str]


def features(snapshot: Mapping[str, Any]) -> Key:
    """Map an introspection snapshot to the three pre-registered discrete features."""
    if snapshot["stopping"]:
        plan = "stopping"
    elif snapshot["plan_len"] is None:
        plan = "none"
    else:
        spare = snapshot["steps_left"] - snapshot["plan_len"]
        plan = "too_long" if spare < 0 else ("tight" if spare <= 2 else "fits")
    counts = snapshot["alert_counts"]
    if counts.get("sensor_isolated", 0) or counts.get("reading_unexplainable", 0):
        alerts = "isolated"
    elif counts.get("bound_widened", 0):
        alerts = "widened"
    elif counts.get("sensors_disagree", 0):
        alerts = "disagree"
    else:
        alerts = "none"
    w = snapshot["width"]
    width = "<=20" if w <= 20 else ("21-40" if w <= 40 else ">40")
    return plan, alerts, width


class SelfModel:
    """Smoothed success frequency per feature combination, fitted from experience."""

    def __init__(self) -> None:
        self.base_rate = 0.5
        self.table: Dict[Key, Tuple[int, int]] = {}  # key -> (successes, count)
        self.fitted = False

    def fit(self, examples: Iterable[Tuple[Key, bool]]) -> "SelfModel":
        counts: Dict[Key, List[int]] = defaultdict(lambda: [0, 0])
        total = succ = 0
        for key, success in examples:
            c = counts[key]
            c[0] += bool(success)
            c[1] += 1
            succ += bool(success)
            total += 1
        if total == 0:
            raise ValueError("cannot fit a self-model on no examples")
        self.base_rate = succ / total
        self.table = {k: (v[0], v[1]) for k, v in counts.items()}
        self.fitted = True
        return self

    def predict(self, key: Key) -> float:
        if not self.fitted:
            raise ValueError("self-model is not fitted")
        s, n = self.table.get(key, (0, 0))
        return (s + SMOOTHING * self.base_rate) / (n + SMOOTHING)


def heuristic(key: Key) -> float:
    """Pre-registered baseline: confident iff a fitting plan exists or the agent is stopping."""
    return 0.95 if key[0] in ("tight", "fits", "stopping") else 0.05


def brier(preds: Sequence[float], labels: Sequence[bool]) -> float:
    return sum((p - float(y)) ** 2 for p, y in zip(preds, labels)) / len(preds)


def ece(preds: Sequence[float], labels: Sequence[bool], bins: int = 10) -> float:
    """Expected calibration error with equal-width bins, weighted by count."""
    groups: Dict[int, List[Tuple[float, bool]]] = defaultdict(list)
    for p, y in zip(preds, labels):
        groups[min(bins - 1, int(p * bins))].append((p, y))
    n = len(preds)
    return sum(len(g) / n * abs(sum(p for p, _ in g) / len(g) - sum(y for _, y in g) / len(g))
               for g in groups.values())


def separation(preds: Sequence[float], labels: Sequence[bool]) -> float:
    pos = [p for p, y in zip(preds, labels) if y]
    neg = [p for p, y in zip(preds, labels) if not y]
    if not pos or not neg:
        return 0.0
    return sum(pos) / len(pos) - sum(neg) / len(neg)
