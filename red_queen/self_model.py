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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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


# M7.3 (pre-registered in docs/PREREG_M7_3.md): "am I making progress?"
PROGRESS_LABELS = ("early", "progressing", "stalled")
PROGRESS_LAG = 2


def _effective_len(plan_len: Optional[int], stopping: bool) -> float:
    if stopping:
        return 0.0
    return float("inf") if plan_len is None else float(plan_len)


def progress_labels(plan_lens: Sequence[Optional[int]], stopping: Sequence[bool]) -> List[str]:
    """Per decision point: early (first 2), progressing (worst-corner plan shorter
    than 2 decision points ago), or stalled. Uses only the agent's own history."""
    eff = [_effective_len(p, s) for p, s in zip(plan_lens, stopping)]
    out = []
    for i, e in enumerate(eff):
        if i < PROGRESS_LAG:
            out.append("early")
        else:
            out.append("progressing" if e < eff[i - PROGRESS_LAG] else "stalled")
    return out


def features_v2(snapshot_key: Key, progress: str) -> Tuple[str, str, str, str]:
    if progress not in PROGRESS_LABELS:
        raise ValueError(f"progress must be one of {PROGRESS_LABELS}")
    return tuple(snapshot_key) + (progress,)


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


class ConfidenceMonitor:
    """M7.1 (pre-registered in docs/PREREG_M7_1.md): an observer for SafeAgent that
    raises ONE 'expect_failure' alert per episode, the first time the self-model's
    confidence drops below `threshold`. Information for a human only: it never
    changes what the agent does. Create a new monitor for each episode."""

    def __init__(self, model: "SelfModel", threshold: float, use_progress: bool = False) -> None:
        if not model.fitted:
            raise ValueError("self-model is not fitted")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be a number in (0, 1)")
        self.model = model
        self.threshold = float(threshold)
        self.use_progress = bool(use_progress)  # M7.3: model must be fitted on features_v2 keys
        self._plan_lens: List[Optional[int]] = []
        self._stopping: List[bool] = []
        self.alerts: List[Dict[str, Any]] = []
        self.confidences: List[float] = []

    def __call__(self, snapshot: Mapping[str, Any]) -> None:
        key = features(snapshot)
        if self.use_progress:
            self._plan_lens.append(snapshot["plan_len"])
            self._stopping.append(bool(snapshot["stopping"]))
            key = features_v2(key, progress_labels(self._plan_lens, self._stopping)[-1])
        p = self.model.predict(key)
        self.confidences.append(p)
        if not self.alerts and p < self.threshold:
            self.alerts.append({"step": int(snapshot["step"]), "kind": "expect_failure",
                                "confidence": round(p, 4), "features": list(key)})

    def report(self) -> Dict[str, Any]:
        return {"threshold": self.threshold, "alerts": [dict(a) for a in self.alerts],
                "min_confidence": round(min(self.confidences), 4) if self.confidences else None}
