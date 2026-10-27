"""M2.1 advisory memory.

`MemoryAdvisor` reads past episodes of the *same* scenario and estimates,
for each (step, action), how often that action had no effect:

    discount(step, action) = blocked attempts / all attempts   (at that step)

The agent's scorer multiplies an action's positive expected gain by
(1 - discount), capped at MAX_MEMORY_DISCOUNT. Memory therefore can only
make an action look less attractive; it can never add, remove or force an
action, and the agent drops the discount once an action is observed to work.

M3.5: `remembered_disturbances()` additionally reports disturbances seen at
a given step in at least half of the used records (known disturbance names
and valid steps only). It does not change `discounts()` or `summary()`.

All retrieved data is treated as untrusted:
  * records marked invalid by the store are ignored
  * events with unknown actions, unknown kinds or bad steps are ignored
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

from .actions import ACTIONS
from .environment import DISTURBANCES
from .memory import MAX_RESULTS, EpisodeMemory
from .scoring import clamp_discount

_COUNTED_KINDS = ("action", "blocked")


class MemoryAdvisor:
    def __init__(self, memory: EpisodeMemory, scenario: str, limit: int = MAX_RESULTS) -> None:
        self.scenario = scenario
        self.records_used = 0
        self.records_ignored_invalid = 0
        self.events_ignored = 0
        attempts: Dict[Tuple[int, str], int] = {}
        blocked: Dict[Tuple[int, str], int] = {}
        seen_disturbances: Dict[Tuple[int, str], int] = {}
        self.disturbance_events_ignored = 0

        for record in memory.search(f"scenario {scenario}", limit=limit):
            if record.scenario != scenario:
                continue  # lexical match on a different scenario: not relevant
            if not record.valid:
                self.records_ignored_invalid += 1
                continue
            self.records_used += 1
            for event in memory.events(record.episode_id):
                step, kind, name = event.get("step"), event.get("kind"), event.get("name")
                if kind == "disturbance":
                    if (name in DISTURBANCES and not isinstance(step, bool)
                            and isinstance(step, int) and step >= 1):
                        seen_disturbances[(step, name)] = seen_disturbances.get((step, name), 0) + 1
                    else:
                        self.disturbance_events_ignored += 1
                    continue
                if kind == "rejected":
                    continue
                if (
                    kind not in _COUNTED_KINDS
                    or name not in ACTIONS
                    or isinstance(step, bool)
                    or not isinstance(step, int)
                    or step < 1
                ):
                    self.events_ignored += 1
                    continue
                key = (step, name)
                attempts[key] = attempts.get(key, 0) + 1
                if kind == "blocked":
                    blocked[key] = blocked.get(key, 0) + 1

        self._table: Dict[Tuple[int, str], float] = {
            key: clamp_discount(blocked.get(key, 0) / count) for key, count in attempts.items()
        }
        # Majority rule per step: at least half of the used records, highest count, then name.
        best: Dict[int, Tuple[int, str]] = {}
        for (step, name), count in sorted(seen_disturbances.items()):
            if 2 * count >= self.records_used and (step not in best or count > best[step][0]):
                best[step] = (count, name)
        self._disturbances: Dict[int, str] = {step: name for step, (_, name) in sorted(best.items())}

    def remembered_disturbances(self) -> Dict[int, str]:
        """M3.5: absolute step -> disturbance name expected after that step's action."""
        return dict(self._disturbances)

    def discounts(self, step: int) -> Mapping[str, float]:
        return {name: d for (s, name), d in sorted(self._table.items()) if s == step and d > 0}

    def summary(self) -> Dict[str, int]:
        return {
            "records_used": self.records_used,
            "records_ignored_invalid": self.records_ignored_invalid,
            "events_ignored": self.events_ignored,
            "nonzero_entries": sum(1 for d in self._table.values() if d > 0),
        }
