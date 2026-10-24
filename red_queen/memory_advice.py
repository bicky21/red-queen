"""M2.1 advisory memory.

`MemoryAdvisor` reads past episodes of the *same* scenario and estimates,
for each (step, action), how often that action had no effect:

    discount(step, action) = blocked attempts / all attempts   (at that step)

The agent's scorer multiplies an action's positive expected gain by
(1 - discount), capped at MAX_MEMORY_DISCOUNT. Memory therefore can only
make an action look less attractive; it can never add, remove or force an
action, and the agent drops the discount once an action is observed to work.

All retrieved data is treated as untrusted:
  * records marked invalid by the store are ignored
  * events with unknown actions, unknown kinds or bad steps are ignored
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

from .actions import ACTIONS
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

        for record in memory.search(f"scenario {scenario}", limit=limit):
            if record.scenario != scenario:
                continue  # lexical match on a different scenario: not relevant
            if not record.valid:
                self.records_ignored_invalid += 1
                continue
            self.records_used += 1
            for event in memory.events(record.episode_id):
                step, kind, name = event.get("step"), event.get("kind"), event.get("name")
                if kind in ("disturbance", "rejected"):
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

    def discounts(self, step: int) -> Mapping[str, float]:
        return {name: d for (s, name), d in sorted(self._table.items()) if s == step and d > 0}

    def summary(self) -> Dict[str, int]:
        return {
            "records_used": self.records_used,
            "records_ignored_invalid": self.records_ignored_invalid,
            "events_ignored": self.events_ignored,
            "nonzero_entries": sum(1 for d in self._table.values() if d > 0),
        }
