"""M2 persistent episodic memory (SQLite, standard library only).

What it does:
  * records finished episodes and their environment events
  * retrieves past episodes by simple lexical (keyword) matching

What it does NOT do (by design, in M2):
  * it never influences the agent's decisions
  * it is never treated as authoritative state

Safeguards:
  * parameterised SQL only (no string-built queries)
  * schema version stored in the file; a mismatch is refused
  * caps on stored episodes (oldest pruned), events per episode,
    query length and number of results
  * every retrieved record is re-validated against the action allowlist
    and marked `valid=False` if it was tampered with or is malformed
  * retrieved records are frozen copies marked `source="memory"`
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .actions import ACTIONS
from .affect import AffectConfig
from .agent import EpisodeResult

SCHEMA_VERSION = 1
DEFAULT_MAX_EPISODES = 1000
DEFAULT_MAX_EVENTS_PER_EPISODE = 500
MAX_QUERY_CHARS = 200
MAX_RESULTS = 20
_TOKEN_RE = re.compile(r"[a-z0-9]+")  # "restart_service" -> "restart", "service"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS episodes (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    scenario         TEXT    NOT NULL,
    affect_enabled   INTEGER NOT NULL,
    config_json      TEXT    NOT NULL,
    success          INTEGER NOT NULL,
    stop_reason      TEXT    NOT NULL,
    action_count     INTEGER NOT NULL,
    actions_json     TEXT    NOT NULL,
    final_state_json TEXT    NOT NULL,
    events_truncated INTEGER NOT NULL,
    summary          TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id  INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    step        INTEGER NOT NULL,
    kind        TEXT    NOT NULL,
    name        TEXT    NOT NULL,
    before_json TEXT    NOT NULL,
    after_json  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_episode ON events(episode_id);
"""


class MemoryStoreError(RuntimeError):
    """Raised for unusable memory files (corrupt, wrong schema version)."""


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def summarize_episode(result: EpisodeResult) -> str:
    """Searchable text built only from facts in the episode record."""
    parts = [
        f"scenario {result.scenario}",
        f"affect {'on' if result.affect_enabled else 'off'}",
        "success" if result.success else "failure",
        f"stop {result.stop_reason}",
        "actions " + " ".join(result.actions) if result.actions else "actions none",
    ]
    for kind in ("blocked", "disturbance", "rejected"):
        names = sorted({e["name"] for e in result.events if e["kind"] == kind})
        if names:
            parts.append(f"{kind} " + " ".join(names))
    return " | ".join(parts)


@dataclass(frozen=True)
class MemoryRecord:
    """A retrieved episode. Untrusted context, never authoritative state."""

    episode_id: int
    scenario: str
    affect_enabled: bool
    config: Dict[str, Any]
    success: bool
    stop_reason: str
    actions: Tuple[str, ...]
    final_state: Dict[str, Any]
    summary: str
    valid: bool
    problems: Tuple[str, ...]
    score: int = 0
    source: str = "memory"


class EpisodeMemory:
    """SQLite-backed store. Use ':memory:' for a throwaway in-RAM store."""

    def __init__(
        self,
        path: Union[str, Path],
        max_episodes: int = DEFAULT_MAX_EPISODES,
        max_events_per_episode: int = DEFAULT_MAX_EVENTS_PER_EPISODE,
    ) -> None:
        if max_episodes < 1 or max_events_per_episode < 0:
            raise ValueError("max_episodes must be >= 1 and max_events_per_episode >= 0")
        self.path = str(path)
        self.max_episodes = max_episodes
        self.max_events_per_episode = max_events_per_episode
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(self.path)
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.executescript(_SCHEMA)
            self._check_schema_version()
        except sqlite3.DatabaseError as exc:
            raise MemoryStoreError(f"Cannot open memory store {self.path!r}: {exc}") from exc

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "EpisodeMemory":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _check_schema_version(self) -> None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        if row is None:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO meta (key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
                )
        elif row[0] != str(SCHEMA_VERSION):
            self._conn.close()
            raise MemoryStoreError(f"Schema version {row[0]!r} is not supported (expected {SCHEMA_VERSION})")

    # -- writing -----------------------------------------------------------

    def record(self, result: EpisodeResult, config: AffectConfig) -> int:
        """Store one finished episode in a single transaction. Returns its id."""
        events = list(result.events)
        truncated = len(events) > self.max_events_per_episode
        events = events[: self.max_events_per_episode]
        config_dict = {
            "enabled": config.enabled,
            "concern_gain": config.concern_gain,
            "urgency_cost_relief": config.urgency_cost_relief,
            "frustration_penalty": config.frustration_penalty,
        }
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO episodes (scenario, affect_enabled, config_json, success, stop_reason,"
                " action_count, actions_json, final_state_json, events_truncated, summary)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    result.scenario,
                    int(result.affect_enabled),
                    json.dumps(config_dict, sort_keys=True),
                    int(result.success),
                    result.stop_reason,
                    result.action_count,
                    json.dumps(list(result.actions)),
                    json.dumps(result.final_state.to_dict(), sort_keys=True),
                    int(truncated),
                    summarize_episode(result),
                ),
            )
            episode_id = int(cur.lastrowid)
            self._conn.executemany(
                "INSERT INTO events (episode_id, step, kind, name, before_json, after_json)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        episode_id,
                        e["step"],
                        e["kind"],
                        e["name"],
                        json.dumps(e["before"], sort_keys=True),
                        json.dumps(e["after"], sort_keys=True),
                    )
                    for e in events
                ],
            )
            self._prune()
        return episode_id

    def _prune(self) -> None:
        """Delete the oldest episodes beyond the cap (events cascade)."""
        self._conn.execute(
            "DELETE FROM episodes WHERE id NOT IN (SELECT id FROM episodes ORDER BY id DESC LIMIT ?)",
            (self.max_episodes,),
        )

    # -- reading -----------------------------------------------------------

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0])

    def event_count(self, episode_id: Optional[int] = None) -> int:
        if episode_id is None:
            return int(self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])
        return int(
            self._conn.execute("SELECT COUNT(*) FROM events WHERE episode_id = ?", (episode_id,)).fetchone()[0]
        )

    def get(self, episode_id: int) -> Optional[MemoryRecord]:
        row = self._conn.execute(_SELECT + " WHERE id = ?", (episode_id,)).fetchone()
        return _to_record(row) if row else None

    def events(self, episode_id: int) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT step, kind, name, before_json, after_json FROM events WHERE episode_id = ? ORDER BY id",
            (episode_id,),
        ).fetchall()
        return [
            {"step": s, "kind": k, "name": n, "before": json.loads(b), "after": json.loads(a)}
            for s, k, n, b, a in rows
        ]

    def search(self, query: str, limit: int = 5) -> List[MemoryRecord]:
        """Lexical retrieval: rank episodes by how many distinct query terms
        appear in their summary. Ties go to the newest episode. Deterministic."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if len(query) > MAX_QUERY_CHARS:
            raise ValueError(f"query longer than {MAX_QUERY_CHARS} characters")
        if not 1 <= limit <= MAX_RESULTS:
            raise ValueError(f"limit must be between 1 and {MAX_RESULTS}")
        terms = set(tokenize(query))
        if not terms:
            return []
        scored = []
        for row in self._conn.execute(_SELECT + " ORDER BY id DESC"):
            score = len(terms & set(tokenize(row[9])))
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda item: (-item[0], -item[1][0]))
        return [_to_record(row, score) for score, row in scored[:limit]]

    def integrity_ok(self) -> bool:
        row = self._conn.execute("PRAGMA integrity_check").fetchone()
        return row is not None and row[0] == "ok"


_SELECT = (
    "SELECT id, scenario, affect_enabled, config_json, success, stop_reason,"
    " action_count, actions_json, final_state_json, summary FROM episodes"
)


def _to_record(row: Sequence[Any], score: int = 0) -> MemoryRecord:
    """Rebuild a record and re-validate it; tampering marks it invalid."""
    (episode_id, scenario, affect_enabled, config_json, success, stop_reason,
     action_count, actions_json, final_state_json, summary) = row
    problems: List[str] = []

    def load(text: str, kind: type, label: str) -> Any:
        try:
            value = json.loads(text)
        except (TypeError, ValueError):
            problems.append(f"{label} is not valid JSON")
            return kind()
        if not isinstance(value, kind):
            problems.append(f"{label} has wrong type")
            return kind()
        return value

    actions = load(actions_json, list, "actions")
    config = load(config_json, dict, "config")
    final_state = load(final_state_json, dict, "final_state")
    bad = [a for a in actions if not isinstance(a, str) or a not in ACTIONS]
    if bad:
        problems.append(f"non-allowlisted actions: {bad!r}")
    if len(actions) != action_count:
        problems.append("action_count does not match actions")
    return MemoryRecord(
        episode_id=int(episode_id),
        scenario=str(scenario),
        affect_enabled=bool(affect_enabled),
        config=config,
        success=bool(success),
        stop_reason=str(stop_reason),
        actions=tuple(str(a) for a in actions),
        final_state=final_state,
        summary=str(summary),
        valid=not problems,
        problems=tuple(problems),
        score=score,
    )
