"""Command line for M2 memory.

  python3 -m red_queen.memory_cli record-all [--db PATH]
  python3 -m red_queen.memory_cli search "restart blocked" [--db PATH] [--limit N]
  python3 -m red_queen.memory_cli stats [--db PATH]

Default database: data/red_queen_memory.sqlite (ignored by git).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .affect import AffectConfig
from .experiments import run_scenario
from .memory import EpisodeMemory, MemoryStoreError
from .scenarios import SCENARIOS

DEFAULT_DB = Path("data") / "red_queen_memory.sqlite"


def record_all(memory: EpisodeMemory) -> List[int]:
    """Run every scenario in both affect modes and store the episodes."""
    ids = []
    for scenario in SCENARIOS:
        for enabled in (False, True):
            config = AffectConfig(enabled=enabled)
            result = run_scenario(scenario, enabled, affect=config)
            ids.append(memory.record(result, config))
    return ids


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="RED QUEEN episodic memory")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("record-all")
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--limit", type=int, default=5)
    sub.add_parser("stats")
    args = parser.parse_args(argv)

    try:
        with EpisodeMemory(args.db) as memory:
            if args.command == "record-all":
                ids = record_all(memory)
                print(f"Recorded {len(ids)} episodes in {args.db} (total {memory.count()})")
            elif args.command == "search":
                results = memory.search(args.query, limit=args.limit)
                if not results:
                    print("No matching episodes.")
                for r in results:
                    flag = "" if r.valid else f"  [INVALID: {'; '.join(r.problems)}]"
                    print(f"#{r.episode_id} score={r.score} {r.summary}{flag}")
            else:
                print(f"db={args.db} episodes={memory.count()} events={memory.event_count()}"
                      f" integrity_ok={memory.integrity_ok()}")
    except (MemoryStoreError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
