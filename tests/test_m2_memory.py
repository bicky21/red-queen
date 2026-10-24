import contextlib
import io
import sqlite3
import tempfile
import unittest
from pathlib import Path

from red_queen.affect import AffectConfig
from red_queen.experiments import run_scenario
from red_queen.memory import (
    MAX_QUERY_CHARS,
    MAX_RESULTS,
    EpisodeMemory,
    MemoryStoreError,
    summarize_episode,
    tokenize,
)
from red_queen.memory_cli import main as cli_main
from red_queen.memory_cli import record_all
from red_queen.scenarios import SCENARIOS, get_scenario
from red_queen.validation import M1_1_FROZEN_SEQUENCES


def _episode(name="flaky_restart", enabled=True):
    config = AffectConfig(enabled=enabled)
    return run_scenario(get_scenario(name), enabled, affect=config), config


class TestHelpers(unittest.TestCase):
    def test_tokenize_splits_underscores_and_case(self):
        self.assertEqual(tokenize("Restart_Service, BLOCKED!"), ["restart", "service", "blocked"])

    def test_summary_contains_only_episode_facts(self):
        result, _ = _episode()
        s = summarize_episode(result)
        for part in ("scenario flaky_restart", "affect on", "success", "blocked restart_service"):
            self.assertIn(part, s)
        self.assertNotIn("disturbance", s)


class TestStore(unittest.TestCase):
    def setUp(self):
        self.memory = EpisodeMemory(":memory:")

    def tearDown(self):
        self.memory.close()

    def test_record_and_get(self):
        result, config = _episode()
        rid = self.memory.record(result, config)
        rec = self.memory.get(rid)
        self.assertTrue(rec.valid)
        self.assertEqual(rec.actions, result.actions)
        self.assertEqual(rec.config["frustration_penalty"], config.frustration_penalty)
        self.assertEqual(rec.source, "memory")

    def test_events_roundtrip(self):
        result, config = _episode("disturbance_mid_recovery", False)
        rid = self.memory.record(result, config)
        self.assertEqual(self.memory.events(rid), list(result.events))

    def test_missing_id(self):
        self.assertIsNone(self.memory.get(999))

    def test_records_are_immutable(self):
        result, config = _episode()
        rec = self.memory.get(self.memory.record(result, config))
        with self.assertRaises(Exception):
            rec.success = False

    def test_search_ranking_and_ties_newest_first(self):
        record_all(self.memory)
        top = self.memory.search("restart blocked", limit=3)
        self.assertEqual([r.score for r in top], [2, 2, 2])
        self.assertEqual([r.episode_id for r in top], sorted([r.episode_id for r in top], reverse=True))

    def test_search_no_terms_or_no_match(self):
        record_all(self.memory)
        self.assertEqual(self.memory.search("!!!"), [])
        self.assertEqual(self.memory.search("zebra"), [])

    def test_search_limits(self):
        with self.assertRaises(ValueError):
            self.memory.search("a" * (MAX_QUERY_CHARS + 1))
        with self.assertRaises(ValueError):
            self.memory.search("x", limit=MAX_RESULTS + 1)
        with self.assertRaises(TypeError):
            self.memory.search(None)

    def test_safety_injection_strings_are_data(self):
        record_all(self.memory)
        n = self.memory.count()
        self.memory.search("x'; DROP TABLE episodes;--")
        self.assertEqual(self.memory.count(), n)

    def test_safety_tampered_actions_flagged(self):
        result, config = _episode()
        rid = self.memory.record(result, config)
        self.memory._conn.execute("UPDATE episodes SET actions_json = ? WHERE id = ?", ('["shell"]', rid))
        rec = self.memory.get(rid)
        self.assertFalse(rec.valid)
        self.assertTrue(any("non-allowlisted" in p for p in rec.problems))

    def test_safety_mismatched_count_flagged(self):
        result, config = _episode()
        rid = self.memory.record(result, config)
        self.memory._conn.execute("UPDATE episodes SET action_count = 99 WHERE id = ?", (rid,))
        self.assertFalse(self.memory.get(rid).valid)


class TestCaps(unittest.TestCase):
    def test_prunes_oldest_and_cascades_events(self):
        with EpisodeMemory(":memory:", max_episodes=3) as memory:
            record_all(memory)
            self.assertEqual(memory.count(), 3)
            kept = {r.episode_id for r in memory.search("scenario", limit=20)}
            total = len(SCENARIOS) * 2
            self.assertEqual(kept, {total, total - 1, total - 2})
            self.assertEqual(memory.event_count(), sum(memory.event_count(i) for i in kept))

    def test_event_truncation(self):
        with EpisodeMemory(":memory:", max_events_per_episode=1) as memory:
            result, config = _episode("compound_failure", False)
            rid = memory.record(result, config)
            self.assertEqual(memory.event_count(rid), 1)
            row = memory._conn.execute("SELECT events_truncated FROM episodes WHERE id = ?", (rid,)).fetchone()
            self.assertEqual(row[0], 1)

    def test_invalid_caps(self):
        with self.assertRaises(ValueError):
            EpisodeMemory(":memory:", max_episodes=0)


class TestFiles(unittest.TestCase):
    def test_persistence_and_schema_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "sub" / "m.sqlite"  # parent folder is created
            with EpisodeMemory(db) as m:
                record_all(m)
            with EpisodeMemory(db) as m:
                self.assertEqual(m.count(), len(SCENARIOS) * 2)
                self.assertTrue(m.integrity_ok())
            conn = sqlite3.connect(db)
            with conn:
                conn.execute("UPDATE meta SET value = '2' WHERE key = 'schema_version'")
            conn.close()
            with self.assertRaises(MemoryStoreError):
                EpisodeMemory(db)

    def test_corrupt_file_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            junk = Path(tmp) / "junk.sqlite"
            junk.write_bytes(b"not a database" * 200)
            with self.assertRaises(MemoryStoreError):
                EpisodeMemory(junk)

    def test_cli_record_search_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "m.sqlite")
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                self.assertEqual(cli_main(["--db", db, "record-all"]), 0)
                self.assertEqual(cli_main(["--db", db, "search", "charge blocked"]), 0)
                self.assertEqual(cli_main(["--db", db, "stats"]), 0)
                self.assertEqual(cli_main(["--db", db, "search", "x", "--limit", "0"]), 2)
            self.assertIn("Recorded 20 episodes", out.getvalue())
            self.assertIn("integrity_ok=True", out.getvalue())
            self.assertIn("limit must be between", err.getvalue())


class TestBehaviourUnchanged(unittest.TestCase):
    def test_agent_unaffected_by_memory(self):
        with EpisodeMemory(":memory:") as memory:
            record_all(memory)
            for s in SCENARIOS:
                with self.subTest(scenario=s.name):
                    self.assertEqual(run_scenario(s, True).to_dict(), run_scenario(s, True).to_dict())

    def test_m1_1_sequences_frozen(self):
        for name, expected in M1_1_FROZEN_SEQUENCES.items():
            for enabled in (False, True):
                with self.subTest(scenario=name, affect=enabled):
                    self.assertEqual(run_scenario(get_scenario(name), enabled).actions, expected)


if __name__ == "__main__":
    unittest.main()
