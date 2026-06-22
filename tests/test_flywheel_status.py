"""Moat status (Theme 2): the cumulative compounding view over persistent memory + the skill ledger."""

import tempfile
import unittest
from pathlib import Path

from virturoid.services.flywheel_status import moat_status
from virturoid.services.memory_db import MemoryDB
from virturoid.services.skill_flywheel import append_manifest


class MoatStatusTests(unittest.TestCase):
    def test_empty_memory_reads_as_honest_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            st = moat_status(Path(tmp))
            self.assertEqual(0, st["counts"]["runs"])
            self.assertEqual([], st["skills"])
            self.assertFalse(st["compounding"])
            self.assertIn("No skills banked yet", st["summary"])

    def test_aggregates_runs_designs_and_manifest_warm_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = Path(tmp) / "memory"
            mem.mkdir()
            with MemoryDB(mem / "virturoid_memory.db") as db:
                db.record_run("p1", "quadruped", "locomotion", {"segments": 18}, 0.0,
                              species="quadruped.composed", succeeded=False, design_source="gene_compiler")
                db.record_run("p2", "legged", "locomotion", {"segments": 24}, 0.5,
                              species="legged6.composed", succeeded=False, design_source="gene_compiler")
            # Two trainable builds banked the same transferable legged policy; the 2nd warm-started.
            append_manifest(mem, {"skill_id": "morph_legged_locomotion", "success_rate": 0.17,
                                  "trainer": "morph_es", "task_type": "locomotion", "warm_started": False})
            append_manifest(mem, {"skill_id": "morph_legged_locomotion", "success_rate": 0.21,
                                  "trainer": "morph_es", "task_type": "locomotion", "warm_started": True})

            st = moat_status(mem)
            self.assertEqual(2, st["counts"]["runs"])
            self.assertEqual(2, st["counts"]["designs"])
            self.assertEqual(1, len(st["skills"]))                       # one distinct skill, 2 builds
            self.assertEqual(2, st["skills"][0]["builds"])
            self.assertAlmostEqual(0.21, st["skills"][0]["best_success"])  # keeps the best
            self.assertEqual(1, st["skills"][0]["warm_started"])
            self.assertEqual(1, st["warm_start"]["warm_started_builds"])
            self.assertEqual(2, st["warm_start"]["trainable_builds"])
            self.assertAlmostEqual(0.5, st["warm_start"]["utilization"])
            self.assertTrue(st["compounding"])

    def test_corrupt_manifest_line_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = Path(tmp) / "memory"
            mem.mkdir()
            (mem / "flywheel_manifest.jsonl").write_text(
                '{"skill_id": "a", "success_rate": 0.3, "warm_started": false}\nNOT JSON\n', encoding="utf-8")
            st = moat_status(mem)
            self.assertEqual(1, len(st["skills"]))   # the valid row survives; the garbage line is ignored


if __name__ == "__main__":
    unittest.main()
