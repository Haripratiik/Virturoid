"""P2 — resurrect the DEAD lessons store (0 rows in the live DB because the reasoned-redesign writer skips the
common LOCOMOTION path). The deploy-select disagreement (`ai_native_tools._record_gait_lesson`) is the honest
physics-verified failure->fix source; this pins the write->recall->surface loop the plan requires: a lesson
written for a class is RECALLED into that class's mined gait hints (grounding the next similar body)."""
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")


def _db():
    from virturoid.services.memory_db import MemoryDB
    return MemoryDB(db_path=Path(tempfile.mkdtemp(prefix="lesson_")) / "m.db")


class LessonsFlywheelTests(unittest.TestCase):
    def test_lesson_write_is_recalled_into_gait_hints(self):
        from virturoid.services.gait_hints import mine_gait_hints
        db = _db()
        # cold: no lessons -> no structural hints
        cold = mine_gait_hints(db, robot_class="quadruped")
        self.assertFalse(any(h.get("kind") == "structural" for h in cold["hints"]))
        # a verified failure->fix lesson lands (as the deploy-select would write it)
        db.record_lesson("quadruped", "banked_gait_underperformed", "deploy_default_crawl",
                         task_type="locomotion", improvement=0.42,
                         root_cause="recalled hint slid; the default crawl walked")
        # recall: the lesson surfaces as a structural hint for the SAME class (grounds the next similar body)
        h = mine_gait_hints(db, robot_class="quadruped")
        struct = [x for x in h["hints"] if x.get("kind") == "structural"]
        self.assertTrue(struct)
        self.assertEqual(struct[0]["failure"], "banked_gait_underperformed")
        self.assertEqual(struct[0]["fix"], "deploy_default_crawl")

    def test_record_lesson_is_idempotent_keep_best(self):
        db = _db()
        db.record_lesson("hexapod", "roll_over", "hip_abduction", improvement=0.3)
        db.record_lesson("hexapod", "roll_over", "hip_abduction", improvement=0.5)   # stronger later fix
        db.record_lesson("hexapod", "roll_over", "hip_abduction", improvement=0.1)   # weaker -> ignored
        rows = db.lessons_for_class("hexapod")
        self.assertEqual(len(rows), 1)                            # one row, not three
        self.assertEqual(rows[0]["improvement"], 0.5)             # keep-best
        self.assertEqual(rows[0]["times_applied"], 3)             # but counts every application


if __name__ == "__main__":
    unittest.main()
