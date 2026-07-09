"""The flywheel SELF-UPDATES on ordinary build+verify (not only under explicit training / a manual GPU night run).

Diagnosed by driving the product: a freshly-built quad produced a CREDIBLE WALK, but the flywheel banked NOTHING
(the persistent bank sat at 0 gaits despite many builds) because only explicit training or the GPU night CLI ever
wrote to it. The bank<->recall mechanism itself was sound; the trigger was missing. verify_robot now auto-banks a
credible walk (keep-best, keyed by morphology), so the flywheel COMPOUNDS on ordinary use, SELF-HEALS if its
memory is wiped, and RECALLS the banked gait as the warm-start hint for the next similar body.
"""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("MUJOCO_GL", "glfw")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "the flywheel loop needs the MuJoCo gait sim")
class FlywheelAutoBankTests(unittest.TestCase):
    def setUp(self):
        # isolate the flywheel to a throwaway DB so the test never touches the real bank
        self._dir = tempfile.mkdtemp()
        self._db = Path(self._dir) / "fw.db"
        self._patch = mock.patch("virturoid.services.memory_db.DEFAULT_DB_PATH", self._db)
        self._patch.start()
        from virturoid.services import session_state as S
        S.reset()

    def tearDown(self):
        self._patch.stop()

    def _gaits(self):
        from virturoid.services.memory_db import MemoryDB
        if not self._db.exists():
            return 0
        with MemoryDB(self._db) as db:
            return db.conn.execute("SELECT COUNT(*) c FROM skills WHERE task_type='locomotion'").fetchone()["c"]

    def _build_and_verify(self, prompt="a quadruped robot dog"):
        from virturoid.services.ai_native_tools import create_robot, verify_robot
        rid = create_robot({"prompt": prompt})["robot_id"]
        return verify_robot({"robot_id": rid, "mode": "quick"})

    def test_ordinary_verify_auto_banks_a_credible_walk(self):
        self.assertEqual(self._gaits(), 0, "starts empty")
        v = self._build_and_verify()
        self.assertTrue(v["verdict"].startswith("CREDIBLE"), f"the taught quad must walk credibly: {v}")
        self.assertTrue(v.get("flywheel_banked"), f"a credible walk must AUTO-BANK on ordinary verify: {v}")
        self.assertEqual(self._gaits(), 1, "the flywheel self-updated on build+verify (was the whole bug)")

    def test_flywheel_self_heals_after_a_memory_wipe(self):
        self._build_and_verify()
        self.assertEqual(self._gaits(), 1)
        from virturoid.services.memory_db import MemoryDB
        with MemoryDB(self._db) as db:                          # wipe the "obvious" banked gait
            db.conn.execute("DELETE FROM skills WHERE task_type='locomotion'")
            db.conn.commit()
        self.assertEqual(self._gaits(), 0, "wiped")
        self._build_and_verify()                               # rebuild -> must re-bank on its own
        self.assertEqual(self._gaits(), 1, "the flywheel re-banked itself after the wipe (self-heal)")

    def test_next_body_recalls_the_banked_gait_as_the_hint(self):
        self._build_and_verify()                               # bank a quad gait
        v2 = self._build_and_verify()                          # a 2nd same-morphology body
        self.assertEqual(v2.get("gait_source"), "learned_flywheel",
                         f"the 2nd body must RECALL the banked gait (the warm-start hint): {v2}")


if __name__ == "__main__":
    unittest.main()
