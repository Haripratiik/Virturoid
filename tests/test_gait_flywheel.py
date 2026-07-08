"""Gait flywheel COMPOUNDS: a learned gait is banked as a specific reusable skill, recalled for the same body,
and warm-starting from it reaches the walk instantly (search cost -> ~0), with a provenance edge recorded.

This is the honest answer to "does the flywheel actually work": banked -> recalled (specific) -> reused (measured).
MuJoCo-gated (physics is the judge). AGENTS.md offline.
"""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "needs MuJoCo — physics is the judge")
class GaitFlywheelTests(unittest.TestCase):
    def _db(self):
        from virturoid.services.memory_db import MemoryDB
        return MemoryDB(db_path=Path(tempfile.mkdtemp(prefix="gaitfw_")) / "m.db")

    def _quad(self):
        from virturoid.services.anatomy_compiler import ensure_walkable_quad
        from virturoid.services.morphology_composer import compose_robot
        p = "a quadruped robot dog"
        return ensure_walkable_quad(compose_robot(p), p)

    def test_bank_only_deployable_and_recall_specific(self):
        from virturoid.services.gait_flywheel import bank_gait, recall_gait
        db, g = self._db(), self._quad()

        class _Fall:  # a fall must NOT be banked (the bank stays a bank of WORKING controllers)
            best_survived, best_forward, best_height_ratio, best_params = False, -0.1, 0.4, {"freq": 1.0}
        self.assertIsNone(bank_gait(db, g, _Fall()))
        self.assertIsNone(recall_gait(db, g))                  # nothing banked yet

        class _Walk:
            best_survived, best_height_ratio = True, 0.85
            best_forward = 1.2
            best_params = {"freq": 1.4, "hip_amp": 1.1, "knee_amp": 0.8, "duty": 0.4, "kp": 200.0, "kd": 7.0}
        sid = bank_gait(db, g, _Walk())
        self.assertTrue(sid and sid.startswith("gait::quadruped::"))   # SPECIFIC key: class + body
        recalled = recall_gait(db, g)                          # retrieve the actual controller params back
        self.assertIsNotNone(recalled)
        self.assertAlmostEqual(recalled["kp"], 200.0, places=3)

    def test_flywheel_compounds_on_reuse(self):
        from virturoid.services.gait_flywheel import learn_gait_flywheel
        from virturoid.services.gait_search import search_gait
        db, g = self._db(), self._quad()
        budget = dict(generations=3, pop=8, steps=600)

        cold = search_gait(g, seed=1, **budget)               # first solve, cold (nothing banked)
        self.assertTrue(cold.best_survived)
        # bank it through the flywheel
        first = learn_gait_flywheel(g, db, seed=1, **budget)
        self.assertTrue(first["banked_skill"])
        self.assertFalse(first["reused_prior"])               # first time: nothing to reuse

        second = learn_gait_flywheel(g, db, seed=2, **budget)  # SAME body again -> must reuse the bank
        self.assertTrue(second["reused_prior"])
        # the recalled prior transfers to (its own) body ~ as well as the banked walk -> instant, no re-search cost.
        self.assertGreaterEqual(second["prior_transfer_forward"], 0.9 * abs(cold.best_forward))
        self.assertTrue(second["survived"])

        # provenance edge recorded -> compounding is MEASURED, not asserted.
        from virturoid.services.robotics_vector_memory import RoboticsVectorMemory
        cs = RoboticsVectorMemory(db).compounding_summary()
        self.assertGreaterEqual(cs.get("edges", 0), 1)


if __name__ == "__main__":
    unittest.main()
