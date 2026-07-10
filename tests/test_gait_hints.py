"""The flywheel-as-HINTS (not copy-paste): hints are auto-mined from banked credible walks (data-derived, no
hardcoding), and warm-start a fresh per-body adaptation — so two different bodies get two different fitted gaits
from the same hints. The moat is transferable principles + adaptation, never a pasted policy.
"""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("MUJOCO_GL", "glfw")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


def _bank(db, cls, gene_id, params, sr):
    from virturoid.services.gait_flywheel import LOCOMOTION
    db.record_skill(f"gait::{cls}::{gene_id}", cls, LOCOMOTION, success_rate=sr,
                    base_config={"gait_params": params, "controller": "crawl_gait"})


class GaitHintsMiningTests(unittest.TestCase):
    def _db(self):
        from virturoid.services.memory_db import MemoryDB
        return MemoryDB(db_path=Path(tempfile.mkdtemp(prefix="hints_")) / "m.db")

    def test_hints_are_mined_from_data_not_hardcoded(self):
        from virturoid.services.gait_hints import mine_gait_hints
        db = self._db()
        # thin corpus -> honest "not enough data" (the default prior, no invented region)
        cold = mine_gait_hints(db, "quadruped")
        self.assertEqual(cold["n"], 0)
        self.assertIn("not enough", cold["note"])
        # bank a few CREDIBLE walks whose freq clusters ~2.0 (deliberately NOT the 1.5 default) -> the mined region
        # must follow the DATA to ~2.0, proving it isn't a hardcoded constant
        for i, f in enumerate((1.9, 2.0, 2.1)):
            _bank(db, "quadruped", f"b{i}", {"freq": f, "hip_amp": 0.8, "knee_amp": 1.1, "duty": 0.3,
                                             "kp": 30.0, "kd": 1.4}, 0.9)
        h = mine_gait_hints(db, "quadruped")
        self.assertEqual(h["n"], 3)
        self.assertAlmostEqual(h["prior"]["freq"], 2.0, delta=0.15)   # region tracked the data, not the 1.5 default
        self.assertTrue(any(hint.get("kind") == "param_region" for hint in h["hints"]))
        # the relational hint is DISCOVERED (all 3 have knee_amp > hip_amp)
        self.assertTrue(any(hint.get("kind") == "relation" for hint in h["hints"]))


@unittest.skipUnless(_MUJOCO, "adaptation runs a short gait search in MuJoCo")
class GaitHintsAdaptationTests(unittest.TestCase):
    def test_two_bodies_get_two_gaits_from_the_same_hints(self):
        from virturoid.services.gait_hints import adapt_gait_from_hints
        from virturoid.services.memory_db import MemoryDB
        from virturoid.services.morphology_composer import compose_robot
        db = MemoryDB(db_path=Path(tempfile.mkdtemp(prefix="hints2_")) / "m.db")
        for i, f in enumerate((1.4, 1.5, 1.6)):
            _bank(db, "quadruped", f"b{i}", {"freq": f, "hip_amp": 0.9, "knee_amp": 1.0, "duty": 0.25,
                                             "kp": 32.0, "kd": 1.5}, 0.9)
        a = adapt_gait_from_hints(compose_robot("a quadruped robot dog that walks"), db,
                                  generations=3, pop=6, steps=400, deploy_steps=600)
        b = adapt_gait_from_hints(compose_robot("a large quadruped robot"), db,
                                  generations=3, pop=6, steps=400, deploy_steps=600)
        self.assertEqual(a["source"], "hint_guided_adaptation")
        self.assertGreater(a["adapted_from_prior_by"], 0.0)          # it MOVED from the prior -> adapted, not copied
        self.assertNotEqual(a["params"], b["params"])                # different bodies -> different fitted gaits


if __name__ == "__main__":
    unittest.main()
