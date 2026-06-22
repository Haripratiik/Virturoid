"""Skill Library — the cross-user / cross-morphology TRAINING flywheel (the moat). The first robot to learn a
skill pays full cold-training cost and BANKS it; a LATER, DIFFERENT robot RECALLS that knowledge and adapts it
with a smaller budget. This locks: (1) cross-morphology recall (a hexapod reuses a quadruped's banked 'walk'),
(2) warm acquisition is cheaper than cold, (3) the bank persists (cross-user). See [[virturoid-flywheel-vision]],
[[task-effectiveness-loop]]. Uses tiny budgets so it runs in-process under pytest."""

import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

_MUJOCO = importlib.util.find_spec("mujoco") is not None


def _radial(n):
    """A legged body with ``n`` radial legs — a DIFFERENT morphology than the composed quadruped."""
    from virturoid.services.morphology_builder import MorphologyBuilder
    b = MorphologyBuilder("legged", base_mount="free", species=f"radial{n}")
    b.base("torso", shape="box", length=0.08, radius=0.18, mass=3.0)
    for i in range(n):
        a = 2 * math.pi * i / n
        b.limb(f"leg{i}", [{"length": 0.12, "axis": (0, 1, 0), "torque": 14.0},
                           {"length": 0.12, "axis": (0, 1, 0), "torque": 12.0}],
               parent="torso", mount_offset=(0.16 * math.cos(a), 0.16 * math.sin(a), -0.04),
               mount_euler=(math.pi, 0, 0))
    return b.build()


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class SkillLibraryTests(unittest.TestCase):
    def test_unknown_skill_rejected(self):
        from virturoid.services.skill_library import skill_of
        self.assertIsNone(skill_of("teleport"))
        self.assertEqual(skill_of("walk")[0], "locomotion")
        self.assertEqual(skill_of("pick_up")[0], "grasp")

    def test_cross_morphology_skill_reuse_and_warm_is_cheaper(self):
        from virturoid.services.memory_db import MemoryDB
        from virturoid.services.morphology_composer import compose_robot
        from virturoid.services.skill_library import acquire_skill, recall_skill
        quad = compose_robot("a robot dog that walks", llm=None)
        hexapod = _radial(6)                                   # a structurally DIFFERENT legged body
        tiny_cold, tiny_warm = (3, 8, 150, 1), (2, 8, 150, 1)  # keep the test fast
        with tempfile.TemporaryDirectory() as d:
            with MemoryDB(Path(d) / "skills.db") as db:
                # nothing banked yet -> no skill to recall
                self.assertIsNone(recall_skill(quad, "walk", db=db)[0])

                # USER 1: a quadruped learns 'walk' from scratch (cold) and BANKS it. target high -> always trains.
                r1 = acquire_skill(quad, "walk", db=db, models_dir=d, target=99.0,
                                   cold=tiny_cold, warm=tiny_warm, workers=1)
                self.assertEqual(r1["mode"], "cold-trained")
                self.assertFalse(r1["warm_started"])
                self.assertEqual(r1["kind"], "legged")

                # the skill is now banked + recallable by a DIFFERENT morphology (cross-morphology transfer)
                pol, meta = recall_skill(hexapod, "walk", db=db)
                self.assertIsNotNone(pol, "a hexapod must be able to recall the quadruped's banked walk skill")
                self.assertEqual(meta["task_type"], "locomotion")

                # USER 2: a hexapod ACQUIRES 'walk' — it RECALLS user 1's knowledge and only warm-adapts.
                r2 = acquire_skill(hexapod, "walk", db=db, models_dir=d, target=99.0,
                                   cold=tiny_cold, warm=tiny_warm, workers=1)
                self.assertTrue(r2["warm_started"], "user 2 must reuse the banked skill, not start cold")
                self.assertEqual(r2["mode"], "warm-adapted")
                self.assertLess(r2["generations"], r1["generations"], "warm adaptation must be cheaper than cold")

    def test_reuse_when_banked_skill_already_clears_target(self):
        # If a banked skill already clears the target on this body, acquisition REUSES it with NO training.
        from virturoid.services.memory_db import MemoryDB
        from virturoid.services.morphology_composer import compose_robot
        from virturoid.services.skill_library import acquire_skill
        quad = compose_robot("a robot dog that walks", llm=None)
        with tempfile.TemporaryDirectory() as d:
            with MemoryDB(Path(d) / "skills.db") as db:
                acquire_skill(quad, "walk", db=db, models_dir=d, target=99.0, cold=(3, 8, 150, 1),
                              warm=(2, 8, 150, 1), workers=1)                       # bank something
                r = acquire_skill(quad, "walk", db=db, models_dir=d, target=-99.0,  # trivially-cleared target
                                  cold=(3, 8, 150, 1), warm=(2, 8, 150, 1), workers=1)
                self.assertEqual(r["mode"], "reused")
                self.assertEqual(r["generations"], 0)


if __name__ == "__main__":
    unittest.main()
