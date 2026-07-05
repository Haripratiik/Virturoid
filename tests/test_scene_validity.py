"""S4 physical-validity gate stack. Each gate must accept a good scene and REJECT a specific defect: unit-sanity
flags a 20 m box; settle rejects an interpenetrating spawn (ejection); navigability rejects a walled-in goal and
accepts a reachable one; reachability rejects an out-of-reach object; solvability keeps-if-the-expert-succeeds.
Generated scene families (S3) should largely pass. All fail-closed."""

import unittest

import numpy as np

from virturoid.schemas.scenes import SceneGraph, SceneObject
from virturoid.services.scene_validity import (
    unit_sanity_gate, settle_gate, navigability_gate, reachability_gate, solvability_gate,
    validate_scene_physical)
from virturoid.services.scene_family import generate_family


def _scene(objs, bounds=((-2.5, -2.5, 0), (2.5, 2.5, 2.6)), spawn=(0, 0, 0, 0, 0, 0)):
    return SceneGraph(id="t", name="t", backend_targets=["mujoco"], robot_spawn_xyz_rpy=spawn,
                      objects=objs, bounds=bounds)


class SceneValidityTests(unittest.TestCase):
    def test_unit_sanity_flags_absurd_object(self):
        good = _scene([SceneObject("b", "cube", (0.4, 0, 0, 0, 0, 0), category="block", size_xyz=(0.05, 0.05, 0.05))])
        self.assertTrue(unit_sanity_gate(good)["ok"])
        bad = _scene([SceneObject("b", "cube", (0.4, 0, 0, 0, 0, 0), category="block", size_xyz=(20.0, 0.05, 0.05))])
        r = unit_sanity_gate(bad)
        self.assertFalse(r["ok"])
        self.assertTrue(any("unit error" in v for v in r["violations"]))

    def test_settle_rejects_interpenetrating_spawn(self):
        try:
            import mujoco  # noqa: F401
        except Exception:  # noqa: BLE001
            self.skipTest("mujoco not installed")
        # two heavy boxes spawned at the SAME point -> they explode apart -> settle fails
        overlap = _scene([
            SceneObject("a", "cube", (0.0, 0.0, 0.1, 0, 0, 0), category="block", size_xyz=(0.1, 0.1, 0.1), mass_kg=0.5),
            SceneObject("b", "cube", (0.0, 0.0, 0.1, 0, 0, 0), category="block", size_xyz=(0.1, 0.1, 0.1), mass_kg=0.5)])
        self.assertFalse(settle_gate(overlap)["ok"])
        # a single block resting just above the floor settles fine
        ok = _scene([SceneObject("a", "cube", (0.0, 0.0, 0.06, 0, 0, 0), category="block",
                                 size_xyz=(0.1, 0.1, 0.1), mass_kg=0.2)])
        self.assertTrue(settle_gate(ok)["ok"], settle_gate(ok))

    def test_navigability_accepts_open_and_rejects_walled_goal(self):
        # open scene: spawn at origin, goal at (2,0), no obstacles -> path exists
        openair = _scene([SceneObject("goal", "zone", (2.0, 0.0, 0.0, 0, 0, 0), size_xyz=(0.4, 0.4, 0.006))])
        r = navigability_gate(openair, robot_radius=0.2)
        self.assertTrue(r["ok"], r)
        self.assertLess(r["ratio"], 1.2)                     # roughly straight shot
        # wall the goal off with a box ring around it -> no path
        walled = _scene([
            SceneObject("goal", "zone", (2.0, 0.0, 0.0, 0, 0, 0), size_xyz=(0.2, 0.2, 0.006)),
            SceneObject("w1", "wall", (2.0, 0.5, 0, 0, 0, 0), category="wall", size_xyz=(2.0, 0.1, 2.4)),
            SceneObject("w2", "wall", (2.0, -0.5, 0, 0, 0, 0), category="wall", size_xyz=(2.0, 0.1, 2.4)),
            SceneObject("w3", "wall", (2.6, 0.0, 0, 0, 0, 0), category="wall", size_xyz=(0.1, 1.0, 2.4)),
            SceneObject("w4", "wall", (1.4, 0.0, 0, 0, 0, 0), category="wall", size_xyz=(0.1, 1.0, 2.4))])
        self.assertFalse(navigability_gate(walled, robot_radius=0.2)["ok"])

    def test_reachability_rejects_far_object(self):
        near = _scene([SceneObject("o", "cube", (0.4, 0.0, 0.0, 0, 0, 0), category="block", size_xyz=(0.05,) * 3)])
        self.assertTrue(reachability_gate(near, base_xy=(0, 0), reach_m=0.55)["ok"])
        far = _scene([SceneObject("o", "cube", (2.5, 0.0, 0.0, 0, 0, 0), category="block", size_xyz=(0.05,) * 3)])
        self.assertFalse(reachability_gate(far, base_xy=(0, 0), reach_m=0.55)["ok"])

    def test_solvability_keep_if_success(self):
        s = _scene([SceneObject("o", "cube", (0.4, 0, 0, 0, 0, 0), category="block", size_xyz=(0.05,) * 3)])
        self.assertTrue(solvability_gate(s, lambda sc: True)["ok"])
        self.assertFalse(solvability_gate(s, lambda sc: False)["ok"])
        self.assertFalse(solvability_gate(s, lambda sc: 1 / 0)["ok"])   # fail-closed on error

    def test_generated_families_mostly_pass(self):
        # the S3 generator should produce scenes that clear the gates (its whole point)
        try:
            import mujoco  # noqa: F401
        except Exception:  # noqa: BLE001
            self.skipTest("mujoco not installed")
        fam = generate_family("navigation", n_train=4, n_held_out=1, seed=11)
        passed = sum(validate_scene_physical(s, robot_radius=0.2, run_settle=False)["ok"] for s in fam.train)
        self.assertGreaterEqual(passed, 3, "most generated nav scenes should be navigable")


if __name__ == "__main__":
    unittest.main()
