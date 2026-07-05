"""S1 per-axis scene schema: a SceneObject may now carry size_xyz (full extents, metres) that OVERRIDES the legacy
scalar-scale sizing, so a scene can express a real 2.4 m-tall / 0.9 m corridor wall. Requirements: (a) objects
without size_xyz render byte-identically to before (back-compat), (b) size_xyz sets the geom half-extents +
resting height exactly, (c) the result compiles in real MuJoCo, (d) the schema rejects malformed size_xyz."""

import unittest

from virturoid.schemas.scenes import SceneGraph, SceneObject
from virturoid.services.mujoco_exporter import _scene_object_xml


def _wall(size_xyz=None, scale=1.0):
    return SceneObject(name="w", object_type="wall", pose_xyz_rpy=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                       scale=scale, size_xyz=size_xyz)


class SceneSchemaDimsTests(unittest.TestCase):
    def test_wall_backcompat_without_size(self):
        # legacy path: 0.32 m-tall (half 0.16), 0.06 m-thick, length = scale
        xml = _scene_object_xml(_wall(scale=1.0), floor=True)
        self.assertIn('size="0.5 0.03 0.16"', xml)
        self.assertIn("0.16", xml.split("pos=")[1][:20])   # rests at z=height/2=0.16

    def test_wall_real_dimensions_via_size_xyz(self):
        # a REAL corridor wall: 2 m long, 0.1 m thick, 2.4 m tall
        xml = _scene_object_xml(_wall(size_xyz=(2.0, 0.1, 2.4)), floor=True)
        self.assertIn('size="1.0 0.05 1.2"', xml)          # half-extents
        self.assertIn("1.2", xml.split("pos=")[1][:24])    # rests on the ground at z=height/2=1.2 m

    def test_manipulable_box_via_size_xyz(self):
        # a YCB cracker box (0.060 x 0.158 x 0.210 m) instead of a 5 cm cube
        obj = SceneObject(name="cracker", object_type="cube", pose_xyz_rpy=(0.4, 0.0, 0.0, 0, 0, 0),
                          material="cardboard", mass_kg=0.411, size_xyz=(0.060, 0.158, 0.210))
        xml = _scene_object_xml(obj)
        self.assertIn('size="0.03 0.079 0.105"', xml)      # half-extents of the real box
        self.assertIn("<freejoint", xml)                   # still a dynamic free body

    def test_compiles_in_mujoco(self):
        try:
            import mujoco
        except Exception:  # noqa: BLE001
            self.skipTest("mujoco not installed")
        wall = _scene_object_xml(_wall(size_xyz=(2.0, 0.1, 2.4)), floor=True)
        obj = _scene_object_xml(SceneObject(name="c", object_type="cube", material="green",
                                            pose_xyz_rpy=(0.0, 0.0, 0.0, 0, 0, 0), size_xyz=(0.06, 0.16, 0.21)))
        xml = f'<mujoco><worldbody>{wall}{obj}</worldbody></mujoco>'   # material="green" -> rgba, no asset needed
        model = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(model)
        for _ in range(20):
            mujoco.mj_step(model, d)
        import numpy as np
        self.assertTrue(np.all(np.isfinite(d.qpos)))

    def test_schema_rejects_bad_size(self):
        good = SceneGraph(id="s", name="s", backend_targets=["mujoco"], robot_spawn_xyz_rpy=(0, 0, 0, 0, 0, 0),
                          objects=[_wall(size_xyz=(1.0, 0.1, 2.0))])
        self.assertTrue(good.validate().ok)
        for bad in [(1.0, 0.0, 2.0), (-1.0, 0.1, 2.0), (1.0, 2.0)]:
            g = SceneGraph(id="s", name="s", backend_targets=["mujoco"], robot_spawn_xyz_rpy=(0, 0, 0, 0, 0, 0),
                           objects=[_wall(size_xyz=bad)])
            self.assertFalse(g.validate().ok, f"should reject size_xyz={bad}")


if __name__ == "__main__":
    unittest.main()
