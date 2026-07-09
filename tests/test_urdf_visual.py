"""The exported URDF must be SELF-CONTAINED and renderable for ANY robot — not just the reference arm.

Bug found by driving Virturoid Studio's 3D viewport (2026-07-09): `urdf_exporter._mesh_name_for_link` defaulted
EVERY unrecognized link to `cad_arm_base.stl`, so a hexapod/quad/humanoid URDF referenced a nonexistent ARM mesh.
The Studio URDFLoader then 404'd on every link (hundreds of console errors, empty viewport) and the exported URDF
wasn't portable. Fix: a link with no real mesh renders as a self-contained primitive box (the collision already
did); only the arm's known links keep their real STL.
"""
import importlib.util
import os
import re
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("MUJOCO_GL", "glfw")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


class MeshNameTests(unittest.TestCase):
    def test_arm_links_keep_meshes_others_get_none(self):
        from virturoid.services.urdf_exporter import _mesh_name_for_link
        self.assertEqual(_mesh_name_for_link("base_link"), "cad_arm_base.stl")   # the reference arm ships this STL
        self.assertEqual(_mesh_name_for_link("gripper_link"), "cad_gripper_mount.stl")
        for leg in ("torso", "leg0_0", "leg1_l_2", "head_0", "front_left_leg"):
            self.assertIsNone(_mesh_name_for_link(leg), f"'{leg}' must NOT fall back to an arm mesh")


@unittest.skipUnless(_MUJOCO, "composing + exporting a robot needs MuJoCo")
class UrdfVisualTests(unittest.TestCase):
    def _urdf_text(self, prompt):
        from virturoid.services.agent_design_tools import export_held
        from virturoid.services.ai_native_tools import create_robot
        rid = create_robot({"prompt": prompt})["robot_id"]
        path = export_held({"robot_id": rid, "formats": ["urdf"]})["artifacts"]["urdf"]
        return open(path, encoding="utf-8").read()

    def test_hexapod_urdf_is_self_contained_boxes_not_arm_mesh(self):
        txt = self._urdf_text("a hexapod robot")
        self.assertNotIn("cad_arm_base", txt, "a hexapod must not reference the hardcoded arm mesh")
        self.assertNotIn("<mesh", txt, "a mesh-less body must reference NO meshes (self-contained)")
        self.assertIn("<box", txt, "visual geometry must be a self-contained primitive box")
        # every link still has a visual + a collision (renderable + simulatable)
        self.assertEqual(txt.count("<visual>"), txt.count("<collision>"))
        self.assertGreater(txt.count("<visual>"), 6)   # torso + 6 legs' worth of links


if __name__ == "__main__":
    unittest.main()
