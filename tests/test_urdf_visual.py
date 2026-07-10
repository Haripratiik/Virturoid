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

    def test_hexapod_urdf_ships_resolvable_shape_meshes_not_arm_mesh(self):
        # The differentiation bridge (ISSUES A2): the exported URDF's VISUAL is now each segment's baked
        # shape-program mesh (so Studio renders the real body, not a capsule), COLLISION stays a primitive.
        # The old "self-contained" property is preserved as the stronger "every mesh ref RESOLVES next to the
        # urdf" — the actual guard against the 404 flood the original bug caused.
        import os

        from virturoid.services.agent_design_tools import export_held
        from virturoid.services.ai_native_tools import create_robot
        rid = create_robot({"prompt": "a hexapod robot"})["robot_id"]
        path = export_held({"robot_id": rid, "formats": ["urdf"]})["artifacts"]["urdf"]
        txt = open(path, encoding="utf-8").read()
        self.assertNotIn("cad_arm_base", txt, "a hexapod must not reference the hardcoded arm mesh")
        refs = re.findall(r'<mesh filename="([^"]+)"', txt)
        self.assertTrue(refs, "the differentiated body must ship shape-program visual meshes")
        base = os.path.dirname(path)
        for ref in refs:                                # NO 404s: every referenced STL ships next to robot.urdf
            self.assertTrue(os.path.exists(os.path.join(base, ref)),
                            f"URDF mesh ref must resolve relative to robot.urdf (self-contained): {ref}")
        self.assertIn("<cylinder", txt, "collision geometry stays a self-contained primitive (physics untouched)")
        self.assertGreater(txt.count("<visual>"), 6)     # torso + 6 legs' worth of links
        self.assertGreater(txt.count("<collision>"), 6)


@unittest.skipUnless(_MUJOCO, "gene_to_urdf transcribes the compiled MuJoCo model")
class GeneUrdfLayoutTests(unittest.TestCase):
    """The URDF is transcribed from the compiled model, so a legged body's legs SPREAD to their real mounts —
    not stacked vertically at the torso tip (the arm-centric compute_arm_layout produced a single origin)."""

    def test_hexapod_legs_spread_not_stacked(self):
        import xml.dom.minidom as minidom

        from virturoid.services.gene_urdf import gene_to_urdf
        from virturoid.services.morphology_composer import compose_robot
        urdf = gene_to_urdf(compose_robot("a hexapod robot", ensure_walkable=True))
        minidom.parseString(urdf)                                   # valid XML
        origins = re.findall(r'<origin xyz="([^"]+)"', urdf)
        self.assertGreater(len(set(origins)), 3, "legs must have distinct joint origins, not one stacked value")
        lateral = [o for o in origins if abs(float(o.split()[1])) > 0.05]
        self.assertGreaterEqual(len(lateral), 4, "legs must spread laterally (nonzero Y), not stack up +z")
        self.assertIn("<cylinder", urdf)                           # real limb geometry, not a pile of boxes
        self.assertNotIn("cad_arm_base", urdf)
        self.assertEqual(urdf.count("<joint"), urdf.count("</joint>"))

    def test_mesh_dir_bakes_shape_programs_default_stays_primitive(self):
        # back-compat + the bridge in one place: no mesh_dir => byte-identical primitive URDF (a missing
        # build123d never breaks export); with mesh_dir => every segment's shape program is baked + resolvable.
        import os
        import re
        import tempfile

        from virturoid.services.gene_urdf import gene_to_urdf
        from virturoid.services.morphology_composer import compose_robot
        g = compose_robot("an octopus robot")
        self.assertNotIn("<mesh", gene_to_urdf(g), "default (no mesh_dir) stays self-contained primitives")
        md = tempfile.mkdtemp()
        refs = re.findall(r'<mesh filename="meshes/([^"]+)"', gene_to_urdf(g, mesh_dir=md))
        self.assertGreater(len(refs), 8, "an octopus's differentiated body bakes many shape meshes")
        for r in refs:
            self.assertTrue(os.path.exists(os.path.join(md, r)), f"baked STL must exist on disk: {r}")

    def test_gene_urdf_uses_named_mujoco_material_colours(self):
        from virturoid.services.gene_urdf import gene_to_urdf
        from virturoid.services.anatomy_compiler import build_from_anatomy

        gene = build_from_anatomy({"robot_class": "legged", "parts": [
            {"name": "mantle", "role": "body", "aspect": "round", "size": 0.4, "girth": 0.3},
            {"name": "arm", "role": "tentacle", "parent": "mantle", "attach": "front_bottom",
             "size": 0.2, "girth": 0.03, "segments": 2, "joint": "revolute"},
        ]})
        urdf = gene_to_urdf(gene)
        colours = re.findall(r'<color rgba="([^\"]+)"', urdf)
        self.assertGreater(len(set(colours)), 1, "named MuJoCo materials must survive URDF export")


if __name__ == "__main__":
    unittest.main()
