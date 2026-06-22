"""Offline guard for the M2 push-skill scene (scripts/mjx_push_skill.py).

The push trainer looks up the ``ee_site`` and the box's ``free_box`` free joint and relies on
contacts being ENABLED. Those assumptions only fail at GPU runtime otherwise — this checks them on
CPU so a broken scene never wastes a GPU run (or hangs the box again)."""

import importlib.util
import unittest

_MUJOCO = importlib.util.find_spec("mujoco") is not None


class PushSceneTests(unittest.TestCase):
    def _push_xml(self):
        from virturoid.fixtures.gene_library import tabletop_arm_gene
        from virturoid.services.gene_compiler import compile_gene_with_scene
        from virturoid.services.task_runtime import generate_task_scenes, select_task_spec

        gene = tabletop_arm_gene()
        spec = select_task_spec("place the box on the target")
        scenes = generate_task_scenes(gene, spec, count=1)
        objs = [o for o in scenes[0].objects if o.object_type == "cube"]
        self.assertTrue(objs, "push scene must contain a manipulable cube")
        return compile_gene_with_scene(gene, objs), objs

    def test_scene_has_box_and_ee_site_in_xml(self):
        xml, objs = self._push_xml()
        self.assertIn("ee_site", xml)
        # the box is a free body the arm can shove
        self.assertIn(f"free_{objs[0].name}", xml)

    @unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
    def test_model_lookups_the_trainer_relies_on(self):
        import mujoco

        xml, _ = self._push_xml()
        mj = mujoco.MjModel.from_xml_string(xml)
        ee = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        box_jid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, "free_box")
        self.assertGreaterEqual(ee, 0, "trainer needs an ee_site")
        self.assertGreaterEqual(box_jid, 0, "trainer needs the free_box joint to read box pose")
        # contacts must be ON for a push task (reach disables them; push must not)
        self.assertFalse(bool(mj.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT))
        # actuated arm joints exist to drive the push
        self.assertGreater(mj.nu, 0)


if __name__ == "__main__":
    unittest.main()
