"""Import an EXISTING robot model (MJCF/URDF) and iterate on it — the "I already have a robot, improve
it" half of the goal: load it, report structure, learn control on it, and replay the learned motion."""

import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

_MUJOCO = importlib.util.find_spec("mujoco") is not None


def _write_existing_model(tmp: str) -> str:
    """Save a composed robot's MJCF to disk to stand in for a user's CAD-exported model."""
    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    from virturoid.services.morphology_composer import compose_robot

    p = Path(tmp) / "existing.xml"
    p.write_text(compile_gene_to_mjcf(compose_robot("a quadruped walking robot"), include_floor=True),
                 encoding="utf-8")
    return str(p)


@unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
class ModelImportTests(unittest.TestCase):
    def test_import_reports_structure(self):
        from virturoid.services.model_import import import_model
        with tempfile.TemporaryDirectory() as tmp:
            imp = import_model(_write_existing_model(tmp))
        self.assertTrue(imp["ok"])
        self.assertGreater(imp["parts"], 8)          # high-fidelity body comes through
        self.assertGreater(imp["actuated"], 0)       # has motors -> learnable
        self.assertTrue(imp["free_base"])            # can move -> locomotion makes sense
        self.assertIn("<mujoco", imp["mjcf"])
        self.assertIn('type="plane"', imp["mjcf"])   # ground ensured

    def test_import_then_learn_and_replay(self):
        from virturoid.services.learn_locomotion import learn_locomotion, rollout_view
        from virturoid.services.model_import import import_model
        with tempfile.TemporaryDirectory() as tmp:
            imp = import_model(_write_existing_model(tmp))
            res = learn_locomotion(imp["mjcf"], generations=2, pop=4, steps=80,
                                   models_dir=tmp, species="imported_test")
            self.assertIsNotNone(res["policy"])              # trained a control policy ON THE IMPORTED BODY
            self.assertTrue(math.isfinite(res["score"]))
            self.assertEqual(res["species"], "imported_test")
            self.assertTrue(Path(res["npz_path"]).exists())  # saved -> reusable
            view, xml = rollout_view(imp["mjcf"], res["policy"], steps=60)
            self.assertGreater(len(view["frames"]), 1)       # replayable learned motion

    def test_urdf_import_synthesizes_actuators(self):
        # A URDF imports its joints but NO MuJoCo motors; import_model adds a motor per movable joint so the
        # imported CAD robot is actually controllable (can be iterated on), not merely simulatable.
        urdf = """<?xml version="1.0"?>
<robot name="t">
  <link name="base"><inertial><mass value="1"/>
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
    <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision></link>
  <link name="arm"><inertial><mass value="0.5"/><origin xyz="0 0 0.1"/>
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.005"/></inertial>
    <collision><origin xyz="0 0 0.1"/><geometry><cylinder radius="0.03" length="0.2"/></geometry></collision></link>
  <joint name="j1" type="revolute"><parent link="base"/><child link="arm"/><origin xyz="0 0 0.1"/>
    <axis xyz="0 1 0"/><limit lower="-2" upper="2" effort="15" velocity="2"/></joint>
</robot>"""
        from virturoid.services.model_import import import_model
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.urdf"; p.write_text(urdf, encoding="utf-8")
            imp = import_model(str(p))
        self.assertTrue(imp["ok"], imp.get("note"))
        self.assertGreaterEqual(imp["actuated"], 1)        # motor synthesized for the revolute joint
        self.assertIn("<actuator>", imp["mjcf"])

    def test_unreadable_path_is_handled(self):
        from virturoid.services.model_import import import_model
        bad = import_model("does/not/exist.xml")
        self.assertFalse(bad["ok"])
        self.assertIn("not found", bad["note"])


if __name__ == "__main__":
    unittest.main()
