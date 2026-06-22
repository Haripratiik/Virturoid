"""Compositional building blocks: agents assemble ANY robot from a few primitives -> valid gene."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from virturoid.services.morphology_builder import MorphologyBuilder
from virturoid.services.morphology_embedding import distance, embed_gene

_MUJOCO = importlib.util.find_spec("mujoco") is not None


def _arm():
    return (MorphologyBuilder("manipulator", base_mount="table", gene_id="b_arm")
            .base("base").link("shoulder", 0.25, axis=(0, 0, 1)).link("upper", 0.25, axis=(0, 1, 0))
            .link("fore", 0.2, axis=(0, 1, 0)).gripper().build())


def _quadruped():
    b = MorphologyBuilder("quadruped", base_mount="floor", gene_id="b_quad").base(
        "torso", shape="box", length=0.3, radius=0.1, mass=4.0)
    for i, (sx, sy) in enumerate([(0.12, 0.08), (0.12, -0.08), (-0.12, 0.08), (-0.12, -0.08)]):
        b.limb(f"leg{i}", [dict(length=0.12, axis=(0, 1, 0)), dict(length=0.14, axis=(0, 1, 0))],
               parent="torso", mount_offset=(sx, sy, 0))
    return b.build()


class MorphologyBuilderTests(unittest.TestCase):
    def test_serial_gripper_arm_is_valid(self):
        arm = _arm()
        self.assertEqual(arm.validate(), [])
        self.assertEqual(arm.end_effector_type, "gripper")
        self.assertEqual(len(arm.actuated_joints()), 5)            # 3 revolute + 2 finger
        self.assertIsNotNone(arm.end_effector())

    def test_quadruped_from_limbs_is_valid_and_branched(self):
        quad = _quadruped()
        self.assertEqual(quad.validate(), [])
        self.assertEqual(len(quad.actuated_joints()), 8)           # 4 legs x 2 joints
        # torso is a branch point with 4 children (the legs)
        torso_children = [s for s in quad.segments if s.parent == "torso"]
        self.assertEqual(len(torso_children), 4)

    def test_multi_finger_hand_is_valid_and_distinct(self):
        from virturoid.services.morphology_embedding import distance, embed_gene
        two = (MorphologyBuilder("manipulator", gene_id="h2").base("b").link("s", 0.25, axis=(0, 0, 1))
               .link("u", 0.25, axis=(0, 1, 0)).hand(n_fingers=2).build())
        three = (MorphologyBuilder("manipulator", gene_id="h3").base("b").link("s", 0.25, axis=(0, 0, 1))
                 .link("u", 0.25, axis=(0, 1, 0)).hand(n_fingers=3).build())
        self.assertEqual(three.validate(), [])
        self.assertEqual(sum(1 for s in three.segments if s.joint_type == "prismatic"), 3)
        self.assertEqual(two.end_effector_type, "gripper")          # n=2 hand == parallel gripper
        self.assertGreater(distance(embed_gene(three), embed_gene(two)), 0.0)   # dexterity changes morphology
        with self.assertRaises(ValueError):
            MorphologyBuilder().base("b").hand(n_fingers=1)         # a hand needs >= 2 fingers

    def test_builder_requires_base_first(self):
        with self.assertRaises(ValueError):
            MorphologyBuilder().link("x", 0.2)

    def test_wheel_is_a_cylinder_continuous_revolute(self):
        g = (MorphologyBuilder("mobile_base", base_mount="free", gene_id="rover").base("chassis")
             .wheel("wheel_l", mount_offset=(0.1, 0.12, -0.04)).wheel("wheel_r", mount_offset=(0.1, -0.12, -0.04))
             .build())
        wheels = [s for s in g.segments if s.name.startswith("wheel")]
        self.assertEqual(len(wheels), 2)
        self.assertTrue(all(w.shape == "cylinder" and w.joint_type == "revolute" for w in wheels))
        self.assertTrue(all(w.joint_lower is None and w.joint_upper is None for w in wheels))  # continuous
        self.assertTrue(all(abs(w.mount_euler[0]) > 1.0 for w in wheels))  # laid flat -> horizontal spin axis
        # both wheels attach to the chassis (cursor stays on the base)
        self.assertTrue(all(w.parent == "chassis" for w in wheels))

    @unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
    def test_free_base_gets_a_freejoint_and_compiles(self):
        import mujoco
        from virturoid.services.gene_compiler import compile_gene_to_mjcf
        g = (MorphologyBuilder("mobile_base", base_mount="free", gene_id="rover2").base("chassis")
             .wheel("w0", mount_offset=(0.1, 0.12, -0.04)).wheel("w1", mount_offset=(-0.1, 0.12, -0.04))
             .build())
        xml = compile_gene_to_mjcf(g)
        self.assertIn("freejoint", xml)
        mj = mujoco.MjModel.from_xml_string(xml)
        self.assertGreaterEqual(mj.nq, 7 + 2)   # free base (7) + 2 wheel hinges

    def test_distinct_morphologies_are_far_in_embedding_space(self):
        self.assertGreater(distance(embed_gene(_arm()), embed_gene(_quadruped())), 2.0)

    @unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
    def test_composed_robots_compile_to_mujoco(self):
        import mujoco
        from virturoid.services.gene_compiler import compile_gene_to_mjcf
        for g in (_arm(), _quadruped()):
            mj = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(g))
            self.assertGreaterEqual(mj.nu, len(g.actuated_joints()))

    def test_composed_robot_auto_places_as_new_species(self):
        from virturoid.services.memory_db import MemoryDB
        from virturoid.services.species_discovery import auto_place_species
        with tempfile.TemporaryDirectory() as tmp, MemoryDB(Path(tmp) / "m.db") as db:
            auto_place_species(_arm(), db)
            placed = auto_place_species(_quadruped(), db)          # far from the arm -> a new species
            self.assertEqual(placed["action"], "created")


if __name__ == "__main__":
    unittest.main()
