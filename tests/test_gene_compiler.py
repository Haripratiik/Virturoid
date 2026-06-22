import importlib.util
import unittest

from virturoid.fixtures.gene_library import (
    humanoid_spray_paint_gene,
    humanoid_upper_body_gene,
    tabletop_arm_gene,
)
from virturoid.schemas.gene import GeneSegment, RobotGene, amend_gene
from virturoid.services.gene_compiler import compile_gene_to_mjcf

_MUJOCO = importlib.util.find_spec("mujoco") is not None


class GeneSchemaTests(unittest.TestCase):
    def test_seed_genes_are_valid(self):
        for gene in (tabletop_arm_gene(), humanoid_upper_body_gene(), humanoid_spray_paint_gene()):
            self.assertEqual([], gene.validate(), f"{gene.id}: {gene.validate()}")

    def test_invalid_genes_are_rejected(self):
        # unknown parent
        g = RobotGene(id="bad", species="x.y", robot_class="manipulator", segments=[
            GeneSegment("root", parent=None, is_end_effector=True),
            GeneSegment("link", parent="ghost"),
        ])
        self.assertTrue(any("unknown parent" in i for i in g.validate()))

        # no end effector
        g2 = RobotGene(id="bad2", species="x.y", robot_class="manipulator", segments=[
            GeneSegment("root", parent=None),
        ])
        self.assertTrue(any("end effector" in i for i in g2.validate()))

        # two roots
        g3 = RobotGene(id="bad3", species="x.y", robot_class="manipulator", segments=[
            GeneSegment("a", parent=None, is_end_effector=True),
            GeneSegment("b", parent=None),
        ])
        self.assertTrue(any("one root" in i for i in g3.validate()))

    def test_cycle_is_rejected(self):
        g = RobotGene(id="cyc", species="x.y", robot_class="manipulator", segments=[
            GeneSegment("a", parent="b", is_end_effector=True),
            GeneSegment("b", parent="a"),
        ])
        self.assertTrue(any("cycle" in i or "root" in i for i in g.validate()))


class AmendmentTests(unittest.TestCase):
    def test_amend_reuses_body_and_applies_diff(self):
        spray = humanoid_spray_paint_gene()
        base = humanoid_upper_body_gene()
        # Same number of segments (body reused), branched from the humanoid node.
        self.assertEqual(len(base.segments), len(spray.segments))
        self.assertEqual("humanoid.upper_body.single_arm", spray.parent_species)
        self.assertEqual("spray_nozzle", spray.end_effector_type)
        # Only the diffed fields changed.
        self.assertEqual(0.30, spray.segment("forearm").length_m)
        self.assertEqual("cylinder", spray.segment("hand").shape)
        self.assertEqual([], spray.validate())
        self.assertEqual("gene_humanoid_upper_body_single_arm", spray.metadata["amended_from"])

    def test_amend_can_carry_to_new_task_without_breaking_validity(self):
        base = tabletop_arm_gene()
        longer = amend_gene(base, new_id="g2", species="fixed_arm.three_dof.long_reach",
                            segment_overrides={"forearm": {"length_m": 0.45}})
        self.assertEqual(0.45, longer.segment("forearm").length_m)
        self.assertEqual(0.357, base.segment("forearm").length_m)  # parent unchanged
        self.assertEqual([], longer.validate())


class GeneSerializationTests(unittest.TestCase):
    def test_gene_roundtrips_through_dict(self):
        for g in (tabletop_arm_gene(), humanoid_upper_body_gene()):
            d = g.to_dict()
            back = RobotGene.from_dict(d)
            self.assertEqual([], back.validate())
            self.assertEqual(g.species, back.species)
            self.assertEqual([s.name for s in g.segments], [s.name for s in back.segments])
            self.assertEqual(g.segment("forearm").length_m, back.segment("forearm").length_m)

    def test_species_tree_stores_and_reuses_full_gene(self):
        import tempfile
        from pathlib import Path

        from virturoid.services.memory_db import MemoryDB

        g = humanoid_upper_body_gene()
        with tempfile.TemporaryDirectory() as tmp:
            with MemoryDB(Path(tmp) / "m.db") as db:
                db.upsert_species_node(g.species, robot_class="humanoid", genes=g.to_dict(), buildable=True)
                stored = db.find_gene_for_class("humanoid")
                self.assertIsNotNone(stored)
                reused = RobotGene.from_dict(stored)
                self.assertEqual([], reused.validate())   # the reused gene is buildable
                self.assertEqual(g.species, reused.species)
                self.assertIsNone(db.find_gene_for_class("quadruped"))  # none stored


class CompileTests(unittest.TestCase):
    def test_distinct_genes_compile_to_distinct_models(self):
        arm = compile_gene_to_mjcf(tabletop_arm_gene())
        humanoid = compile_gene_to_mjcf(humanoid_upper_body_gene())
        self.assertIn("<mujoco", arm)
        self.assertIn("torso", humanoid)
        self.assertNotIn("torso", arm)            # genuinely different morphology
        self.assertNotEqual(arm, humanoid)
        # humanoid has more actuated joints (motors) than the 3-DOF arm
        self.assertGreater(humanoid.count("<motor "), arm.count("<motor "))

    def test_invalid_gene_refuses_to_compile(self):
        bad = RobotGene(id="b", species="x.y", robot_class="manipulator",
                        segments=[GeneSegment("r", parent=None)])  # no end effector
        with self.assertRaises(ValueError):
            compile_gene_to_mjcf(bad)

    @unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
    def test_compiled_models_load_in_mujoco(self):
        import mujoco

        for gene in (tabletop_arm_gene(), humanoid_upper_body_gene(), humanoid_spray_paint_gene()):
            xml = compile_gene_to_mjcf(gene)
            model = mujoco.MjModel.from_xml_string(xml)   # raises if MJCF is invalid
            data = mujoco.MjData(model)
            for _ in range(50):                            # it actually steps (stable enough)
                mujoco.mj_step(model, data)
            self.assertGreater(model.nbody, 1)
            self.assertEqual(model.nu, len(gene.actuated_joints()))  # one actuator per actuated joint
            # The end-effector site exists for controllers/graspers.
            self.assertGreaterEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"), 0)


if __name__ == "__main__":
    unittest.main()
