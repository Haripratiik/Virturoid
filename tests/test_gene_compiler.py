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

    def test_composition_provenance_roundtrips(self):
        g = tabletop_arm_gene()
        g.design_source = "anatomy_generic"
        g.composition_notes = ["LLM anatomy was unavailable; deterministic general compiler used."]
        back = RobotGene.from_dict(g.to_dict())
        self.assertEqual(back.design_source, g.design_source)
        self.assertEqual(back.composition_notes, g.composition_notes)

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

    def test_anatomy_shape_proxies_are_present_in_mjcf(self):
        from virturoid.services.anatomy_compiler import build_from_anatomy

        g = build_from_anatomy({"robot_class": "legged", "parts": [
            {"name": "torso", "role": "body", "size": 0.6, "girth": 0.16},
            {"name": "beam", "role": "arm", "parent": "torso", "attach": "front_top", "aim": "forward",
             "size": 0.24, "girth": 0.05, "joint": "revolute"},
        ]})
        xml = compile_gene_to_mjcf(g)
        self.assertIn('name="torso_geom" type="box"', xml)
        self.assertIn('name="beam_geom" type="box"', xml)

    def test_kinematic_ancestors_are_excluded_from_self_contact(self):
        from virturoid.services.morphology_composer import compose_robot

        gene = compose_robot("a six-legged hexapod walking robot", llm=None)
        xml = compile_gene_to_mjcf(gene)
        # The upper leg's direct parent is already filtered by MuJoCo, but its
        # torso grandparent is not.  That explicit exclusion removes the known
        # torso/leg penetration without turning off world contact.
        self.assertIn('<exclude body1="torso" body2="leg0_1"/>', xml)
        self.assertIn('<geom name="floor"', xml)

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

    @unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
    def test_hexapod_has_no_initial_internal_contacts(self):
        import mujoco

        from virturoid.services.morphology_composer import compose_robot

        gene = compose_robot("a six-legged hexapod walking robot", llm=None)
        model = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene, include_floor=True))
        data = mujoco.MjData(model)
        if model.nkey:
            mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        internal = []
        for idx in range(data.ncon):
            contact = data.contact[idx]
            if model.geom_bodyid[contact.geom1] and model.geom_bodyid[contact.geom2]:
                internal.append((model.geom(contact.geom1).name, model.geom(contact.geom2).name))
        self.assertEqual([], internal)

    @unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
    def test_styling_pass_present_and_physics_neutral(self):
        # R1 realism styling pass: a 3-point studio light rig, an HD offscreen buffer, a concrete floor, and the
        # unified material palette must all be present — AND must be physics-neutral (visual-only geoms/lights add
        # no bodies, joints, or actuators, so grasp/locomotion dynamics stay byte-identical).
        import mujoco

        gene = tabletop_arm_gene()
        xml = compile_gene_to_mjcf(gene, include_floor=True)
        for tok in ('name="key"', 'name="fill"', 'name="rim"', 'offwidth="1920"',
                    'castshadow="true"', 'mat_cf', 'mat_shell'):
            self.assertIn(tok, xml, f"styling token {tok!r} missing from the compiled MJCF")
        self.assertEqual(3, xml.count("<light "), "expected exactly the 3-point rig on the floored render")
        model = mujoco.MjModel.from_xml_string(xml)
        self.assertEqual(model.nlight, 3, "the 3 rig lights are the only lights (physics-neutral additions)")
        self.assertEqual(model.nu, len(gene.actuated_joints()), "styling must not add actuators")
        self.assertEqual(model.nbody, len(gene.segments) + 1, "styling must not add bodies (world + segments)")
        # A measurement pass (no floor) omits scene lights so it never fights a wrapping scene's own rig.
        self.assertEqual(0, compile_gene_to_mjcf(gene, include_floor=False).count("<light "))

    @unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
    def test_r2_fairings_and_boots_are_visual_only(self):
        # R2: a walking limb gets accent bodywork (a fairing on its top segment) + a rubber boot at the foot, so
        # it reads as a designed limb, not a bare capsule chain. All visual-only (mass 0, contype 0) -> physics
        # byte-identical: a legged body compiles to the SAME nbody/nu with and without the decoration.
        import mujoco

        from virturoid.services.anatomy_compiler import generic_creature_gene
        gene = generic_creature_gene("a quadruped robot dog", "quadruped")
        xml = compile_gene_to_mjcf(gene, include_floor=True)
        self.assertIn("_fairing", xml, "a quadruped's legs should get shell fairings")
        self.assertIn("_boot", xml, "a quadruped's feet should get rubber boots")
        # every fairing/boot geom is non-colliding and massless (visual-only).
        for line in xml.splitlines():
            if "_fairing" in line or "_boot" in line:
                self.assertIn('contype="0"', line)
                self.assertIn('mass="0"', line)
        model = mujoco.MjModel.from_xml_string(xml)          # still a valid, loadable model
        self.assertEqual(model.nbody, len(gene.segments) + 1)  # decoration adds geoms, never bodies


if __name__ == "__main__":
    unittest.main()
