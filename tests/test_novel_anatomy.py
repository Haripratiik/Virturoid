"""Tier-3 novel-body archetypes: distinctive exotic morphologies (spider/snake/frog) that the real-model
substrate and the parametric legged walker can't capture — recognizable skeletons that compile, carry a
baked rest pose (emitted as a MuJoCo keyframe), and stand. Hexapod/octopod stay on the parametric path."""

import importlib.util
import unittest

_MUJOCO = importlib.util.find_spec("mujoco") is not None


class NovelDispatchTests(unittest.TestCase):
    def test_distinctive_morphologies_route_to_archetypes(self):
        from virturoid.services.novel_anatomy import novel_archetype_gene
        spider = novel_archetype_gene("a spider robot")
        self.assertIsNotNone(spider)
        self.assertEqual("arachnid.proc", spider.species)
        self.assertEqual([], spider.validate())
        # 8 legs x (femur+tibia) = 16 leg segments + cephalothorax/abdomen/head
        self.assertEqual(16, sum(1 for s in spider.segments if s.name.startswith("leg_")))
        self.assertIn("rest_pose", spider.metadata)
        self.assertEqual("serpent.proc", novel_archetype_gene("a snake").species)
        self.assertEqual("amphibian.proc", novel_archetype_gene("a frog robot").species)

    def test_hexapod_and_generic_stay_on_parametric_path(self):
        from virturoid.services.novel_anatomy import novel_archetype_gene
        # an N-legged WALKER must use the parametric gait-tuned template, not a decorative archetype
        self.assertIsNone(novel_archetype_gene("a six-legged hexapod walking robot"))
        self.assertIsNone(novel_archetype_gene("a quadruped robot"))
        self.assertIsNone(novel_archetype_gene("a humanoid robot"))

    def test_compose_robot_routes_distinctive_shapes_and_generic_animals(self):
        # OFFLINE (no LLM): a DISTINCTIVE shape the general compiler can't make (spider = 8 radial legs) still
        # uses its dedicated archetype, but a common animal ("dog") is NOT hard-routed — it gets a CLASS-GENERAL
        # generic quadruped via the general compiler (no build_dog). With an LLM the anatomy designer supplies
        # the species anatomy for both; that is the primary path.
        import os
        os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
        from virturoid.services.morphology_composer import compose_robot
        spider = compose_robot("a spider robot")
        self.assertEqual("arachnid.proc", spider.species)
        self.assertEqual("archetype", getattr(spider, "design_source", None))
        dog = compose_robot("a dog robot")
        self.assertEqual("anatomy_generic", getattr(dog, "design_source", None))   # NOT a hand-coded dog
        self.assertEqual([], dog.validate())


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class NovelCompileTests(unittest.TestCase):
    def test_archetypes_compile_with_a_valid_rest_pose_keyframe(self):
        import mujoco
        from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
        from virturoid.services.novel_anatomy import build_frog, build_snake, build_spider

        for gene in (build_spider(), build_snake(), build_frog()):
            xml = compile_gene_to_mjcf(gene, spawn_z=standing_spawn_z(gene))
            m = mujoco.MjModel.from_xml_string(xml)
            self.assertEqual(1, m.nkey, f"{gene.species}: expected a rest-pose keyframe")
            # the keyframe qpos length MUST equal the model's nq (else MuJoCo would have refused to load)
            self.assertEqual(m.nq, m.key_qpos.shape[1])
            d = mujoco.MjData(m)
            mujoco.mj_resetDataKeyframe(m, d, 0)
            mujoco.mj_forward(m, d)            # poses without error
            lo = float((d.geom_xpos[:, 2] - m.geom_rbound).min())
            self.assertGreater(lo, -0.05)      # spawns standing, not buried in the floor


if __name__ == "__main__":
    unittest.main()
