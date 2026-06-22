"""Morphology-agnostic transferable grasp (POC-A). Locks the parts that are TRUE independent of how well the
learned residual trains: (1) the grasp handles (fingers, object, TCP) are derived from the graph for ANY
manipulator with no hardcoded geom ids; (2) ``morph_grasp_rollout(policy=None)`` reproduces the scripted
grasp_eval base (so the residual is a clean superset); (3) the N-finger contact count is graph-derived.
The residual is a per-token MorphPolicy so it transfers across arms — see [[task-effectiveness-loop]]."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class GraspSkillTests(unittest.TestCase):
    def _arm(self):
        from virturoid.services.morphology_composer import compose_robot
        return compose_robot("a tabletop robot arm", llm=None)

    def test_encode_grasp_is_morphology_agnostic(self):
        import mujoco

        from virturoid.schemas.scenes import SceneObject
        from virturoid.services.gene_compiler import compile_gene_with_scene
        from virturoid.services.grasp_skill import encode_grasp
        arm = self._arm()
        cube = SceneObject("box", "cube", (0.44, 0.0, 0.05, 0, 0, 0), mass_kg=0.03)
        mj = mujoco.MjModel.from_xml_string(compile_gene_with_scene(arm, [cube]))
        h = encode_grasp(mj, arm)
        self.assertGreater(len(h["arm_u"]), 0)                       # revolute arm joints present
        self.assertGreaterEqual(len(h["fin_u"]), 2)                 # >= 2 gripper fingers (slide joints)
        self.assertGreaterEqual(len(h["finger_geoms"]), 2)         # finger geoms found by graph, not by name
        self.assertGreaterEqual(h["box_geom"], 0)                  # object geom resolved
        self.assertGreaterEqual(h["tcp"], 0)                       # grasp_site present
        self.assertEqual(len(h["arm_tok"]), len(h["arm_u"]))       # token<->arm-slot alignment for the residual

    def test_scripted_base_reproduces_grasp_eval(self):
        from virturoid.services.grasp_skill import morph_grasp_rollout
        arm = self._arm()
        r = morph_grasp_rollout(arm, None, 0.44, 0.0)               # policy=None -> the scripted base
        self.assertTrue(r["success"], msg=f"scripted base must grasp+lift the tabletop arm: {r}")
        self.assertGreaterEqual(r["contacts"], 2)                  # >= 2 finger bodies touched the object
        self.assertGreater(r["lifted"], 0.05)

    def test_residual_policy_is_a_clean_superset(self):
        # a zero-effect rollout with a fresh (near-zero output) policy must still run + return the metric shape
        from virturoid.services.grasp_skill import evaluate_morph_grasp
        from virturoid.services.morph_graph import encode_robot
        from virturoid.services.morph_policy import MorphPolicy
        import mujoco
        from virturoid.schemas.scenes import SceneObject
        from virturoid.services.gene_compiler import compile_gene_with_scene
        arm = self._arm()
        cube = SceneObject("box", "cube", (0.44, 0.0, 0.05, 0, 0, 0), mass_kg=0.03)
        fd = encode_robot(mujoco.MjModel.from_xml_string(compile_gene_with_scene(arm, [cube]))).feature_dim
        out = evaluate_morph_grasp(arm, MorphPolicy(fd, seed=0), positions=[(0.44, 0.0)], residual_scale=0.1)
        self.assertIn("success_rate", out)
        self.assertIn("mean_contacts", out)

    def test_grasp_residual_banks_and_recalls_cross_manipulator(self):
        # bank a grasp residual keyed by structural kind + 'grasp'; a manipulator recalls it (the flywheel write/read)
        from virturoid.services.grasp_skill import bank_grasp_residual, grasp_feature_dim, recall_grasp_residual
        from virturoid.services.memory_db import MemoryDB
        from virturoid.services.morph_policy import MorphPolicy
        arm = self._arm()
        fd = grasp_feature_dim(arm)
        with tempfile.TemporaryDirectory() as d:
            with MemoryDB(Path(d) / "m.db") as db:
                self.assertIsNone(recall_grasp_residual(arm, db)[0])               # nothing banked yet
                bank_grasp_residual(MorphPolicy(fd, seed=0), arm, db, models_dir=d,
                                    residual_scale=0.12, fin_residual_scale=0.04, success_rate=0.6)
                pol, sc = recall_grasp_residual(arm, db)
                self.assertIsNotNone(pol)                                          # recalled the residual
                self.assertEqual(pol.feature_dim, fd)                             # feature-dim compatible
                self.assertAlmostEqual(sc["residual_scale"], 0.12)               # scales round-tripped via base_config
                self.assertAlmostEqual(sc["fin_residual_scale"], 0.04)

    def test_acquire_skill_grasp_reuses_banked_residual(self):
        # acquire_skill routes grasp through the LEARNED-policy flywheel; a banked residual that clears target is reused.
        from virturoid.services.grasp_skill import bank_grasp_residual, grasp_feature_dim
        from virturoid.services.memory_db import MemoryDB
        from virturoid.services.morph_policy import MorphPolicy
        from virturoid.services.skill_library import acquire_skill
        arm = self._arm()
        with tempfile.TemporaryDirectory() as d:
            with MemoryDB(Path(d) / "m.db") as db:
                bank_grasp_residual(MorphPolicy(grasp_feature_dim(arm), seed=0), arm, db, models_dir=d, success_rate=0.6)
                r = acquire_skill(arm, "grasp", db=db, models_dir=d, target=-1.0)  # target trivially met -> reuse, no train
                self.assertEqual(r["mode"], "reused")
                self.assertEqual(r["skill_id"], "morph_manipulator_grasp")
                self.assertEqual(r["task_type"], "grasp")


if __name__ == "__main__":
    unittest.main()
