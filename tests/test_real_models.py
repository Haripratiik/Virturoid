"""STAGE 1: real production models (MuJoCo Menagerie) load, simulate, flow through OUR learning interface,
and route correctly through the unified build entry. Skipped if robot_descriptions isn't installed."""

import importlib.util
import unittest

_MUJOCO = importlib.util.find_spec("mujoco") is not None
_RD = importlib.util.find_spec("robot_descriptions") is not None


@unittest.skipUnless(_MUJOCO and _RD, "MuJoCo + robot_descriptions required.")
class RealModelTests(unittest.TestCase):
    def test_real_quadruped_loads_with_real_assets(self):
        from virturoid.services.real_model_library import load_real_model
        r = load_real_model("a quadruped walking robot")
        self.assertTrue(r["ok"], r.get("note"))
        self.assertGreater(r["meshes"], 0)         # REAL meshes, not primitives
        self.assertGreater(r["actuated"], 0)       # REAL actuators
        self.assertTrue(r["free_base"])            # can locomote

    def test_portable_mjcf_sims_and_uses_learning_interface(self):
        import mujoco
        import numpy as np

        from virturoid.services.morph_graph import encode_robot
        from virturoid.services.morph_policy import MorphPolicy
        from virturoid.services.real_model_library import real_model_mjcf
        info = real_model_mjcf("a quadruped walking robot")
        self.assertTrue(info["ok"])
        self.assertIn('type="plane"', info["mjcf"])            # ground ensured
        m = mujoco.MjModel.from_xml_string(info["mjcf"])       # portable string resolves meshes
        self.assertGreater(m.nmesh, 0)
        graph = encode_robot(m)                                # flows through OUR morph-agnostic interface
        d = mujoco.MjData(m); mujoco.mj_resetData(m, d); mujoco.mj_forward(m, d)
        pol = MorphPolicy(graph.feature_dim, seed=0)
        for _ in range(20):
            graph.apply(m, d, pol.act(graph.observe(m, d)), alpha=0.3); mujoco.mj_step(m, d)
        self.assertTrue(np.all(np.isfinite(d.qpos)))           # the policy drives a REAL robot stably

    def test_build_robot_generates_original_by_default_real_is_opt_in(self):
        # The product CREATES original designs: build_robot generates by default and must NEVER copy a real
        # production model for "a humanoid" (that would be retrieval, not design). Real models are an OPT-IN
        # reference/benchmark only (prefer_real=True).
        from virturoid.services.robot_factory import build_robot
        gen = build_robot("a humanoid robot")
        self.assertEqual(gen["kind"], "procedural")            # default -> ORIGINAL generated design
        self.assertIn("gene", gen)
        self.assertNotIn("mjcf", gen)                          # not a copied real model
        novel = build_robot("a snake-like serpentine slithering robot")
        self.assertEqual(novel["kind"], "procedural")
        self.assertIn("gene", novel)
        # opt-in reference path still works (for "import/benchmark a real robot")
        ref = build_robot("a humanoid robot", prefer_real=True, llm=None)
        self.assertEqual(ref["kind"], "real")
        self.assertIn("mjcf", ref)


if __name__ == "__main__":
    unittest.main()
