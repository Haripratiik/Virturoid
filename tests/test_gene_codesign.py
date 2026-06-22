import importlib.util
import unittest

from virturoid.fixtures.gene_library import tabletop_arm_gene

_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
class GeneCoDesignTests(unittest.TestCase):
    def test_co_design_returns_valid_gene_and_does_not_regress(self):
        # Tiny budget for speed; the contract: returns a valid variant gene + controller
        # params, and the reported best is no worse than the baseline (CEM keeps the best).
        from virturoid.services.gene_build import generate_reachable_scenes
        from virturoid.services.gene_codesign import co_optimize_gene

        g = tabletop_arm_gene()
        scenes = generate_reachable_scenes(g, count=2)
        res = co_optimize_gene(g, scenes, iterations=1, population=2, elite=1)
        self.assertEqual([], res["gene"].validate())          # the co-designed gene is buildable
        self.assertIn("kp", res["controller_params"])
        self.assertGreaterEqual(res["success_rate"], res["baseline_success"])
        # The variant scales body params but stays the same kinematic structure.
        self.assertEqual(len(g.segments), len(res["gene"].segments))


if __name__ == "__main__":
    unittest.main()
