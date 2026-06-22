"""General body co-design (Theme 3): one CEM routine tunes ANY morphology against its topology-dispatched
task metric (locomotion / navigation / grasp / pick-place), never regressing the baseline body."""

import importlib.util
import unittest

_MUJOCO = importlib.util.find_spec("mujoco") is not None


def _gene(rc, prompt="x"):
    from virturoid.services.morphology_composer import compose_from_spec, morphology_from_requirements
    return compose_from_spec(morphology_from_requirements(0.65, 0.25, prompt=prompt, robot_class=rc))


@unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
class GeneralCoDesignTests(unittest.TestCase):
    def test_codesign_quadruped_locomotion_never_regresses_and_returns_valid_gene(self):
        from virturoid.services.gene_codesign import co_design_general
        r = co_design_general(_gene("quadruped"), "a quadruped that walks", iterations=2, population=4, seed=0)
        self.assertGreaterEqual(r["best_value"], r["baseline_value"] - 1e-6)   # never regress
        self.assertEqual(r["gene"].validate(), [])                            # best variant is buildable
        self.assertIn("length_scale", r["changed"])
        self.assertEqual(len(r["history"]), 3)                                # baseline + 2 iterations

    def test_codesign_generalizes_to_a_manipulator(self):
        from virturoid.services.gene_codesign import co_design_general
        r = co_design_general(_gene("manipulator", "sort blocks"), "sort blocks into bins",
                              iterations=1, population=3, seed=0)
        self.assertGreaterEqual(r["best_value"], r["baseline_value"] - 1e-6)
        self.assertEqual(r["gene"].validate(), [])


if __name__ == "__main__":
    unittest.main()
