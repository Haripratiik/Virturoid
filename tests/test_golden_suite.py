"""Golden suite (plan v2 §5.4): sealed regression cases re-run through the independent verifier; a case below
its sealed floor BLOCKS banking (drift alarm). Injected verify -> deterministic, no physics."""

import unittest

from virturoid.services.golden_suite import GoldenCase, run_golden_suite


class GoldenSuiteTests(unittest.TestCase):
    def test_all_above_floor_passes(self):
        cases = [GoldenCase("L1_quad_walk", min_metric=0.0)]
        fake = lambda tid, gene, pol, **k: {"metrics": {"forward_m": 0.3}, "verified_pass": False}
        rep = run_golden_suite(cases, gene_for=lambda t: None, verify=fake)
        self.assertTrue(rep["passed"])
        self.assertEqual(rep["regressions"], [])

    def test_regression_blocks(self):
        cases = [GoldenCase("L1_quad_walk", min_metric=0.0)]
        # a code change made the quad go BACKWARD -> below the 0.0 floor -> regression -> block banking
        fake = lambda tid, gene, pol, **k: {"metrics": {"forward_m": -0.2}, "verified_pass": False}
        rep = run_golden_suite(cases, gene_for=lambda t: None, verify=fake)
        self.assertFalse(rep["passed"])
        self.assertEqual(len(rep["regressions"]), 1)
        self.assertEqual(rep["regressions"][0]["task"], "L1_quad_walk")

    def test_default_cases_exist(self):
        from virturoid.services.golden_suite import GOLDEN_CASES
        self.assertTrue(GOLDEN_CASES)                     # a non-empty sealed set ships by default


if __name__ == "__main__":
    unittest.main()
