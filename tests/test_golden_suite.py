"""Golden suite (plan v2 §5.4): sealed regression cases re-run through the independent verifier; a case below
its sealed floor BLOCKS banking (drift alarm). Injected verify -> deterministic, no physics."""

import tempfile
import unittest
from pathlib import Path

from virturoid.services.golden_suite import (GoldenCase, load_golden_cases, run_golden_suite,
                                             seal_golden_floor)


class GoldenRatchetTests(unittest.TestCase):
    def test_seal_only_ever_raises_and_loads(self):
        # N21: seal a floor for a banked capability; a higher verified value RAISES it, a lower one NEVER lowers it.
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "floors.json")
            f1 = seal_golden_floor("L2_hex_walk", 0.50, metric_key="forward_m", path=p)   # -> 0.9*0.50 = 0.45
            self.assertAlmostEqual(f1, 0.45)
            f2 = seal_golden_floor("L2_hex_walk", 0.80, metric_key="forward_m", path=p)   # higher -> raises to 0.72
            self.assertAlmostEqual(f2, 0.72)
            f3 = seal_golden_floor("L2_hex_walk", 0.30, metric_key="forward_m", path=p)   # lower -> stays 0.72
            self.assertAlmostEqual(f3, 0.72)
            cases = {c.task_id: c for c in load_golden_cases(path=p)}
            self.assertIn("L2_hex_walk", cases)                # the ratcheted case is now protected
            self.assertAlmostEqual(cases["L2_hex_walk"].min_metric, 0.72)
            self.assertIn("L1_quad_walk", cases)               # the base case is never dropped

    def test_load_without_registry_is_base_only(self):
        cases = load_golden_cases(path="build/__nonexistent_floors__.json")
        self.assertTrue(any(c.task_id == "L1_quad_walk" for c in cases))  # base cases always present


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
