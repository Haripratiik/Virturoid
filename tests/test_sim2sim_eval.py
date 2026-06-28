"""§4.2 — sim-to-sim transfer measurement (SIMPLER-style Pearson r + MMRV): the design-stage "Evaluating Bits,
not Atoms" instrument. Metrics are pure functions, unit-tested on synthetic scores."""

import unittest

from virturoid.services.sim2sim_eval import mmrv, pearson_r, run_sim2sim, transfer_report


class MetricTests(unittest.TestCase):
    def test_pearson(self):
        self.assertAlmostEqual(pearson_r([1, 2, 3, 4], [2, 4, 6, 8]), 1.0, places=6)     # perfectly correlated
        self.assertAlmostEqual(pearson_r([1, 2, 3, 4], [4, 3, 2, 1]), -1.0, places=6)    # perfectly inverted
        self.assertEqual(pearson_r([1, 1, 1], [1, 2, 3]), 0.0)                           # constant -> degenerate
        self.assertEqual(pearson_r([1], [1]), 0.0)                                       # < 2 points

    def test_mmrv_zero_when_ranking_holds(self):
        self.assertEqual(mmrv([3, 2, 1], [6, 4, 2]), 0.0)          # same order, no violation
        self.assertEqual(mmrv([1, 2, 3], [1, 2, 3]), 0.0)

    def test_mmrv_flags_inversion(self):
        # nominal says A(10) >= B(9) but perturbed says B(5) > A(1): A's violation = 5 - 1 = 4; B's = 0; mean = 2
        self.assertAlmostEqual(mmrv([10, 9], [1, 5]), 2.0)

    def test_transfer_report_verdict(self):
        strong = transfer_report([1, 2, 3, 4], [1.0, 2.1, 2.9, 4.0])
        self.assertGreaterEqual(strong.pearson_r, 0.99)
        self.assertEqual(strong.mmrv, 0.0)
        self.assertIn("strong", strong.verdict)
        weak = transfer_report([1, 2, 3, 4], [4, 3, 2, 1])
        self.assertLess(weak.pearson_r, 0.0)
        self.assertIn("weak", weak.verdict)
        d = strong.to_dict()
        self.assertEqual(d["n"], 4)
        self.assertEqual(len(d["candidates"]), 4)
        self.assertIn("pearson_r", d)

    def test_run_sim2sim_drives_evaluators(self):
        res = run_sim2sim([1, 2, 3], eval_nominal=lambda x: x, eval_perturbed=lambda x: x * 0.9,
                          label_fn=lambda x: f"c{x}")
        self.assertEqual(res.labels, ["c1", "c2", "c3"])
        self.assertGreaterEqual(res.pearson_r, 0.99)              # a scaled copy is perfectly rank-preserving
        self.assertEqual(res.mmrv, 0.0)


if __name__ == "__main__":
    unittest.main()
