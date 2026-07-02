"""Plan v3 M5 — sim2real reality-gap report. Measures nominal vs a held-out (wider-than-training) DR
distribution and reports the transfer/robustness numbers. Deterministic; CPU MuJoCo."""

import unittest

from virturoid.services.morph_policy import CPG_DEFAULT
from virturoid.services.sim2real import HELD_OUT_DR, sim2real_transfer_report
from virturoid.services.steerable_body import steerable_quadruped
from virturoid.services.virt_bench_arms import _zero_policy_with_cpg


class Sim2RealTests(unittest.TestCase):
    def test_report_structure_and_bounds(self):
        gene = steerable_quadruped(n_legs=4)
        pol = _zero_policy_with_cpg(gene, CPG_DEFAULT)          # a walkable CPG quad -> nonzero nominal
        r = sim2real_transfer_report(gene, pol, n=4, steps=150, seed=0)
        for k in ("nominal", "held_out_dr", "transfer_survival", "forward_retention", "reality_gap_m"):
            self.assertIn(k, r)
        self.assertTrue(0.0 <= r["transfer_survival"] <= 1.0)
        self.assertIsInstance(r["reality_gap_m"], float)
        # the held-out distribution is WIDER than the training DR (probes UNMODELED dynamics)
        self.assertEqual(r["held_out_dr"]["distribution"]["friction"], HELD_OUT_DR["friction"])
        self.assertGreater(HELD_OUT_DR["gain"], 0.15)          # wider than recipe_robustness's training default

    def test_deterministic(self):
        gene = steerable_quadruped(n_legs=4)
        a = sim2real_transfer_report(gene, None, n=3, steps=120, seed=1)
        b = sim2real_transfer_report(gene, None, n=3, steps=120, seed=1)
        self.assertEqual(a["transfer_survival"], b["transfer_survival"])
        self.assertEqual(a["reality_gap_m"], b["reality_gap_m"])


if __name__ == "__main__":
    unittest.main()
