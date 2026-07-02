"""Plan v3 M4 — the reproducibility capsule. The manifest hash is deterministic (the replication check) and the
report generator is a pure function over the head-to-head result (unit-tests without physics)."""

import unittest

from virturoid.run_benchmark import build_report
from virturoid.services.prereg import build_prereg_manifest


def _synthetic_h2h(with_baseline=False):
    def row(tid, a0, a, ap, b):
        r = {"task": tid, "A_pass": a, "Aplus_pass": ap, "B_pass": b}
        if with_baseline:
            r["A0_pass"] = a0
        return r

    def hon(a0, a, ap, b):
        h = {"A": {"claimed": a, "verified": a, "overclaim": 0},
             "A+": {"claimed": ap, "verified": ap, "overclaim": 0},
             "B": {"claimed": b + 1, "verified": b, "overclaim": 1}}   # B over-claims by 1 (deploy gap)
        if with_baseline:
            h["A0"] = {"claimed": a0, "verified": a0, "overclaim": 0}
        return h

    per_seed = [
        {"rows": [row("L2_hex_walk", 0, 0, 0, 1)], "honesty": hon(0, 0, 0, 1)},
        {"rows": [row("L2_hex_walk", 0, 0, 1, 1)], "honesty": hon(0, 0, 1, 1)},
    ]
    agg = {"harness_delta": {"mean": 0.5, "sem": 0.5, "ci95": 0.98, "iqm": 0.5, "n": 2},
           "transfer_delta": {"mean": 0.5, "sem": 0.5, "ci95": 0.98, "iqm": 0.5, "n": 2},
           "A_solved": {"mean": 0.0, "n": 2}, "Aplus_solved": {"mean": 0.5, "n": 2},
           "B_solved": {"mean": 1.0, "n": 2}}
    return {"seeds": [20260701, 20260702], "n_seeds": 2, "per_seed": per_seed, "aggregate": agg}


class ManifestReproducibilityTests(unittest.TestCase):
    def test_manifest_hash_is_deterministic(self):
        a = build_prereg_manifest(seeds=[1, 2, 3], split="held_out")
        b = build_prereg_manifest(seeds=[1, 2, 3], split="held_out")
        self.assertEqual(a["manifest_hash"], b["manifest_hash"])   # re-run before any edit -> same hash
        self.assertIn("verifier_sha256", a)
        self.assertNotEqual(a["manifest_hash"], "")

    def test_manifest_changes_with_arms(self):
        base = build_prereg_manifest(arms=("A", "A+", "B"), seeds=[1])
        withb = build_prereg_manifest(arms=("A0", "A", "A+", "B"), seeds=[1])
        self.assertNotEqual(base["manifest_hash"], withb["manifest_hash"])   # adding Arm 0 is a different protocol


class ReportTests(unittest.TestCase):
    def test_build_report_three_arm(self):
        prereg = build_prereg_manifest(seeds=[20260701, 20260702], split="held_out")
        md, report = build_report(_synthetic_h2h(), prereg, with_baseline=False)
        self.assertIn("harness_delta", md)
        self.assertIn("L2_hex_walk", md)
        self.assertIn(report["manifest_hash"], md)                 # provenance stamped in the report
        self.assertEqual(report["honesty"]["B"]["overclaim"], 2)   # 1 per seed x 2 seeds
        self.assertIn("B (search+memory)", md)
        self.assertNotIn("A0", report["per_task"]["L2_hex_walk"])  # no baseline column when off

    def test_build_report_with_baseline(self):
        prereg = build_prereg_manifest(arms=("A0", "A", "A+", "B"), seeds=[20260701, 20260702])
        md, report = build_report(_synthetic_h2h(with_baseline=True), prereg, with_baseline=True)
        self.assertIn("baseline_delta", md)
        self.assertIn("Claude+MCP", md)
        self.assertIn("A0", report["honesty"])
        self.assertEqual(report["per_task"]["L2_hex_walk"]["A0"], 0)   # baseline solved it in 0/2 seeds


if __name__ == "__main__":
    unittest.main()
