"""Pre-registration manifest (plan v2 §3.7/M5): freezes tasks+seeds+budgets+GRADER-CODE hashes + declared
predictions into a deterministic, hashable manifest for a credible head-to-head."""

import json
import unittest

from virturoid.services.prereg import build_prereg_manifest, write_prereg


class PreregTests(unittest.TestCase):
    def test_manifest_is_complete_and_deterministic(self):
        m1 = build_prereg_manifest()
        for k in ("tasks", "task_registry_sha256", "verifier_sha256", "arms_sha256", "seeds",
                  "budget_per_arm", "metrics", "delta_formulas", "predictions", "manifest_hash"):
            self.assertIn(k, m1)
        self.assertGreaterEqual(m1["seed_count"], 3)          # MLE-bench canon: >=3 seeds
        self.assertEqual(len(m1["verifier_sha256"]), 64)      # the grader is hashed (versioned)
        m2 = build_prereg_manifest()
        self.assertEqual(m1["manifest_hash"], m2["manifest_hash"])   # deterministic -> replication check

    def test_declares_arms_and_predictions(self):
        m = build_prereg_manifest()
        self.assertEqual(set(m["arms"]), {"A", "A+", "B"})    # the three ablation arms
        self.assertIn("transfer_delta", m["predictions"])
        self.assertIn("honesty_delta", m["metrics"])

    def test_write_prereg_roundtrips(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "prereg.json"
            h = write_prereg(str(p))
            loaded = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(loaded["manifest_hash"], h)      # the written hash matches the manifest


if __name__ == "__main__":
    unittest.main()
