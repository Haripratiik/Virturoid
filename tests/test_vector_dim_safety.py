"""Wave 4 — embedding plumbing correctness. Two latent bugs the audit flagged: (1) nearest() SILENTLY
zip-truncated a mismatched-dim vector into a bogus similarity; (2) the skill sub-space rode on the metric-aware
embed_body, so a future metric adoption (different dim) would desync it. Pin both: mismatched vectors are SKIPPED
(not truncated), and the skill body block stays the fixed 29-D baseline regardless of the metric."""
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")


class VectorDimSafetyTests(unittest.TestCase):
    def _vm(self):
        from virturoid.services.memory_db import MemoryDB
        from virturoid.services.robotics_vector_memory import RoboticsVectorMemory
        db = MemoryDB(db_path=Path(tempfile.mkdtemp(prefix="dim_")) / "m.db")
        return RoboticsVectorMemory(db)

    def test_nearest_skips_mismatched_dim_instead_of_truncating(self):
        from virturoid.services.robotics_vector_memory import BODY
        vm = self._vm()
        vm.upsert(BODY, "same_dim", [1.0, 0.0, 0.0])
        vm.upsert(BODY, "wrong_dim", [1.0, 0.0])                  # a stale/other-version vector (dim 2 vs 3)
        hits = vm.nearest(BODY, [1.0, 0.0, 0.0], k=5)
        ids = [h["obj_id"] for h in hits]
        self.assertIn("same_dim", ids)
        self.assertNotIn("wrong_dim", ids)                       # skipped, NOT truncated into a fake 1.0 match

    def test_skill_body_block_is_metric_independent_and_fixed_dim(self):
        from virturoid.services.morphology_composer import compose_robot
        from virturoid.services.morphology_embedding import FEATURE_NAMES, embed_gene
        from virturoid.services.robotics_vector_memory import embed_skill
        g = compose_robot("a small quadruped robot dog")
        z = embed_skill("locomotion quadruped", g, success_rate=0.9)
        # z = baseline body (len FEATURE_NAMES) + text (64) + 1 fingerprint — a fixed, metric-independent length
        self.assertEqual(len(z), len(FEATURE_NAMES) + 64 + 1)
        # and the gene-absent skill has the SAME total length (so they stay comparable)
        z0 = embed_skill("locomotion quadruped", None, success_rate=0.9)
        self.assertEqual(len(z), len(z0))
        _ = embed_gene(g)                                         # baseline embed still importable/used


if __name__ == "__main__":
    unittest.main()
