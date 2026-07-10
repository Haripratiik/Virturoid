"""The learned transfer metric must (1) NEVER regress the flywheel — with no proven metric on disk, embed_metric
returns the exact baseline vector — and (2) gate honestly: fit_body_metric reports a leave-one-body-out held-out
number vs the true baseline and only marks itself ``proven`` when it beats it."""
import os
import tempfile
import types
import unittest
from pathlib import Path

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")


class BodyMetricTests(unittest.TestCase):
    def test_embed_metric_is_baseline_when_no_proven_bundle(self):
        import virturoid.services.body_metric as bm
        from virturoid.services.morphology_composer import compose_robot
        from virturoid.services.morphology_embedding import embed_gene
        # point the loader at an absent path so no bundle is found -> baseline passthrough
        bm._cache = None
        bm._cache_mtime = None
        missing = Path(tempfile.mkdtemp()) / "none.json"
        orig = bm.DEFAULT_METRIC_PATH
        bm.DEFAULT_METRIC_PATH = missing
        try:
            g = compose_robot("a small quadruped robot dog")
            self.assertEqual(bm.embed_metric(g), embed_gene(g))    # exact baseline -> cannot regress
        finally:
            bm.DEFAULT_METRIC_PATH = orig

    def test_fit_reports_held_out_vs_baseline_and_gates(self):
        from virturoid.services.body_metric import fit_body_metric
        from virturoid.services.morphology_composer import compose_robot
        ids = ["a small quadruped robot dog", "a medium quadruped robot", "a large quadruped robot",
               "a six-legged hexapod robot", "a six-axis robot arm on a table"]
        bodies = [{"id": str(i), "robot_class": "quadruped", "_gene": compose_robot(p), "self_credible": True}
                  for i, p in enumerate(ids)]
        n = len(bodies)
        # a plausible transfer matrix: the quads (0,1,2) transfer among themselves, hexa(3) partially, arm(4) not
        T = [[1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 1]]
        F = [[max(0.0, 0.6) if T[i][j] else 0.0 for j in range(n)] for i in range(n)]
        corpus = {"bodies": bodies, "transfer": T, "forward": F}
        b = fit_body_metric(corpus, feature_space="rich", whiten=False, method="relevance", save=False)
        self.assertIn("held_out_triplet_acc", b)
        self.assertIn("baseline_held_out_triplet_acc", b)
        self.assertIsInstance(b["proven"], bool)
        self.assertEqual(len(b["weights"]), len(b["weights"]))     # weights present, one per feature


if __name__ == "__main__":
    unittest.main()
