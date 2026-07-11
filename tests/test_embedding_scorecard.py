"""P6 — VIRT-Bench-Transfer as a STANDING CI gate. The moat's headline claim (distance predicts physics-verified
transfer) is measured on a committed physics-real fixture corpus every run, so a regression toward chance BLOCKS.
Nobody else benchmarks controller-transfer-between-morphologies (Dec-2025 co-design survey), so this is our
regression gate and a future public artifact."""
import os
import unittest
from pathlib import Path

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")

_FIXTURE = Path(__file__).parent / "fixtures" / "transfer_corpus_fixture.json"


class EmbeddingScorecardTests(unittest.TestCase):
    def test_shipped_embedding_beats_the_transfer_floor(self):
        from virturoid.services.embedding_scorecard import TRIPLET_FLOOR, embedding_transfer_scorecard
        sc = embedding_transfer_scorecard(_FIXTURE)
        self.assertEqual(sc["status"], "scored")
        self.assertIsNotNone(sc["coarse_triplet_ranking_acc"])
        # THE GATE: distance must predict transfer clearly above chance on the physics-verified corpus
        self.assertGreaterEqual(sc["coarse_triplet_ranking_acc"], TRIPLET_FLOOR)
        self.assertTrue(sc["gate_pass"])
        self.assertIn(sc["embedding_active"], ("baseline_29d", "learned_metric"))

    def test_missing_corpus_is_an_honest_skip_not_a_failure(self):
        from virturoid.services.embedding_scorecard import embedding_transfer_scorecard
        sc = embedding_transfer_scorecard("build/data/__does_not_exist__.json")
        self.assertEqual(sc["status"], "no_corpus")
        self.assertTrue(sc["gate_pass"])                         # absent corpus doesn't fail CI; it asks to build one


if __name__ == "__main__":
    unittest.main()
