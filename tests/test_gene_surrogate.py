import unittest

from virturoid.fixtures.gene_library import humanoid_upper_body_gene, tabletop_arm_gene
from virturoid.services.gene_surrogate import (
    FEATURE_NAMES,
    GeneFitnessSurrogate,
    gene_features,
    train_from_rows,
)


class GeneSurrogateTests(unittest.TestCase):
    def test_features_are_fixed_length_and_distinct_per_morphology(self):
        fa = gene_features(tabletop_arm_gene())
        fh = gene_features(humanoid_upper_body_gene())
        self.assertEqual(len(FEATURE_NAMES), len(fa))
        self.assertNotEqual(fa, fh)  # arm vs humanoid have different features

    def test_surrogate_learns_a_monotonic_relationship(self):
        # Synthetic flywheel where success rises with reach — surrogate must recover the trend.
        rows = []
        for reach in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            feats = [5, 3, reach, 1.0, 12.0, 0.35, 0.025]
            rows.append({"features": feats, "success_rate": min(1.0, max(0.0, (reach - 0.2) / 0.5))})
        s = train_from_rows(rows)
        lo = s.predict([5, 3, 0.25, 1.0, 12.0, 0.35, 0.025])
        hi = s.predict([5, 3, 0.65, 1.0, 12.0, 0.35, 0.025])
        self.assertGreater(hi, lo)  # more reach -> higher predicted success

    def test_rank_orders_candidates_as_a_cheap_screen(self):
        rows = [{"features": [5, 3, r, 1.0, 12.0, 0.35, 0.025], "success_rate": r} for r in (0.1, 0.5, 0.9)]
        s = train_from_rows(rows)
        # Build three genes implicitly via features by faking predict inputs through rank:
        ranked = s.rank([tabletop_arm_gene(), humanoid_upper_body_gene()])
        self.assertEqual(2, len(ranked))
        self.assertGreaterEqual(ranked[0][1], ranked[1][1])  # sorted high to low

    def test_untrained_predict_raises(self):
        with self.assertRaises(RuntimeError):
            GeneFitnessSurrogate().predict(tabletop_arm_gene())


if __name__ == "__main__":
    unittest.main()
