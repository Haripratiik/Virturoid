"""Embedding-quality benchmark core (Move 2 acceptance criterion): held-out kNN success correlation."""

import unittest

from virturoid.services.embedding_benchmark import (
    knn_predict, knn_success_correlation, pearson, spearman,
)


class EmbeddingBenchmarkTests(unittest.TestCase):
    def test_knn_predicts_from_nearest_neighbors(self):
        train, y = [[0.0], [1.0], [2.0], [3.0]], [0.0, 0.3, 0.6, 0.9]
        self.assertAlmostEqual(knn_predict(train, y, [2.1], 1), 0.6)      # nearest is 2.0 -> 0.6
        self.assertAlmostEqual(knn_predict(train, y, [1.5], 2), 0.45)     # mean(0.3, 0.6)

    def test_correlation_high_when_embedding_tracks_success(self):
        # success == the coordinate, so kNN in this space recovers it -> high correlation
        train = [[float(i)] for i in range(10)]
        ty = [i / 9 for i in range(10)]
        test = [[i + 0.5] for i in range(9)]
        tesy = [(i + 0.5) / 9 for i in range(9)]
        r = knn_success_correlation(train, ty, test, tesy, k=2)
        self.assertGreater(r["pearson"], 0.9)
        self.assertEqual(r["n"], 9)

    def test_correlation_low_when_embedding_is_uninformative(self):
        # identical vectors -> kNN can't distinguish -> ~no correlation with success
        train, ty = [[1.0]] * 10, [i / 9 for i in range(10)]
        test, tesy = [[1.0]] * 8, [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]
        self.assertLess(abs(knn_success_correlation(train, ty, test, tesy, k=3)["pearson"]), 0.5)

    def test_pearson_and_spearman(self):
        self.assertAlmostEqual(pearson([1, 2, 3], [2, 4, 6]), 1.0)          # perfect linear
        self.assertAlmostEqual(spearman([1, 2, 3], [10, 20, 30]), 1.0)      # perfect monotonic


if __name__ == "__main__":
    unittest.main()
