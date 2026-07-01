"""Embedding-quality benchmark (Move 2 acceptance criterion): does an embedding retrieve USEFUL neighbors?

The honest gate for a learned ``z_body`` is not "does it agree with the hand-crafted vector" — a valid latent
can capture different structure — but a DOWNSTREAM one: on held-out designs, does k-NN retrieval in the
embedding space recover designs whose success predicts the query's success? (The Theme-3 DoD: a good design
embedding warm-starts by *quality*.) This module is the pure, testable core of that benchmark: given train/test
(vector, success) pairs, predict each test design's success from its k nearest TRAIN neighbors and correlate the
prediction with the truth. A higher correlation = the embedding places similar-success designs nearer = better
for warm-start. Pure-Python (no torch/numpy); the script wires it to the learned latent vs the deterministic
morphology vector on a real labeled dataset.
"""

from __future__ import annotations

import math


def _l2(a: list[float], b: list[float]) -> float:
    m = min(len(a), len(b))
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(m)))


def knn_predict(train_vecs: list[list[float]], train_y: list[float], query: list[float], k: int) -> float:
    """Predict a query's success as the mean success of its ``k`` nearest TRAIN neighbors (L2)."""
    if not train_vecs:
        return 0.0
    order = sorted(range(len(train_vecs)), key=lambda i: _l2(query, train_vecs[i]))
    top = order[:max(1, min(k, len(order)))]
    return sum(train_y[i] for i in top) / len(top)


def pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2:
        return 0.0
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    va = math.sqrt(sum((x - ma) ** 2 for x in a))
    vb = math.sqrt(sum((x - mb) ** 2 for x in b))
    return cov / (va * vb) if va > 0 and vb > 0 else 0.0


def spearman(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2:
        return 0.0

    def _rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for pos, idx in enumerate(order):
            r[idx] = pos
        return r

    ra, rb = _rank(a), _rank(b)
    return pearson([float(x) for x in ra], [float(x) for x in rb])


def knn_success_correlation(train_vecs: list[list[float]], train_y: list[float],
                            test_vecs: list[list[float]], test_y: list[float], *, k: int = 5) -> dict:
    """Held-out embedding quality: correlate k-NN-predicted success (from train neighbors) with the actual
    test success. Returns ``{pearson, spearman, n, k}`` — higher = the embedding is better for warm-start."""
    preds = [knn_predict(train_vecs, train_y, q, k) for q in test_vecs]
    return {"pearson": round(pearson(preds, test_y), 4), "spearman": round(spearman(preds, test_y), 4),
            "n": len(test_y), "k": k}
