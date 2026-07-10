"""The embedding eval harness is the SHIP-GATE for the moat, so it must itself be trustworthy: a PERFECT embedding
(transferring pairs close, non-transferring far) must score triplet_ranking_acc == 1.0, and an ADVERSARIAL one
(the opposite) must score 0.0. This pins the metric's meaning so a later 'improvement' can't game it."""
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")


def _mini_corpus():
    """3 real genes A,B,C with a hand-set transfer matrix: A's gait transfers to B (close) but NOT C."""
    from virturoid.services.morphology_composer import compose_robot
    ga = compose_robot("a small quadruped robot dog")
    gb = compose_robot("a large quadruped robot")
    gc = compose_robot("a six-axis robot arm on a table")
    bodies = [
        {"id": "A", "robot_class": "quadruped", "_gene": ga, "self_credible": True},
        {"id": "B", "robot_class": "quadruped", "_gene": gb, "self_credible": True},
        {"id": "C", "robot_class": "manipulator", "_gene": gc, "self_credible": True},
    ]
    # A->B transfers (1), A->C does not (0); B,C sources too (self=1). rows: [A,B,C]
    T = [[1, 1, 0], [1, 1, 0], [0, 0, 1]]
    F = [[0.8, 0.7, 0.0], [0.6, 0.9, 0.0], [0.0, 0.0, 0.5]]
    return {"bodies": bodies, "transfer": T, "forward": F}


class EmbeddingEvalTests(unittest.TestCase):
    def test_perfect_embedding_scores_one_adversarial_scores_zero(self):
        from virturoid.services.embedding_eval import evaluate_embedding
        corpus = _mini_corpus()
        # perfect: A,B identical direction (close), C orthogonal (far) -> distance predicts transfer exactly
        vecs = {"A": [1.0, 0.0], "B": [0.99, 0.01], "C": [0.0, 1.0]}
        good = evaluate_embedding(lambda g: vecs[_idof(corpus, g)], corpus)
        self.assertEqual(good["triplet_ranking_acc"], 1.0)         # every (A, B+, C-) ranked correctly
        self.assertEqual(good["precision_at"][1], 1.0)
        # adversarial: put the NON-transferring C close to A and the transferring B far -> must score 0
        adv = {"A": [1.0, 0.0], "B": [0.0, 1.0], "C": [0.99, 0.01]}
        bad = evaluate_embedding(lambda g: adv[_idof(corpus, g)], corpus)
        self.assertEqual(bad["triplet_ranking_acc"], 0.0)
        self.assertLess(bad["precision_at"][1], good["precision_at"][1])

    def test_metrics_present_and_typed(self):
        from virturoid.services.embedding_eval import evaluate_embedding
        corpus = _mini_corpus()
        m = evaluate_embedding(lambda g: [1.0, 0.0], corpus)       # degenerate embed -> no crash, defined keys
        for key in ("triplet_ranking_acc", "precision_at", "within_legged_cos", "kendall_tau_prox_forward"):
            self.assertIn(key, m)


def _idof(corpus, gene):
    for b in corpus["bodies"]:
        if b["_gene"] is gene:
            return b["id"]
    raise KeyError("gene not in corpus")


if __name__ == "__main__":
    unittest.main()
