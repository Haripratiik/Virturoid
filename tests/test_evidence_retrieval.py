"""T-C/T-D: evidence-bundle retrieval (R4) + universal transfer screening (R3).

retrieve_evidence turns skill recall into ranked, audit-ready evidence (held-out success, failure clusters,
IO compatibility, match kind). screen_transfer generalizes zero-shot transfer screening to any task and rejects
negative transfer. Pure/offline (AGENTS.md).
"""
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")

from virturoid.services.evidence_retrieval import retrieve_evidence, screen_transfer  # noqa: E402


class _Cand:
    def __init__(self, id_, score):
        self.id = id_
        self._score = score


class EvidenceRetrievalTests(unittest.TestCase):
    def _db(self):
        from virturoid.services.memory_db import MemoryDB
        db = MemoryDB(db_path=Path(tempfile.mkdtemp(prefix="mem_")) / "m.db")
        # two exact class+task matches (IO-compatible), one same-class other-task, one cross-class same-task.
        db.record_skill("s_quad_walk", "quadruped", "locomotion", success_rate=0.8,
                        obs_dim=20, act_dim=8, region={"x": [0, 1]}, reward_spec={"forward": 1.0})
        db.record_skill("s_quad_walk_hi", "quadruped", "locomotion", success_rate=0.9,
                        obs_dim=20, act_dim=8, region={"x": [0, 2]})
        db.record_skill("s_quad_jump", "quadruped", "jump", success_rate=0.6, obs_dim=20, act_dim=8)
        db.record_skill("s_hex_walk", "hexapod", "locomotion", success_rate=0.7, obs_dim=30, act_dim=12)
        db.record_failures(1, "quadruped", "locomotion",
                           [{"label": "fell_over", "count": 3}, {"label": "upright_but_slow", "count": 1}])
        return db

    def test_best_evidence_ranks_first_with_io_and_kind(self):
        db = self._db()
        bundles = retrieve_evidence(db, "quadruped", "locomotion", obs_dim=20, act_dim=8, k=5)
        self.assertEqual(bundles[0].candidate_skill_id, "s_quad_walk_hi")   # highest held-out success
        self.assertEqual(bundles[0].match_kind, "exact_io")
        self.assertTrue(bundles[0].io_compatible)
        self.assertEqual(bundles[0].heldout_success, 0.9)
        # scores are monotonically non-increasing (ranked by evidence).
        scores = [b.evidence_score for b in bundles]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_failure_clusters_and_reward_provenance_attached(self):
        db = self._db()
        top = retrieve_evidence(db, "quadruped", "locomotion", obs_dim=20, act_dim=8)[0]
        self.assertIn("fell_over", top.failure_clusters)          # recall returns evidence, not just a policy
        walk = next(b for b in retrieve_evidence(db, "quadruped", "locomotion")
                    if b.candidate_skill_id == "s_quad_walk")
        self.assertEqual(walk.reward_spec_ref, {"forward": 1.0})  # reward provenance carried

    def test_cross_class_candidate_included_but_lower(self):
        db = self._db()
        bundles = retrieve_evidence(db, "quadruped", "locomotion", obs_dim=20, act_dim=8, k=10)
        kinds = {b.candidate_skill_id: b.match_kind for b in bundles}
        self.assertEqual(kinds["s_hex_walk"], "cross_class")
        self.assertEqual(kinds["s_quad_jump"], "same_class")
        # the cross-class hexapod ranks below the exact quadruped walkers.
        ids = [b.candidate_skill_id for b in bundles]
        self.assertLess(ids.index("s_quad_walk_hi"), ids.index("s_hex_walk"))

    def test_no_match_returns_empty(self):
        db = self._db()
        self.assertEqual(retrieve_evidence(db, "snake", "swim"), [])


class TransferScreeningTests(unittest.TestCase):
    def test_rejects_negative_transfer_and_ranks(self):
        cands = [_Cand("a", 0.5), _Cand("b", -0.1), _Cand("c", 0.3)]
        out = screen_transfer(cands, screen_fn=lambda c: c._score, reject_below=0.0)
        self.assertEqual(out["best"]["candidate"], "a")           # highest zero-shot score
        self.assertEqual(len(out["survivors"]), 2)               # a, c
        self.assertEqual(out["negative_transfer_rejected"], 1)   # b (negative transfer) rejected pre-training
        self.assertEqual([s["candidate"] for s in out["survivors"]], ["a", "c"])

    def test_erroring_candidate_is_rejected_not_fatal(self):
        def screen(c):
            if c.id == "bad":
                raise RuntimeError("rollout blew up")
            return c._score

        out = screen_transfer([_Cand("ok", 0.4), _Cand("bad", 0.0)], screen_fn=screen)
        self.assertEqual(out["best"]["candidate"], "ok")
        self.assertEqual(out["negative_transfer_rejected"], 1)   # the erroring candidate is rejected

    def test_all_rejected_has_no_best(self):
        out = screen_transfer([_Cand("a", -1.0)], screen_fn=lambda c: c._score)
        self.assertIsNone(out["best"])


if __name__ == "__main__":
    unittest.main()
