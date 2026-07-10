"""The transfer ledger is how the embedding upgrades itself with usage — so its contracts must hold: trials
round-trip into a fit corpus; the ratchet ADOPTS ONLY a proven metric (an unproven fit must write nothing); and
bank_gait(cross_eval=True) densifies the ledger with real physics outcomes."""
import importlib.util
import os
import tempfile
import types
import unittest
from pathlib import Path

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("MUJOCO_GL", "glfw")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


def _db():
    from virturoid.services.memory_db import MemoryDB
    return MemoryDB(db_path=Path(tempfile.mkdtemp(prefix="ledger_")) / "m.db")


def _fake_result(fwd, credible=True):
    return {"survived": True, "credible": credible, "forward": fwd, "height_ratio": 0.8}


class TransferLedgerTests(unittest.TestCase):
    def test_trials_round_trip_into_a_fit_corpus(self):
        from virturoid.services.morphology_composer import compose_robot
        from virturoid.services.transfer_ledger import corpus_from_ledger, record_transfer_trial
        db = _db()
        a = compose_robot("a small quadruped robot dog")
        b = compose_robot("a six-legged hexapod robot")
        gp = {"freq": 1.5, "hip_amp": 0.9, "knee_amp": 1.0, "duty": 0.25, "kp": 32.0, "kd": 1.5}
        for i in range(4):
            record_transfer_trial(db, src_gene=a, dst_gene=b, gait_params=gp, result=_fake_result(0.8))
            record_transfer_trial(db, src_gene=b, dst_gene=a, gait_params=gp,
                                  result=_fake_result(0.05, credible=False))
        corpus = corpus_from_ledger(db, min_trials=8)
        self.assertIsNotNone(corpus)
        self.assertEqual(len(corpus["bodies"]), 2)
        ids = [x["id"] for x in corpus["bodies"]]
        ia, ib = ids.index(a.id), ids.index(b.id)
        self.assertEqual(corpus["transfer"][ia][ib], 1)            # a's gait walked on b
        self.assertEqual(corpus["transfer"][ib][ia], 0)            # b's did not walk on a
        self.assertLess(corpus_from_ledger(db, min_trials=99) or 0, 1)   # thin -> None (falsy)

    def test_ratchet_adopts_only_a_proven_metric(self):
        import virturoid.services.body_metric as bm
        from virturoid.services.morphology_composer import compose_robot
        from virturoid.services.transfer_ledger import ratchet_metric, record_transfer_trial
        db = _db()
        genes = [compose_robot(p) for p in ("a small quadruped robot dog", "a medium quadruped robot",
                                            "a six-legged hexapod robot", "a six-axis robot arm on a table")]
        gp = {"freq": 1.5, "hip_amp": 0.9, "knee_amp": 1.0, "duty": 0.25, "kp": 32.0, "kd": 1.5}
        for s in genes[:3]:
            for d in genes:
                if s is not d:
                    record_transfer_trial(db, src_gene=s, dst_gene=d, gait_params=gp,
                                          result=_fake_result(0.7, credible=(d is not genes[3])))
        # isolate the metric file; the CONTRACT: file exists <=> the fit was proven
        tmp = Path(tempfile.mkdtemp(prefix="metric_")) / "body_metric.json"
        orig = bm.DEFAULT_METRIC_PATH
        bm.DEFAULT_METRIC_PATH = tmp
        bm._cache = None
        bm._cache_mtime = None
        try:
            out = ratchet_metric(db, min_trials=8)
            self.assertIn(out["action"], ("adopted", "kept_baseline"))
            self.assertEqual(out["action"] == "adopted", tmp.exists(),
                             "an unproven metric must never be written (and a proven one must be)")
            # thin ledger -> honest skip
            skinny = ratchet_metric(_db(), min_trials=8)
            self.assertEqual(skinny["action"], "skipped")
        finally:
            bm.DEFAULT_METRIC_PATH = orig
            bm._cache = None
            bm._cache_mtime = None


@unittest.skipUnless(_MUJOCO, "cross-eval replays gaits in MuJoCo")
class BankCrossEvalTests(unittest.TestCase):
    def test_bank_gait_cross_eval_densifies_the_ledger(self):
        from virturoid.services.gait_flywheel import bank_gait
        from virturoid.services.morphology_composer import compose_robot
        from virturoid.services.transfer_ledger import _conn
        db = _db()
        quad = compose_robot("a small quadruped robot dog")
        hexa = compose_robot("a six-legged hexapod robot")
        gp = {"freq": 1.5, "hip_amp": 0.9, "knee_amp": 1.0, "duty": 0.25, "kp": 32.0, "kd": 1.5}
        r = types.SimpleNamespace(best_survived=True, best_forward=1.0, best_credible=True,
                                  best_params=gp, best_height_ratio=0.8)
        self.assertIsNotNone(bank_gait(db, quad, r))               # first bank: no neighbours yet
        self.assertIsNotNone(bank_gait(db, hexa, r, cross_eval=True))   # second: cross-evals vs the quad
        n = _conn(db).execute("SELECT COUNT(*) FROM transfer_trials").fetchone()[0]
        self.assertGreaterEqual(n, 1, "cross_eval should have recorded physics-verified trials")


if __name__ == "__main__":
    unittest.main()
