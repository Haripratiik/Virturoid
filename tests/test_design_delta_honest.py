"""P2 — the design flywheel recorded `best - baseline` (the search's gain over its OWN start, ~0.008) under
kind='warm_start', over-claiming a warm-vs-cold value it never measured (ISSUES E5/E6). Fix: default records the
gain under an honest kind that does NOT imply warm-vs-cold; the TRUE warm-start delta (warm_best - cold_best) is
recorded only when measure_delta runs the cold reference. Mocked co-design keeps this a fast unit test."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")


class DesignDeltaHonestTests(unittest.TestCase):
    def _setup(self):
        from virturoid.services.memory_db import MemoryDB
        from virturoid.services.morphology_composer import compose_robot
        from virturoid.services.task_matched_eval import robot_kind
        db = MemoryDB(db_path=Path(tempfile.mkdtemp(prefix="dflyw_")) / "m.db")
        gene = compose_robot("a small quadruped robot dog")
        task = robot_kind(gene)
        # a warm prior so co_design_with_memory takes the warm-started branch
        db.record_run(prompt="prior", robot_class=gene.robot_class, task_type=task,
                      converged_design={"seg": 1}, success_rate=0.7, species="quad.prior")
        return db, gene

    def _fake_codesign(self):
        def fake(gene, prompt, **kw):
            if kw.get("warm_start") is None:                     # the COLD reference
                return {"changed": {}, "best_value": 0.60, "baseline_value": 0.60}
            return {"changed": {"x": 1}, "best_value": 0.90, "baseline_value": 0.89}   # the warm run
        return fake

    def test_default_records_honest_search_gain_not_warmstart(self):
        import virturoid.services.gene_codesign as gc
        from virturoid.services.design_flywheel import co_design_with_memory
        db, gene = self._setup()
        with mock.patch.object(gc, "co_design_general", self._fake_codesign()):
            r = co_design_with_memory(gene, "a small quadruped robot dog", db, measure_delta=False)
        self.assertTrue(r["warm_started"])
        self.assertAlmostEqual(r["provenance_delta"], 0.01, places=3)    # search gain 0.90-0.89, honestly labelled
        self.assertIsNone(r["warmstart_vs_cold"])                        # NOT claimed
        kinds = [row["kind"] for row in db.conn.execute("SELECT kind FROM provenance").fetchall()]
        self.assertIn("design_search_gain", kinds)
        self.assertNotIn("warm_start", kinds)                            # no misleading warm_start edge

    def test_measure_delta_records_true_warm_vs_cold(self):
        import virturoid.services.gene_codesign as gc
        from virturoid.services.design_flywheel import co_design_with_memory
        db, gene = self._setup()
        with mock.patch.object(gc, "co_design_general", self._fake_codesign()):
            r = co_design_with_memory(gene, "a small quadruped robot dog", db, measure_delta=True)
        self.assertAlmostEqual(r["warmstart_vs_cold"], 0.30, places=3)   # 0.90 (warm) - 0.60 (cold): the REAL value
        self.assertAlmostEqual(r["provenance_delta"], 0.30, places=3)
        rows = db.conn.execute("SELECT kind, delta, meta FROM provenance").fetchall()
        ws = [row for row in rows if row["kind"] == "warm_start"]
        self.assertTrue(ws)
        self.assertAlmostEqual(ws[0]["delta"], 0.30, places=3)
        self.assertIn("warm_vs_cold", ws[0]["meta"])                     # provenance says HOW it was measured


if __name__ == "__main__":
    unittest.main()
