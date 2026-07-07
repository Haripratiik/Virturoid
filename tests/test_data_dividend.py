"""T-B: DataDividendRecord + ledger (the flywheel "what did this run improve" record).

Locks the dossier / Training-Improvement Data-Dividend semantics: measured delta, reuse-only-on-improvement,
direction-aware key metric, permission gating, and a round-trip ledger. Pure/offline (AGENTS.md).
"""
import os
import tempfile
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")

from virturoid.schemas.data_dividend import DataDividendRecord  # noqa: E402
from virturoid.services.data_dividend import (  # noqa: E402
    compute_dividend,
    dividend_summary,
    read_dividends,
    record_dividend,
)


class ComputeDividendTests(unittest.TestCase):
    def test_measured_delta_and_reusable_on_improvement(self):
        rec = compute_dividend(
            run_id="run1", improved_prior_type="skill", improved_prior_ref="grasp_v2",
            before_metrics={"heldout_success": 0.60, "steps": 100000},
            after_metrics={"heldout_success": 0.82, "steps": 90000},
            key_metric="heldout_success")
        self.assertAlmostEqual(rec.measured_delta["heldout_success"], 0.22, places=6)
        self.assertAlmostEqual(rec.measured_delta["steps"], -10000, places=6)
        self.assertTrue(rec.reusable_by_default)               # key metric improved -> reusable

    def test_no_improvement_is_not_reusable(self):
        rec = compute_dividend(
            run_id="run2", improved_prior_type="reward", improved_prior_ref="r1",
            before_metrics={"heldout_success": 0.70}, after_metrics={"heldout_success": 0.65},
            key_metric="heldout_success")
        self.assertFalse(rec.reusable_by_default)               # regressed -> not banked as reusable

    def test_lower_is_better_metric(self):
        # steps-to-gate: a DROP is an improvement.
        rec = compute_dividend(
            run_id="run3", improved_prior_type="transfer_rule", improved_prior_ref="t1",
            before_metrics={"steps_to_gate": 50000}, after_metrics={"steps_to_gate": 12000},
            key_metric="steps_to_gate", higher_is_better=False)
        self.assertTrue(rec.reusable_by_default)
        self.assertEqual(rec.measured_delta["steps_to_gate"], -38000)

    def test_permission_no_reuse_blocks_reusability(self):
        rec = compute_dividend(
            run_id="run4", improved_prior_type="skill", improved_prior_ref="s1",
            before_metrics={"success": 0.1}, after_metrics={"success": 0.9},
            key_metric="success", permission_scope="no_reuse")
        self.assertFalse(rec.reusable_by_default)                # improved, but permission forbids reuse
        self.assertTrue(rec.validate().ok)

    def test_only_shared_numeric_metrics_get_a_delta(self):
        rec = compute_dividend(
            run_id="run5", improved_prior_type="body", improved_prior_ref="b1",
            before_metrics={"a": 1.0, "note": "x"}, after_metrics={"a": 2.0, "b": 5.0, "note": "y"},
            key_metric="a")
        self.assertEqual(set(rec.measured_delta), {"a"})         # 'b' not in before, 'note' not numeric

    def test_validation_flags_unknown_prior_type(self):
        rec = DataDividendRecord(id="dd_x", run_id="r", improved_prior_type="made_up")
        result = rec.validate()
        self.assertTrue(any(i.code == "unknown_prior_type" for i in result.issues))


class LedgerTests(unittest.TestCase):
    def test_record_read_summary_roundtrip(self):
        mem = tempfile.mkdtemp(prefix="dividend_")
        r1 = compute_dividend(run_id="r1", improved_prior_type="skill", improved_prior_ref="s1",
                              before_metrics={"success": 0.5}, after_metrics={"success": 0.8},
                              key_metric="success")
        r2 = compute_dividend(run_id="r2", improved_prior_type="reward", improved_prior_ref="rw1",
                              before_metrics={"success": 0.8}, after_metrics={"success": 0.75},
                              key_metric="success")
        record_dividend(r1, memory_dir=mem)
        record_dividend(r2, memory_dir=mem)
        rows = read_dividends(memory_dir=mem)
        self.assertEqual(len(rows), 2)
        summary = dividend_summary(memory_dir=mem)
        self.assertEqual(summary["total_dividends"], 2)
        self.assertEqual(summary["reusable_by_default"], 1)      # only r1 improved
        self.assertEqual(summary["by_prior_type"], {"skill": 1, "reward": 1})

    def test_missing_ledger_is_empty(self):
        mem = tempfile.mkdtemp(prefix="dividend_empty_")
        self.assertEqual(read_dividends(memory_dir=mem), [])
        self.assertEqual(dividend_summary(memory_dir=mem)["total_dividends"], 0)


if __name__ == "__main__":
    unittest.main()
