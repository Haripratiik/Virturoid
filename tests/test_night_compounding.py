"""Plan v3 M3 — multi-night compounding measurement. Pure aggregation over night snapshots; no GPU."""

import os
import tempfile
import unittest

from virturoid.services.night_compounding import (
    append_night, compounding_series, load_nights, night_snapshot, render_ascii)


class NightCompoundingTests(unittest.TestCase):
    def test_snapshot_from_report_dict(self):
        rep = {"night": {"banked": 2, "novel": 1, "qd": {"filled": 3, "coverage": 0.25, "annecs_v": 2,
                                                         "qd_score": 1.4}}}
        s = night_snapshot(rep, night=1)
        self.assertEqual(s["banked"], 2)
        self.assertEqual(s["coverage"], 0.25)
        self.assertEqual(s["annecs_v"], 2)

    def test_snapshot_from_object(self):
        class _N:                                            # run_night returns counters as an object
            banked, novel = 1, 1
            qd = {"filled": 1, "coverage": 0.1, "annecs_v": 1, "qd_score": 0.5}
        s = night_snapshot({"night": _N()}, night=2)
        self.assertEqual(s["banked"], 1)
        self.assertEqual(s["coverage"], 0.1)

    def test_compounding_series_grows(self):
        snaps = [{"night": 1, "banked": 1, "coverage": 0.10, "annecs_v": 1},
                 {"night": 2, "banked": 2, "coverage": 0.25, "annecs_v": 2},
                 {"night": 3, "banked": 1, "coverage": 0.30, "annecs_v": 1}]
        c = compounding_series(snaps)
        self.assertEqual([s["cumulative_banked"] for s in c["series"]], [1, 3, 4])
        self.assertTrue(c["compounding"])                    # cumulative banked accumulates
        self.assertTrue(c["banked_compounds"])
        self.assertTrue(c["coverage_compounds"])             # coverage also grows (shared-archive case)
        self.assertIn("COMPOUNDING", render_ascii(c))

    def test_coverage_regression_flagged_but_banked_compounds(self):
        # coverage per-night oscillates when the QD archive isn't SHARED across nights; "compounding" tracks the
        # always-valid BANKED signal (capabilities accumulate in shared memory), coverage is reported separately.
        snaps = [{"night": 1, "banked": 1, "coverage": 0.30}, {"night": 2, "banked": 1, "coverage": 0.20}]
        c = compounding_series(snaps)
        self.assertTrue(c["coverage_regressed"])
        self.assertFalse(c["coverage_compounds"])            # coverage did NOT grow
        self.assertTrue(c["compounding"])                    # but cumulative banked (1->2) DID accumulate

    def test_append_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "nights.jsonl")
            append_night({"night": {"banked": 1, "qd": {"coverage": 0.1, "annecs_v": 1}}}, log)
            append_night({"night": {"banked": 2, "qd": {"coverage": 0.2, "annecs_v": 1}}}, log)
            nights = load_nights(log)
            self.assertEqual([n["night"] for n in nights], [1, 2])   # auto-numbered sequentially
            self.assertEqual(compounding_series(nights)["cumulative_banked"], 3)

    def test_empty_is_safe(self):
        c = compounding_series([])
        self.assertEqual(c["n_nights"], 0)
        self.assertFalse(c["compounding"])
        self.assertIn("no nights", render_ascii(c))

    def test_run_night_series_driver(self):
        # the multi-night DRIVER runs n nights (injected run_one) + records each -> the accumulated series.
        from virturoid.services.night_compounding import run_night_series
        reports = iter([
            {"night": {"banked": 1, "qd": {"coverage": 0.10, "annecs_v": 1}}},
            {"night": {"banked": 2, "qd": {"coverage": 0.22, "annecs_v": 2}}},
            {"night": {"banked": 1, "qd": {"coverage": 0.30, "annecs_v": 1}}},
        ])
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "series.jsonl")
            c = run_night_series(lambda: next(reports), 3, log_path=log)
            self.assertEqual(c["n_nights"], 3)
            self.assertEqual(c["cumulative_banked"], 4)      # 1+2+1
            self.assertTrue(c["compounding"])                # grows + coverage climbs
            self.assertEqual(len(load_nights(log)), 3)       # each night recorded


if __name__ == "__main__":
    unittest.main()
