"""§4.4 — the flywheel-compounding chart turns a run_flywheel report into the moat KPI (banked assets growing,
warm-start rising) with a dependency-free ASCII chart."""

import unittest

from virturoid.services.flywheel_chart import flywheel_compounding_chart, render_ascii_chart

_REPORT = {"cycles": [
    {"cycle": 0, "success_rate": 0.40, "warm_start_available": False, "banked_after": {"designs": 1, "skills": 0}},
    {"cycle": 1, "success_rate": 0.70, "warm_start_available": True,  "banked_after": {"designs": 2, "skills": 1}},
    {"cycle": 2, "success_rate": 0.85, "warm_start_available": True,  "banked_after": {"designs": 3, "skills": 2}},
], "totals": {}, "compounding": True}


class FlywheelChartTests(unittest.TestCase):
    def test_compounding_series_and_kpis(self):
        ch = flywheel_compounding_chart(_REPORT)
        self.assertEqual(ch["n_cycles"], 3)
        self.assertEqual(ch["banked_start"], 1)
        self.assertEqual(ch["banked_end"], 5)                 # 3 designs + 2 skills
        self.assertEqual(ch["banked_growth"], 4)
        self.assertTrue(ch["compounding"])
        self.assertAlmostEqual(ch["warm_start_fraction"], round(2 / 3, 3))
        self.assertIn("COMPOUNDING", ch["headline"])

    def test_flat_run_is_not_compounding(self):
        flat = {"cycles": [
            {"cycle": 0, "success_rate": 0.5, "warm_start_available": False, "banked_after": {"designs": 1}},
            {"cycle": 1, "success_rate": 0.5, "warm_start_available": False, "banked_after": {"designs": 1}},
        ], "compounding": False}
        ch = flywheel_compounding_chart(flat)
        self.assertEqual(ch["banked_growth"], 0)
        self.assertFalse(ch["compounding"])

    def test_ascii_chart_renders_bars(self):
        txt = render_ascii_chart(flywheel_compounding_chart(_REPORT))
        self.assertIn("Flywheel compounding", txt)
        self.assertIn("cycle", txt)
        self.assertIn("#", txt)

    def test_empty_report(self):
        ch = flywheel_compounding_chart({"cycles": []})
        self.assertEqual(ch["n_cycles"], 0)
        self.assertIn("no cycles", ch["headline"])


if __name__ == "__main__":
    unittest.main()
