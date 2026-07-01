"""QD archive (WS2/N2): MAP-Elites cells keep the elite per behavior niche; the dashboard reads ANNECS-V,
QD-score, and coverage off it. Drives the archive with synthetic descriptors (no physics)."""

import unittest

from virturoid.services.qd_archive import QDArchive


class QDArchiveTests(unittest.TestCase):
    def _archive(self):
        return QDArchive(dims=[("n_legs", 2, 10), ("speed", 0.0, 1.0)], bins=8)

    def test_elite_per_cell_keeps_best(self):
        a = self._archive()
        r1 = a.add("hex_slow", (6, 0.10), fitness=0.3)
        self.assertTrue(r1["added"] and r1["novel_cell"])
        # same cell, higher fitness -> replaces
        r2 = a.add("hex_fast", (6, 0.12), fitness=0.6)      # (6,0.12) bins to the same cell as (6,0.10)
        self.assertTrue(r2["added"] and r2["replaced"] and not r2["novel_cell"])
        self.assertEqual(a.cells[r2["cell"]].item, "hex_fast")
        # same cell, lower fitness -> rejected
        r3 = a.add("hex_mid", (6, 0.11), fitness=0.5)
        self.assertFalse(r3["added"])
        self.assertEqual(a.best().item, "hex_fast")

    def test_distinct_cells_grow_annecs_and_qd(self):
        a = self._archive()
        a.add("quad", (4, 0.5), 0.5)
        a.add("hex", (6, 0.5), 0.7)
        a.add("octo", (8, 0.5), 0.4)
        self.assertEqual(a.novel_cells_filled(), 3)          # three distinct niches filled
        self.assertAlmostEqual(a.qd_score(), 0.5 + 0.7 + 0.4)
        self.assertEqual(len(a.elites()), 3)
        self.assertGreater(a.coverage(), 0.0)
        snap = a.snapshot()
        self.assertEqual(snap["filled"], 3)
        self.assertEqual(snap["annecs_v"], 3)
        self.assertAlmostEqual(snap["best_fitness"], 0.7)

    def test_annecs_is_monotone_even_when_cell_replaced(self):
        a = self._archive()
        a.add("x", (4, 0.5), 0.5)
        a.add("y", (4, 0.5), 0.9)                            # replaces the SAME cell -> annecs must NOT increment
        self.assertEqual(a.novel_cells_filled(), 1)

    def test_out_of_range_descriptor_clamps_into_edge_cell(self):
        a = self._archive()
        r = a.add("giant", (99, 5.0), 0.2)                   # way out of range -> clamped, not a crash
        self.assertTrue(r["added"])
        self.assertEqual(r["cell"], (7, 7))                  # top edge cells (bins-1)


if __name__ == "__main__":
    unittest.main()
