"""Design Brain summary: the moat measured in one call (MAP-Elites archive + provenance compounding)."""

import tempfile
import unittest
from pathlib import Path

from virturoid.fixtures.gene_library import quadruped_gene, tabletop_arm_gene
from virturoid.services.design_brain import design_brain_summary
from virturoid.services.map_elites_archive import MapElitesArchive
from virturoid.services.memory_db import MemoryDB


class DesignBrainTests(unittest.TestCase):
    def test_empty_memory_dir_reads_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = design_brain_summary(tmp)
            self.assertEqual(s["archive_coverage"], 0)
            self.assertEqual(s["provenance_edges"], 0)
            self.assertIsNone(s["mean_delta"])
            self.assertIn("headline", s)

    def test_aggregates_archive_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp)
            arc = MapElitesArchive()
            arc.insert(tabletop_arm_gene(), 0.5)          # arm niche
            arc.insert(quadruped_gene(), 0.7)             # distinct quad niche
            arc.save(memory_dir / "design_archive.json")
            with MemoryDB(memory_dir / "virturoid_memory.db") as db:
                vm = db.vector_memory()
                vm.record_provenance("design", "b1", parent_type="design", parent_id="a", delta=0.2)
                vm.record_provenance("design", "b2", parent_type="design", parent_id="a", delta=0.1)
            s = design_brain_summary(memory_dir)
            self.assertEqual(s["archive_coverage"], 2)            # two illuminated niches
            self.assertAlmostEqual(s["qd_score"], 1.2, places=4)  # 0.5 + 0.7
            self.assertEqual(s["provenance_edges"], 2)
            self.assertEqual(s["seeded_builds"], 2)
            self.assertAlmostEqual(s["mean_delta"], 0.15, places=4)   # measured compounding
            self.assertIn("niches illuminated", s["headline"])


if __name__ == "__main__":
    unittest.main()
