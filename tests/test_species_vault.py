"""Obsidian-style species vault: linked notes whose graph mirrors the morphology vector space."""

import tempfile
import unittest
from pathlib import Path

from virturoid.fixtures.gene_library import humanoid_upper_body_gene, tabletop_arm_gene
from virturoid.services.design_critic import add_parallel_gripper
from virturoid.services.memory_db import MemoryDB
from virturoid.services.species_discovery import auto_place_species
from virturoid.services.species_vault import export_species_vault


class SpeciesVaultTests(unittest.TestCase):
    def test_vault_exports_notes_with_morphology_neighbor_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = MemoryDB(Path(tmp) / "m.db")
            for g in (tabletop_arm_gene(), add_parallel_gripper(tabletop_arm_gene()),
                      humanoid_upper_body_gene()):
                auto_place_species(g, db)
            vault = Path(tmp) / "vault"
            n = export_species_vault(db, vault)
            db.close()

            self.assertEqual(n, 3)
            self.assertTrue((vault / "Species Tree.md").exists())
            note = (vault / "fixed_arm.three_dof.tabletop.gripper.md").read_text(encoding="utf-8")
            self.assertIn("## Morphology neighbors (vector space)", note)
            # the gripper-arm's NEAREST neighbor is the plain arm (a wikilink), not the far humanoid
            arm_idx = note.find("[[fixed_arm.three_dof.tabletop]]")
            hum_idx = note.find("[[humanoid.upper_body.single_arm]]")
            self.assertGreater(arm_idx, 0)
            self.assertTrue(hum_idx == -1 or arm_idx < hum_idx)   # arm listed first (closer)


if __name__ == "__main__":
    unittest.main()
