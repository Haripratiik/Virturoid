import tempfile
import unittest
from pathlib import Path

from virturoid.services.memory_db import MemoryDB
from virturoid.services.species_maintenance import prune_tree
from virturoid.services.species_vault import export_species_vault


def _seed(db):
    # A small tree with a near-duplicate pair of unused, non-buildable leaves.
    db.upsert_species_node("fixed_arm.three_dof.tabletop", parent_species="fixed_arm",
                           robot_class="manipulator", morphology_tags=["serial_chain", "tabletop"],
                           buildable=True, source="catalog")
    db.upsert_species_node("humanoid.arm.left", parent_species="humanoid", robot_class="humanoid",
                           morphology_tags=["humanoid", "single_arm"], buildable=False, source="requested")
    db.upsert_species_node("humanoid.arm.right", parent_species="humanoid", robot_class="humanoid",
                           morphology_tags=["humanoid", "single_arm"], buildable=False, source="requested")


class TrimMergeTests(unittest.TestCase):
    def _db(self, tmp):
        return MemoryDB(Path(tmp) / "mem.db")

    def test_merges_near_duplicate_siblings(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self._db(tmp) as db:
                _seed(db)
                result = prune_tree(db)
                # The two identical-tag humanoid siblings collapse into one.
                self.assertEqual(1, len(result.merged))
                live = {n["species_pattern"] for n in db.species_tree_nodes()}
                self.assertTrue(("humanoid.arm.left" in live) ^ ("humanoid.arm.right" in live))

    def test_merges_by_morphology_distance_when_genes_present(self):
        from virturoid.fixtures.gene_library import tabletop_arm_gene
        from virturoid.services.design_critic import lengthen_reach
        with tempfile.TemporaryDirectory() as tmp:
            with self._db(tmp) as db:
                arm = tabletop_arm_gene()
                near = lengthen_reach(arm, factor=1.02, new_id="arm2", species=None)  # ~identical morphology
                # two non-buildable sibling species that DISAGREE on tags but are near in vector space
                db.upsert_species_node("manip.a", parent_species="manip", robot_class="manipulator",
                                       morphology_tags=["alpha"], genes=arm.to_dict(), buildable=False)
                db.upsert_species_node("manip.b", parent_species="manip", robot_class="manipulator",
                                       morphology_tags=["beta"], genes=near.to_dict(), buildable=False)
                result = prune_tree(db, dry_run=True)
                # tags disagree ({alpha} vs {beta}, Jaccard 0), so ONLY the morphology vector path can
                # flag these as near-duplicates — a merge here proves embedding-driven maintenance.
                self.assertEqual(1, len(result.merged))
                self.assertIn(result.merged[0][0], {"manip.a", "manip.b"})

    def test_never_merges_across_robot_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self._db(tmp) as db:
                db.upsert_species_node("a.x", parent_species="root", robot_class="manipulator",
                                       morphology_tags=["t"], buildable=False)
                db.upsert_species_node("b.x", parent_species="root", robot_class="humanoid",
                                       morphology_tags=["t"], buildable=False)
                result = prune_tree(db)
                self.assertEqual([], result.merged)  # different classes -> not merged

    def test_never_trims_node_with_builds(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self._db(tmp) as db:
                db.upsert_species_node("solo.leaf", parent_species="root", robot_class="manipulator",
                                       morphology_tags=["t"], buildable=False)
                db.record_run("p", "manipulator", "pick_place_sort",
                              {"forearm_mm": 340, "wrist_mm": 160, "actuator_torque_nm": 12}, 0.9,
                              species="solo.leaf")
                result = prune_tree(db)
                self.assertNotIn("solo.leaf", result.trimmed)
                self.assertIsNotNone(db.species_node("solo.leaf"))

    def test_never_trims_buildable(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self._db(tmp) as db:
                db.upsert_species_node("real.species", parent_species="root", robot_class="manipulator",
                                       morphology_tags=["t"], buildable=True)
                result = prune_tree(db)
                self.assertNotIn("real.species", result.trimmed)

    def test_dry_run_does_not_mutate(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self._db(tmp) as db:
                _seed(db)
                before = len(db.species_tree_nodes())
                result = prune_tree(db, dry_run=True)
                self.assertTrue(result.merged or result.trimmed)
                self.assertEqual(before, len(db.species_tree_nodes()))  # unchanged

    def test_trims_only_subsumed_leaf_keeps_novel(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self._db(tmp) as db:
                # A dominating sibling whose tags are a superset of the redundant one.
                db.upsert_species_node("arm.full", parent_species="root", robot_class="manipulator",
                                       morphology_tags=["serial", "wrist", "gripper"], buildable=False)
                db.upsert_species_node("arm.subset", parent_species="root", robot_class="manipulator",
                                       morphology_tags=["serial", "wrist"], buildable=False)  # subsumed
                # A genuinely novel leaf with a unique tag — must be kept.
                db.upsert_species_node("arm.novel", parent_species="root", robot_class="manipulator",
                                       morphology_tags=["serial", "force_torque_unique"], buildable=False)
                result = prune_tree(db)
                self.assertIn("arm.subset", result.trimmed)
                self.assertNotIn("arm.novel", result.trimmed)
                self.assertIsNotNone(db.species_node("arm.novel"))


class VaultTests(unittest.TestCase):
    def test_exports_linked_markdown_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            with MemoryDB(Path(tmp) / "mem.db") as db:
                _seed(db)
                db.add_species_tip("humanoid.arm.left", "Use a longer forearm for shelf reach.", audience="builder")
                vault = Path(tmp) / "vault"
                n = export_species_vault(db, vault)
                self.assertEqual(3, n)
                self.assertTrue((vault / "Species Tree.md").exists())
                note = (vault / "humanoid.arm.left.md").read_text(encoding="utf-8")
                self.assertIn("[[humanoid]]", note)               # parent wikilink
                self.assertIn("longer forearm for shelf reach", note)  # tip surfaced
                self.assertIn("species: humanoid.arm.left", note)  # frontmatter


if __name__ == "__main__":
    unittest.main()
