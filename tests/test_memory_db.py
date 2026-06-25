import tempfile
import unittest
from pathlib import Path

from virturoid.services.memory_db import MemoryDB


class MemoryDBTests(unittest.TestCase):
    def _db(self, tmp):
        return MemoryDB(Path(tmp) / "mem.db")

    def test_record_run_creates_run_design_and_species(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self._db(tmp) as db:
                run_id = db.record_run(
                    "sort red and blue blocks", "manipulator", "pick_place_sort",
                    {"forearm_mm": 340, "wrist_mm": 160, "actuator_torque_nm": 12}, 0.92,
                    species="serial_arm_3dof", target_success_rate=0.8, succeeded=True,
                    backend="openai", design_source="ai_designer+codesign",
                )
                self.assertGreater(run_id, 0)
                stats = db.stats()
                self.assertEqual(1, stats["runs"])
                self.assertEqual(1, stats["designs"])
                self.assertEqual(1, stats["species"])

    def test_best_design_returns_highest_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self._db(tmp) as db:
                db.record_run("p", "manipulator", "pick_place_sort", {"forearm_mm": 300, "wrist_mm": 150, "actuator_torque_nm": 12}, 0.7)
                db.record_run("p", "manipulator", "pick_place_sort", {"forearm_mm": 340, "wrist_mm": 160, "actuator_torque_nm": 12}, 0.95)
                best = db.best_design("manipulator", "pick_place_sort")
                self.assertAlmostEqual(0.95, best["success_rate"])
                self.assertEqual(340, best["converged_design"]["forearm_mm"])

    def test_known_failures_aggregate(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self._db(tmp) as db:
                rid = db.record_run("p", "manipulator", "pick_place_sort", {"forearm_mm": 300, "wrist_mm": 150, "actuator_torque_nm": 12}, 0.5)
                db.record_failures(rid, "manipulator", "pick_place_sort",
                                   [{"label": "missed_grasp", "count": 4}, {"label": "wrong_bin", "count": 1}])
                rid2 = db.record_run("p", "manipulator", "pick_place_sort", {"forearm_mm": 310, "wrist_mm": 150, "actuator_torque_nm": 12}, 0.6)
                db.record_failures(rid2, "manipulator", "pick_place_sort", [{"label": "missed_grasp", "count": 2}])
                fails = db.known_failures("manipulator", "pick_place_sort")
                self.assertEqual("missed_grasp", fails[0]["label"])
                self.assertEqual(6, fails[0]["total"])

    def test_record_failures_accepts_objects(self):
        class Cluster:
            def __init__(self, label, count):
                self.failure_label = label
                self.count = count

        with tempfile.TemporaryDirectory() as tmp:
            with self._db(tmp) as db:
                rid = db.record_run("p", "manipulator", "pick_place_sort", {"forearm_mm": 300, "wrist_mm": 150, "actuator_torque_nm": 12}, 0.5)
                db.record_failures(rid, "manipulator", "pick_place_sort", [Cluster("dropped", 3)])
                self.assertEqual(3, db.known_failures("manipulator", "pick_place_sort")[0]["total"])

    def test_transfer_candidates_best_per_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self._db(tmp) as db:
                db.record_run("a", "manipulator", "pick_place_sort", {"forearm_mm": 340, "wrist_mm": 160, "actuator_torque_nm": 12}, 0.9)
                db.record_run("b", "manipulator", "pick_place_box", {"forearm_mm": 320, "wrist_mm": 150, "actuator_torque_nm": 12}, 0.8)
                cands = db.transfer_candidates("manipulator")
                keys = {c["key"] for c in cands}
                self.assertIn("manipulator__pick_place_sort", keys)
                self.assertTrue(all("converged_design" in c for c in cands))

    def test_similar_runs_ranks_by_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self._db(tmp) as db:
                db.record_run("sort red and blue blocks into bins", "manipulator", "pick_place_sort", {"forearm_mm": 340, "wrist_mm": 160, "actuator_torque_nm": 12}, 0.9)
                db.record_run("navigate a mobile robot to a goal", "mobile_base", "navigation", None, 0.8)
                hits = db.similar_runs("sort colored blocks into bins", limit=3)
                self.assertTrue(hits)
                self.assertEqual("pick_place_sort", hits[0]["task_type"])

    def test_training_dataset_filters_and_respects_permission(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self._db(tmp) as db:
                db.record_run("p1", "manipulator", "pick_place_sort", {"forearm_mm": 340, "wrist_mm": 160, "actuator_torque_nm": 12}, 0.95)
                db.record_run("p2", "manipulator", "pick_place_sort", {"forearm_mm": 200, "wrist_mm": 100, "actuator_torque_nm": 8}, 0.3)
                db.record_run("p3", "manipulator", "pick_place_sort", {"forearm_mm": 350, "wrist_mm": 160, "actuator_torque_nm": 14}, 0.9,
                              training_permission="no_training")
                rows = db.training_dataset(min_success=0.8)
                prompts = {r["prompt"] for r in rows}
                self.assertIn("p1", prompts)
                self.assertNotIn("p2", prompts)   # below threshold
                self.assertNotIn("p3", prompts)   # opted out of training
                self.assertIn("design", rows[0])

    def test_export_training_jsonl(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ds.jsonl"
            with self._db(tmp) as db:
                db.record_run("p1", "manipulator", "pick_place_sort", {"forearm_mm": 340, "wrist_mm": 160, "actuator_torque_nm": 12}, 0.95)
                n = db.export_training_jsonl(out, min_success=0.8)
            self.assertEqual(1, n)
            rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
            self.assertEqual(340, rows[0]["design"]["forearm_mm"])

    def test_species_tree_upsert_and_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self._db(tmp) as db:
                db.upsert_species_node("fixed_arm.three_dof.tabletop", parent_species="fixed_arm",
                                       robot_class="manipulator", morphology_tags=["serial_chain"],
                                       buildable=True, source="catalog")
                db.upsert_species_node("humanoid.arm.single", parent_species="humanoid",
                                       robot_class="humanoid", morphology_tags=["humanoid"],
                                       genes={"links": ["a"], "joints": [], "end_effectors": ["a"]},
                                       buildable=False, source="species_agent")
                nodes = db.species_tree_nodes()
                self.assertEqual(2, len(nodes))
                h = db.species_node("humanoid.arm.single")
                self.assertEqual("humanoid", h["parent_species"])
                self.assertFalse(h["buildable"])
                self.assertEqual(["humanoid"], h["morphology_tags"])
                self.assertIn("links", h["genes"])
                # idempotent: upsert again updates, doesn't duplicate
                db.upsert_species_node("humanoid.arm.single", robot_class="humanoid",
                                       morphology_tags=["humanoid", "single_arm"], buildable=True)
                self.assertEqual(2, len(db.species_tree_nodes()))
                self.assertTrue(db.species_node("humanoid.arm.single")["buildable"])

    def test_find_gene_for_class_needs_buildable_and_genes(self):
        """Flywheel reuse regression: a stored gene is reusable ONLY when its species node is BOTH buildable
        AND carries a full gene (segments). The bug was that a DISCOVERED species banked the gene but stayed
        buildable=0, so find_gene_for_class never returned it (memory_reused was always False)."""
        gene = {"species": "quadruped.composed", "segments": [{"name": "torso"}]}
        with tempfile.TemporaryDirectory() as tmp:
            with self._db(tmp) as db:
                # gene present but NOT buildable -> not reusable (the old broken state)
                db.upsert_species_node("quadruped.composed", robot_class="quadruped", genes=gene, buildable=False)
                self.assertIsNone(db.find_gene_for_class("quadruped"))
                # buildable AND gene -> reusable (the fix: a discovered species is buildable from its own gene)
                db.upsert_species_node("quadruped.composed", robot_class="quadruped", genes=gene, buildable=True)
                found = db.find_gene_for_class("quadruped")
                self.assertIsNotNone(found)
                self.assertTrue(found.get("segments"))

    def test_persists_across_connections(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "mem.db"
            with MemoryDB(p) as db:
                db.record_run("p", "manipulator", "pick_place_sort", {"forearm_mm": 340, "wrist_mm": 160, "actuator_torque_nm": 12}, 0.9)
            with MemoryDB(p) as db2:
                self.assertEqual(1, db2.stats()["runs"])
                self.assertIsNotNone(db2.best_design("manipulator", "pick_place_sort"))


if __name__ == "__main__":
    unittest.main()
