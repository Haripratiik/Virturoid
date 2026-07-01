"""Robotics vector memory (Pillar 2 core): the unified body/skill/episode/task retrieval index.

Pure-Python (no numpy/torch/MuJoCo on the default path): embeddings are deterministic + dependency-free,
the store is SQLite cosine-kNN, and provenance edges make warm-start compounding measurable. Also
verifies the non-breaking upgrade of ``memory_db.similar_runs`` to embedding cosine (catches sub-word
matches Jaccard misses) with the Jaccard fallback intact.
"""

import tempfile
import unittest
from pathlib import Path

from virturoid.fixtures.gene_library import tabletop_arm_gene, humanoid_upper_body_gene
from virturoid.services.design_critic import add_parallel_gripper
from virturoid.services.memory_db import MemoryDB
from virturoid.services.robotics_vector_memory import (
    BODY, SKILL, TASK, RUN, RoboticsVectorMemory, BODY_DIM,
    cosine, embed_body, embed_episode, embed_skill, embed_task, embed_text,
)


class EmbeddingTests(unittest.TestCase):
    def test_text_embedding_is_deterministic_and_captures_similarity(self):
        a = embed_text("sort red and blue blocks into bins")
        self.assertEqual(a, embed_text("sort red and blue blocks into bins"))   # deterministic
        near = cosine(a, embed_text("sort colored blocks into bins"))           # overlapping words
        far = cosine(a, embed_text("navigate a mobile robot down a hallway"))   # unrelated
        self.assertGreater(near, far)
        self.assertGreater(near, 0.2)

    def test_text_embedding_catches_subword_similarity(self):
        # char trigrams make "grasp" ~ "grasping" — a pure token set (Jaccard) would score these 0
        self.assertGreater(cosine(embed_text("grasp"), embed_text("grasping")), 0.3)

    def test_body_embedding_places_similar_bodies_closer(self):
        arm = tabletop_arm_gene()
        arm_grip = add_parallel_gripper(arm)
        humanoid = humanoid_upper_body_gene()
        near = cosine(embed_body(arm), embed_body(arm_grip))
        far = cosine(embed_body(arm), embed_body(humanoid))
        self.assertGreater(near, far)      # arm+gripper more like the arm than a humanoid is

    def test_body_embedding_accepts_a_learned_latent(self):
        # the GPU-phase seam: a learned graph latent slots in behind the same signature
        latent = [0.1, 0.2, 0.3, 0.4]
        v = embed_body(tabletop_arm_gene(), latent=latent)
        self.assertEqual(len(v), len(latent))
        self.assertAlmostEqual(sum(x * x for x in v) ** 0.5, 1.0, places=5)   # L2-normalized

    def test_skill_embedding_is_fixed_length(self):
        with_gene = embed_skill("grasp a cube", tabletop_arm_gene(), success_rate=0.9)
        without_gene = embed_skill("grasp a cube", None, success_rate=0.9)
        self.assertEqual(len(with_gene), len(without_gene))                    # comparable sub-space
        self.assertGreater(len(with_gene), BODY_DIM)                           # body ⊕ task ⊕ fingerprint

    def test_episode_embedding_separates_walk_from_fall(self):
        walk_a = embed_episode({"cadence": 12.0, "upright_frac": 0.9, "forward_m": 0.8})
        walk_b = embed_episode({"cadence": 11.0, "upright_frac": 0.88, "forward_m": 0.75})
        fell = embed_episode({"cadence": 0.0, "upright_frac": 0.1, "forward_m": -0.2})
        self.assertGreater(cosine(walk_a, walk_b), cosine(walk_a, fell))


class VectorStoreTests(unittest.TestCase):
    def _mem(self, tmp):
        return MemoryDB(Path(tmp) / "mem.db")

    def test_nearest_ranks_by_cosine_and_respects_namespaces(self):
        with tempfile.TemporaryDirectory() as tmp, self._mem(tmp) as db:
            vm = RoboticsVectorMemory(db)
            vm.upsert(BODY, "east", [1.0, 0.0, 0.0])
            vm.upsert(BODY, "north", [0.0, 1.0, 0.0])
            vm.upsert(BODY, "east2", [0.9, 0.1, 0.0])
            # a different sub-space with the SAME obj_id must not collide or leak into body queries
            vm.upsert(TASK, "east", [1.0, 0.0, 0.0])
            hits = vm.nearest(BODY, [1.0, 0.0, 0.0], k=2)
            self.assertEqual([h["obj_id"] for h in hits], ["east", "east2"])
            self.assertEqual(vm.count(BODY), 3)
            self.assertEqual(vm.count(TASK), 1)
            # exclude_id drops the query object itself
            self.assertNotIn("east", [h["obj_id"] for h in vm.nearest(BODY, [1.0, 0.0, 0.0], exclude_id="east")])

    def test_upsert_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp, self._mem(tmp) as db:
            vm = RoboticsVectorMemory(db)
            vm.upsert(BODY, "x", [1.0, 0.0])
            vm.upsert(BODY, "x", [0.0, 1.0])       # overwrite, not duplicate
            self.assertEqual(vm.count(BODY), 1)
            self.assertAlmostEqual(vm.get(BODY, "x")["vec"][1], 1.0, places=5)

    def test_provenance_ledger_measures_compounding(self):
        with tempfile.TemporaryDirectory() as tmp, self._mem(tmp) as db:
            vm = RoboticsVectorMemory(db)
            vm.record_provenance(SKILL, "grasp_v2", parent_type=SKILL, parent_id="grasp_v1",
                                 kind="warm_start", delta=0.2)
            vm.record_provenance(BODY, "arm_v2", parent_type=BODY, parent_id="arm_v1",
                                 kind="amend", delta=0.1)
            summary = vm.compounding_summary()
            self.assertEqual(summary["edges"], 2)
            self.assertEqual(summary["seeded_builds"], 2)
            self.assertAlmostEqual(summary["mean_delta"], 0.15, places=4)
            self.assertAlmostEqual(summary["positive_fraction"], 1.0, places=4)
            self.assertEqual(len(vm.provenance_for(SKILL, "grasp_v2")), 1)

    def test_index_runs_backfills_and_retrieves(self):
        with tempfile.TemporaryDirectory() as tmp, self._mem(tmp) as db:
            db.record_run("teach the arm to grasp a cube", "manipulator", "grasp", None, 0.9)
            db.record_run("navigate a mobile robot to a goal", "mobile_base", "navigation", None, 0.8)
            vm = RoboticsVectorMemory(db)
            self.assertEqual(vm.index_runs(), 2)          # both embedded
            self.assertEqual(vm.index_runs(), 0)          # incremental: nothing new the second time
            # a lexically-overlapping query ("grasp"/"cube") retrieves the manipulation run; deeper
            # semantic matching ("pick up an object" -> grasp) is the frozen-LM upgrade (GPU phase)
            hits = vm.nearest(RUN, embed_task("grasp the cube"), k=1)
            self.assertEqual(hits[0]["meta"]["task_type"], "grasp")   # nearer the manipulation run

    def test_index_species_bodies(self):
        from virturoid.services.species_discovery import auto_place_species
        with tempfile.TemporaryDirectory() as tmp, self._mem(tmp) as db:
            auto_place_species(tabletop_arm_gene(), db)
            auto_place_species(humanoid_upper_body_gene(), db)
            vm = RoboticsVectorMemory(db)
            self.assertEqual(vm.index_species_bodies(), 2)
            # a gripper-arm query retrieves the arm species ahead of the humanoid
            hits = vm.nearest(BODY, embed_body(add_parallel_gripper(tabletop_arm_gene())), k=2)
            self.assertEqual(hits[0]["meta"]["robot_class"], "manipulator")

    def test_index_skills_enables_cross_body_retrieval(self):
        from virturoid.services.species_discovery import auto_place_species
        with tempfile.TemporaryDirectory() as tmp, self._mem(tmp) as db:
            arm_sp = auto_place_species(tabletop_arm_gene(), db)["species_pattern"]
            quad_sp = auto_place_species(humanoid_upper_body_gene(), db)["species_pattern"]
            db.record_skill("grasp.arm", "manipulator", "grasp", success_rate=0.9, species=arm_sp)
            db.record_skill("reach.other", "humanoid", "reach", success_rate=0.8, species=quad_sp)
            vm = RoboticsVectorMemory(db)
            self.assertEqual(vm.index_skills(), 2)
            # a new gripper-arm doing grasp retrieves the arm's grasp skill ahead of the other body's
            hits = vm.nearest_skills(add_parallel_gripper(tabletop_arm_gene()), "grasp", k=2)
            self.assertEqual(hits[0]["obj_id"], "grasp.arm")
            self.assertEqual(hits[0]["meta"]["task_type"], "grasp")


class SimilarRunsUpgradeTests(unittest.TestCase):
    def _mem(self, tmp):
        return MemoryDB(Path(tmp) / "mem.db")

    def test_semantic_similar_runs_beats_jaccard_on_subword_match(self):
        with tempfile.TemporaryDirectory() as tmp, self._mem(tmp) as db:
            db.record_run("teach the arm to grasp a cube", "manipulator", "grasp", None, 0.9)
            db.record_run("navigate a mobile robot to a goal", "mobile_base", "navigation", None, 0.8)
            # "grasping objects" shares NO whole token with either run -> Jaccard finds nothing
            self.assertEqual(db.similar_runs("grasping objects", semantic=False), [])
            # the embedding path catches the sub-word "grasp" ~ "grasping" and retrieves the grasp run
            hits = db.similar_runs("grasping objects", semantic=True)
            self.assertTrue(hits)
            self.assertEqual(hits[0]["task_type"], "grasp")

    def test_semantic_path_preserves_existing_ranking(self):
        # the original test_memory_db expectation must still hold under the default (semantic) path
        with tempfile.TemporaryDirectory() as tmp, self._mem(tmp) as db:
            db.record_run("sort red and blue blocks into bins", "manipulator", "pick_place_sort", None, 0.9)
            db.record_run("navigate a mobile robot to a goal", "mobile_base", "navigation", None, 0.8)
            hits = db.similar_runs("sort colored blocks into bins", limit=3)
            self.assertTrue(hits)
            self.assertEqual(hits[0]["task_type"], "pick_place_sort")

    def test_robot_class_filter_holds_in_semantic_path(self):
        with tempfile.TemporaryDirectory() as tmp, self._mem(tmp) as db:
            db.record_run("grasp a cube", "manipulator", "grasp", None, 0.9)
            db.record_run("grasp a cube with a mobile manipulator", "mobile_base", "grasp", None, 0.8)
            hits = db.similar_runs("grasp an object", robot_class="manipulator")
            self.assertTrue(hits)
            self.assertTrue(all(h["robot_class"] == "manipulator" for h in hits))


class LearnedLatentGateTests(unittest.TestCase):
    def test_spearman_rank_correlation(self):
        from virturoid.services.robotics_vector_memory import _spearman
        self.assertAlmostEqual(_spearman([1, 2, 3, 4], [1, 2, 3, 4]), 1.0)
        self.assertAlmostEqual(_spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)

    def test_body_latent_falls_back_safely(self):
        # returns None (→ deterministic embed_gene) when no TRUSTED model exists, or a float list when one
        # does; must never crash regardless of what model file is (or isn't) on disk (the never-worse gate)
        from virturoid.services.robotics_vector_memory import _body_latent
        v = _body_latent(tabletop_arm_gene())
        self.assertTrue(v is None or (isinstance(v, list) and all(isinstance(x, float) for x in v)))

    def test_embed_body_uses_deterministic_when_untrusted(self):
        # with no trusted learned model, the body embedding is exactly the deterministic morphology vector
        from virturoid.services.robotics_vector_memory import _body_latent
        if _body_latent(tabletop_arm_gene()) is None:      # untrusted/absent model → the shipped path
            self.assertEqual(embed_body(tabletop_arm_gene()),
                             embed_body(tabletop_arm_gene(), latent=None))


class LearnedLatentSeamTests(unittest.TestCase):
    def test_gnn_embed_exposes_a_pooled_latent_when_torch_present(self):
        try:
            import torch  # noqa: F401
        except Exception:  # noqa: BLE001 - torch is optional; the CPU path never needs it
            self.skipTest("torch not installed (learned z_body is GPU-phase; CPU path uses embed_gene)")
        from virturoid.services.gene_surrogate_nn import GeneGNN
        gnn = GeneGNN(hidden=16, device="cpu")
        latent = gnn.embed(tabletop_arm_gene())
        self.assertEqual(len(latent), 16)
        self.assertEqual(latent, gnn.embed(tabletop_arm_gene()))          # deterministic for fixed weights
        # and it plugs straight into the body embedder
        self.assertEqual(len(embed_body(tabletop_arm_gene(), latent=latent)), 16)


if __name__ == "__main__":
    unittest.main()
