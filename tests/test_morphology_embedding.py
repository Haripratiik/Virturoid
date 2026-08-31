"""Morphology vector space + autonomous species discovery: any gene -> embedding -> self-placed tree.

Pure-Python (no MuJoCo): the building blocks compose into a gene, the gene embeds into a fixed vector,
and novel morphologies are auto-classified/created in the species tree by nearest-neighbour."""

import tempfile
import unittest
from pathlib import Path

from virturoid.fixtures.gene_library import tabletop_arm_gene, humanoid_upper_body_gene
from virturoid.schemas.gene import amend_gene
from virturoid.services.design_critic import add_parallel_gripper, lengthen_reach
from virturoid.services.memory_db import MemoryDB
from virturoid.services.morphology_embedding import FEATURE_NAMES, distance, embed_gene
from virturoid.services.species_discovery import auto_place_species, morphology_tags, nearest_species


class MorphologyEmbeddingTests(unittest.TestCase):
    def test_embedding_is_fixed_length_and_deterministic(self):
        a = embed_gene(tabletop_arm_gene())
        b = embed_gene(tabletop_arm_gene())
        self.assertEqual(len(a), len(FEATURE_NAMES))
        self.assertEqual(a, b)                                 # deterministic
        self.assertEqual(distance(a, b), 0.0)

    def test_similar_morphologies_are_closer_than_different_ones(self):
        arm = tabletop_arm_gene()
        arm_grip = add_parallel_gripper(arm)
        humanoid = humanoid_upper_body_gene()
        d_close = distance(embed_gene(arm), embed_gene(arm_grip))
        d_far = distance(embed_gene(arm), embed_gene(humanoid))
        self.assertLess(d_close, d_far)                        # arm+gripper nearer to arm than a humanoid is
        self.assertLess(d_close, 2.0)
        # SEPARATION IS A RATIO, not a distance in absolute units. This line read ``d_far > 2.5`` — a bar on an
        # un-normalized Euclidean distance, which any robot can clear by simply being heavier. Measured
        # 2026-08-08 with the old linear mass map, ``total_mass`` supplied 2.78 of that 3.60: the arm weighs
        # 0.925 kg and the humanoid 9.25 kg, so ~74% of the squared "different morphology" distance was the 10x
        # mass, not the morphology. Mass is log-compressed now (see ``_LOG_SCALED``) and the same pair reads
        # d_close 1.186 / d_far 2.465, a 2.08x separation earned from structure. A ratio is what "closer than"
        # means and it cannot be gamed by scale.
        self.assertGreater(d_far, 1.8 * d_close)


class SpeciesDiscoveryTests(unittest.TestCase):
    def _db(self, tmp):
        return MemoryDB(Path(tmp) / "m.db")

    def test_autonomous_tree_grows_by_morphology(self):
        with tempfile.TemporaryDirectory() as tmp, self._db(tmp) as db:
            arm = tabletop_arm_gene()
            first = auto_place_species(arm, db)
            self.assertEqual(first["action"], "created")        # empty tree -> first species

            # a near-identical parameter variant CLASSIFIES into the existing species (no new node)
            variant = lengthen_reach(arm, factor=1.05, new_id="arm_v", species=None)
            v = auto_place_species(variant, db)
            self.assertEqual(v["action"], "classified")
            self.assertEqual(v["species_pattern"], first["species_pattern"])
            self.assertEqual(len(db.species_tree_nodes()), 1)

            # a gripper arm is a NEW species parented under its closest relative (the arm)
            grip = auto_place_species(add_parallel_gripper(arm), db)
            self.assertEqual(grip["action"], "created")
            self.assertEqual(grip["parent"], first["species_pattern"])

            # a humanoid is a NEW species, NOT parented under the arm (far in morphology space)
            hum = auto_place_species(humanoid_upper_body_gene(), db)
            self.assertEqual(hum["action"], "created")
            self.assertNotEqual(hum["parent"], first["species_pattern"])
            self.assertEqual(len(db.species_tree_nodes()), 3)

    def test_nearest_species_ranks_by_distance(self):
        with tempfile.TemporaryDirectory() as tmp, self._db(tmp) as db:
            arm = tabletop_arm_gene()
            auto_place_species(arm, db)
            auto_place_species(humanoid_upper_body_gene(), db)
            ranked = nearest_species(add_parallel_gripper(arm), db.species_tree_nodes())
            self.assertTrue(ranked)
            self.assertEqual(ranked[0]["robot_class"], "manipulator")   # closest is the arm, not the humanoid
            self.assertTrue(all(ranked[i]["distance"] <= ranked[i + 1]["distance"] for i in range(len(ranked) - 1)))

    def test_transfer_candidates_surface_nearest_body_and_skill(self):
        from virturoid.services.species_discovery import transfer_candidates
        with tempfile.TemporaryDirectory() as tmp, self._db(tmp) as db:
            arm = tabletop_arm_gene()
            auto_place_species(arm, db)                          # an arm species with a stored gene
            auto_place_species(humanoid_upper_body_gene(), db)   # a far species
            db.record_skill("grasp.arm", "manipulator", "grasp", success_rate=0.9,
                            params_path="p.npz", species="manipulator", obs_dim=34, act_dim=5)
            # a new gripper-arm should transfer from the nearest prior robot (the arm), not the humanoid
            cands = transfer_candidates(add_parallel_gripper(arm), db, task_type="grasp")
            self.assertTrue(cands)
            self.assertEqual(cands[0]["robot_class"], "manipulator")
            self.assertIsNotNone(cands[0]["gene"])               # the body to amend / warm-start
            self.assertIsNotNone(cands[0]["banked_skill"])       # the policy to warm-start

    def test_morphology_tags_are_readable(self):
        tags = morphology_tags(add_parallel_gripper(tabletop_arm_gene()))
        self.assertIn("manipulator", tags)
        self.assertTrue(any(t.startswith("dof") for t in tags))
        self.assertTrue(any(t.startswith("fingers") for t in tags))


if __name__ == "__main__":
    unittest.main()
