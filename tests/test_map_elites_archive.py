"""MAP-Elites design archive (Pillar 1 Curator): illuminate diverse trainable bodies, per-niche elitism.

Pure-Python (gene construction needs no MuJoCo): different body plans land in different niches, insertion
keeps the best per cell, and QD-score/coverage track illumination.
"""

import tempfile
import unittest
from pathlib import Path

from virturoid.fixtures.gene_library import quadruped_gene, tabletop_arm_gene
from virturoid.services.design_critic import add_parallel_gripper
from virturoid.services.map_elites_archive import MapElitesArchive, descriptor


class DescriptorTests(unittest.TestCase):
    def test_different_body_plans_get_different_niches(self):
        self.assertNotEqual(descriptor(tabletop_arm_gene()), descriptor(quadruped_gene()))

    def test_descriptor_is_deterministic(self):
        self.assertEqual(descriptor(quadruped_gene()), descriptor(quadruped_gene()))


class ArchiveTests(unittest.TestCase):
    def test_per_cell_elitism(self):
        arc = MapElitesArchive()
        arm = tabletop_arm_gene()
        self.assertEqual(arc.insert(arm, 0.4), "added")
        self.assertEqual(arc.insert(arm, 0.6, controller={"kp": 30}), "improved")   # higher score wins the cell
        self.assertEqual(arc.insert(arm, 0.5), "rejected")                          # lower score can't displace
        self.assertEqual(arc.coverage(), 1)                                         # still one niche
        self.assertAlmostEqual(arc.nearest_elite(arm)["score"], 0.6)
        self.assertEqual(arc.nearest_elite(arm)["controller"], {"kp": 30})

    def test_illumination_across_niches(self):
        arc = MapElitesArchive()
        arc.insert(tabletop_arm_gene(), 0.5)
        arc.insert(quadruped_gene(), 0.7)
        self.assertEqual(arc.coverage(), 2)                    # two distinct body plans -> two niches
        self.assertAlmostEqual(arc.qd_score(), 1.2)            # sum of elite scores
        self.assertAlmostEqual(arc.best()["score"], 0.7)       # the quad is the current best
        self.assertEqual(arc.summary()["coverage"], 2)

    def test_warm_start_seed_by_niche(self):
        arc = MapElitesArchive()
        arc.insert(quadruped_gene(), 0.8, controller={"freq": 1.5, "leg_flip": True})
        # a same-plan body retrieves its niche's elite to warm-start from; an empty niche returns None
        self.assertIsNotNone(arc.nearest_elite(quadruped_gene()))
        self.assertIsNone(arc.nearest_elite(tabletop_arm_gene()))

    def test_save_and_load_round_trips(self):
        arc = MapElitesArchive()
        arc.insert(tabletop_arm_gene(), 0.5, controller={"kp": 12})
        arc.insert(add_parallel_gripper(tabletop_arm_gene()), 0.6)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "archive.json"
            arc.save(p)
            back = MapElitesArchive.load(p)
        self.assertEqual(back.coverage(), arc.coverage())
        self.assertAlmostEqual(back.qd_score(), arc.qd_score())
        self.assertEqual(back.best()["controller"] or {}, arc.best()["controller"] or {})


if __name__ == "__main__":
    unittest.main()
