"""S5 scene-batch packing + S7 scene metrics. Packing must turn a family's TRAIN pool into fixed-K-slot per-env
arrays (poses/sizes/presence) with balanced env coverage and the held-out pool EXCLUDED. Metrics must reward a
structurally-diverse family (distinct_frac high) and flag collisions / out-of-bounds / instability."""

import unittest

import numpy as np

from virturoid.services.scene_family import generate_family
from virturoid.services.scene_batch import pack_scenes, assign_scenes_to_envs, build_training_batch
from virturoid.services.scene_metrics import (
    collision_metrics, out_of_bounds_rate, structural_diversity, family_metrics)
from virturoid.schemas.scenes import SceneGraph, SceneObject


def _scene(objs, bounds=((-1, -1, 0), (1, 1, 1))):
    return SceneGraph(id="t", name="t", backend_targets=["mujoco"], robot_spawn_xyz_rpy=(0, 0, 0, 0, 0, 0),
                      objects=objs, bounds=bounds)


class SceneBatchTests(unittest.TestCase):
    def test_pack_fixed_slots_and_presence(self):
        fam = generate_family("pick_place_sort", n_train=6, n_held_out=2, seed=1)
        batch = pack_scenes(fam.train)
        self.assertEqual(batch.poses.shape[0], 6)
        self.assertEqual(batch.poses.shape[2], 7)                 # xyz + quat
        self.assertEqual(batch.sizes.shape[1], batch.k_slots)
        # presence sums to the real packable-object count per scene; parked slots sit far below the floor
        for i in range(batch.n_scenes):
            n_present = int(batch.presence[i].sum())
            self.assertGreater(n_present, 0)
            parked = batch.poses[i, batch.presence[i] == 0.0]
            if len(parked):
                self.assertTrue(np.all(parked[:, 2] < -1.0))

    def test_env_assignment_is_balanced_and_deterministic(self):
        a = assign_scenes_to_envs(64, 6, seed=3)
        b = assign_scenes_to_envs(64, 6, seed=3)
        self.assertTrue(np.array_equal(a, b))                     # deterministic
        counts = np.bincount(a, minlength=6)
        self.assertLessEqual(counts.max() - counts.min(), 1)      # balanced coverage

    def test_training_batch_excludes_heldout(self):
        fam = generate_family("navigation", n_train=5, n_held_out=3, seed=4)
        out = build_training_batch(fam, n_envs=40, seed=0)
        self.assertEqual(out["batch"].n_scenes, 5)                # only the train pool is packed
        self.assertEqual(len(out["held_out_ids"]), 3)
        packed_ids = set(out["batch"].scene_ids)
        self.assertEqual(len(packed_ids & set(out["held_out_ids"])), 0)   # held-out never enters training
        self.assertGreaterEqual(out["coverage"]["min_envs_per_scene"], 1)


class SceneMetricsTests(unittest.TestCase):
    def test_collision_and_oob(self):
        overlap = _scene([SceneObject("a", "cube", (0.0, 0.0, 0.0, 0, 0, 0), size_xyz=(0.2, 0.2, 0.05)),
                          SceneObject("b", "cube", (0.05, 0.0, 0.0, 0, 0, 0), size_xyz=(0.2, 0.2, 0.05))])
        self.assertTrue(collision_metrics(overlap)["collides"])
        clear = _scene([SceneObject("a", "cube", (-0.5, 0, 0, 0, 0, 0), size_xyz=(0.05,) * 3),
                        SceneObject("b", "cube", (0.5, 0, 0, 0, 0, 0), size_xyz=(0.05,) * 3)])
        self.assertFalse(collision_metrics(clear)["collides"])
        oob = _scene([SceneObject("a", "cube", (5.0, 0, 0, 0, 0, 0), size_xyz=(0.05,) * 3)])
        self.assertGreater(out_of_bounds_rate(oob), 0.0)

    def test_structural_diversity_rewards_variety(self):
        diverse = generate_family("pick_place_sort", n_train=8, n_held_out=2, seed=5)
        self.assertEqual(structural_diversity(diverse)["distinct_frac"], 1.0)   # all 8 structurally unique

    def test_family_metrics_report(self):
        fam = generate_family("pick_place_sort", n_train=6, n_held_out=2, seed=6)
        m = family_metrics(fam, run_settle=False)
        self.assertTrue(m["disjoint_split"])
        self.assertEqual(m["structural_diversity"]["distinct_frac"], 1.0)
        self.assertGreaterEqual(m["valid_rate"], 0.5)             # a good generator makes mostly-valid scenes
        self.assertLessEqual(m["mean_r_out"], 0.2)

    def test_scene_ci_gate(self):
        from virturoid.services.scene_metrics import run_scene_ci
        rep = run_scene_ci(n_train=8, n_held_out=3, seed=0, run_settle=False)
        self.assertIn("summary", rep)
        for row in rep["tasks"]:                                  # every standard task family is diverse + disjoint
            self.assertTrue(row["disjoint_split"], row["task_type"])
            self.assertGreaterEqual(row["structural_diversity"]["distinct_frac"], 0.8, row["task_type"])


if __name__ == "__main__":
    unittest.main()
