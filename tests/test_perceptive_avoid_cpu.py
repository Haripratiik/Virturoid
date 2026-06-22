"""Perceptive MorphPolicy (Theme 1 + perception Rung A): the learned policy can now act on SENSED ranges,
and the ES sense-and-avoid training loop runs end-to-end on CPU — over the SAME elitist optimizer as
locomotion, so warm-start/transfer/monotonicity all carry over. Proves the perception channel trains
(non-regression via elitism) without a slow/flaky 'solves the maze' claim. Relates to perception-next-keystone."""

import importlib.util
import math
import unittest

_MUJOCO = importlib.util.find_spec("mujoco") is not None


def _radial(n):
    from virturoid.services.morphology_builder import MorphologyBuilder
    b = MorphologyBuilder("legged", base_mount="free", species=f"r{n}")
    b.base("torso", shape="box", length=0.08, radius=0.18, mass=3.0)
    for i in range(n):
        a = 2 * math.pi * i / n
        b.limb(f"leg{i}", [{"length": 0.12, "axis": (0, 1, 0), "torque": 14.0},
                           {"length": 0.12, "axis": (0, 1, 0), "torque": 12.0}],
               parent="torso", mount_offset=(0.16 * math.cos(a), 0.16 * math.sin(a), -0.04),
               mount_euler=(math.pi, 0, 0))
    return b.build()


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class PerceptiveAvoidTests(unittest.TestCase):
    def test_avoid_score_runs_and_feeds_the_perception_token(self):
        from virturoid.services.morph_policy import MorphPolicy
        from virturoid.services.morph_trainer import avoid_score
        from virturoid.services.policy_flywheel import _feature_dim_for

        body = _radial(4)
        fdim = _feature_dim_for(body)
        score = avoid_score(body, MorphPolicy(fdim, seed=0), goal_xy=(2.0, 0.0), n_rays=8, steps=120)
        self.assertIsInstance(score, float)
        self.assertGreater(score, -10.0)        # a finite, non-blowup score on the sense-and-avoid task

    def test_training_the_perceptive_policy_does_not_regress(self):
        # Same elitist ES loop as locomotion via fitness_fn -> the trained perceptive policy is >= the
        # gen-0 baseline (monotonic). Deterministic, so this is a non-flaky proof the loop trains.
        from virturoid.services.morph_policy import MorphPolicy
        from virturoid.services.morph_trainer import avoid_score, train_morph_avoid
        from virturoid.services.policy_flywheel import _feature_dim_for

        body = _radial(4)
        fdim = _feature_dim_for(body)
        baseline = avoid_score(body, MorphPolicy(fdim, seed=0), goal_xy=(2.0, 0.0), n_rays=8, steps=120)
        trained, hist = train_morph_avoid([body], goal_xy=(2.0, 0.0), feature_dim=fdim, n_rays=8,
                                          generations=3, pop=6, steps=120, seed=0)
        self.assertEqual(len(hist), 4)           # baseline + 3 generations
        trained_score = avoid_score(body, trained, goal_xy=(2.0, 0.0), n_rays=8, steps=120)
        self.assertGreaterEqual(trained_score + 1e-6, baseline)   # elitism: never worse than the start

    def test_perceptive_training_preserves_feature_dim_and_transfer_contract(self):
        # The perceptive policy is still a plain MorphPolicy (feature_dim unchanged) -> it drives ANY body
        # and round-trips through the flat ES vector, so transfer + banking keep working.
        import numpy as np
        from virturoid.services.morph_policy import MorphPolicy
        from virturoid.services.morph_trainer import train_morph_avoid
        from virturoid.services.policy_flywheel import _feature_dim_for

        body = _radial(4)
        fdim = _feature_dim_for(body)
        trained, _ = train_morph_avoid([body], goal_xy=(2.0, 0.0), feature_dim=fdim, n_rays=8,
                                       generations=1, pop=4, steps=100, seed=0)
        self.assertEqual(fdim, trained.feature_dim)
        # flat ES vector round-trips (banking/transfer rely on it), now including the perception weights.
        flat = trained.get_params()
        self.assertEqual(flat.size, trained.n_params)
        trained.set_params(flat)
        self.assertTrue(np.allclose(trained.get_params(), flat))
        # drives a DIFFERENT morphology (hexapod) — feature_dim is constant, so perception transfers too.
        from virturoid.services.morph_policy import rollout_morph
        r = rollout_morph(_radial(6), trained, steps=150)
        self.assertTrue(r["finite"])


if __name__ == "__main__":
    unittest.main()
