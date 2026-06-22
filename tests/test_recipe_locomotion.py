"""The anti-collapse RECIPE locomotion path (see memory locomotion-collapse-fix): position-PD to the default
standing pose + obs normalization + terminate-on-fall + clipped-non-negative velocity-tracking reward. The
default torque-residual-on-gravity-comp ``rollout_morph`` collapses to a fall-forward FLOP (z 0.305 -> 0.08);
the recipe is what actually keeps a body UPRIGHT and walking. These tests pin (a) the recipe primitives run and
return well-formed stats, (b) the obs normalizer is applied, (c) the recipe training loop executes (serial +
recipe-aware), and — the crux — (d) the PD-to-default self-righting prior keeps even a RANDOM policy standing
where the torque-residual control does not."""

import importlib.util
import unittest

_MUJOCO = importlib.util.find_spec("mujoco") is not None


def _quad():
    from virturoid.services.morphology_composer import compose_robot
    return compose_robot("a robot dog that walks", llm=None)


@unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
class RecipeRolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from virturoid.services.morph_graph import encode_robot
        from virturoid.services.morph_policy import compiled_model, robot_mjcf
        cls.g = _quad()
        cls.fd = encode_robot(compiled_model(robot_mjcf(cls.g))).feature_dim

    def test_recipe_rollout_wellformed(self):
        from virturoid.services.morph_policy import MorphPolicy, recipe_rollout_morph
        pol = MorphPolicy(self.fd, seed=1)
        r = recipe_rollout_morph(self.g, pol, steps=200)
        self.assertTrue(r["finite"])
        for k in ("gait", "forward", "height_ratio", "alive", "survived", "speed"):
            self.assertIn(k, r)
        self.assertGreaterEqual(r["gait"], 0.0)          # reward is clipped non-negative (kills the suicide-lunge)
        self.assertLessEqual(r["alive"], 200)
        self.assertEqual(r["feature_dim"], self.fd)

    def test_obs_normalizer_shape_and_applied(self):
        import numpy as np
        from virturoid.services.morph_policy import MorphPolicy, recipe_obs_normalizer, recipe_rollout_morph
        mean, std = recipe_obs_normalizer(self.g, n_pol=2, steps=80)
        self.assertEqual(mean.shape, (self.fd,))
        self.assertEqual(std.shape, (self.fd,))
        self.assertTrue(np.all(np.isfinite(mean)) and np.all(std > 0))
        # the normalizer must actually feed the policy: a non-trivial normalizer changes the trajectory/reward
        pol = MorphPolicy(self.fd, seed=2)
        raw = recipe_rollout_morph(self.g, pol, steps=200)["gait"]
        nrm = recipe_rollout_morph(self.g, pol, steps=200, normalizer=(mean, std))["gait"]
        self.assertNotAlmostEqual(raw, nrm, places=4)

    def test_recipe_forward_score_matches_rollout(self):
        from virturoid.services.morph_policy import MorphPolicy, recipe_rollout_morph
        from virturoid.services.morph_trainer import recipe_forward_score
        pol = MorphPolicy(self.fd, seed=3)
        a = recipe_forward_score(self.g, pol, steps=200)
        b = recipe_rollout_morph(self.g, pol, steps=200)["gait"]
        self.assertAlmostEqual(a, b, places=6)

    def test_record_frames(self):
        from virturoid.services.morph_policy import MorphPolicy, recipe_rollout_morph
        pol = MorphPolicy(self.fd, seed=4)
        r = recipe_rollout_morph(self.g, pol, steps=60, record_frames=True, frame_every=5)
        self.assertGreater(len(r["frames"]), 0)

    def test_recipe_training_runs(self):
        from virturoid.services.morph_policy import recipe_obs_normalizer
        from virturoid.services.morph_trainer import train_morph_es
        norm = recipe_obs_normalizer(self.g, n_pol=2, steps=60)
        pol, hist = train_morph_es([self.g], feature_dim=self.fd, generations=1, pop=4, steps=120,
                                   seed=0, workers=1, recipe=True, normalizer=norm)
        self.assertEqual(pol.feature_dim, self.fd)
        self.assertEqual(len(hist), 2)                   # f0 + one generation

    def test_pd_to_default_self_rights_a_random_policy(self):
        """The crux of the fix: under PD-to-default control a RANDOM policy hovers near the standing pose
        (self-righting prior), so it stays UPRIGHT — where the torque-residual-on-grav-comp control (the old
        default ``rollout_morph``) lets the same random policy collapse. Recipe height_ratio must beat it."""
        from virturoid.services.morph_policy import MorphPolicy, recipe_rollout_morph, rollout_morph
        steps = 250
        pol = MorphPolicy(self.fd, seed=7)
        recipe_hr = recipe_rollout_morph(self.g, pol, steps=steps)["height_ratio"]
        residual_hr = rollout_morph(self.g, pol, steps=steps)["height_ratio"]
        self.assertGreater(recipe_hr, 0.5, f"PD-to-default should keep a random policy standing (hr={recipe_hr})")
        self.assertGreaterEqual(recipe_hr, residual_hr,
                                f"recipe hr {recipe_hr} should be >= residual hr {residual_hr}")


if __name__ == "__main__":
    unittest.main()
