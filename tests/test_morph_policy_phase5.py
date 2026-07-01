"""Phase 5: MorphPolicy opt-in FiLM + topology-attention-bias (the SOTA tokenizer upgrades), CPU-validated.

The critical property: these are IDENTITY when off/untrained (zero-init), so default + banked policies are
byte-identical and unchanged — the upgrades only bite once GPU/ES training moves their weights.
"""

import unittest

import numpy as np

from virturoid.services.morph_policy import MorphPolicy
from virturoid.services.topo_pe import hop_distance_matrix


def _obs(n=4, fd=24, seed=1):
    return np.random.default_rng(seed).normal(0, 1, (n, fd))


class MorphPolicyPhase5Tests(unittest.TestCase):
    def test_default_policy_is_unchanged(self):
        p = MorphPolicy(24, seed=0)
        self.assertFalse(p.film)
        self.assertFalse(p.topo_bias)
        self.assertEqual(p._order, list(MorphPolicy._ORDER))        # no extra params by default
        self.assertEqual(p.get_params().size, p.n_params)

    def test_film_off_is_byte_identical(self):
        obs = _obs()
        off = MorphPolicy(24, seed=0)
        on = MorphPolicy(24, seed=0, film=True)                     # same seed -> base weights identical
        # Wfilm is zero-init -> FiLM is identity -> forward matches the plain policy exactly
        self.assertTrue(np.allclose(on.act(obs), off.act(obs)))
        self.assertGreater(on.n_params, off.n_params)              # but the capacity is there to train

    def test_film_changes_output_once_trained(self):
        obs = _obs()
        off = MorphPolicy(24, seed=0)
        on = MorphPolicy(24, seed=0, film=True)
        on._arrs["Wfilm"][...] = np.random.default_rng(2).normal(0, 0.5, on._arrs["Wfilm"].shape)
        self.assertFalse(np.allclose(on.act(obs), off.act(obs)))   # a trained FiLM modulates the tokens

    def test_topo_bias_off_and_zero_is_identity(self):
        obs = _obs(n=4)
        hop = hop_distance_matrix([-1, 0, 1, 0])
        off = MorphPolicy(24, seed=0)
        on = MorphPolicy(24, seed=0, topo_bias=True)
        # zero-init bias table -> even WITH a hop matrix, attention is unchanged
        self.assertTrue(np.allclose(on.act(obs, hop=hop), off.act(obs)))
        # and a None hop is always identity regardless of training
        on._arrs["Wtopo"][...] = np.random.default_rng(3).normal(0, 0.5, on._arrs["Wtopo"].shape)
        self.assertTrue(np.allclose(on.act(obs, hop=None), off.act(obs)))

    def test_topo_bias_changes_attention_once_trained(self):
        obs = _obs(n=4)
        hop = hop_distance_matrix([-1, 0, 1, 0])
        off = MorphPolicy(24, seed=0)
        on = MorphPolicy(24, seed=0, topo_bias=True)
        on._arrs["Wtopo"][...] = np.array([0.0, 1.5, -1.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertFalse(np.allclose(on.act(obs, hop=hop), off.act(obs)))

    def test_topo_bias_policy_deploys_through_recipe_rollout(self):
        """A topo-bias policy must run the recipe deploy path with its hop matrix wired (train==deploy)."""
        from virturoid.fixtures.gene_library import quadruped_gene
        from virturoid.services.morph_graph import encode_robot
        from virturoid.services.morph_policy import compiled_model, recipe_rollout_morph, robot_mjcf
        gene = quadruped_gene()
        g = encode_robot(compiled_model(robot_mjcf(gene)))
        pol = MorphPolicy(g.feature_dim, topo_bias=True)        # zero Wtopo -> runs; hop must be computed + passed
        r = recipe_rollout_morph(gene, pol, steps=60)
        self.assertTrue(r["finite"])
        self.assertIn("forward", r)

    def test_npz_round_trip_preserves_opt_in_weights(self):
        import tempfile
        from pathlib import Path
        p = MorphPolicy(24, seed=0, film=True, topo_bias=True)
        p._arrs["Wfilm"][...] = np.random.default_rng(4).normal(0, 0.3, p._arrs["Wfilm"].shape)
        p._arrs["Wtopo"][...] = np.random.default_rng(5).normal(0, 0.3, p._arrs["Wtopo"].shape)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.npz"
            p.to_npz(path)
            q = MorphPolicy.from_npz(path)
            self.assertTrue(q.film and q.topo_bias)
            self.assertTrue(np.allclose(q.get_params(), p.get_params()))
            obs = _obs()
            self.assertTrue(np.allclose(q.act(obs), p.act(obs)))


if __name__ == "__main__":
    unittest.main()
