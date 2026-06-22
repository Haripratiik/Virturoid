"""Theme 1 keystone: a morphology-agnostic policy interface — ONE policy (fixed feature_dim, one
parameter set) drives ANY composed body (any token count). Validated CPU-first before MJX/PPO."""

import importlib.util
import math
import unittest

_MUJOCO = importlib.util.find_spec("mujoco") is not None


def _radial_legged(n):
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


def _gene(rc, prompt="x"):
    from virturoid.services.morphology_composer import compose_from_spec, morphology_from_requirements
    return compose_from_spec(morphology_from_requirements(0.65, 0.25, prompt=prompt, robot_class=rc))


class ParamTests(unittest.TestCase):
    def test_params_roundtrip(self):
        import numpy as np
        from virturoid.services.morph_policy import MorphPolicy
        p = MorphPolicy(23, seed=3)
        flat = p.get_params()
        self.assertEqual(flat.size, p.n_params)
        flat2 = flat + 0.5
        p.set_params(flat2)
        self.assertTrue(np.allclose(p.get_params(), flat2))

    def test_from_npz_roundtrip(self):
        import os
        import tempfile

        import numpy as np
        from virturoid.services.morph_policy import MorphPolicy
        p = MorphPolicy(24, hidden=32, seed=5)
        a = p._arrs
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "p.npz")
            np.savez(path, meta=np.array([24, 32, 8, 0.5]), **{k: a[k] for k in MorphPolicy._ORDER})
            q = MorphPolicy.from_npz(path)               # the GPU→CPU transfer loader
        self.assertEqual(q.feature_dim, 24)
        self.assertTrue(np.allclose(q.get_params(), p.get_params()))


class PerceptionTokenTests(unittest.TestCase):
    """The exteroceptive perception token must add range-sensing WITHOUT touching feature_dim, the proprio
    forward, or old banked .npz (perception-blind backcompat). CPU/NumPy only — no MuJoCo needed."""

    def _obs(self, n=5, fd=24, seed=1):
        import numpy as np
        return np.random.default_rng(seed).normal(0, 1, (n, fd))

    def test_feature_dim_and_param_count(self):
        from virturoid.services.morph_policy import MorphPolicy
        p = MorphPolicy(24, hidden=32, seed=0)
        self.assertEqual(24, p.feature_dim)                       # per-token width unchanged
        # base 4929 (We 768 + Wq/Wk/Wv/Wo 4*1024 + Wh 32 + bh 1 + be 32) + Wrange 10*32 + brange 32.
        self.assertEqual(4929 + p.percept_dim * 32 + 32, p.n_params)
        self.assertEqual(p.n_params, p.get_params().size)

    def test_ranges_none_is_byte_identical_to_proprio_forward(self):
        import numpy as np
        from virturoid.services.morph_policy import MorphPolicy
        p = MorphPolicy(24, seed=2)
        obs = self._obs()
        # The perception path is strictly opt-in: no ranges == the original forward, exactly.
        self.assertTrue(np.array_equal(p.act(obs), p.act(obs, ranges=None)))

    def test_perception_token_changes_actions(self):
        import numpy as np
        from virturoid.services.morph_policy import MorphPolicy
        p = MorphPolicy(24, seed=3)
        obs = self._obs()
        a0 = p.act(obs)
        a1 = p.act(obs, ranges=np.full(8, 3.0), cmd=[0.8, 0.0])   # open all around
        a2 = p.act(obs, ranges=np.full(8, 0.1), cmd=[0.0, 0.8])   # walled in
        self.assertEqual(a1.shape, (obs.shape[0],))              # action head still per joint token
        self.assertFalse(np.allclose(a0, a1))                    # perception influences the action
        self.assertFalse(np.allclose(a1, a2))                    # different percepts -> different actions

    def test_old_npz_without_perception_keys_loads_perception_blind(self):
        import os
        import tempfile

        import numpy as np
        from virturoid.services.morph_policy import MorphPolicy
        p = MorphPolicy(24, hidden=32, seed=5)
        old_keys = ("We", "be", "Wq", "Wk", "Wv", "Wo", "Wh", "bh")   # pre-perception format (8 keys)
        obs = self._obs()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "old.npz")
            np.savez(path, meta=np.array([24, 32, 8, 0.5]), **{k: p._arrs[k] for k in old_keys})
            q = MorphPolicy.from_npz(path)                       # must NOT KeyError on missing Wrange/brange
        self.assertEqual(24, q.feature_dim)
        # Loaded the shared proprio weights -> perception-blind act() is identical to the source policy.
        self.assertTrue(np.array_equal(q.act(obs), p.act(obs)))


@unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
class MorphGraphTests(unittest.TestCase):
    def _encode(self, gene):
        import mujoco
        from virturoid.services.gene_compiler import compile_gene_to_mjcf
        from virturoid.services.morph_graph import encode_robot
        return encode_robot(mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene)))

    def test_feature_dim_constant_across_morphologies(self):
        bodies = [_gene("quadruped"), _gene("manipulator", "grasp"), _gene("mobile_base"),
                  _radial_legged(6), _radial_legged(8)]
        dims = {self._encode(g).feature_dim for g in bodies}
        self.assertEqual(len(dims), 1, f"feature_dim must be identical across bodies, got {dims}")

    def test_token_count_and_parents_valid(self):
        g = self._encode(_radial_legged(6))
        self.assertEqual(g.n_tokens, 12)              # 6 legs x 2 joints
        for p in g.parent:
            self.assertTrue(-1 <= p < g.n_tokens)

    def test_one_policy_drives_any_body(self):
        from virturoid.services.morph_policy import MorphPolicy, rollout_morph
        fdim = self._encode(_gene("quadruped")).feature_dim
        policy = MorphPolicy(fdim, seed=1)            # ONE policy / one parameter set
        for g in [_gene("quadruped"), _gene("manipulator", "grasp"), _radial_legged(6), _radial_legged(8)]:
            r = rollout_morph(g, policy, steps=200, alpha=0.4)
            self.assertTrue(r["finite"], f"policy destabilized a body: {r}")
            self.assertEqual(r["feature_dim"], fdim)

    def test_one_policy_trains_across_a_distribution(self):
        # Theme 1: TRAIN one policy over a DISTRIBUTION of different-DOF bodies (8-DOF quadruped + 12-DOF
        # hexapod) and it drives them all — the morphology-agnostic learning loop, not just the interface.
        from virturoid.services.morph_policy import rollout_morph
        from virturoid.services.morph_trainer import train_morph_es
        bodies = [_gene("quadruped"), _radial_legged(6)]
        fdim = self._encode(bodies[0]).feature_dim
        policy, hist = train_morph_es(bodies, feature_dim=fdim, generations=2, pop=4, steps=120)
        self.assertEqual(len(hist), 3)                # baseline + 2 generations of mean fitness
        for g in bodies:                              # the one trained policy controls every body in the set
            r = rollout_morph(g, policy, steps=150)
            self.assertTrue(r["finite"])
            self.assertEqual(r["feature_dim"], fdim)


if __name__ == "__main__":
    unittest.main()
