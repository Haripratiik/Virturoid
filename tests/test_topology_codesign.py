"""Topology co-design V1: invent leg count, scored ZERO-SHOT by one shared transferable policy
(no per-candidate training), never-regress. Gated on a banked policy (models/morph_quad_att.npz)."""

import importlib.util
import unittest
from pathlib import Path

_MUJOCO = importlib.util.find_spec("mujoco") is not None
_NPZ = Path(__file__).resolve().parent.parent / "models" / "morph_quad_att.npz"


@unittest.skipUnless(_MUJOCO and _NPZ.exists(), "needs MuJoCo + a banked transferable policy")
class TopologyCodesignTests(unittest.TestCase):
    def test_leg_count_search_zero_shot(self):
        from virturoid.services.morph_policy import MorphPolicy
        from virturoid.services.topology_codesign import search_leg_count

        pol = MorphPolicy.from_npz(str(_NPZ))                 # ONE policy scores every topology
        res = search_leg_count(pol, n_range=range(3, 7), steps=400, baseline_n=4)
        self.assertEqual(set(res["history"]), {3, 4, 5, 6})    # all candidates scored
        self.assertIn(res["best_n_legs"], (3, 4, 5, 6))
        self.assertEqual(res["best_score"], max(res["history"].values()))  # picks the best
        self.assertEqual(res["best_gene"].validate(), [])     # winner is a buildable gene
        # at least one topology is viable (finite, upright) under the shared policy
        self.assertGreater(max(res["history"].values()), float("-inf"))

    def test_2d_morphology_search_legs_and_dofs(self):
        from virturoid.services.morph_policy import MorphPolicy
        from virturoid.services.topology_codesign import search_morphology

        pol = MorphPolicy.from_npz(str(_NPZ))
        res = search_morphology(pol, leg_counts=range(3, 6), dof_options=(2, 3), steps=300)
        self.assertEqual(len(res["history"]), 6)              # 3 leg counts x 2 DOF options
        self.assertIsInstance(res["best"], tuple)             # (n_legs, dofs)
        self.assertEqual(res["best_gene"].validate(), [])
        self.assertEqual(res["best_score"], max(res["history"].values()))

    def test_never_regresses_off_baseline(self):
        from virturoid.services.morph_policy import MorphPolicy
        from virturoid.services.steerable_body import steerable_quadruped
        from virturoid.services.topology_codesign import search_topology

        pol = MorphPolicy.from_npz(str(_NPZ))
        cands = {"a": steerable_quadruped(n_legs=4), "b": steerable_quadruped(n_legs=4)}  # equal
        res = search_topology(cands, pol, steps=300, baseline="a")
        self.assertEqual(res["best"], "a")                    # ties keep the incumbent (no regress)


if __name__ == "__main__":
    unittest.main()
