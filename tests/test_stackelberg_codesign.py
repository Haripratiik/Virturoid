"""Stackelberg best-response co-design (Pillar 1 keystone): score a body by the best controller you can
train on it, not a fixed CPG.

The honest-score ordering is a fast pure test; the best-response search + leader-follower loop are
MuJoCo-gated and run with tiny settings (a handful of short rollouts) to stay quick.
"""

import importlib.util
import unittest

from virturoid.fixtures.gene_library import quadruped_gene, tabletop_arm_gene
from virturoid.services.stackelberg_codesign import (
    best_response_locomotion, best_response_score, co_design_stackelberg, honest_locomotion_score,
)

_MUJOCO = importlib.util.find_spec("mujoco") is not None


class HonestScoreTests(unittest.TestCase):
    def test_walk_outranks_slide_outranks_fall(self):
        walk = honest_locomotion_score({"survived": True, "forward": 0.6, "cadence": 6.0, "upright_frac": 0.75})
        slide = honest_locomotion_score({"survived": True, "forward": 0.6, "cadence": 0.0, "upright_frac": 0.80})
        fall = honest_locomotion_score({"survived": False, "forward": 0.5, "cadence": 2.0, "upright_frac": 0.30})
        self.assertGreater(walk, slide)      # a real stepping gait beats an upright slide (same forward travel)
        self.assertGreater(slide, fall)      # and an upright slide beats a forward lunge-and-fall
        self.assertGreater(walk, 0.5)

    def test_forward_alone_does_not_win(self):
        # a fast SLIDE (cadence 0) must not beat a slower real WALK — the anti-Goodhart property
        fast_slide = honest_locomotion_score({"survived": True, "forward": 1.2, "cadence": 0.0, "upright_frac": 0.9})
        real_walk = honest_locomotion_score({"survived": True, "forward": 0.5, "cadence": 5.0, "upright_frac": 0.7})
        self.assertGreater(real_walk, fast_slide)


@unittest.skipUnless(_MUJOCO, "MuJoCo required for rollout-based best-response")
class BestResponseTests(unittest.TestCase):
    def test_best_response_returns_a_controller_and_beats_the_fixed_default(self):
        from virturoid.services.morph_policy import CPG_DEFAULT, recipe_rollout_morph
        gene = quadruped_gene()
        # honest score of the FIXED default controller (what the old eval used)
        default = honest_locomotion_score(recipe_rollout_morph(gene, steps=150, kp=32.0, kd=1.5,
                                                               cpg=dict(CPG_DEFAULT), seed=0))
        br = best_response_locomotion(gene, samples=2, iters=1, steps=150, seed=0)
        self.assertIn("leg_flip", br["controller"])
        self.assertIn("freq", br["controller"])
        # the Stackelberg guarantee: the best-response is never worse than the fixed controller
        # (the default is among the candidates), so the body's fitness reflects its trainability
        self.assertGreaterEqual(br["score"], default - 1e-6)

    def test_best_response_score_dispatches_by_morphology(self):
        legged = best_response_score(quadruped_gene(), samples=2, iters=1, steps=150, seed=0)
        self.assertEqual(legged["kind"], "legged")
        self.assertIsNotNone(legged["controller"])          # legged bodies get a trained follower
        arm = best_response_score(tabletop_arm_gene(), prompt="sort blocks into bins")
        self.assertEqual(arm["kind"], "manipulator")
        self.assertIsNone(arm["controller"])                # non-legged: fixed-controller fallback (for now)

    def test_co_design_stackelberg_never_regresses_and_returns_a_follower(self):
        gene = quadruped_gene()
        out = co_design_stackelberg(gene, iterations=1, population=2, br_samples=2, br_iters=1,
                                    br_steps=150, seed=0)
        self.assertGreaterEqual(out["best_value"], out["baseline_value"])   # seed body always a candidate
        self.assertIsNotNone(out["best_controller"])
        self.assertIn("leg_flip", out["best_controller"])
        self.assertIn("length_scale", out["changed"])
        self.assertTrue(out["gene"].segments)               # returns a valid, buildable body


if __name__ == "__main__":
    unittest.main()
