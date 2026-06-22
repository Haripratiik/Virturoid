"""Adaptive per-joint PD gains — the "any robot" generalization of the locomotion recipe (task #22).

The scalar recipe (kp=32, kd=1.5) is tuned for the reference quadruped's joint inertia. ``recipe_gains`` derives
per-joint gains from each DOF's effective inertia so the closed-loop response (natural frequency wn, damping zeta)
is CONSTANT across morphologies — at the reference inertia it reproduces ~32/1.5/0.4 (the ANCHOR, so the tuned
quad never regresses), and a humanoid's heavy hips get the stiffness they need to hold against gravity. These
tests pin: (a) the anchor, (b) shapes/bounds, (c) inertia-scaling, (d) the auto-gate (quad off / humanoid on),
(e) scalar parity (default path byte-unchanged), (f) the npz flag round-trip, (g) banked flag drives the rollout,
and (h) the train==replay gains invariant (the viewport replays with the SAME gains training used)."""

import importlib.util
import unittest

_MUJOCO = importlib.util.find_spec("mujoco") is not None


def _compiled(gene):
    import mujoco
    from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
    from virturoid.services.morph_graph import encode_robot
    xml = compile_gene_to_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene, meshed=False))
    m = mujoco.MjModel.from_xml_string(xml); m.opt.iterations = 20
    return m, encode_robot(m)


def _quad():
    from virturoid.fixtures.gene_library import quadruped_gene
    return quadruped_gene()


def _humanoid():
    from virturoid.services.morphology_composer import compose_robot
    return compose_robot("a humanoid robot", llm=None)


@unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
class RecipeGainsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gq = _quad()
        cls.mq, cls.grq = _compiled(cls.gq)

    def test_reference_quad_reproduces_anchor(self):
        """The whole design depends on this: at the reference inertia the adaptive gains == the tuned scalar, so
        enabling them never moves the anchor body. Tight tolerances lock the invariant (the cap is floored at
        _KP_REF so it can't clip the median quad joint below 32)."""
        import numpy as np
        from virturoid.services.morph_policy import recipe_gains
        kp, kd, asc = recipe_gains(self.mq, self.grq)
        self.assertAlmostEqual(float(np.median(kp)), 32.0, delta=0.05)     # exact (cap floored at _KP_REF)
        self.assertAlmostEqual(float(np.median(kd)), 1.5, delta=0.02)      # ~1.5 (only the _I_REF freeze rounding)
        self.assertTrue(np.allclose(asc, 0.4))

    def test_gains_shapes_and_bounds(self):
        import numpy as np
        from virturoid.services.morph_policy import recipe_gains
        kp, kd, asc = recipe_gains(self.mq, self.grq)
        for v in (kp, kd, asc):
            self.assertEqual(v.shape, (self.grq.n_tokens,))
        self.assertTrue(np.all(kp >= 2.0) and np.all(kp <= 192.0))
        self.assertTrue(np.all(kd >= 0.1) and np.all(kd <= 24.0))

    def test_kp_scales_with_inertia(self):
        """A humanoid's heavy joints must get materially higher stiffness than the quad's — that is the point."""
        import numpy as np
        from virturoid.services.morph_policy import recipe_gains
        kp_q, _, _ = recipe_gains(self.mq, self.grq)
        mh, grh = _compiled(_humanoid())
        kp_h, _, _ = recipe_gains(mh, grh)
        self.assertGreater(float(np.median(kp_h)), float(np.median(kp_q)) * 1.2)

    def test_adaptive_recommended_gate(self):
        """The auto-gate keeps the near-reference quad on scalar gains and turns adaptive ON for the humanoid."""
        from virturoid.services.morph_policy import adaptive_recommended
        self.assertFalse(adaptive_recommended(self.mq, self.grq))
        mh, grh = _compiled(_humanoid())
        self.assertTrue(adaptive_recommended(mh, grh))

    def test_light_jointed_legged_bodies_stay_scalar(self):
        """The gate is ONE-SIDED: a body LIGHTER than the reference (a dog / a spider) is only over-stiffened by
        the scalar kp=32 (safe), so it must KEEP the validated scalar gait — never silently swapped to a softer,
        never-validated adaptive controller. Regression guard for the auto-gate."""
        from virturoid.services.morphology_composer import compose_robot
        from virturoid.services.morph_policy import adaptive_recommended
        for prompt in ("a dog robot", "a spider robot"):
            m, gr = _compiled(compose_robot(prompt, llm=None))
            self.assertFalse(adaptive_recommended(m, gr), f"{prompt!r} must keep scalar gains")

    def test_scalar_path_is_unchanged(self):
        """Default (non-adaptive) rollout must be byte-identical to a hand-broadcast scalar — the existing recipe
        path cannot shift. A fresh policy has adaptive_gains=False, so the default IS the scalar path."""
        from virturoid.services.morph_policy import MorphPolicy, recipe_rollout_morph
        pol = MorphPolicy(self.grq.feature_dim, seed=11)
        self.assertFalse(pol.adaptive_gains)
        a = recipe_rollout_morph(self.gq, pol, steps=200)["gait"]
        b = recipe_rollout_morph(self.gq, pol, steps=200, kp=32.0, kd=1.5, ascale=0.4)["gait"]
        self.assertAlmostEqual(a, b, places=9)

    def test_npz_round_trips_adaptive_flag(self):
        import os
        import tempfile

        import numpy as np
        from virturoid.services.morph_policy import MorphPolicy
        p = MorphPolicy(self.grq.feature_dim, seed=5)
        p.adaptive_gains = True
        norm = (np.zeros(self.grq.feature_dim), np.ones(self.grq.feature_dim))
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "p.npz")
            p.to_npz(path, 0.5, normalizer=norm)
            q = MorphPolicy.from_npz(path)
            self.assertTrue(q.adaptive_gains)
            # an old 4-element-meta npz (no adaptive slot) must still load and default to scalar.
            old = os.path.join(td, "old.npz")
            np.savez(old, meta=np.array([self.grq.feature_dim, 32, 8, 0.5]),
                     **{k: p._arrs[k] for k in MorphPolicy._ORDER})
            self.assertFalse(MorphPolicy.from_npz(old).adaptive_gains)

    def test_banked_flag_drives_adaptive_rollout(self):
        """A policy carrying adaptive_gains=True makes recipe_rollout_morph take the adaptive path WITHOUT an
        explicit adaptive= arg (mirrors how the banked obs normalizer is auto-applied)."""
        from virturoid.services.morph_policy import MorphPolicy, recipe_rollout_morph
        mh, grh = _compiled(_humanoid())
        gh = _humanoid()
        pol = MorphPolicy(grh.feature_dim, seed=7)
        scalar = recipe_rollout_morph(gh, pol, steps=200)["gait"]
        pol.adaptive_gains = True
        flagged = recipe_rollout_morph(gh, pol, steps=200)["gait"]
        explicit = recipe_rollout_morph(gh, MorphPolicy(grh.feature_dim, seed=7), steps=200, adaptive=True)["gait"]
        self.assertNotAlmostEqual(scalar, flagged, places=5)              # the flag changed the control
        self.assertAlmostEqual(flagged, explicit, places=9)              # flag == explicit adaptive=True

    def test_replay_site_honors_adaptive_flag(self):
        """The invariant that prevents a train/replay mismatch: the REAL viewport replay site (rollout_view) must
        drive an adaptive policy with the inertia-scaled gains, NOT the hardcoded scalar 32/1.5. We prove it by
        rolling the SAME policy weights out through rollout_view once flagged adaptive and once scalar — if the
        replay honored the flag the recorded trajectories diverge; if it ignored it (the bug) they'd be identical."""
        import numpy as np
        from virturoid.services.learn_locomotion import rollout_view
        from virturoid.services.morph_policy import MorphPolicy
        gh = _humanoid()
        _, grh = _compiled(gh)

        def view_for(adaptive):
            p = MorphPolicy(grh.feature_dim, seed=3)
            p.obs_mean = np.zeros(grh.feature_dim); p.obs_std = np.ones(grh.feature_dim)  # mark as a recipe policy
            p.adaptive_gains = adaptive
            v, _ = rollout_view(gh, p, steps=80, frame_every=4)
            return np.asarray(v["frames"][-1], dtype=float)                  # last recorded geom-pose frame

        scalar_last, adaptive_last = view_for(False), view_for(True)
        self.assertEqual(scalar_last.shape, adaptive_last.shape)
        self.assertFalse(np.allclose(scalar_last, adaptive_last),
                         "rollout_view ignored adaptive_gains — replayed the adaptive policy with scalar 32/1.5")


if __name__ == "__main__":
    unittest.main()
