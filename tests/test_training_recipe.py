"""The GPU-training recipe is chosen AUTOMATICALLY per body (one source of truth), not hand-toggled per call-site.

Enforces the deploy-safe INVARIANT phase_obs==cpg: a gait-phase clock is only fed at deploy WITH a CPG source
(recipe_rollout_morph gates it on cpg_on), so phase_obs WITHOUT cpg silently drops the clock at deploy and the
gait drifts out of phase and falls -- the exact deploy gap diagnosed on the humanoid. The recipe makes that
mismatch unrepresentable, and auto-enables adaptive (inertia-scaled) gains for heavy/humanoid bodies.
"""
import importlib.util
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("MUJOCO_GL", "glfw")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "recipe selection compiles the body")
class TrainingRecipeTests(unittest.TestCase):
    def _recipe(self, prompt):
        from virturoid.services.gpu_trainer import default_training_recipe
        from virturoid.services.morphology_composer import compose_robot
        return default_training_recipe(compose_robot(prompt, ensure_walkable=True))

    def test_phase_obs_equals_cpg_invariant_holds_for_every_body(self):
        # the deploy-safe invariant: never a phase clock without a CPG source (that was the humanoid deploy gap)
        for p in ("a humanoid robot", "a quadruped robot dog", "a hexapod robot", "a robot arm that grasps",
                  "a quadcopter drone", "a snake robot"):
            r = self._recipe(p)
            self.assertEqual(r["phase_obs"], r["cpg"], f"phase_obs must equal cpg for {p}: {r}")

    def test_multi_leg_bodies_get_the_cpg_gait_prior(self):
        self.assertTrue(self._recipe("a quadruped robot dog")["cpg"])   # >=3 legs -> trot-CPG helps
        self.assertTrue(self._recipe("a hexapod robot")["cpg"])

    def test_biped_skips_the_quad_trot_cpg(self):
        # MEASURED: a humanoid trained WITHOUT cpg deployed 0.34m stable vs 0.16m with the quad-trot CPG prior
        self.assertFalse(self._recipe("a humanoid robot")["cpg"], "a 2-leg body should not get the quad-trot CPG")

    def test_non_legged_bodies_get_no_cpg(self):
        self.assertFalse(self._recipe("a robot arm that grasps")["cpg"])
        self.assertFalse(self._recipe("a quadcopter drone")["cpg"])

    def test_humanoid_auto_enables_adaptive_gains_quad_does_not(self):
        # a heavy/humanoid body needs inertia-scaled per-joint gains; a near-reference quad keeps the scalar gains
        self.assertTrue(self._recipe("a humanoid robot")["adaptive"], "a humanoid must auto-get adaptive gains")
        self.assertFalse(self._recipe("a quadruped robot dog")["adaptive"], "a quad keeps the validated scalar gains")

    def test_recipe_carries_the_deploy_gap_deltas(self):
        r = self._recipe("a quadruped robot dog")
        for k in ("dr", "contact_dr", "sphere_feet"):
            self.assertTrue(r[k], f"{k} (a deploy-gap delta) must be on")
        self.assertEqual(r["decimation"], 4)                   # 50 Hz control (train==deploy)


if __name__ == "__main__":
    unittest.main()
