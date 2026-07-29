"""The assisted training loop measures each candidate with the CONTROLLER IT WAS TRAINED UNDER.

A recipe/CPG policy (banked GPU policies, recipe-ES policies) driven by the bare-residual ``rollout_morph`` runs
WITHOUT its gait prior and produces a different, invalid measurement. train_assisted used to measure EVERY candidate
that way, so it under-measured banked/GPU recipe policies -> it skipped reuse and retrained needlessly (the same misrouting
``learn_locomotion.locomotion_episode`` already fixed). ``_measure_travel`` routes a recipe/cpg policy to
``recipe_rollout_morph`` (deploy == measure) and a plain policy to ``rollout_morph``.
"""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("MUJOCO_GL", "glfw")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "needs MuJoCo — the rollout IS the measurement")
class MeasureTravelRoutingTests(unittest.TestCase):
    def _quad_and_policies(self):
        from virturoid.services.morph_graph import encode_robot
        from virturoid.services.morph_policy import CPG_DEFAULT, MorphPolicy, compiled_model, robot_mjcf
        from virturoid.services.morphology_composer import compose_robot
        g = compose_robot("a quadruped robot dog", ensure_walkable=True)
        graph = encode_robot(compiled_model(robot_mjcf(g)))
        cpg = MorphPolicy(graph.feature_dim, seed=0)
        cpg.cpg = CPG_DEFAULT                                     # mark it a recipe/CPG policy (carries a gait prior)
        plain = MorphPolicy(graph.feature_dim, seed=0)           # a bare-residual policy (no recipe metadata)
        return g, cpg, plain

    def test_cpg_policy_is_measured_via_the_recipe_rollout(self):
        from virturoid.services.assisted_trainer import _measure_travel
        from virturoid.services.morph_policy import recipe_rollout_morph, rollout_morph
        g, cpg, _ = self._quad_and_policies()
        recipe = recipe_rollout_morph(g, cpg, steps=800)         # the correct controller for a CPG policy
        wrong = rollout_morph(g, cpg, steps=800)                 # the OLD, buggy no-CPG measurement
        m = _measure_travel(g, cpg, steps=800)
        # routed to recipe_rollout_morph -> its forward matches that rollout exactly (deterministic, same controller)
        self.assertAlmostEqual(m["forward"], float(recipe["forward"]), places=5)
        # Routing matters: only the recipe result is admissible for a policy trained with a CPG prior. The exact
        # failure mode of the mismatched controller may change as token features improve, but its result must differ.
        self.assertTrue(m["upright"], "a CPG policy stays upright under recipe_rollout_morph")
        self.assertNotAlmostEqual(float(recipe["forward"]), float(wrong["forward"]), places=3,
                                  msg="the two rollouts must differ for a CPG policy, else the routing is moot")

    def test_plain_policy_is_measured_via_the_plain_rollout(self):
        from virturoid.services.assisted_trainer import _measure_travel
        from virturoid.services.morph_policy import rollout_morph
        g, _, plain = self._quad_and_policies()
        plainr = rollout_morph(g, plain, steps=800)
        m = _measure_travel(g, plain, steps=800)
        self.assertAlmostEqual(m["forward"], float(plainr["forward"]), places=5)
        self.assertEqual(m["upright"], bool(plainr.get("upright")))

    def test_deployed_rollout_marks_which_controller_was_measured(self):
        from virturoid.services.morph_policy import rollout_deployed_morph_policy

        g, cpg, plain = self._quad_and_policies()
        self.assertEqual("recipe_cpg", rollout_deployed_morph_policy(g, cpg, steps=300)["deployment_controller"])
        self.assertEqual("residual", rollout_deployed_morph_policy(g, plain, steps=300)["deployment_controller"])

    def test_recipe_diagnosis_uses_the_rollout_speed_not_a_magic_divisor(self):
        from virturoid.services.assisted_trainer import _recipe_diagnosis

        diag = _recipe_diagnosis({"forward": 0.9, "speed": 0.5, "cadence": 2.0,
                                  "upright_frac": 0.95, "height_ratio": 0.9, "survived": True})
        self.assertEqual(0.5, diag["speed"])
        self.assertIn("0.50m/s", diag["summary"])

    def test_gpu_deploy_selection_can_keep_an_earlier_checkpoint(self):
        from virturoid.services.assisted_trainer import select_best_gpu_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            final = Path(tmp) / "gpu_quad.npz"
            early = Path(tmp) / "gpu_quad_it40.npz"
            final.write_bytes(b"final"); early.write_bytes(b"early")

            selected = select_best_gpu_checkpoint(
                object(), str(final),
                load_policy=lambda path: Path(path).name,
                measure=lambda policy: {
                    "forward": 0.82 if policy.endswith("it40.npz") else 0.31,
                    "upright": policy.endswith("it40.npz"),
                },
            )

        self.assertIsNotNone(selected)
        self.assertTrue(selected["path"].endswith("gpu_quad_it40.npz"))
        self.assertAlmostEqual(0.82, selected["forward"])


if __name__ == "__main__":
    unittest.main()
