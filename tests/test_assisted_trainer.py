"""The assisted training loop measures each candidate with the CONTROLLER IT WAS TRAINED UNDER.

A recipe/CPG policy (banked GPU policies, recipe-ES policies) driven by the bare-residual ``rollout_morph`` runs
WITHOUT its gait prior and reads as FALLEN. train_assisted used to measure EVERY candidate that way, so it
under-measured banked/GPU recipe policies -> it skipped reuse and retrained needlessly (the same misrouting
``learn_locomotion.locomotion_episode`` already fixed). ``_measure_travel`` routes a recipe/cpg policy to
``recipe_rollout_morph`` (deploy == measure) and a plain policy to ``rollout_morph``.
"""
import importlib.util
import os
import unittest

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
        # routing MATTERS: a CPG policy holds up under its own rollout but the bare-residual one FELLS it
        self.assertTrue(m["upright"], "a CPG policy stays upright under recipe_rollout_morph")
        self.assertFalse(bool(wrong.get("upright")), "the no-CPG rollout wrongly reads the CPG policy as fallen")
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


if __name__ == "__main__":
    unittest.main()
