"""GEN-1 (docs/generality_plan.md): structural appendage discovery must classify ANY body's chains by
FUNCTION from contact + kinematics, never from names or robot_class. These are the acceptance tests that
made the keystone honest — a hexapod's 6 legs, a spider's 8, a snake's spine, a rover's wheels, an arm's
reach, all found where the old leg-name regex saw only leg{n}_{l|r} quads (measured: 0/20 tokens on the LLM
hexapod). Offline (llm=None) so they are deterministic in CI.
"""
import importlib.util
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class AppendageDiscoveryTests(unittest.TestCase):
    def _map(self, prompt):
        from virturoid.services.appendage_map import build_appendage_map
        from virturoid.services.morph_policy import compiled_model, robot_mjcf
        from virturoid.services.morphology_composer import compose_robot
        return build_appendage_map(compiled_model(robot_mjcf(compose_robot(prompt, llm=None))))

    def test_leg_count_from_structure_not_names(self):
        # the core generality claim: n_legs is READ from the body, so an 8-leg spider is 8 (the old name
        # regex counted 2 because the LLM/archetype names weren't leg{n}_{l|r}).
        self.assertEqual(self._map("a quadruped robot dog").n_legs, 4)
        self.assertEqual(self._map("a hexapod").n_legs, 6)
        self.assertEqual(self._map("a spider").n_legs, 8)

    def test_leg_joint_roles_are_axis_derived(self):
        # each leg exposes abduction (world fore-aft x) + stride/lift (world lateral y) from the WORLD axis, so
        # the gait engine (GEN-2) can swing the stride joints and hold abduction — no per-body joint tables.
        legs = self._map("a quadruped robot dog").legs
        self.assertTrue(legs and all(lg.stride_tokens for lg in legs), "every leg must have a drivable stride joint")
        self.assertIn("abduction", legs[0].roles)

    def test_snake_is_a_spine_not_legs(self):
        am = self._map("a snake robot")
        self.assertEqual(am.n_legs, 0)
        self.assertIsNotNone(am.spine)
        self.assertEqual(am.kind(), "limbless")

    def test_rover_wheels_from_continuous_joints(self):
        am = self._map("a wheeled rover")
        self.assertGreaterEqual(am.n_wheels, 2, "unlimited grounded hinges are wheels")
        self.assertEqual(am.n_legs, 0)
        self.assertEqual(am.kind(), "wheeled")

    def test_arm_reaches_up_not_grounded(self):
        am = self._map("a robot arm that sorts objects")
        self.assertEqual(am.n_legs, 0, "an arm reaches into free space -> not a leg")
        self.assertGreaterEqual(am.n_arms, 1)
        self.assertEqual(am.kind(), "manipulator")

    def test_foot_positions_are_support_polygon_vertices(self):
        am = self._map("a quadruped robot dog")
        xy = am.foot_xy()
        self.assertEqual(len(xy), 4, "one support-polygon vertex per leg")
        self.assertTrue(all(len(p) == 2 for p in xy))


if __name__ == "__main__":
    unittest.main()
