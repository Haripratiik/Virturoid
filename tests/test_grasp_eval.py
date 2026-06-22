"""Real grasp-lift (findings §12): a composed grasp arm actually grasps and LIFTS a box by friction
(no pinning). This is the end-to-end check behind 'build a grasping robot from one prompt'."""

import importlib.util
import unittest

_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
class GraspLiftTests(unittest.TestCase):
    def test_composed_grasp_arm_grasps_and_lifts(self):
        from virturoid.services.grasp_eval import evaluate_grasp_lift
        from virturoid.services.morphology_composer import compose_robot

        gene = compose_robot("grasp and lift a box on a table")
        r = evaluate_grasp_lift(gene, positions=[(0.40, 0.0), (0.44, 0.06), (0.48, -0.06)])
        # a real friction grasp: vertical approach, both fingers contact, box comes off the table.
        self.assertGreaterEqual(r["success_rate"], 0.66, f"composed arm should grasp: {r}")
        self.assertGreater(r["mean_lift"], 0.05)
        self.assertLess(r["mean_tilt_deg"], 5.0)        # top-down approach

    def test_metric_reports_no_grasp_site_gracefully(self):
        """A mobile base has no gripper TCP — the metric reports failure, it does not crash."""
        from virturoid.services.grasp_eval import grasp_lift_attempt
        from virturoid.services.morphology_composer import compose_robot

        base = compose_robot("a mobile base to deliver parts")
        out = grasp_lift_attempt(base, 0.4, 0.0)
        self.assertFalse(out["success"])
        self.assertEqual(out["reason"], "no_grasp_site")


if __name__ == "__main__":
    unittest.main()
