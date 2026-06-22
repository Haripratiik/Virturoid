"""Task-matched evaluation for spray robots (§11): surface coverage, not pick-place success."""

import importlib.util
import unittest

_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
class SprayCoverageTests(unittest.TestCase):
    def test_composed_spray_arm_covers_a_reachable_panel(self):
        from virturoid.services.morphology_composer import compose_robot
        from virturoid.services.spray_coverage import evaluate_spray_coverage

        g = compose_robot("spray paint a panel with 0.7 m reach")
        self.assertEqual(g.end_effector_type, "spray_nozzle")     # composed as a spray robot, not a gripper
        r = evaluate_spray_coverage(g, grid=4)
        self.assertEqual(r["total"], 16)
        self.assertGreater(r["coverage"], 0.3)                    # the nozzle sweeps + covers the panel
        self.assertTrue(0.0 <= r["coverage"] <= 1.0)

    def test_coverage_is_deterministic(self):
        from virturoid.services.morphology_composer import compose_robot
        from virturoid.services.spray_coverage import evaluate_spray_coverage

        g = compose_robot("spray paint a panel with 0.7 m reach")
        a = evaluate_spray_coverage(g, grid=3)
        b = evaluate_spray_coverage(g, grid=3)
        self.assertEqual(a["coverage"], b["coverage"])            # reproducible (§21.4)


if __name__ == "__main__":
    unittest.main()
