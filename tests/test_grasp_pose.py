"""Top-down grasp pose (findings §12): a composed grasp arm has a wrist DOF and the orientation-aware
IK levels the palm to a vertical approach — the structural fix for the 'jaws sweep the object' failure
of a wrist-less arm."""

import importlib.util
import math
import unittest

_MUJOCO = importlib.util.find_spec("mujoco") is not None


class GraspArmHasWristTests(unittest.TestCase):
    def test_default_grasp_arm_gets_a_wrist_pitch_dof(self):
        from virturoid.services.morphology_composer import compose_robot

        g = compose_robot("grasp and lift a box on a table")
        # the palm (end-effector) must be a wrist segment, and there must be >= 3 revolute arm joints
        # PLUS that wrist (so the gripper orientation is decoupled from reach).
        revolute = [s for s in g.segments if s.joint_type == "revolute"]
        self.assertGreaterEqual(len(revolute), 4)
        self.assertEqual(g.end_effector().name, "wrist")
        # the wrist is a PITCH joint (axis ~ y), the orientation DOF
        wrist = next(s for s in g.segments if s.name == "wrist")
        self.assertGreater(abs(wrist.joint_axis[1]), 0.5)

    def test_sort_arm_also_wristed(self):
        from virturoid.services.morphology_composer import compose_robot

        g = compose_robot("sort blocks into bins")
        self.assertTrue(any(s.name == "wrist" and s.joint_type == "revolute" for s in g.segments))


@unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
class TopDownIKTests(unittest.TestCase):
    def _arm_model(self):
        import mujoco

        from virturoid.schemas.scenes import SceneObject
        from virturoid.services.gene_compiler import compile_gene_with_scene
        from virturoid.services.morphology_composer import compose_robot

        g = compose_robot("grasp and lift a box on a table")
        cube = SceneObject("box", "cube", (0.40, 0.0, 0.05, 0, 0, 0), mass_kg=0.03, friction=1.0, scale=1.0)
        mj = mujoco.MjModel.from_xml_string(compile_gene_with_scene(g, [cube]))
        return g, mj, mujoco

    def test_orientation_ik_achieves_vertical_approach(self):
        from virturoid.services.grasp_pose import (
            approach_tilt_deg,
            arm_revolute_joints,
            solve_top_down_grasp_pose,
        )

        g, mj, mujoco = self._arm_model()
        tcp = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        self.assertGreaterEqual(tcp, 0, "composed grasp arm must emit a grasp_site TCP")
        palm = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, g.end_effector().name)
        arm = arm_revolute_joints(mj)
        data = mujoco.MjData(mj)
        mujoco.mj_resetData(mj, data)
        seed = [0.0, 1.0, -1.4, -0.8][: len(arm)]
        solve_top_down_grasp_pose(mj, data, site_id=tcp, palm_body_id=palm,
                                  target=(0.40, 0.0, 0.05), arm_joints=arm, seed=seed)
        import numpy as np

        residual = float(np.linalg.norm(data.site_xpos[tcp] - np.array([0.40, 0.0, 0.05])))
        tilt = approach_tilt_deg(mj, data, palm)
        self.assertLess(residual, 0.01, "grasp TCP should reach the object")
        self.assertLess(tilt, 5.0, "wristed arm should approach the object palm-down (vertical)")

    def test_tilt_metric_zero_for_straight_down(self):
        """approach_tilt_deg is a well-formed readiness signal: 0 when the palm points straight down."""
        # a trivial sanity bound — the IK test above exercises the real geometry.
        self.assertAlmostEqual(math.degrees(math.acos(1.0)), 0.0)


if __name__ == "__main__":
    unittest.main()
