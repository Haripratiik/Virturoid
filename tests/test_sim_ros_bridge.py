"""§4.7 — the closed-loop deploy bridge + SafetyFilter: the "sim controls the robot, safely" loop that proves
the design->build->run path on CPU (MuJoCo as virtual hardware), the same shape the exported ROS2 node runs
against ros2_control on metal."""

import unittest

from virturoid.services.morphology_composer import compose_robot
from virturoid.services.sim_ros_bridge import (
    SafetyFilter, SimHardwareBridge, hold_pose_command, run_sim_ros_demo,
)


class SafetyFilterTests(unittest.TestCase):
    def test_clamps_joint_limits_and_flags(self):
        sf = SafetyFilter(lower=[-1.0, -1.0], upper=[1.0, 1.0], vel_limit=100.0)   # high vel -> only limits bite
        out, viol = sf.clamp([5.0, -5.0], q=[0.0, 0.0], dt=0.01)
        self.assertEqual(viol, 2)
        self.assertLessEqual(out[0], 1.0)
        self.assertGreaterEqual(out[1], -1.0)

    def test_in_range_command_passes_untouched(self):
        sf = SafetyFilter(lower=[-1.0], upper=[1.0], vel_limit=100.0)
        out, viol = sf.clamp([0.3], q=[0.3], dt=0.01)
        self.assertEqual(viol, 0)
        self.assertAlmostEqual(out[0], 0.3)

    def test_rate_limit_caps_the_step(self):
        sf = SafetyFilter(lower=[-10.0], upper=[10.0], vel_limit=1.0)              # 1 rad/s
        out, viol = sf.clamp([5.0], q=[0.0], dt=0.1)                               # max step = 0.1 rad
        self.assertEqual(viol, 1)
        self.assertAlmostEqual(out[0], 0.1, places=5)


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.gene = compose_robot("a quadruped walking robot", llm=None)

    def test_closed_loop_runs_and_state_matches_joints(self):
        b = SimHardwareBridge(self.gene)
        res = b.run(hold_pose_command(b), steps=120)
        self.assertTrue(res["finite"])
        self.assertEqual(res["n_joints"], len(res["joint_names"]))
        self.assertEqual(len(res["final_state"]["position"]), res["n_joints"])

    def test_safety_blocks_a_reckless_command(self):
        b = SimHardwareBridge(self.gene)
        reckless = lambda state: [100.0] * b.model.nu          # far past every joint limit  # noqa: E731
        res = b.run(reckless, steps=60)
        self.assertGreater(res["total_violations"], 0)         # the SafetyFilter engaged
        self.assertTrue(res["finite"])                         # and the 'hardware' did not blow up
        self.assertTrue(all(abs(p) < 5.0 for p in res["final_state"]["position"]),
                        "safety must keep joints near their limits, not at the commanded 100 rad")

    def test_demo_convenience_runs(self):
        res = run_sim_ros_demo(self.gene, steps=50)
        self.assertTrue(res["finite"])
        self.assertIn("total_violations", res)


if __name__ == "__main__":
    unittest.main()
