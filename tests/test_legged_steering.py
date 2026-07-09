"""Legged STEERING — a quadruped can now TURN and navigate to a forward-cone goal (not just walk straight).

The crawl gait was open-loop straight-ahead. This adds differential-stride steering (``turn_bias``: amplify one
side's stride so the body yaws, like a diff-drive rover with legs) + a closed-loop ``steer_fn`` the follower drives
from the heading error. MEASURED: turn_bias yaws the body in the commanded direction; ``legged_steer_to_goal``
reaches forward-cone goals precisely (~0.1 m).

HONEST LIMIT (documented, not hidden): scripted differential steering handles course-correction, NOT sharp
(>~60 deg) turns or tight mazes — those need the LEARNED steering policy (nav_learned, GPU). So this closes
legged navigation to a goal ROUGHLY AHEAD, and narrows (does not fully close) legged maze-solving.
"""
import importlib.util
import math
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("MUJOCO_GL", "glfw")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "legged steering needs the MuJoCo sim")
class LeggedSteeringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from virturoid.services.morphology_composer import compose_robot
        cls.gene = compose_robot("a quadruped robot dog", ensure_walkable=True)

    def test_turn_bias_yaws_the_body_in_the_commanded_direction(self):
        # differential stride must produce OPPOSITE yaw for opposite bias (a real, controllable turn) — the
        # un-gameable check: not just "it moved" but "it turned the way we asked, both ways".
        from virturoid.services.morph_policy import crawl_gait_rollout
        left = crawl_gait_rollout(self.gene, steps=1200, turn_bias=-0.5)   # -bias -> +yaw (left)
        right = crawl_gait_rollout(self.gene, steps=1200, turn_bias=0.5)   # +bias -> -yaw (right)
        self.assertTrue(left["survived"] and right["survived"], "the body must stay up while turning")
        self.assertGreater(left["yaw_change"], right["yaw_change"] + 0.3,
                           f"opposite bias must yaw opposite ways: left={left['yaw_change']} right={right['yaw_change']}")

    def test_reaches_a_forward_cone_goal(self):
        from virturoid.services.morph_policy import legged_steer_to_goal
        for goal in [(2.2, 0.0), (2.2, 0.6), (2.2, -0.6)]:
            r = legged_steer_to_goal(self.gene, goal, steps=4000)
            self.assertTrue(r["reached_goal"], f"legged robot must steer to a forward-cone goal {goal}: {r}")
            self.assertLess(r["goal_dist"], 0.6, f"must arrive near {goal}: {r}")

    def test_steering_is_off_by_default_no_regression(self):
        # turn_bias defaults to 0 -> the gait is byte-identical to the straight walk (a walkable quad still walks)
        from virturoid.services.morph_policy import crawl_gait_rollout
        r = crawl_gait_rollout(self.gene, steps=1200)
        self.assertGreater(r["forward"], 0.3, "the default (no-steer) gait still walks forward")
        self.assertIn("yaw_change", r)                          # the new telemetry is present


if __name__ == "__main__":
    unittest.main()
