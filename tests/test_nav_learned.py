"""Learned steerable legged navigation (W3). The pure-pursuit command geometry is tested directly; the
closed-loop maze solve is gated on a trained velocity-conditioned policy (models/morph_quad_vel.npz)."""

import importlib.util
import math
import unittest
from pathlib import Path

_MUJOCO = importlib.util.find_spec("mujoco") is not None
_VEL_NPZ = Path(__file__).resolve().parent.parent / "models" / "morph_frog_vel.npz"


class PurePursuitGeometryTests(unittest.TestCase):
    def test_straight_ahead_drives_forward_no_turn(self):
        from virturoid.services.nav_learned import pure_pursuit_cmd
        vx, wz, _ = pure_pursuit_cmd((0, 0), 0.0, [(0, 0), (3, 0)], 1, cruise_v=0.8)
        self.assertGreater(vx, 0.6)               # aligned -> near cruise speed
        self.assertAlmostEqual(wz, 0.0, places=4)  # no turn needed

    def test_target_to_the_left_turns_left_and_slows(self):
        from virturoid.services.nav_learned import pure_pursuit_cmd
        vx, wz, _ = pure_pursuit_cmd((0, 0), 0.0, [(0, 0), (0, 3)], 1, cruise_v=0.8, wz_max=0.8)
        self.assertGreater(wz, 0.5)               # turn left (positive yaw rate)
        self.assertLess(vx, 0.2)                  # facing 90deg off -> creep forward

    def test_target_behind_turns_hard_barely_advances(self):
        from virturoid.services.nav_learned import pure_pursuit_cmd
        vx, wz, _ = pure_pursuit_cmd((0, 0), 0.0, [(0, 0), (-3, 0)], 1, cruise_v=0.8, wz_max=0.8)
        self.assertAlmostEqual(abs(wz), 0.8, places=4)  # saturated turn
        self.assertLess(vx, 0.15)

    def test_waypoint_advances_within_lookahead(self):
        from virturoid.services.nav_learned import pure_pursuit_cmd
        wpts = [(0, 0), (0.3, 0), (2.0, 0)]       # start near wp1 -> should skip to wp2
        _, _, tgt = pure_pursuit_cmd((0, 0), 0.0, wpts, 1, lookahead=0.6)
        self.assertEqual(tgt, 2)


@unittest.skipUnless(_MUJOCO and _VEL_NPZ.exists(),
                     "needs MuJoCo + a trained velocity policy (models/morph_frog_vel.npz)")
class LearnedMazeSolveTests(unittest.TestCase):
    def test_legged_robot_follows_maze_path(self):
        from virturoid.services.morph_policy import MorphPolicy
        from virturoid.services.nav_learned import solve_maze_learned
        from virturoid.services.steerable_body import steerable_quadruped

        pol = MorphPolicy.from_npz(str(_VEL_NPZ))
        frog = steerable_quadruped()
        out = solve_maze_learned(frog, pol, n=2, seed=1, walls=False, max_steps=4000)
        # acceptance: the learned policy makes real progress along the planned path (steerable locomotion)
        self.assertGreater(out["progress"], 0.0)
        self.assertNotEqual(out["status"], "not_legged")


if __name__ == "__main__":
    unittest.main()
