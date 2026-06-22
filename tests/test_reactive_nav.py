"""Reactive sense-and-avoid command geometry (perception Rung A): from a range ring + goal bearing,
pick a velocity command that heads to the goal when clear and turns away from blocked fronts. Tested
directly (no sim/policy needed)."""

import math
import unittest

_N = 8
_ANG = [2 * math.pi * i / _N for i in range(_N)]  # ray 0 = forward, ray 2 = +90deg (left), ray 4 = behind


class ReactiveCommandTests(unittest.TestCase):
    def test_goal_ahead_all_open_drives_straight(self):
        from virturoid.services.reactive_nav import reactive_command
        vx, wz = reactive_command([3.0] * _N, _ANG, 0.0, max_range=3.0, cruise_v=0.8)
        self.assertGreater(vx, 0.6)
        self.assertAlmostEqual(wz, 0.0, places=3)

    def test_front_blocked_turns_away_and_slows(self):
        from virturoid.services.reactive_nav import reactive_command
        ranges = [0.15, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]   # only the forward ray is blocked
        vx, wz = reactive_command(ranges, _ANG, 0.0, max_range=3.0, cruise_v=0.8)
        self.assertGreater(abs(wz), 0.3)     # steer off the blocked heading
        self.assertLess(vx, 0.4)             # don't barrel into the wall

    def test_goal_behind_turns_around(self):
        from virturoid.services.reactive_nav import reactive_command
        vx, wz = reactive_command([3.0] * _N, _ANG, math.pi, max_range=3.0)
        self.assertGreater(abs(wz), 0.5)     # large turn toward a goal behind

    def test_command_stays_in_trained_range(self):
        from virturoid.services.reactive_nav import reactive_command
        for bearing in [-2.0, -0.5, 0.0, 1.0, 3.0]:
            ranges = [0.3, 1.0, 3.0, 2.0, 0.5, 3.0, 1.5, 0.4]
            vx, wz = reactive_command(ranges, _ANG, bearing, max_range=3.0, cruise_v=0.8, wz_max=0.8)
            self.assertGreaterEqual(vx, 0.0)
            self.assertLessEqual(vx, 0.8)
            self.assertLessEqual(abs(wz), 0.8)


if __name__ == "__main__":
    unittest.main()
