"""Perception Rung A: a body's rangefinder ring senses its environment — open space reads max range,
and walls in a maze are sensed (some rays come back short). Validates the exteroception interface
CPU-first before any GPU training."""

import importlib.util
import unittest

_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class ExteroceptionTests(unittest.TestCase):
    def _frog(self):
        from virturoid.services.steerable_body import steerable_quadruped
        return steerable_quadruped()

    def test_open_ground_reads_max_no_self_hits(self):
        import mujoco

        from virturoid.services.exteroception import read_ranges, with_rangefinders
        m = mujoco.MjModel.from_xml_string(with_rangefinders(self._frog(), n_rays=8))
        d = mujoco.MjData(m); mujoco.mj_resetData(m, d); mujoco.mj_forward(m, d)
        n_rf = sum(1 for s in range(m.nsensor) if m.sensor_type[s] == mujoco.mjtSensor.mjSENS_RANGEFINDER)
        self.assertEqual(n_rf, 8)
        r = read_ranges(m, d, n_rays=8, max_range=3.0)
        self.assertEqual(r.shape, (8,))
        self.assertTrue((r > 2.99).all(), f"open ground should read max (no self-hits): {r}")

    def test_maze_walls_are_sensed(self):
        import mujoco

        from virturoid.services.exteroception import read_ranges, with_rangefinders
        from virturoid.services.maze import generate_maze
        maze = generate_maze(2, seed=1)
        m = mujoco.MjModel.from_xml_string(with_rangefinders(self._frog(), walls=maze, n_rays=8))
        d = mujoco.MjData(m); mujoco.mj_resetData(m, d); mujoco.mj_forward(m, d)
        r = read_ranges(m, d, n_rays=8, max_range=3.0)
        self.assertTrue((r < 2.99).any(), f"walls should be sensed (some ray short): {r}")
        self.assertTrue((r >= 0.0).all() and (r <= 3.0).all())


if __name__ == "__main__":
    unittest.main()
