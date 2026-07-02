"""Terrain difficulty ladder (plan v2 §5.3 / G3): flat->rough->stairs->gaps as plain box geoms that COMPILE in
MuJoCo, with a fixed geom count per level (per-env difficulty scales sizes, not counts -> no MJX recompile)."""

import importlib.util
import unittest

from virturoid.services.terrain import (TERRAIN_LEVELS, flat_floor, gaps, rough_floor, scene_with_terrain,
                                        stairs, terrain_mjcf)

_MUJOCO = importlib.util.find_spec("mujoco") is not None
_BALL = '<body name="b" pos="0 0 0.5"><freejoint/><geom type="sphere" size="0.05" condim="3"/></body>'


class TerrainTests(unittest.TestCase):
    def test_levels_known_and_dispatch(self):
        self.assertEqual(TERRAIN_LEVELS, ("flat", "rough", "stairs", "gaps"))
        for lvl in TERRAIN_LEVELS:
            self.assertIn("geom", terrain_mjcf(lvl))
        with self.assertRaises(ValueError):
            terrain_mjcf("lava")

    def test_difficulty_scales_the_knob(self):
        # a harder rough terrain has taller pads; harder gaps are wider (monotone in difficulty)
        easy = terrain_mjcf("gaps", difficulty=0.0)
        hard = terrain_mjcf("gaps", difficulty=1.0)
        self.assertNotEqual(easy, hard)

    def test_fixed_geom_count_across_difficulty(self):
        # count must NOT change with difficulty (an MJX recompile trigger); only sizes/heights change
        c0 = rough_floor(amplitude=0.02).count("<geom")
        c1 = rough_floor(amplitude=0.10).count("<geom")
        self.assertEqual(c0, c1)

    @unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
    def test_each_level_compiles_in_mujoco(self):
        import mujoco
        for lvl in TERRAIN_LEVELS:
            xml = scene_with_terrain(_BALL, lvl, difficulty=0.7, seed=1)
            m = mujoco.MjModel.from_xml_string(xml)      # raises if the MJCF is malformed
            self.assertGreater(m.ngeom, 1)               # terrain geoms + the ball
            mujoco.mj_step(m, mujoco.MjData(m))          # one physics step runs (contacts resolve)


if __name__ == "__main__":
    unittest.main()
