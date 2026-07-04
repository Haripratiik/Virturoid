"""B2 terrain DSL: every atomic tile must (a) produce a heightfield in [0,1] that grows monotonically harder with
difficulty, (b) compile into a real MuJoCo model as an hfield floor, and the Rudin curriculum must promote
survivors + demote fallers so the fleet's mean level tracks competence."""

import unittest

import numpy as np

from virturoid.services.terrain_dsl import (
    TILE_TYPES, Terrain, make_tile, compose_row, hfield_asset_xml, hfield_geom_xml, hfield_data, RudinCurriculum)


class TerrainTileTests(unittest.TestCase):
    def test_all_tiles_build_in_range(self):
        for tile in TILE_TYPES:
            t = make_tile(tile, difficulty=0.5, seed=1, n=32)
            self.assertEqual(t.heights.shape, (32, 32))
            self.assertGreaterEqual(t.heights.min(), 0.0)
            self.assertLessEqual(t.heights.max(), 1.0)
            self.assertTrue(np.all(t.friction > 0.0))

    def test_difficulty_is_monotone_rougher(self):
        # a harder tile should have >= the roughness (height spread) of an easier one, for the elevation tiles
        for tile in ("slope", "stairs", "waves", "rough", "stones"):
            easy = np.std(make_tile(tile, difficulty=0.1, seed=2, n=48).heights)
            hard = np.std(make_tile(tile, difficulty=0.9, seed=2, n=48).heights)
            self.assertGreater(hard + 1e-9, easy, f"{tile} did not get rougher with difficulty")

    def test_flat_is_flat_and_friction_dips(self):
        self.assertAlmostEqual(float(np.ptp(make_tile("flat").heights)), 0.0, places=6)
        fr = make_tile("friction", difficulty=0.8, seed=3, n=48)
        self.assertLess(fr.friction.min(), 1.0)          # slippery patches exist
        self.assertAlmostEqual(float(fr.heights.max()), 0.0, places=6)   # friction tile stays flat

    def test_compose_row_concatenates(self):
        row = compose_row([make_tile("flat", n=24), make_tile("stairs", n=24, difficulty=0.5),
                           make_tile("waves", n=24, difficulty=0.5)])
        self.assertEqual(row.heights.shape, (24, 72))
        self.assertAlmostEqual(row.size_m, 24.0)          # 3 * default 8m

    def test_determinism(self):
        a = make_tile("rough", difficulty=0.7, seed=42, n=32).heights
        b = make_tile("rough", difficulty=0.7, seed=42, n=32).heights
        self.assertTrue(np.array_equal(a, b))

    def test_compiles_as_mujoco_hfield(self):
        try:
            import mujoco
        except Exception:  # noqa: BLE001
            self.skipTest("mujoco not installed")
        t = make_tile("stairs", difficulty=0.6, seed=5, n=32)
        xml = (f'<mujoco><asset><material name="mat_ground" rgba="0.4 0.4 0.4 1"/>'
               f'{hfield_asset_xml(t, name="terr")}</asset>'
               f'<worldbody>{hfield_geom_xml(t, name="terr")}'
               f'<body pos="0 0 1"><freejoint/><geom type="sphere" size="0.1"/></body>'
               f'</worldbody></mujoco>')
        model = mujoco.MjModel.from_xml_string(xml)
        # assign the elevation data post-compile (the DSL's documented contract) and step
        data_flat = hfield_data(t)
        self.assertEqual(data_flat.size, model.hfield_nrow[0] * model.hfield_ncol[0])
        model.hfield_data[:] = data_flat
        d = mujoco.MjData(model)
        for _ in range(50):
            mujoco.mj_step(model, d)
        self.assertTrue(np.all(np.isfinite(d.qpos)))     # a ball settles on the stairs, no NaN blowup


class RudinCurriculumTests(unittest.TestCase):
    def test_promote_and_demote(self):
        cur = RudinCurriculum(n_envs=4, n_levels=10)
        self.assertTrue(np.all(cur.levels == 0))
        # env0 walks far + survives -> promote; env1 falls -> demote (already floored); env2 barely moves -> demote
        info = cur.update(distance_frac=np.array([0.9, 0.5, 0.1, 0.7]),
                          alive=np.array([True, False, True, True]))
        self.assertEqual(cur.levels[0], 1)               # promoted
        self.assertEqual(cur.levels[1], 0)               # fell, floored at 0
        self.assertEqual(cur.levels[2], 0)               # too slow, floored
        self.assertEqual(cur.levels[3], 1)               # promoted
        self.assertEqual(info["promoted"], 2)

    def test_mean_level_tracks_competence(self):
        cur = RudinCurriculum(n_envs=8, n_levels=6)
        for _ in range(10):                               # a fleet that always succeeds should climb the ladder
            cur.update(distance_frac=np.full(8, 0.95), alive=np.full(8, True))
        self.assertEqual(cur.levels.max(), 5)             # capped at n_levels-1
        self.assertGreater(cur.difficulty().mean(), 0.8)


class CurriculumAuthorTests(unittest.TestCase):
    def test_authors_task_curriculum(self):
        from virturoid.services.terrain_dsl import author_curriculum
        cur = author_curriculum("climb stairs", n_envs=16, n_levels=8)
        self.assertIn("stairs", cur["tiles"])                    # stairs task selects the stairs tile
        self.assertEqual(cur["curriculum"].n_envs, 16)
        easy = cur["tile_for"](0, seed=0)                        # level 0 course is (near) flat
        hard = cur["tile_for"](7, seed=1)                        # top level is rougher
        self.assertGreaterEqual(float(np.std(hard.heights)), float(np.std(easy.heights)))


if __name__ == "__main__":
    unittest.main()
