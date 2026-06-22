"""Display normalization for the real-model substrate: real Menagerie models render dark (near-black
materials); normalize_display lifts them to legible grey + neutral lighting WITHOUT touching geometry,
mass, or joints — so the viewport shows production robots clearly while physics/training stay identical."""

import importlib.util
import unittest

_MUJOCO = importlib.util.find_spec("mujoco") is not None

_DARK_MJCF = """
<mujoco model="dark">
  <asset><material name="blk" rgba="0.1 0.1 0.1 1"/></asset>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="b" pos="0 0 0.3">
      <freejoint/>
      <geom name="lit" type="box" size="0.1 0.1 0.1" material="blk" mass="1.0"/>
      <geom name="bare" type="sphere" size="0.05" rgba="0.05 0.05 0.05 1" pos="0.2 0 0" mass="0.3"/>
    </body>
  </worldbody>
</mujoco>
"""


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class NormalizeDisplayTests(unittest.TestCase):
    def test_lifts_dark_surfaces_but_leaves_physics_identical(self):
        import mujoco
        import numpy as np
        from virturoid.services.real_model_library import normalize_display

        m = mujoco.MjModel.from_xml_string(_DARK_MJCF)
        before = {
            "geom_size": m.geom_size.copy(), "geom_pos": m.geom_pos.copy(),
            "body_mass": m.body_mass.copy(), "body_inertia": m.body_inertia.copy(),
            "jnt_axis": m.jnt_axis.copy(), "dof_damping": m.dof_damping.copy(),
        }
        mat_before = m.mat_rgba.copy()
        report = normalize_display(m)

        # materials + bare-geom colour got lifted out of near-black
        self.assertGreaterEqual(report["materials_lifted"], 1)
        self.assertGreater(float(m.mat_rgba[0][:3].max()), 0.18)
        self.assertFalse(np.allclose(mat_before[0][:3], m.mat_rgba[0][:3]))
        # PHYSICS bit-identical (display-only): sizes, positions, mass, inertia, joints untouched
        for k, v in before.items():
            self.assertTrue(np.array_equal(getattr(m, k), v), f"{k} changed — display normalization touched physics")

    def test_safe_on_already_bright_model(self):
        import mujoco
        from virturoid.services.real_model_library import normalize_display
        m = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body><freejoint/>'
            '<geom type="box" size="0.1 0.1 0.1" rgba="0.7 0.7 0.7 1" mass="1"/></body></worldbody></mujoco>')
        rep = normalize_display(m)            # nothing near-black -> no lift, no crash
        self.assertEqual(rep["geoms_lifted"], 0)


if __name__ == "__main__":
    unittest.main()
