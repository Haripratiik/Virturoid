"""S2 code-owned dimension priors: the table must return realistic canonical sizes, CLAMP absurd LLM-proposed
dimensions into the cited band (and log the honesty event), infer plausible mass, and — the unit-sanity job —
FLAG a 10 cm wall / 20 m box and identify the classic mm/inch rescale signatures."""

import unittest

from virturoid.services.dimension_priors import (
    PRIORS, DimPrior, default_size, snap_to_prior, mass_for, check_dimensions, rescale_signature,
    robot_scene_ratio_ok)


class DimensionPriorsTests(unittest.TestCase):
    def test_defaults_are_realistic(self):
        # convention: size_xyz = (x, y, z) with Z = height. A corridor is >=0.915 m WIDE (x) and ceiling-tall (z)
        w, d, h = default_size("corridor")
        self.assertGreaterEqual(w, 0.915)                    # ADA min corridor width on the x axis
        self.assertGreaterEqual(h, 2.03)                     # at least door-height tall (z)
        self.assertGreaterEqual(default_size("table")[2], 0.71)   # real table HEIGHT (z), not a 2.5cm slab
        self.assertEqual(default_size("ycb.mug"), (0.080, 0.080, 0.082))

    def test_snap_clamps_absurd_and_logs(self):
        # LLM asks for a 0.1 m-tall "wall" (height is the z axis) -> clamped up into [0.9, 4.0], logged
        r = snap_to_prior("wall", (3.0, 0.12, 0.1))
        self.assertTrue(r.clamped)
        self.assertGreaterEqual(r.size_xyz[2], 0.9)          # height clamped up
        self.assertTrue(any("out of band" in e for e in r.events))
        # a plausible wall is left alone
        self.assertFalse(snap_to_prior("wall", (3.0, 0.12, 2.4)).clamped)

    def test_snap_default_when_no_proposal(self):
        r = snap_to_prior("table")
        self.assertEqual(r.size_xyz, default_size("table"))
        self.assertFalse(r.clamped)

    def test_unknown_category_is_soft(self):
        r = snap_to_prior("unicorn_horn", (0.1, 0.4, 0.1))
        self.assertEqual(r.size_xyz, (0.1, 0.4, 0.1))         # proposed passed through, flagged not crashed
        self.assertTrue(r.events)

    def test_mass_inference_in_band(self):
        m = mass_for("ycb.cracker_box", default_size("ycb.cracker_box"))
        self.assertGreaterEqual(m, 0.35); self.assertLessEqual(m, 0.45)   # real cracker-box mass band
        # a scaled-up box can't imply an absurd mass outside the band
        self.assertLessEqual(mass_for("ycb.foam_brick", (0.2, 0.2, 0.2)), 0.04)

    def test_unit_sanity_flags_and_rescale_signature(self):
        # a 10 cm-tall wall is caught; a wall in millimetres (x1000) is identified as the mm->m signature
        self.assertFalse(check_dimensions("wall", (3000.0, 2440.0, 120.0))["ok"])   # mm values
        sig = rescale_signature("wall", (3000.0, 2440.0, 120.0))
        self.assertIsNotNone(sig)
        self.assertIn("x1000", sig)
        # a plausible wall (length, thickness, HEIGHT) passes and has no rescale suggestion
        self.assertTrue(check_dimensions("wall", (3.0, 0.12, 2.44))["ok"])
        self.assertIsNone(rescale_signature("wall", (3.0, 0.12, 2.44)))

    def test_robot_scene_ratio_guard(self):
        self.assertTrue(robot_scene_ratio_ok(0.3, 8.0)["ok"])          # 0.3 m robot in an 8 m scene
        self.assertFalse(robot_scene_ratio_ok(12.0, 8.0)["ok"])        # robot bigger than scene -> flagged
        self.assertFalse(robot_scene_ratio_ok(0.01, 100.0)["ok"])      # speck in a stadium -> flagged

    def test_all_priors_are_wellformed(self):
        for cat, p in PRIORS.items():
            self.assertIsInstance(p, DimPrior)
            self.assertEqual(len(p.default), 3)
            for (lo, hi), dv in zip(p.bounds, p.default):
                self.assertLessEqual(lo, dv + 1e-9, f"{cat}: default below band")
                self.assertLessEqual(dv, hi + 1e-9, f"{cat}: default above band")


if __name__ == "__main__":
    unittest.main()
