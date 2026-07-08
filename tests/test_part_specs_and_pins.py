"""Part fidelity + part selection: every part carries REAL structured specs (motor torque/RPM, camera resolution,
lidar channels/range, ...) and a user can PIN an exact part for a category (the BOM swaps to it, bad pins rejected).
Offline, pure catalog/BOM logic; no MuJoCo.
"""
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")


class PartSpecTests(unittest.TestCase):
    def test_motor_specs_are_structured_with_rpm(self):
        from virturoid.services.component_catalog import part_specs
        s = part_specs("AK80-9")
        self.assertEqual(s["category"], "actuator")
        for k in ("peak_torque_nm", "rated_torque_nm", "no_load_rpm", "voltage_v", "gear_ratio"):
            self.assertIn(k, s)
        self.assertGreater(s["no_load_rpm"], 0)                 # RPM is derived + surfaced, not hidden in rad/s

    def test_lidar_and_camera_specs_are_real_numbers(self):
        from virturoid.services.component_catalog import part_specs
        lid = part_specs("Ouster OS1-32")
        self.assertEqual(lid["channels"], 32)
        self.assertEqual(lid["range_m"], 120.0)
        cam = part_specs("RealSense D435i")
        self.assertIn("rgb_mp", cam)
        self.assertIn("fps", cam)
        self.assertIn("fov_deg", cam)

    def test_list_parts_filters_by_category(self):
        from virturoid.services.component_catalog import list_parts
        self.assertEqual(len(list_parts("lidar")), 4)
        self.assertTrue(all(p["category"] == "actuator" for p in list_parts("actuator")))
        self.assertGreater(len(list_parts()), len(list_parts("camera")))   # 'all' > one category

    def test_unknown_part_returns_none(self):
        from virturoid.services.component_catalog import part_specs
        self.assertIsNone(part_specs("Flux Capacitor"))


class PartPinTests(unittest.TestCase):
    def _rover(self):
        from virturoid.fixtures.gene_library import tabletop_arm_gene
        from virturoid.services.grounded_physics import ground_gene
        g = tabletop_arm_gene()
        ground_gene(g)
        return g

    def test_pin_swaps_the_part_in_the_bom(self):
        from virturoid.services.bom_builder import build_bom
        bom = build_bom(self._rover(), task="navigate", pins={"lidar": "Ouster OS1-32", "actuator": "T-Motor AK80-9"})
        self.assertEqual(bom["pins"]["applied"], [{"category": "lidar", "part": "Ouster OS1-32"},
                                                  {"category": "actuator", "part": "T-Motor AK80-9"}])
        lidar_lines = [ln for ln in bom["lines"] if ln["category"] == "lidar"]
        self.assertEqual(lidar_lines[0]["part"], "Ouster OS1-32")
        act_lines = [ln for ln in bom["lines"] if ln["category"] == "actuator"]
        self.assertTrue(all(ln["part"] == "T-Motor AK80-9" for ln in act_lines))

    def test_bad_pin_is_rejected_not_faked(self):
        from virturoid.services.bom_builder import build_bom
        bom = build_bom(self._rover(), pins={"lidar": "Nonexistent", "camera": "T-Motor AK80-9"})
        reasons = {r["part"]: r["reason"] for r in bom["pins"]["rejected"]}
        self.assertIn("Nonexistent", reasons)
        self.assertIn("actuator, not a camera", reasons["T-Motor AK80-9"])

    def test_pin_travels_on_gene_metadata(self):
        from virturoid.services.bom_builder import build_bom
        g = self._rover()
        g.metadata["pinned_parts"] = {"lidar": "Livox Mid-360"}
        bom = build_bom(g, task="navigate")                    # honored from metadata, no pins arg
        self.assertEqual([ln for ln in bom["lines"] if ln["category"] == "lidar"][0]["part"], "Livox Mid-360")


if __name__ == "__main__":
    unittest.main()
