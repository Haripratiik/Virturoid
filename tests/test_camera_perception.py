"""CV is a REAL part of a camera-equipped robot: the robot gets a FUNCTIONAL onboard camera whose FOV + render
resolution come from the actual camera PART, and the tiny CV encoder perceives through it. Before this, the camera
was cosmetic and every robot navigated by the rangefinder ring.

MuJoCo-gated (compiles + renders the robot's own camera); the pure resolution math always runs.
"""
import importlib.util
import math
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("MUJOCO_GL", "glfw")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


class RenderResolutionTests(unittest.TestCase):
    def test_render_px_scales_with_megapixels(self):
        from virturoid.services.camera_perception import render_px_for_camera
        from virturoid.services.component_catalog import resolve_part
        lo = render_px_for_camera(resolve_part("Arducam OV9281"))     # 1 MP
        hi = render_px_for_camera(resolve_part("Luxonis OAK-D Pro"))  # 12 MP
        self.assertLess(lo, hi, "a higher-megapixel camera must feed the encoder a sharper (larger) render")
        self.assertTrue(64 <= lo <= hi <= 256)                        # clamped to a useful band


@unittest.skipUnless(_MUJOCO, "compiling + rendering the robot's own camera needs MuJoCo")
class FunctionalCameraTests(unittest.TestCase):
    def _cam_robot(self):
        from virturoid.services.morphology_composer import compose_robot
        return compose_robot("a mobile robot that navigates with a camera")

    def test_camera_robot_gets_a_functional_camera_with_part_fov(self):
        import mujoco
        from virturoid.services.gene_compiler import gene_to_meshed_mjcf, standing_spawn_z
        gene = self._cam_robot()
        m = mujoco.MjModel.from_xml_string(gene_to_meshed_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene)))
        cams = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)]
        self.assertIn("robot_cam", cams, "a camera-equipped robot must carry a FUNCTIONAL robot_cam")
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "robot_cam")
        self.assertGreater(float(m.cam_fovy[cid]), 0)                 # a real FOV from the part, not zero/default

    def test_camera_fov_comes_from_the_pinned_part(self):
        # pin a specific camera -> its datasheet FOV drives the sim camera (a real "features affect the camera" proof)
        import mujoco
        from virturoid.services.camera_perception import robot_camera_part
        from virturoid.services.gene_compiler import gene_to_meshed_mjcf, standing_spawn_z
        from virturoid.services.sensor_geometry import _camera_fovy_deg
        gene = self._cam_robot()
        gene.metadata["pinned_parts"] = {"camera": "Logitech C920"}
        part = robot_camera_part(gene)
        self.assertEqual(part.name, "Logitech C920")
        # the compiled camera's fovy should match the pinned part's vertical FOV (C920 ~ 78 deg-class, not the default 60)
        expected = _camera_fovy_deg(part)
        m = mujoco.MjModel.from_xml_string(gene_to_meshed_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene)))
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "robot_cam")
        self.assertAlmostEqual(float(m.cam_fovy[cid]), expected, delta=0.5)

    def test_robot_perceives_a_target_through_its_own_camera(self):
        from virturoid.services.camera_perception import robot_sees_target
        r = robot_sees_target(self._cam_robot())
        self.assertTrue(r["has_camera"])
        self.assertEqual(r["perception"], "onboard_camera + tiny_cv")
        self.assertTrue(r["sees"], f"the robot's own camera + CV must detect the target, got {r}")
        self.assertGreater(r["render_px"], 0)


if __name__ == "__main__":
    unittest.main()
