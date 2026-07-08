"""In-house COMPUTER VISION training on a BUILT robot's OWN onboard camera.

The un-gameable claim: render the robot's functional ``robot_cam`` (mounted + FOV'd from its REAL camera part) over
many target placements, encode each frame with the tiny 2-conv CNN, and fit a readout that reads the target's
bearing — then measure the HELD-OUT bearing error vs a predict-the-mean baseline. If the trained readout beats the
baseline, the CV genuinely learned to perceive through THIS robot's camera (not the rangefinder ring, not a generic
scene). The ``train_camera_policy`` agent tool also BANKS the learned vision on the robot so it deploys with it.

MuJoCo-gated (compiles + renders the robot's own camera); the honest no-camera branch always runs.
"""
import importlib.util
import os
import unittest
from unittest import mock

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("MUJOCO_GL", "glfw")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


class HonestNoCameraTests(unittest.TestCase):
    """A robot that carries no camera trains no vision — we say so, we don't fabricate a model."""

    def test_evaluate_is_honest_without_a_camera(self):
        from virturoid.services import robot_vision
        with mock.patch("virturoid.services.camera_perception.robot_camera_part", return_value=None):
            r = robot_vision.evaluate_robot_vision(object())
        self.assertFalse(r["has_camera"])
        self.assertFalse(r["trained_in_house"])
        self.assertIn("note", r)

    def test_tool_rejects_a_robot_with_no_camera(self):
        from virturoid.services import input_training_tools as itt
        from virturoid.services import session_state as S
        from virturoid.services.morphology_composer import compose_robot
        rid = S.put_robot(compose_robot("a quadruped robot dog"), label="nocam")
        with mock.patch("virturoid.services.camera_perception.robot_camera_part", return_value=None):
            out = itt._train_camera_policy({"robot_id": rid})
        self.assertIn("error", out)
        self.assertNotIn("banked", out)


@unittest.skipUnless(_MUJOCO, "training the robot's own camera needs MuJoCo renders")
class InHouseCameraTrainingTests(unittest.TestCase):
    def _cam_robot(self):
        from virturoid.services.morphology_composer import compose_robot
        return compose_robot("a mobile robot that navigates with a camera")

    def test_trains_vision_on_the_robots_own_camera(self):
        # the honest, un-gameable proof: held-out bearing error beats a predict-the-mean baseline
        from virturoid.services.robot_vision import evaluate_robot_vision
        r = evaluate_robot_vision(self._cam_robot(), n_train=220, seed=0)
        self.assertTrue(r["has_camera"])
        self.assertTrue(r["trained_in_house"])
        self.assertGreater(r["render_px"], 0)
        self.assertTrue(r["learned_perception"], f"CV must beat the predict-mean baseline on the robot's camera: {r}")
        self.assertGreaterEqual(r["improvement_x"], 1.5)
        self.assertLess(r["bearing_mae_deg"], 12.0, f"held-out bearing error too high to call it perception: {r}")

    def test_render_resolution_tracks_the_camera_part(self):
        # a higher-megapixel camera feeds the encoder a sharper (larger) render — the camera's specs are real
        from virturoid.services.camera_perception import render_px_for_camera, robot_camera_part
        gene = self._cam_robot()
        px_default = render_px_for_camera(robot_camera_part(gene))
        gene.metadata["pinned_parts"] = {"camera": "Luxonis OAK-D Pro"}   # 12 MP
        px_hi = render_px_for_camera(robot_camera_part(gene))
        self.assertGreaterEqual(px_hi, px_default)
        self.assertTrue(64 <= px_default <= px_hi <= 256)

    def test_tool_trains_and_banks_the_vision_policy(self):
        from virturoid.services import input_training_tools as itt
        from virturoid.services import session_state as S
        rid = S.put_robot(self._cam_robot(), label="cv")
        out = itt._train_camera_policy({"robot_id": rid, "n_train": 220, "seed": 0})
        self.assertTrue(out["trained_in_house"])
        self.assertTrue(out["banked"])
        self.assertTrue(out["learned"], f"the banked vision must clear the baseline: {out}")
        vp = (S.get_robot(rid).metadata or {}).get("vision_policy")
        self.assertIsNotNone(vp, "the learned vision must be banked on the robot so it deploys with it")
        self.assertTrue(vp["trained"])
        self.assertGreater(len(vp["readout"]), 8, "a real fitted readout, not a stub")
        self.assertEqual(vp["camera_part"], out["camera_part"])

    def test_banked_vision_is_consumed_by_perception(self):
        # the trained CV is actually USED downstream: robot_sees_target prefers the robot's own learned readout
        from virturoid.services.camera_perception import robot_sees_target
        from virturoid.services.robot_vision import train_robot_vision
        gene = self._cam_robot()
        before = robot_sees_target(gene)
        self.assertFalse(before.get("vision_trained"), "an untrained robot must not claim learned perception")
        self.assertEqual(before["perception"], "onboard_camera + tiny_cv")
        tr = train_robot_vision(gene, n=200, seed=0)
        gene.metadata["vision_policy"] = {"trained": True, "enc_seed": tr["enc_seed"],
                                          "readout": [float(v) for v in tr["readout"]]}
        after = robot_sees_target(gene)
        self.assertTrue(after["vision_trained"], f"the banked learned vision must drive perception: {after}")
        self.assertEqual(after["perception"], "learned_onboard_camera + tiny_cv")
        self.assertTrue(after["sees"])


if __name__ == "__main__":
    unittest.main()
