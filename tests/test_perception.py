import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from virturoid.mvp import write_mvp_robot_arm_project

_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
class PerceptionInTheLoopTests(unittest.TestCase):
    def _scene(self, out):
        import mujoco

        ss = json.loads((out / "simulation" / "baseline_scene_set.json").read_text(encoding="utf-8"))
        scene = ss["scenes"][0]
        xml = out / "simulation" / "mujoco" / "scenes" / "baseline" / f"{scene['id']}.xml"
        model = mujoco.MjModel.from_xml_path(str(xml))
        objects = {o["name"]: tuple(o["pose_xyz_rpy"][:3]) for o in scene["objects"]}
        return model, objects

    def test_synthetic_perception_detects_blocks_and_drives_grasp(self):
        from virturoid.services.perception_adapter import SyntheticPerception
        from virturoid.services.pick_place_controller import run_pick_place_episode

        with tempfile.TemporaryDirectory() as tmpdir:
            out = write_mvp_robot_arm_project(Path(tmpdir) / "robot")
            model, objects = self._scene(out)

            # With clean perception the arm still sorts (perception-in-the-loop).
            outcome = run_pick_place_episode(
                model, {"objects": objects}, perception=SyntheticPerception(seed=2, dropout=0.0, misdetect=0.0)
            )
            self.assertEqual(2, outcome["block_count"])
            self.assertEqual("success", outcome["status"])

    def test_perception_dropout_causes_missed_grasp(self):
        from virturoid.services.perception_adapter import SyntheticPerception
        from virturoid.services.pick_place_controller import run_pick_place_episode

        with tempfile.TemporaryDirectory() as tmpdir:
            out = write_mvp_robot_arm_project(Path(tmpdir) / "robot")
            model, objects = self._scene(out)

            # If the sensor never detects a block, the robot cannot act on it.
            outcome = run_pick_place_episode(
                model, {"objects": objects}, perception=SyntheticPerception(seed=2, dropout=1.0)
            )
            self.assertEqual("failure", outcome["status"])
            self.assertEqual("missed_grasp", outcome["failure_label"])
            self.assertTrue(all(not b.get("detected", True) for b in outcome["per_block"].values()))

    def test_perception_evaluation_reports_backend(self):
        from virturoid.services.perception_adapter import SyntheticPerception
        from virturoid.services.physics_evaluator import run_physics_pick_place_evaluation

        with tempfile.TemporaryDirectory() as tmpdir:
            out = write_mvp_robot_arm_project(Path(tmpdir) / "robot")
            report = run_physics_pick_place_evaluation(
                out, scene_uris=["simulation/baseline_scene_set.json"], perception=SyntheticPerception(seed=3)
            )
            self.assertTrue(report.validate().ok)
            self.assertIn("+perception", report.controller)
            self.assertTrue(any("synthetic_sensor" in n for n in report.notes))


if __name__ == "__main__":
    unittest.main()
