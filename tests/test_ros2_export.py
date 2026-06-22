"""§24 / §30: the exported ROS2 package is structurally real and RUNS the exported controller."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from virturoid.schemas.trained_policy import ActionDimension, TrainedPolicy
from virturoid.services.controller_exporter import export_controller_bundle
from virturoid.services.ros2_exporter import export_ros2_package, maybe_export_ros2_package


def _fake_package(root: Path):
    """Minimal package on disk: a robot_genome + a tiny scene set (enough for the ROS2 export)."""
    (root / "robot").mkdir(parents=True, exist_ok=True)
    (root / "simulation").mkdir(parents=True, exist_ok=True)
    genome = {"id": "test_arm", "joints": [{"name": "base_yaw"}, {"name": "shoulder_pitch"},
                                           {"name": "elbow_pitch"}]}
    (root / "robot" / "robot_genome.json").write_text(json.dumps(genome), encoding="utf-8")
    (root / "simulation" / "scene_set.json").write_text(json.dumps(
        {"scenes": [{"objects": [{"pose_xyz_rpy": [0.42, -0.05, 0.05, 0, 0, 0]}]}]}), encoding="utf-8")


def _policy() -> TrainedPolicy:
    joints = ["base_yaw", "shoulder_pitch", "elbow_pitch"]
    return TrainedPolicy(
        id="p1", name="reach", robot_genome_id="test_arm", joint_names=joints,
        action_dimensions=[ActionDimension(j, -1.0, 1.0) for j in joints],
        control_frequency_hz=20.0, weights=[[0.1, 0.0, 0.0], [0.0, 0.1, 0.2], [0.0, -0.1, 0.1]],
        input_features=["x", "y", "bias"],
        safety_clamps={"joint_position_limits": [[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]]})


def _run_bundled_test(root: Path, fn: str) -> None:
    """Import and run a function from the package's generated regression test (no ROS2 needed)."""
    spec = importlib.util.spec_from_file_location("ros2_regr", root / "test" / "test_task_regression.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    getattr(mod, fn)()


class Ros2ExportTests(unittest.TestCase):
    def test_structure_without_controller(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); _fake_package(root)
            pkg = export_ros2_package(root)
            for rel in ("package.xml", "setup.py", "launch/evaluate.launch.py", "config/robot.yaml",
                        "virturoid_robot/evaluation_node.py", "test/test_task_regression.py"):
                self.assertTrue((pkg / rel).exists(), rel)
            cfg = json.loads((pkg / "config" / "robot.yaml").read_text())
            self.assertEqual(cfg["joints"], ["base_yaw", "shoulder_pitch", "elbow_pitch"])
            self.assertFalse(cfg["has_controller"])
            # node + test compile as valid Python
            compile((pkg / "virturoid_robot" / "evaluation_node.py").read_text(), "node.py", "exec")
            _run_bundled_test(pkg, "test_config_has_joints")

    def test_embeds_and_runs_the_exported_controller(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); _fake_package(root)
            export_controller_bundle(root, _policy())          # write software/controller/*
            pkg = maybe_export_ros2_package(root)
            self.assertIsNotNone(pkg)
            cfg = json.loads((pkg / "config" / "robot.yaml").read_text())
            self.assertTrue(cfg["has_controller"])
            self.assertEqual(cfg["target_positions"], [[0.42, -0.05]])   # pulled from the scene set
            self.assertTrue((pkg / "virturoid_robot" / "controller.py").exists())
            self.assertTrue((pkg / "virturoid_robot" / "policy_params.json").exists())
            # the bundled regression test runs the REAL controller and checks clamped joint targets
            _run_bundled_test(pkg, "test_controller_runs_if_present")

    def test_maybe_export_is_safe_without_genome(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(maybe_export_ros2_package(Path(tmp)))    # no genome -> no crash, no package


if __name__ == "__main__":
    unittest.main()
