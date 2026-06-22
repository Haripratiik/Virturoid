import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

from virturoid.mvp import write_mvp_robot_arm_project
from virturoid.services.backend_adapters import available_backends, backend_registry
from virturoid.services.ros2_exporter import export_ros2_package

_MUJOCO = importlib.util.find_spec("mujoco") is not None


class Ros2ExportTests(unittest.TestCase):
    def test_ros2_package_is_structurally_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = write_mvp_robot_arm_project(Path(tmpdir) / "robot")
            root = export_ros2_package(out, package_name="virturoid_robot")

            for rel in ["package.xml", "setup.py", "launch/evaluate.launch.py", "config/robot.yaml",
                        "virturoid_robot/evaluation_node.py", "test/test_task_regression.py"]:
                self.assertTrue((root / rel).exists(), rel)

            # package.xml parses and names the package; config lists the real joints.
            xml = ElementTree.parse(root / "package.xml").getroot()
            self.assertEqual("virturoid_robot", xml.find("name").text)
            config = json.loads((root / "config" / "robot.yaml").read_text(encoding="utf-8"))
            self.assertEqual(["base_yaw", "shoulder_pitch", "elbow_pitch"], config["joints"])
            # The generated node is valid Python.
            compile((root / "virturoid_robot" / "evaluation_node.py").read_text(encoding="utf-8"), "node.py", "exec")


class BackendRegistryTests(unittest.TestCase):
    def test_registry_reports_all_backends(self):
        registry = backend_registry()
        self.assertEqual({"mujoco", "isaac", "gazebo"}, set(registry))
        # Isaac/Gazebo are interface-only here; MuJoCo status reflects availability.
        self.assertIn(registry["isaac"].status, {"available", "interface_only"})
        self.assertIn(registry["gazebo"].status, {"available", "interface_only"})
        self.assertTrue(registry["mujoco"].capabilities)

    @unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
    def test_mujoco_is_available(self):
        self.assertIn("mujoco", available_backends())
        self.assertEqual("implemented", backend_registry()["mujoco"].status)


if __name__ == "__main__":
    unittest.main()
