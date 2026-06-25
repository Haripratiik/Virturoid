"""Real ROS2 package export (Phase 5, plan §30 / §24 second demo).

Generates an installable ROS2 (ament_python) package from the Robot Genome and the exported control
program: package.xml, setup.py, a launch file, config YAML, an evaluation node, and a regression test.
When a controller bundle is present in the package (``software/controller/``), the node runs the
*actual* exported ``ReachController`` and publishes the joint targets it infers for a sequence of target
positions — so the exported artifact really runs the controller (quality bar §22), not a zero stub. The
controller is pure-stdlib, so the bundled regression test exercises it without a ROS2 runtime.
"""

from __future__ import annotations

import json
from pathlib import Path


def export_ros2_package(package_dir: Path, package_name: str = "virturoid_robot") -> Path:
    package_dir = Path(package_dir)
    genome = json.loads((package_dir / "robot" / "robot_genome.json").read_text(encoding="utf-8"))
    joints = [j["name"] for j in genome.get("joints", [])]

    root = package_dir / "export" / "ros2" / package_name
    (root / package_name).mkdir(parents=True, exist_ok=True)
    (root / "launch").mkdir(parents=True, exist_ok=True)
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "test").mkdir(parents=True, exist_ok=True)
    (root / "resource").mkdir(parents=True, exist_ok=True)

    # Embed the exported controller (if the package has one) so the node runs the real policy.
    bundle_dir = package_dir / "software" / "controller"
    has_controller = (bundle_dir / "policy_params.json").exists() and (bundle_dir / "controller.py").exists()
    policy_type = "reach"
    if has_controller:
        try:
            policy_type = json.loads((bundle_dir / "policy_params.json").read_text(encoding="utf-8")).get(
                "policy_type", "reach")
        except Exception:  # noqa: BLE001 - default to the reach harness if the bundle params are unreadable
            policy_type = "reach"
    targets = _harness_targets(package_dir)
    if has_controller:
        (root / package_name / "controller.py").write_text(
            (bundle_dir / "controller.py").read_text(encoding="utf-8"), encoding="utf-8")
        (root / package_name / "policy_params.json").write_text(
            (bundle_dir / "policy_params.json").read_text(encoding="utf-8"), encoding="utf-8")

    (root / "package.xml").write_text(_PACKAGE_XML.format(name=package_name), encoding="utf-8")
    (root / "setup.py").write_text(_SETUP_PY.format(name=package_name), encoding="utf-8")
    (root / "setup.cfg").write_text(_SETUP_CFG.format(name=package_name), encoding="utf-8")
    (root / "resource" / package_name).write_text("", encoding="utf-8")
    (root / package_name / "__init__.py").write_text("", encoding="utf-8")
    (root / package_name / "evaluation_node.py").write_text(_NODE_PY, encoding="utf-8")
    (root / "launch" / "evaluate.launch.py").write_text(_LAUNCH_PY.format(name=package_name), encoding="utf-8")
    (root / "config" / "robot.yaml").write_text(
        json.dumps({"robot_genome_id": genome.get("id"), "joints": joints, "control_frequency_hz": 20.0,
                    "has_controller": has_controller, "policy_type": policy_type, "target_positions": targets},
                   indent=2),
        encoding="utf-8",
    )
    (root / "test" / "test_task_regression.py").write_text(_TEST_PY, encoding="utf-8")
    (root / "README.md").write_text(
        f"# {package_name}\n\nGenerated ROS2 package for `{genome.get('id')}`"
        + ((f" — runs the exported {'GaitController (trot gait)' if policy_type == 'trot_cpg_gait' else 'ReachController'}.\n\n")
           if has_controller else " (no controller bundle; node publishes a neutral pose).\n\n")
        + "```\ncolcon build --packages-select " + package_name + "\nros2 launch " + package_name
        + " evaluate.launch.py\n```\n", encoding="utf-8",
    )
    return root


def maybe_export_ros2_package(package_dir, package_name: str = "virturoid_robot"):
    """Export a ROS2 package if the build has a Robot Genome; never raise into the build pipeline.

    Returns the package root path on success, else None (e.g. genome not written yet, or no joints).
    Lets every trained build emit a runnable ROS2 harness (§24/§30) without making the export a hard
    dependency of the build succeeding.
    """
    package_dir = Path(package_dir)
    if not (package_dir / "robot" / "robot_genome.json").exists():
        return None
    try:
        return export_ros2_package(package_dir, package_name)
    except Exception:  # noqa: BLE001 - ROS2 export must never break the core build
        return None


def _harness_targets(package_dir: Path) -> list[list[float]]:
    """Target block (x, y) positions for the harness to drive the controller through. Pulls real object
    positions from a generated scene set when available, else a small default sweep of the workspace."""
    for rel in ("simulation/holdout_scene_set.json", "simulation/scene_set.json",
                "simulation/baseline_scene_set.json"):
        path = package_dir / rel
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        pts: list[list[float]] = []
        for scene in (data.get("scenes") or [])[:5]:
            for obj in scene.get("objects", []):
                pose = obj.get("pose_xyz_rpy") or obj.get("pose")
                if pose and len(pose) >= 2:
                    pts.append([round(float(pose[0]), 4), round(float(pose[1]), 4)])
                    break
        if pts:
            return pts
    return [[0.40, -0.10], [0.40, 0.10], [0.45, 0.0]]


_PACKAGE_XML = """<?xml version="1.0"?>
<package format="3">
  <name>{name}</name>
  <version>0.1.0</version>
  <description>Virturoid-generated robot evaluation package.</description>
  <maintainer email="robots@virturoid.local">Virturoid</maintainer>
  <license>Apache-2.0</license>
  <exec_depend>rclpy</exec_depend>
  <exec_depend>sensor_msgs</exec_depend>
  <exec_depend>trajectory_msgs</exec_depend>
  <test_depend>python3-pytest</test_depend>
  <export><build_type>ament_python</build_type></export>
</package>
"""

_SETUP_PY = """from setuptools import setup

package_name = "{name}"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/evaluate.launch.py"]),
        ("share/" + package_name + "/config", ["config/robot.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={{"console_scripts": ["evaluation_node = {name}.evaluation_node:main"]}},
)
"""

_SETUP_CFG = """[develop]
script_dir=$base/lib/{name}
[install]
install_scripts=$base/lib/{name}
"""

_NODE_PY = '''"""Virturoid evaluation node: runs the exported controller and publishes its joint targets.

If a controller bundle was exported with this package, the node loads the real ReachController and,
each tick, infers joint position targets for the next target position and publishes them as a
JointTrajectory. With no controller it publishes a neutral pose. This is a runnable harness for the
exported policy (plan §24), not a stub.
"""

import json
from pathlib import Path

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

_PKG = Path(__file__).resolve().parent


class EvaluationNode(Node):
    def __init__(self):
        super().__init__("virturoid_evaluation_node")
        config = json.loads((_PKG.parents[1] / "config" / "robot.yaml").read_text())
        self.joints = config["joints"]
        self.targets = config.get("target_positions") or [[0.4, 0.0]]
        self.policy_type = config.get("policy_type", "reach")
        self.i = 0
        self.t = 0.0
        self.dt = 1.0 / config.get("control_frequency_hz", 20.0)
        self.controller = None
        if config.get("has_controller") and (_PKG / "controller.py").exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("vq_controller", _PKG / "controller.py")
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            cls = mod.GaitController if self.policy_type == "trot_cpg_gait" else mod.ReachController
            self.controller = cls.from_file(str(_PKG / "policy_params.json"))
            self.joints = self.controller.joint_names
        self.pub = self.create_publisher(JointTrajectory, "/joint_trajectory_controller/joint_trajectory", 10)
        self.create_timer(self.dt, self.tick)

    def tick(self):
        msg = JointTrajectory()
        msg.joint_names = self.joints
        point = JointTrajectoryPoint()
        if self.controller is not None:
            if self.policy_type == "trot_cpg_gait":
                targets = self.controller.infer(self.t); self.t += self.dt
            else:
                target = self.targets[self.i % len(self.targets)]; self.i += 1
                targets = self.controller.infer(target)
            point.positions = [float(targets[j]) for j in self.joints]
        else:
            point.positions = [0.0 for _ in self.joints]
        msg.points = [point]
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = EvaluationNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
'''

_LAUNCH_PY = '''from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package="{name}", executable="evaluation_node", name="virturoid_evaluation_node", output="screen"),
    ])
'''

_TEST_PY = '''"""Regression test: config loads, joints agree, and (if present) the exported controller RUNS.

Runs without a ROS2 install — the ReachController is pure stdlib — so `colcon test` / `pytest` exercises
the actual exported policy: it must infer one joint position target per joint, each within its limits.
"""

import json
import importlib.util
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1]


def _config():
    return json.loads((_PKG / "config" / "robot.yaml").read_text())


def test_config_has_joints():
    config = _config()
    assert config["joints"], "robot config must list joints"
    assert config["control_frequency_hz"] > 0


def test_controller_runs_if_present():
    config = _config()
    pkg = _PKG / _PKG.name
    if not config.get("has_controller") or not (pkg / "controller.py").exists():
        return  # no controller bundle exported with this package
    spec = importlib.util.spec_from_file_location("vq_controller", pkg / "controller.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    if config.get("policy_type") == "trot_cpg_gait":
        controller = mod.GaitController.from_file(str(pkg / "policy_params.json"))
        for t in (0.0, 0.1, 0.25, 0.5):
            out = controller.infer(t)
            assert set(out) == set(controller.joint_names), "controller must output every joint"
            for j, limit in zip(controller.joint_names, controller.limits):
                assert limit[0] - 1e-6 <= out[j] <= limit[1] + 1e-6, f"{j} target out of limits"
    else:
        controller = mod.ReachController.from_file(str(pkg / "policy_params.json"))
        for target in config.get("target_positions", [[0.4, 0.0]]):
            out = controller.infer(target)
            assert set(out) == set(controller.joint_names), "controller must output every joint"
            for j, (low, high) in zip(controller.joint_names, controller.position_limits):
                assert low - 1e-6 <= out[j] <= high + 1e-6, f"{j} target out of limits"
'''
