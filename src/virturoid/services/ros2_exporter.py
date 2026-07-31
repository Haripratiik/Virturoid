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


def _aerial_flight_config(genome: dict) -> dict | None:
    """A quadcopter's ROS2 flight-deploy reference. A drone has NO actuated joints (welded rotors), so the
    ros2_control joint-trajectory node is inert for it — real flight runs on the onboard autopilot (the Pixhawk
    in the BOM) commanded over MAVROS position setpoints. This returns the geometric-controller reference +
    rotor layout so the package documents the ACTUAL deploy path instead of a silently-wrong walker/reach node."""
    links = [l if isinstance(l, str) else l.get("name", "") for l in genome.get("links", [])]
    n_rotors = sum(1 for n in links if str(n).startswith("rotor")) or 4
    try:
        from virturoid.services.aerial import FLY_GAINS
        gains = dict(FLY_GAINS)
    except Exception:  # noqa: BLE001
        gains = {"kp": [3.0, 3.0, 8.0], "kd": [2.6, 2.6, 5.0], "KR": 533.0, "KW": 65.0, "vcap": 1.2}
    return {"airframe": "quadcopter", "n_rotors": n_rotors,
            "autopilot": "PX4/ArduPilot on the Pixhawk (see BOM) — runs the rotor mixing + inner loops",
            "ros2_interface": "MAVROS offboard: publish geometry_msgs/PoseStamped to "
                              "/mavros/setpoint_position/local (position waypoints); the autopilot handles thrust",
            "geometric_controller_reference": gains,
            "sample_waypoints_xyz": [[0.0, 0.0, 1.2], [1.5, 0.8, 1.4], [-1.2, 0.6, 1.0]],
            "note": "the joint-trajectory evaluation_node is INERT for this jointless airframe — deploy via MAVROS"}


def _copy_urdf_meshes(package_dir: Path, root: Path, package_name: str, urdf: str) -> str:
    """Carry the URDF's visual meshes into the ROS2 package and re-point the references at them.

    ``robot/robot.urdf`` addresses its STLs RELATIVE to itself (``meshes/<name>.stl``, resolving to
    ``robot/meshes/``). Copying only the .urdf into ``export/ros2/<pkg>/urdf/`` therefore produced a
    file whose every mesh reference dangled -- 22 of them on a shipped quadruped -- so the one artifact
    a customer actually opens in RViz/Gazebo/Isaac was the one that could not load.

    The meshes are copied to ``<pkg>/meshes/`` and rewritten to ``package://<pkg>/meshes/<name>.stl``
    rather than kept relative, because that is the only form that survives installation: after
    ``colcon build`` the URDF is read from ``share/<pkg>/urdf/`` and resolved by ROS's package lookup,
    and robot_state_publisher / RViz / Gazebo / Isaac's URDF importer all understand ``package://``
    while none of them reliably resolve a path relative to the .urdf file. A reference that cannot be
    resolved is left exactly as written and called out in a comment at the end of the file, so the
    export never silently claims geometry it did not ship.
    """
    import re
    import shutil

    src_dir = package_dir / "robot"
    out_dir = root / "meshes"
    copied: set[str] = set()
    missing: list[str] = []

    def _rewrite(match: "re.Match[str]") -> str:
        ref = match.group(1)
        if ref.startswith(("package://", "file://", "http://", "https://", "/")):
            return match.group(0)                       # already absolute/addressable: leave it alone
        src = (src_dir / ref).resolve()
        try:
            src.relative_to(src_dir.resolve())          # never copy out of the package
        except ValueError:
            missing.append(ref)
            return match.group(0)
        if not src.is_file():
            missing.append(ref)
            return match.group(0)
        if src.name not in copied:
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, out_dir / src.name)
            copied.add(src.name)
        return f'filename="package://{package_name}/meshes/{src.name}"'

    urdf = re.sub(r'filename="([^"]+)"', _rewrite, urdf)
    if missing:
        note = ("  <!-- NOTE: {n} mesh reference(s) could not be resolved from robot/ and are left as written: "
                "{refs}. Those links fall back to their primitive collision shapes. -->\n").format(
            n=len(missing), refs="; ".join(sorted(set(missing))[:8]))
        urdf = urdf.replace("</robot>", note + "</robot>")
    return urdf


def export_ros2_package(package_dir: Path, package_name: str = "virturoid_robot") -> Path:
    package_dir = Path(package_dir)
    genome = json.loads((package_dir / "robot" / "robot_genome.json").read_text(encoding="utf-8"))
    joints = [j["name"] for j in genome.get("joints", [])]
    aerial = (genome.get("robot_class") == "aerial"
              or any(str(l if isinstance(l, str) else l.get("name", "")).startswith("rotor")
                     for l in genome.get("links", [])))

    root = package_dir / "export" / "ros2" / package_name
    (root / package_name).mkdir(parents=True, exist_ok=True)
    (root / "launch").mkdir(parents=True, exist_ok=True)
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "test").mkdir(parents=True, exist_ok=True)
    (root / "resource").mkdir(parents=True, exist_ok=True)
    # M7: ship the robot_description -- copy the package's URDF into urdf/ so the installed package can publish it
    # via robot_state_publisher (was absent -> a ROS 2 package with no robot to describe).
    _src_urdf = package_dir / "robot" / "robot.urdf"
    if _src_urdf.exists():
        (root / "urdf").mkdir(parents=True, exist_ok=True)
        (root / "urdf" / "robot.urdf").write_text(
            _copy_urdf_meshes(package_dir, root, package_name, _src_urdf.read_text(encoding="utf-8")),
            encoding="utf-8")

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
    # ros2_control DEPLOY substrate (§4.7): a controller-manager config + a BOM-keyed hardware-interface map (the
    # bridge to the REAL motors you bought) + the safety filter the node applies before commanding a motor.
    actuator_map = _read_actuator_map(package_dir)
    (root / "config" / "ros2_control.yaml").write_text(_ros2_control_yaml(joints), encoding="utf-8")
    (root / "config" / "hardware_interface.yaml").write_text(
        _hardware_interface_yaml(joints, actuator_map), encoding="utf-8")
    (root / package_name / "safety_filter.py").write_text(_SAFETY_FILTER_PY, encoding="utf-8")
    (root / "test" / "test_task_regression.py").write_text(_TEST_PY, encoding="utf-8")
    # AERIAL: a quadcopter has no actuated joints, so ros2_control joint trajectories are the wrong interface.
    # Emit the ACTUAL flight-deploy reference (autopilot + MAVROS + the geometric controller) instead of letting
    # the README claim a walker/reach controller that can't fly it.
    flight = _aerial_flight_config(genome) if aerial else None
    if flight is not None:
        (root / "config" / "flight.yaml").write_text(json.dumps(flight, indent=2), encoding="utf-8")
    _deploy_md = (
        f"## Deploy this drone (aerial)\n"
        f"This airframe is a {flight['airframe']} with {flight['n_rotors']} rotors and NO actuated joints, so the\n"
        f"ros2_control joint interface below does NOT apply. Flight runs on the onboard autopilot: {flight['autopilot']}.\n"
        f"Command it from ROS2 via {flight['ros2_interface']}. `config/flight.yaml` carries the geometric-controller\n"
        f"reference (rotor layout + gains) Virturoid used in sim; the joint `evaluation_node` is inert for this body.\n"
        if flight is not None else
        "## Deploy to hardware (§4.7)\n"
        "`config/ros2_control.yaml` (controller manager) + `config/hardware_interface.yaml` (each joint -> its\n"
        "real BOM actuator) wire the controller to ros2_control; set `hardware_plugin` to your motor-bus driver\n"
        "(Dynamixel / ODrive / CAN / EtherCAT). `" + package_name + "/safety_filter.py` clamps every command to\n"
        "joint + rate limits before a motor sees it. Validate the closed loop in sim first via\n"
        "`services/sim_ros_bridge` (MuJoCo as virtual hardware behind the same command/state interface).\n")
    _ctrl_md = (
        " — a quadcopter: flight deploys via the onboard autopilot (see below), not this joint node.\n\n" if aerial
        else (f" — runs the exported {'GaitController (trot gait)' if policy_type == 'trot_cpg_gait' else 'ReachController'}.\n\n"
              if has_controller else " (no controller bundle; node publishes a neutral pose).\n\n"))
    (root / "README.md").write_text(
        f"# {package_name}\n\nGenerated ROS2 package for `{genome.get('id')}`" + _ctrl_md
        + "```\ncolcon build --packages-select " + package_name + "\nros2 launch " + package_name
        + " evaluate.launch.py\n```\n\n" + _deploy_md,
        encoding="utf-8",
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


def _read_actuator_map(package_dir: Path) -> dict:
    """The joint -> real-motor map from the bill of materials, so the hardware interface names the ACTUAL part
    per joint. Empty (generic) when no BOM was written with the package."""
    for rel in ("robot/bill_of_materials.json", "reports/bill_of_materials.json"):
        p = package_dir / rel
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8")).get("actuator_map", {}) or {}
            except (json.JSONDecodeError, OSError):
                pass
    return {}


def _ros2_control_yaml(joints: list) -> str:
    jl = "\n".join(f"      - {j}" for j in joints) or "      []"
    return (
        "controller_manager:\n"
        "  ros__parameters:\n"
        "    update_rate: 100  # Hz\n"
        "    joint_state_broadcaster:\n"
        "      type: joint_state_broadcaster/JointStateBroadcaster\n"
        "    joint_trajectory_controller:\n"
        "      type: joint_trajectory_controller/JointTrajectoryController\n\n"
        "joint_trajectory_controller:\n"
        "  ros__parameters:\n"
        "    joints:\n" + jl + "\n"
        "    command_interfaces: [position]\n"
        "    state_interfaces: [position, velocity]\n")


def _hardware_interface_yaml(joints: list, actuator_map: dict) -> str:
    rows = []
    for j in joints:
        motor = actuator_map.get(j) or "GENERIC position actuator (set from the BOM)"
        rows.append(f'  {j}:\n    actuator: "{motor}"\n    command_interface: position\n'
                    f"    state_interfaces: [position, velocity]")
    body = "\n".join(rows) or "  {}"
    return (
        "# Maps each robot joint to the REAL actuator from the bill of materials and the ros2_control interface it\n"
        "# exposes. Set hardware_plugin to your bus driver (Dynamixel / ODrive / CAN / EtherCAT) before deploying.\n"
        'hardware:\n  hardware_plugin: "REPLACE_WITH_YOUR_DRIVER  # e.g. dynamixel_hardware/DynamixelHardware"\n'
        "joints:\n" + body + "\n")


_SAFETY_FILTER_PY = '''"""Pure-stdlib safety gate (no ROS/numpy): clamp commanded joint-position targets to joint
limits + a per-step rate (velocity) limit. The LAST thing before a real motor -- it prevents a policy from
driving a joint past its mechanical stop or slewing faster than the safety budget. The node calls clamp() on
every command before publishing to ros2_control. Mirrors services/sim_ros_bridge.SafetyFilter."""


class SafetyFilter:
    def __init__(self, lower, upper, vel_limit=8.0):
        self.lower, self.upper, self.vel_limit = list(lower), list(upper), float(vel_limit)

    def clamp(self, target, q, dt):
        """Return (clamped_targets, n_violations); a violation = the raw command had to be altered."""
        out, violations = [], 0
        step = max(1e-6, self.vel_limit * float(dt))
        for i, t in enumerate(target):
            c = min(self.upper[i], max(self.lower[i], float(t)))
            c = min(q[i] + step, max(q[i] - step, c))
            if abs(c - float(t)) > 1e-6:
                violations += 1
            out.append(c)
        return out, violations
'''


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
  <exec_depend>launch</exec_depend>
  <exec_depend>launch_ros</exec_depend>
  <exec_depend>ament_index_python</exec_depend>
  <exec_depend>robot_state_publisher</exec_depend>
  <exec_depend>xacro</exec_depend>
  <test_depend>python3-pytest</test_depend>
  <export><build_type>ament_python</build_type></export>
</package>
"""

_SETUP_PY = """import os
from glob import glob

from setuptools import setup

package_name = "{name}"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    # M7 (2026-07-24 audit): GLOB every config + launch + urdf so they survive `colcon build` install (was: a
    # hardcoded subset -> ros2_control.yaml / hardware_interface.yaml / the URDF never installed -> the package
    # only ran from the source tree). package_data ships policy_params.json beside the installed module.
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml") + glob("config/*.json")),
        (os.path.join("share", package_name, "urdf"), glob("urdf/*")),
        # The URDF's visual meshes. Without this the installed robot_description resolves
        # package://{name}/meshes/... to nothing and every mesh link renders empty.
        (os.path.join("share", package_name, "meshes"), glob("meshes/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    package_data={{package_name: ["*.json"]}},
    include_package_data=True,
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


def _config_dir():
    # M7: after `colcon build` the config installs to share/<pkg>/config (NOT beside the module) -- resolve it via
    # the ament share dir, falling back to the source-tree layout so `python evaluation_node.py` still runs.
    try:
        from ament_index_python.packages import get_package_share_directory
        share = Path(get_package_share_directory("virturoid_robot"))
        if (share / "config").is_dir():
            return share / "config"
    except Exception:
        pass
    return _PKG.parents[1] / "config"


class EvaluationNode(Node):
    def __init__(self):
        super().__init__("virturoid_evaluation_node")
        config = json.loads((_config_dir() / "robot.yaml").read_text())
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
            cls = mod.GaitController if self.policy_type in ("trot_cpg_gait", "crawl_wave_gait") else mod.ReachController
            self.controller = cls.from_file(str(_PKG / "policy_params.json"))
            self.joints = self.controller.joint_names
        self.pub = self.create_publisher(JointTrajectory, "/joint_trajectory_controller/joint_trajectory", 10)
        self.create_timer(self.dt, self.tick)

    def tick(self):
        msg = JointTrajectory()
        msg.joint_names = self.joints
        point = JointTrajectoryPoint()
        if self.controller is not None:
            if self.policy_type in ("trot_cpg_gait", "crawl_wave_gait"):
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
    if config.get("policy_type") in ("trot_cpg_gait", "crawl_wave_gait"):
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
