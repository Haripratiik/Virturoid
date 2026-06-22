"""MJCF/URDF export for a differential-drive mobile base (#5).

The arm exporter assumes a serial chain; a wheeled base needs a floating chassis
with two powered wheels on a floor. This module emits that model so the second
robot class is genuinely simulatable, not just a schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

from virturoid.schemas.robot import RobotGenome
from virturoid.schemas.scenes import SceneGraph, SceneObject, SceneSet
from virturoid.services.mobile_base_builder import TRACK_HALF_M, WHEEL_HALF_WIDTH_M, WHEEL_RADIUS_M

_CHASSIS_HALF = (0.15, 0.12, 0.045)


def mobile_base_to_mjcf(robot: RobotGenome, scene: SceneGraph | None = None) -> str:
    r = WHEEL_RADIUS_M
    hw = WHEEL_HALF_WIDTH_M
    track = TRACK_HALF_M
    chassis_z = r + 0.02
    cx, cy, cz = _CHASSIS_HALF
    model_name = f"{robot.name} {scene.id}" if scene else robot.name
    lines = [
        f'<mujoco model="{escape(model_name)}">',
        '  <compiler angle="radian" />',
        '  <option timestep="0.002" gravity="0 0 -9.81" />',
        "  <default>",
        '    <joint damping="0.2" armature="0.01" />',
        '    <geom friction="1.2 0.05 0.01" />',
        "  </default>",
        "  <worldbody>",
        '    <light name="key" pos="0 0 3" dir="0 0 -1" />',
        '    <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0" rgba="0.6 0.6 0.62 1" />',
        f'    <body name="base_link" pos="0 0 {round(chassis_z, 4)}">',
        '      <freejoint name="chassis_free" />',
        f'      <geom name="chassis" type="box" size="{cx} {cy} {cz}" mass="2.4" rgba="0.25 0.3 0.45 1" />',
        # Passive casters keep the two-wheel base upright.
        f'      <geom name="caster_front" type="sphere" size="0.02" pos="{cx - 0.02} 0 {-cz}" friction="0.1 0.05 0.01" rgba="0.2 0.2 0.2 1" />',
        f'      <geom name="caster_rear" type="sphere" size="0.02" pos="{-(cx - 0.02)} 0 {-cz}" friction="0.1 0.05 0.01" rgba="0.2 0.2 0.2 1" />',
        f'      <geom name="sensor_mast" type="box" size="0.02 0.02 0.1" pos="{cx - 0.04} 0 {cz + 0.1}" rgba="0.3 0.3 0.3 1" />',
        _wheel_body("left_wheel_link", "left_wheel", track, r, hw, chassis_z),
        _wheel_body("right_wheel_link", "right_wheel", -track, r, hw, chassis_z),
        "    </body>",
        *_scene_objects_xml(scene.objects if scene else []),
        "  </worldbody>",
        "  <actuator>",
        '    <motor name="left_wheel_motor" joint="left_wheel" gear="1" forcerange="-22 22" />',
        '    <motor name="right_wheel_motor" joint="right_wheel" gear="1" forcerange="-22 22" />',
        "  </actuator>",
        "</mujoco>",
    ]
    return "\n".join(lines) + "\n"


def write_mobile_base_scene_collection(robot: RobotGenome, scene_set: SceneSet, output_dir: Path) -> Path:
    scene_dir = output_dir / "simulation" / "mujoco" / "scenes" / scene_set.purpose
    scene_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for scene in scene_set.scenes:
        relative_uri = f"simulation/mujoco/scenes/{scene_set.purpose}/{scene.id}.xml"
        scene_path = output_dir / relative_uri
        scene_path.write_text(mobile_base_to_mjcf(robot, scene), encoding="utf-8")
        entries.append(
            {
                "scene_set_id": scene_set.id,
                "scene_id": scene.id,
                "purpose": scene_set.purpose,
                "mujoco_xml": relative_uri,
                "object_count": len(scene.objects),
            }
        )

    index = {
        "id": f"compiled_mujoco_{robot.id}",
        "robot_genome_id": robot.id,
        "backend": "mujoco",
        "scene_count": len(entries),
        "scenes": entries,
        "notes": [
            "Each navigation scene has a concrete MJCF XML file.",
            "Mobile-base scenes include start/goal zones and static obstacles.",
        ],
    }
    index_path = output_dir / "simulation" / "mujoco" / "compiled_scene_index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index_path


def _wheel_body(link: str, joint: str, y: float, r: float, hw: float, chassis_z: float) -> str:
    # Wheel center sits at axle height (z = r) relative to the chassis body origin.
    z = round(r - chassis_z, 4)
    y1 = round(y - hw, 4)
    y2 = round(y + hw, 4)
    return (
        f'      <body name="{escape(link)}" pos="0 {y} {z}">\n'
        f'        <joint name="{escape(joint)}" type="hinge" axis="0 1 0" />\n'
        f'        <geom name="{escape(link)}_geom" type="cylinder" fromto="0 {y1 - y} 0 0 {y2 - y} 0" '
        f'size="{r}" mass="0.35" rgba="0.1 0.1 0.1 1" />\n'
        "      </body>"
    )


def _scene_objects_xml(objects: list[SceneObject]) -> list[str]:
    return [_scene_object_xml(item) for item in objects]


def _scene_object_xml(item: SceneObject) -> str:
    x, y, _, _, _, yaw = item.pose_xyz_rpy
    if item.object_type == "zone":
        radius = round(0.18 * (item.scale or 1.0), 4)
        rgba = "0.1 0.3 0.9 0.22" if item.name == "start_zone" else "0.1 0.8 0.25 0.25"
        return (
            f'    <geom name="{escape(item.name)}" type="cylinder" size="{radius} 0.002" '
            f'pos="{x} {y} 0.003" rgba="{rgba}" contype="0" conaffinity="0" />'
        )
    if item.object_type == "obstacle":
        half = round(0.07 * (item.scale or 1.0), 4)
        height = round(0.10 * (item.scale or 1.0), 4)
        z = round(height, 4)
        friction_attr = f' friction="{item.friction} 0.05 0.01"' if item.friction else ""
        return (
            f'    <geom name="{escape(item.name)}" type="box" size="{half} {half} {height}" '
            f'pos="{x} {y} {z}" euler="0 0 {round(yaw, 4)}"{friction_attr} rgba="0.35 0.35 0.38 1" />'
        )
    return (
        f'    <geom name="{escape(item.name)}" type="sphere" size="0.04" '
        f'pos="{x} {y} 0.04" rgba="0.5 0.5 0.5 1" />'
    )


def mobile_base_to_urdf(robot: RobotGenome) -> str:
    lines = [f'<robot name="{escape(robot.name)}">']
    for link in robot.links:
        lines.extend(
            [
                f'  <link name="{escape(link)}">',
                '    <inertial><origin xyz="0 0 0" /><mass value="0.5" />'
                '<inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001" /></inertial>',
                "  </link>",
            ]
        )
    for joint in robot.joints:
        axis = joint.axis_xyz or (0.0, 1.0, 0.0)
        y = TRACK_HALF_M if "left" in joint.name else -TRACK_HALF_M
        lines.extend(
            [
                f'  <joint name="{escape(joint.name)}" type="continuous">',
                f'    <parent link="{escape(joint.parent_link)}" />',
                f'    <child link="{escape(joint.child_link)}" />',
                f'    <origin xyz="0 {y} 0" rpy="0 0 0" />',
                f'    <axis xyz="{axis[0]} {axis[1]} {axis[2]}" />',
                "  </joint>",
            ]
        )
    lines.append("</robot>")
    return "\n".join(lines) + "\n"


def drive_rollout(mjcf_path: Path, left_torque: float, right_torque: float, steps: int = 600) -> dict:
    """Step the base under wheel torques; return base displacement and yaw change."""
    import math

    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    start = np.array(data.qpos[:3], dtype=float)
    start_yaw = _yaw(data.qpos[3:7])
    stable = True
    for _ in range(steps):
        data.ctrl[0] = left_torque
        data.ctrl[1] = right_torque
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            stable = False
            break
    end = np.array(data.qpos[:3], dtype=float)
    displacement = end - start
    return {
        "forward_distance_m": round(float(displacement[0]), 4),
        "lateral_distance_m": round(float(displacement[1]), 4),
        "planar_distance_m": round(float(math.hypot(displacement[0], displacement[1])), 4),
        "yaw_change_rad": round(float(_yaw(data.qpos[3:7]) - start_yaw), 4),
        "stable": stable,
    }


def _yaw(quat) -> float:
    import math

    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def write_mobile_base_files(robot: RobotGenome, output_dir: Path) -> dict:
    robot_dir = output_dir / "robot"
    sim_dir = output_dir / "simulation" / "mujoco"
    robot_dir.mkdir(parents=True, exist_ok=True)
    sim_dir.mkdir(parents=True, exist_ok=True)
    urdf_path = robot_dir / "robot.urdf"
    mjcf_path = sim_dir / "mobile_base_scene.xml"
    urdf_path.write_text(mobile_base_to_urdf(robot), encoding="utf-8")
    mjcf_path.write_text(mobile_base_to_mjcf(robot), encoding="utf-8")
    return {"urdf": urdf_path, "mjcf": mjcf_path}
