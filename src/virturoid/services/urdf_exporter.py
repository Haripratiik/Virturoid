from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from virturoid.schemas.cad import CadModel
from virturoid.schemas.robot import RobotGenome, RobotJoint
from virturoid.services.robot_kinematics import LinkLayout, compute_arm_layout, iter_links


def robot_genome_to_urdf(
    robot: RobotGenome,
    mesh_prefix: str = "../cad/mesh/visual",
    cad_models: list[CadModel] | None = None,
) -> str:
    """Generate a URDF from a RobotGenome using the shared kinematic layout.

    Link origins, lengths, and masses come from the same layout the MJCF
    exporter uses, so the two robot descriptions stay consistent with the CAD.
    """
    layout_root = compute_arm_layout(robot, cad_models)
    layout_by_link = {link.name: link for link in iter_links(layout_root)}

    lines = [f'<robot name="{escape(robot.name)}">']
    for link_name in robot.links:
        layout = layout_by_link.get(link_name)
        lines.extend(_link_to_urdf_lines(link_name, layout, mesh_prefix))
    for joint in robot.joints:
        layout = layout_by_link.get(joint.child_link)
        lines.extend(_joint_to_urdf_lines(joint, layout))
    lines.append("</robot>")
    return "\n".join(lines) + "\n"


def write_robot_urdf(
    robot: RobotGenome,
    output_dir: Path,
    cad_models: list[CadModel] | None = None,
) -> Path:
    path = output_dir / "robot" / "robot.urdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(robot_genome_to_urdf(robot, cad_models=cad_models), encoding="utf-8")
    return path


def _link_to_urdf_lines(link_name: str, layout: LinkLayout | None, mesh_prefix: str) -> list[str]:
    mesh_name = _mesh_name_for_link(link_name)
    mass = round(layout.mass_kg, 4) if layout else 0.1
    length = layout.length_m if layout else 0.1
    half = layout.half_xy_m if layout else 0.025
    # Thin-box inertia about the link centroid (length along local z).
    ixx = round(mass * (half * half + length * length / 3.0) / 3.0, 6)
    izz = round(mass * (half * half) * 2.0 / 3.0, 6)
    center_z = round(length / 2.0, 4)
    return [
        f'  <link name="{escape(link_name)}">',
        "    <inertial>",
        f'      <origin xyz="0 0 {center_z}" rpy="0 0 0" />',
        f'      <mass value="{mass}" />',
        f'      <inertia ixx="{ixx}" ixy="0" ixz="0" iyy="{ixx}" iyz="0" izz="{izz}" />',
        "    </inertial>",
        "    <visual>",
        '      <origin xyz="0 0 0" rpy="0 0 0" />',
        f'      <geometry><mesh filename="{mesh_prefix}/{mesh_name}" /></geometry>',
        "    </visual>",
        "    <collision>",
        '      <origin xyz="0 0 0" rpy="0 0 0" />',
        f'      <geometry><mesh filename="{mesh_prefix}/{mesh_name}" /></geometry>',
        "    </collision>",
        "  </link>",
    ]


def _joint_to_urdf_lines(joint: RobotJoint, layout: LinkLayout | None) -> list[str]:
    axis = joint.axis_xyz or (0.0, 0.0, 1.0)
    origin = layout.origin_xyz if layout else (0.0, 0.0, 0.1)
    limit = joint.limit
    lower = limit.lower if limit and limit.lower is not None else -3.14
    upper = limit.upper if limit and limit.upper is not None else 3.14
    velocity = limit.velocity if limit and limit.velocity is not None else 1.0
    effort = limit.effort if limit and limit.effort is not None else 1.0
    return [
        f'  <joint name="{escape(joint.name)}" type="{escape(joint.joint_type)}">',
        f'    <parent link="{escape(joint.parent_link)}" />',
        f'    <child link="{escape(joint.child_link)}" />',
        f'    <origin xyz="{origin[0]} {origin[1]} {origin[2]}" rpy="0 0 0" />',
        f'    <axis xyz="{axis[0]} {axis[1]} {axis[2]}" />',
        f'    <limit lower="{lower}" upper="{upper}" effort="{effort}" velocity="{velocity}" />',
        "  </joint>",
    ]


def _mesh_name_for_link(link_name: str) -> str:
    mapping = {
        "base_link": "cad_arm_base.stl",
        "upper_link": "cad_upper_link.stl",
        "forearm_link": "cad_forearm_link.stl",
        "wrist_link": "cad_wrist_camera_mount.stl",
        "gripper_link": "cad_gripper_mount.stl",
    }
    return mapping.get(link_name, "cad_arm_base.stl")
