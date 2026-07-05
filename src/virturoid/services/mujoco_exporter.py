from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

from virturoid.schemas.cad import CadModel
from virturoid.schemas.robot import RobotGenome, RobotJoint, RobotSensor
from virturoid.schemas.scenes import SceneGraph, SceneObject, SceneSet
from virturoid.services.robot_kinematics import TABLE_TOP_Z, LinkLayout, compute_arm_layout

# Geom half-sizes (m) for manipulable objects and target containers.
_CUBE_HALF = 0.025
_CONTAINER_HALF = (0.06, 0.06, 0.03)
_STATIC_TYPES = {"container", "conveyor", "surface"}


def robot_and_scene_to_mjcf(
    robot: RobotGenome,
    scene_set: SceneSet,
    cad_models: list[CadModel] | None = None,
) -> str:
    """Generate an MJCF for the MVP robot and the first scene of the set."""
    scene = scene_set.scenes[0] if scene_set.scenes else None
    return robot_and_single_scene_to_mjcf(
        robot, scene, scene_set.id if scene_set else "scene_set", cad_models
    )


def robot_and_single_scene_to_mjcf(
    robot: RobotGenome,
    scene: SceneGraph | None,
    scene_set_id: str,
    cad_models: list[CadModel] | None = None,
) -> str:
    model_name = f"{robot.name} {scene.id}" if scene else robot.name
    layout = compute_arm_layout(robot, cad_models)
    lines = [
        f'<mujoco model="{escape(model_name)}">',
        '  <compiler angle="radian" coordinate="local" />',
        '  <option timestep="0.002" gravity="0 0 -9.81" />',
        f"  <!-- scene_set_id={escape(scene_set_id)} scene_id={escape(scene.id if scene else 'none')} -->",
        "  <default>",
        '    <joint damping="0.8" armature="0.01" />',
        '    <geom friction="1 0.1 0.01" />',
        "  </default>",
        "  <asset>",
        '    <material name="mat_red" rgba="0.8 0.1 0.1 1" />',
        '    <material name="mat_blue" rgba="0.1 0.2 0.8 1" />',
        '    <material name="mat_gray" rgba="0.5 0.5 0.5 1" />',
        '    <material name="mat_link" rgba="0.35 0.35 0.38 1" />',
        "  </asset>",
        "  <worldbody>",
        '    <light name="key" pos="0 0 2" dir="0 0 -1" />',
        f'    <geom name="table" type="box" size="0.7 0.45 {TABLE_TOP_Z}" pos="0.4 0 0" rgba="0.7 0.7 0.65 1" />',
        _body_xml(layout, robot.sensors, indent=4),
    ]
    if scene:
        lines.extend(_scene_objects_xml(scene.objects))
    lines.extend(
        [
            "  </worldbody>",
            "  <actuator>",
            *[_joint_motor_xml(joint) for joint in robot.joints],
            *(_gripper_motor_xml() if robot.end_effectors else []),
            "  </actuator>",
            *_sensor_output_xml(robot),
            "</mujoco>",
        ]
    )
    return "\n".join(lines) + "\n"


def write_mujoco_xml(
    robot: RobotGenome,
    scene_set: SceneSet,
    output_dir: Path,
    cad_models: list[CadModel] | None = None,
) -> Path:
    path = output_dir / "simulation" / "mujoco" / "mvp_scene.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(robot_and_scene_to_mjcf(robot, scene_set, cad_models), encoding="utf-8")
    return path


def write_mujoco_scene_collection(
    robot: RobotGenome,
    scene_sets: list[SceneSet],
    output_dir: Path,
    cad_models: list[CadModel] | None = None,
) -> Path:
    scene_dir = output_dir / "simulation" / "mujoco" / "scenes"
    scene_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for scene_set in scene_sets:
        for scene in scene_set.scenes:
            relative_uri = f"simulation/mujoco/scenes/{scene_set.purpose}/{scene.id}.xml"
            scene_path = output_dir / relative_uri
            scene_path.parent.mkdir(parents=True, exist_ok=True)
            scene_path.write_text(
                robot_and_single_scene_to_mjcf(robot, scene, scene_set.id, cad_models),
                encoding="utf-8",
            )
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
            "Each generated training scene has a concrete MJCF XML file.",
            "Bodies form a serial kinematic chain; manipulable objects are dynamic free bodies.",
        ],
    }
    index_path = output_dir / "simulation" / "mujoco" / "compiled_scene_index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index_path


def _body_xml(layout: LinkLayout, sensors: list[RobotSensor], indent: int) -> str:
    """Recursively emit the nested kinematic chain as MJCF bodies."""
    pad = " " * indent
    x, y, z = layout.origin_xyz
    length = layout.length_m
    half = layout.half_xy_m
    lines = [f'{pad}<body name="{escape(layout.name)}" pos="{x} {y} {z}">']

    if layout.joint is not None:
        lines.append(f"{pad}  {_joint_xml(layout.joint)}")

    # Box geom extending from the joint origin along local +z, with CAD mass.
    lines.append(
        f'{pad}  <geom name="{escape(layout.name)}_geom" type="box" '
        f'size="{half} {half} {round(length / 2, 4)}" pos="0 0 {round(length / 2, 4)}" '
        f'mass="{round(layout.mass_kg, 4)}" material="mat_link" />'
    )

    # End-effector reference site at the distal tip (used by the runner/controller).
    if layout.is_end_effector:
        lines.append(
            f'{pad}  <site name="ee_site" pos="0 0 {round(length, 4)}" size="0.012" rgba="0.1 0.9 0.5 1" />'
        )
        lines.append(
            f'{pad}  <site name="grasp_site" pos="0 0 {round(length - 0.055, 4)}" size="0.01" rgba="1 0.7 0.1 1" />'
        )
        lines.extend(f"{pad}  {row}" for row in _parallel_jaw_gripper_xml(length))

    # Mount any sensors declared on this link so they move with the chain.
    for sensor in sensors:
        if sensor.parent_link == layout.name:
            lines.extend(f"{pad}  {row}" for row in _sensor_mount_xml(sensor))

    for child in layout.children:
        lines.append(_body_xml(child, sensors, indent + 2))

    lines.append(f"{pad}</body>")
    return "\n".join(lines)


def _joint_xml(joint: RobotJoint) -> str:
    axis = joint.axis_xyz or (0.0, 0.0, 1.0)
    limit = joint.limit
    if limit and limit.lower is not None and limit.upper is not None:
        range_attr = f' limited="true" range="{limit.lower} {limit.upper}"'
    else:
        range_attr = ' limited="false"'
    return (
        f'<joint name="{escape(joint.name)}" type="hinge" '
        f'axis="{axis[0]} {axis[1]} {axis[2]}" pos="0 0 0"{range_attr} />'
    )


def _joint_motor_xml(joint: RobotJoint) -> str:
    effort = joint.limit.effort if joint.limit and joint.limit.effort else None
    force_attr = f' forcerange="{-effort} {effort}"' if effort else ""
    return f'    <motor name="{escape(joint.name)}_motor" joint="{escape(joint.name)}" gear="1"{force_attr} />'


def _gripper_motor_xml() -> list[str]:
    return [
        '    <motor name="left_gripper_slide_motor" joint="left_gripper_slide" gear="1" forcerange="-25 25" />',
        '    <motor name="right_gripper_slide_motor" joint="right_gripper_slide" gear="1" forcerange="-25 25" />',
    ]


def _parallel_jaw_gripper_xml(parent_length: float) -> list[str]:
    z = round(parent_length, 4)
    return [
        f'<body name="left_gripper_finger" pos="0 0.055 {z}">',
        '  <joint name="left_gripper_slide" type="slide" axis="0 -1 0" limited="true" range="0 0.04" damping="2.0" />',
        '  <geom name="left_gripper_pad" type="box" size="0.012 0.008 0.055" pos="0 0 -0.045" mass="0.025" rgba="0.12 0.12 0.14 1" friction="1.5 0.1 0.01" />',
        '</body>',
        f'<body name="right_gripper_finger" pos="0 -0.055 {z}">',
        '  <joint name="right_gripper_slide" type="slide" axis="0 1 0" limited="true" range="0 0.04" damping="2.0" />',
        '  <geom name="right_gripper_pad" type="box" size="0.012 0.008 0.055" pos="0 0 -0.045" mass="0.025" rgba="0.12 0.12 0.14 1" friction="1.5 0.1 0.01" />',
        '</body>',
    ]


_RGBA = {"red": "0.8 0.15 0.15 1", "blue": "0.15 0.25 0.8 1", "green": "0.2 0.7 0.3 1",
         "orange": "0.9 0.55 0.15 1", "yellow": "0.85 0.75 0.2 1", "gray": "0.5 0.5 0.55 1"}


def _rgba_for(item: SceneObject, default: str = "gray") -> str:
    m = (item.material or "").lower()
    for key in ("red", "blue", "green", "orange", "yellow"):
        if key in m:
            return _RGBA[key]
    return _RGBA[default]


def _scene_objects_xml(objects: list[SceneObject]) -> list[str]:
    # A scene that brings its own ground plane or walls is a FLOOR scene (navigation / maze): targets and
    # obstacles sit on the floor at full size, not pinned to the tabletop. Tabletop scenes are unaffected.
    floor = any(getattr(o, "object_type", None) in ("floor", "wall") for o in objects)
    return [_scene_object_xml(item, floor=floor) for item in objects]


def _half_xyz(item: SceneObject, fx: float, fy: float, fz: float) -> tuple[float, float, float]:
    """Half-extents (hx, hy, hz) for a box geom. When ``item.size_xyz`` (full extents along local x/y/z) is set it
    OVERRIDES the legacy scale-derived fallback ``(fx, fy, fz)`` — this is the single point that lets a scene carry
    real dimensions. size_xyz absent -> returns the fallback unchanged, so existing scenes are byte-identical."""
    if item.size_xyz is not None:
        ex, ey, ez = item.size_xyz
        return round(ex / 2.0, 4), round(ey / 2.0, 4), round(ez / 2.0, 4)
    return fx, fy, fz


def _scene_object_xml(item: SceneObject, floor: bool = False) -> str:
    x, y, _, _, _, yaw = item.pose_xyz_rpy
    name = escape(item.name)
    s = item.scale or 1.0

    # ---- floor-scene structure: ground arena + maze/boundary walls ----
    if item.object_type == "floor":
        _fx = round(max(0.2, s), 3)                        # scale = ground X half-extent in metres
        hx, hy, hz = _half_xyz(item, _fx, round(_fx * (item.friction or 0.8), 3), 0.02)
        # friction carries the Y/X aspect so the legacy floor matches the course shape (size_xyz overrides it)
        return (f'    <geom name="{name}" type="box" size="{hx} {hy} {hz}" pos="{x} {y} {round(-hz, 4)}" '
                f'rgba="0.82 0.82 0.85 1"/>')
    if item.object_type == "wall":
        hx, hy, hz = _half_xyz(item, round(0.5 * s, 4), 0.03, 0.16)   # size_xyz = (length, thickness, HEIGHT)
        return (f'    <geom name="{name}" type="box" size="{hx} {hy} {hz}" pos="{x} {y} {round(hz, 4)}" '
                f'euler="0 0 {round(yaw, 4)}" rgba="{_rgba_for(item, "gray")}"/>')

    # ---- tabletop open-top bin (sorting / box handling) ----
    if item.object_type == "container":
        material = "mat_red" if (item.material and "red" in item.material) else (
            "mat_blue" if (item.material and "blue" in item.material) else "mat_gray")
        return _open_bin_xml(item.name, x, y, material)

    # ---- elevated shelf / platform to lift a box onto (a low steel pedestal on the tabletop) ----
    if item.object_type == "platform":
        hx, hy, h = _half_xyz(item, round(0.13 * s, 4), round(0.11 * s, 4), round(0.05 * s, 4))
        return (f'    <geom name="{name}" type="box" size="{hx} {hy} {h}" '
                f'pos="{x} {y} {round(TABLE_TOP_Z + h, 4)}" rgba="0.30 0.42 0.55 1"/>')

    # ---- target pad: a flat coloured marker (go here / place here / stack here) ----
    if item.object_type == "zone":
        _fh = round((0.16 if floor else 0.06) * s, 4)
        hx, hy, hz = _half_xyz(item, _fh, _fh, 0.006)
        base_z = 0.0 if floor else TABLE_TOP_Z
        return (f'    <geom name="{name}" type="box" size="{hx} {hy} {hz}" '
                f'pos="{x} {y} {round(base_z + hz, 4)}" rgba="{_rgba_for(item, "green")}"/>')

    # ---- floor obstacle / pillar to steer around (navigation / maze) ----
    if floor and item.object_type == "obstacle":
        h = round(0.12 * s, 4)
        # footprint stays sized to the robot (course stays navigable), but keep the LEGACY obstacle SHORTER than the
        # mobile base's ~0.18 m roofline. A full cube here towered ~0.7 m and hid the low rover behind it; a
        # low crate steers the same and leaves the robot the tallest, clearly-visible thing in the scene.
        hx, hy, hz = _half_xyz(item, h, h, round(min(h, 0.07), 4))
        return (f'    <geom name="{name}" type="box" size="{hx} {hy} {hz}" pos="{x} {y} {round(hz, 4)}" '
                f'euler="0 0 {round(yaw, 4)}" rgba="0.5 0.5 0.55 1"/>')

    # ---- tabletop static furniture (conveyor / surface) ----
    if item.object_type in _STATIC_TYPES:
        material = "mat_gray"
        if item.material and "red" in item.material:
            material = "mat_red"
        if item.material and "blue" in item.material:
            material = "mat_blue"
        hx, hy, hz = _CONTAINER_HALF
        z = TABLE_TOP_Z + hz
        return (
            f'    <geom name="{name}" type="box" size="{hx} {hy} {hz}" '
            f'pos="{x} {y} {round(z, 4)}" material="{material}" />'
        )

    # ---- manipulable object: a dynamic free body resting on the table ----
    # red/blue/gray keep their shared materials (so sorting scenes are byte-identical); cardboard boxes render
    # tan and green/orange objects show their real colour via rgba.
    mm = (item.material or "").lower()
    if "cardboard" in mm or "wood" in mm:
        color_attr = ' rgba="0.72 0.55 0.35 1"'
    elif "green" in mm:
        color_attr = f' rgba="{_RGBA["green"]}"'
    elif "orange" in mm or "yellow" in mm:
        color_attr = f' rgba="{_RGBA["orange"]}"'
    else:
        material = "mat_gray"
        if "red" in mm:
            material = "mat_red"
        if "blue" in mm:
            material = "mat_blue"
        color_attr = f' material="{material}"'
    _h = round(_CUBE_HALF * s, 4)
    hx, hy, hz = _half_xyz(item, _h, _h, _h)             # size_xyz -> a real box (e.g. a YCB cracker box), else a cube
    z = round(TABLE_TOP_Z + hz + 0.001, 4)               # rest on the table at its OWN half-height
    mass = item.mass_kg if item.mass_kg else 0.05
    friction_attr = f' friction="{item.friction} 0.1 0.01"' if item.friction else ""
    return (
        f'    <body name="obj_{name}" pos="{x} {y} {z}" euler="0 0 {round(yaw, 4)}">\n'
        f'      <freejoint name="free_{name}" />\n'
        f'      <geom name="{name}" type="box" size="{hx} {hy} {hz}" '
        f'mass="{round(mass, 4)}"{friction_attr}{color_attr} />\n'
        "    </body>"
    )


def _open_bin_xml(name: str, x: float, y: float, material: str) -> str:
    """An open-top bin (floor + four low walls) so placed blocks rest inside it."""
    inner = 0.07  # inner half-width
    wall = 0.006
    h = 0.04  # wall half-height
    floor_z = round(TABLE_TOP_Z + 0.004, 4)
    wall_z = round(TABLE_TOP_Z + h, 4)
    n = escape(name)
    outer = inner + wall
    return "\n".join(
        [
            f'    <geom name="{n}" type="box" size="{inner} {inner} 0.004" pos="{x} {y} {floor_z}" material="{material}" />',
            f'    <geom name="{n}_wall_px" type="box" size="{wall} {outer} {h}" pos="{round(x + inner, 4)} {y} {wall_z}" material="{material}" />',
            f'    <geom name="{n}_wall_nx" type="box" size="{wall} {outer} {h}" pos="{round(x - inner, 4)} {y} {wall_z}" material="{material}" />',
            f'    <geom name="{n}_wall_py" type="box" size="{outer} {wall} {h}" pos="{x} {round(y + inner, 4)} {wall_z}" material="{material}" />',
            f'    <geom name="{n}_wall_ny" type="box" size="{outer} {wall} {h}" pos="{x} {round(y - inner, 4)} {wall_z}" material="{material}" />',
        ]
    )


def _sensor_mount_xml(sensor: RobotSensor) -> list[str]:
    sensor_type = _sensor_type(sensor.name, sensor.sensor_component_id)
    x, y, z, roll, pitch, yaw = sensor.transform_xyz_rpy
    if sensor_type in {"rgbd_camera", "rgb_camera"}:
        return [
            f'<camera name="{escape(sensor.name)}" pos="{x} {y} {z}" xyaxes="0 -1 0 0 0 1" '
            f'mode="fixed" user="{roll} {pitch} {yaw}" />'
        ]
    if sensor_type == "lidar":
        return [
            f'<site name="{escape(sensor.name)}_ray_origin" pos="{x} {y} {z}" size="0.01" '
            'rgba="0.1 0.8 0.7 1" />'
        ]
    return []


def _sensor_output_xml(robot: RobotGenome) -> list[str]:
    sensor_lines = []
    for sensor in robot.sensors:
        sensor_type = _sensor_type(sensor.name, sensor.sensor_component_id)
        if sensor_type == "rgbd_camera":
            sensor_lines.extend(
                [
                    f'    <framepos name="{escape(sensor.name)}_pose" objtype="camera" objname="{escape(sensor.name)}" />',
                    f'    <framequat name="{escape(sensor.name)}_orientation" objtype="camera" objname="{escape(sensor.name)}" />',
                ]
            )
        elif sensor_type == "rgb_camera":
            sensor_lines.append(
                f'    <framepos name="{escape(sensor.name)}_pose" objtype="camera" objname="{escape(sensor.name)}" />'
            )
        elif sensor_type == "lidar":
            sensor_lines.append(
                f'    <rangefinder name="{escape(sensor.name)}_range" site="{escape(sensor.name)}_ray_origin" />'
            )
    if not sensor_lines:
        return []
    return [
        "  <sensor>",
        *sensor_lines,
        "  </sensor>",
    ]


def _sensor_type(sensor_name: str, component_id: str) -> str:
    key = f"{sensor_name} {component_id}".lower()
    if "lidar" in key:
        return "lidar"
    if "rgbd" in key or "depth" in key:
        return "rgbd_camera"
    if "rgb" in key or "camera" in key:
        return "rgb_camera"
    return "unknown"
