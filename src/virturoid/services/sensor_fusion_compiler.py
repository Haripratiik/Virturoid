"""The sensor-fusion compiler (agentic platform plan WS-S / S1): the deployable state-estimation stack, written
from the robot's OWN bill of materials — no human hand-authors an EKF config.

A customer's robot carries an IMU, wheel/joint encoders, a camera, maybe a LiDAR. Making those sensors agree on
"where am I and which way am I facing" is the sensor-fusion problem, and today a roboticist hand-writes a
``robot_localization`` YAML — the infamous 15-boolean state matrix — plus an IMU orientation filter and an
odometry source, per robot, by hand. This module DERIVES that stack from the BOM the platform already built:

  * it reads the SAME sensor suite the BOM/render use (``bom_builder._sensor_suite``) — so the fusion config
    references EXACTLY the sensors the robot has, never a fabricated one;
  * it places each sensor on the correct parent LINK using the same host-resolution as the render
    (``sensor_geometry._host``), so every TF frame points at a body part that exists;
  * it picks the estimator by what the robot physically IS (``task_matched_eval.robot_kind``): a wheeled base
    gets 2-D wheel-odometry + IMU heading; a legged body gets contact-aided leg-odometry + full-3-D AHRS; a
    fixed-base arm gets NO base EKF (it doesn't localize) — and says so, honestly;
  * it builds the 15-state ``odomN_config`` / ``imuN_config`` boolean matrices from what each sensor actually
    OBSERVES (an AHRS contributes absolute roll/pitch; a raw 6-DOF IMU gets a Madgwick filter first), not a
    copied template;
  * it reports what is MISSING and the consequence (no odometry source → position drifts on IMU alone), because
    a fusion config that hides an unobservable state is a lie the robot pays for at deploy time.

Pure CPU, no ROS import, deterministic, LLM-free — it's a compiler, not a guess. The output is a set of files
(``config/ekf.yaml``, ``config/imu_filter.yaml``, ``config/sensors.yaml``, ``launch/sensor_fusion.launch.py``)
plus an honest manifest, dropped into an exported package by ``write_sensor_fusion``.
"""

from __future__ import annotations

import json
from pathlib import Path

# The robot_localization state vector, in its canonical order. Every EKF sensor config is a 15-boolean mask
# over THIS order — fusing a state means "trust this sensor for this quantity".
_STATE_ORDER = ("x", "y", "z", "roll", "pitch", "yaw",
                "vx", "vy", "vz", "vroll", "vpitch", "vyaw",
                "ax", "ay", "az")


def _frame_id(seg_name: str) -> str:
    """A URDF/TF-legal frame id from a segment name (lowercase, non-alnum -> underscore)."""
    return "".join(c if c.isalnum() else "_" for c in (seg_name or "base").lower()).strip("_") or "base_link"


def _mask(states) -> list[bool]:
    """A 15-bool robot_localization config from the set of state names it should fuse."""
    want = set(states)
    return [s in want for s in _STATE_ORDER]


def _imu_is_ahrs(comp) -> bool:
    """True when the IMU outputs an ABSOLUTE orientation itself (a 9-DOF / onboard-fusion AHRS like the BNO055
    or VN-100), so we can fuse roll/pitch directly. A raw 6-DOF accel+gyro IMU cannot, and needs a Madgwick
    filter upstream to produce an orientation quaternion first."""
    specs = getattr(comp, "specs", None) or {}
    if specs.get("onboard_fusion") or int(specs.get("dof", 0) or 0) >= 9:
        return True
    headline = (getattr(comp, "spec", "") or "").lower()
    return "ahrs" in headline or "fusion" in headline


def _mounting_offset(host, cat: str, mount: str) -> tuple[float, float, float]:
    """A grounded (x-forward, y-left, z-up REP-103) translation of the sensor frame relative to its parent link,
    from the host segment's real dimensions — cameras/F-T on the forward face, IMU/LiDAR on top. Not fabricated:
    the same forward/up placement the render uses, expressed in the deploy base frame."""
    L = float(getattr(host, "length_m", 0.2) or 0.2)
    r = max(0.02, float(getattr(host, "radius_m", 0.05) or 0.05))
    m = (mount or "").lower()
    if cat in ("camera", "thermal", "force_torque") or "wrist" in m or "eye-in-hand" in m or "forward" in m or "front" in m:
        return (round(L / 2.0, 4), 0.0, round(r, 4))     # forward face, a little above centerline
    if cat in ("lidar",) or "mast" in m or "top" in m:
        return (0.0, 0.0, round(r + 0.05, 4))            # on a short mast on top
    return (0.0, 0.0, round(r, 4))                        # imu/gps: on top of the body, near the CoM


_TOPIC = {
    "camera": ("sensor_msgs/msg/Image", "/camera/color/image_raw"),
    "lidar": ("sensor_msgs/msg/PointCloud2", "/scan/points"),
    "imu": ("sensor_msgs/msg/Imu", "/imu/data_raw"),
    "force_torque": ("geometry_msgs/msg/WrenchStamped", "/ft_sensor/wrench"),
    "gps": ("sensor_msgs/msg/NavSatFix", "/gps/fix"),
    "thermal": ("sensor_msgs/msg/Image", "/thermal/image_raw"),
    "microphone": ("audio_common_msgs/msg/AudioData", "/audio"),
}


def compile_sensor_fusion(gene, *, task: str = "") -> dict:
    """Derive the whole state-estimation stack for ``gene`` from its BOM sensors. Returns a manifest dict:
    ``kind``, the resolved ``sensors`` (each with its parent frame + what it contributes), the generated
    ``files`` (config YAMLs + launch), the ``fused_states`` actually observed, and an honest ``missing`` list."""
    from virturoid.services.bom_builder import _sensor_suite
    from virturoid.services.component_catalog import component
    from virturoid.services.sensor_geometry import _host
    from virturoid.services.task_matched_eval import robot_kind

    kind = robot_kind(gene)
    scale_kg = sum(float(getattr(s, "mass_kg", 0.0) or 0.0) for s in gene.segments)
    caps = sorted(getattr(gene, "capabilities", []) or [])
    suite = _sensor_suite(gene.robot_class, caps, task, scale_kg)

    sensors: list[dict] = []
    imus: list[dict] = []
    has_camera = has_lidar = has_ft = False
    for name, qty, mount in suite:
        comp = component(name)
        cat = comp.category if comp else "imu"
        host = _host(cat, mount, gene)
        parent = _frame_id(getattr(host, "name", "base") if host is not None else "base")
        child = {"camera": "camera_link", "lidar": "lidar_link", "imu": "imu_link",
                 "force_torque": "ft_sensor_link", "gps": "gps_link", "thermal": "thermal_link",
                 "microphone": "mic_link"}.get(cat, f"{cat}_link")
        msg_type, topic = _TOPIC.get(cat, ("sensor_msgs/msg/Imu", "/sensor"))
        entry = {"part": name, "vendor": getattr(comp, "vendor", "") if comp else "", "category": cat,
                 "qty": int(qty), "mount": mount, "parent_frame": parent, "child_frame": child,
                 "topic": topic, "msg_type": msg_type, "offset_xyz": _mounting_offset(host, cat, mount)}
        if cat == "imu":
            entry["ahrs"] = _imu_is_ahrs(comp)
            imus.append(entry)
        elif cat == "camera":
            has_camera = True
        elif cat == "lidar":
            has_lidar = True
        elif cat == "force_torque":
            has_ft = True
        sensors.append(entry)

    primary_imu = imus[0] if imus else None
    needs_madgwick = bool(primary_imu) and not primary_imu.get("ahrs", False)

    files: dict[str, str] = {}
    fused: dict[str, list[str]] = {}          # state_name -> [sensor sources fusing it]
    missing: list[str] = []
    notes: list[str] = []

    # --- the estimator, chosen by what the robot physically IS ---------------------------------------------
    two_d = kind in ("mobile", "aquatic")     # a ground/water base estimates a planar pose; a legged/aerial body is full-3D
    base_frame = "base_link"

    if kind == "manipulator":
        # A fixed-base arm does NOT localize a moving base — an EKF here would estimate a pose that never changes.
        # The honest stack is joint-state (encoders) + optional wrist wrench; say so instead of shipping a dead EKF.
        notes.append("Fixed-base manipulator: no base-pose EKF (the base does not move). State estimation is "
                     "joint_states from the encoders; a wrist F/T sensor is republished as a WrenchStamped.")
        if not has_ft:
            missing.append("No wrist force/torque sensor in the BOM: contact/force feedback is unavailable for "
                           "compliant or force-limited manipulation.")
        if has_camera:
            notes.append("Eye-in-hand camera present: its TF is published on the end-effector link for hand-eye "
                         "use (calibrate the exact transform on hardware).")
        # no ekf.yaml for a fixed base — that's the honest choice
    else:
        # --- odometry source (the velocity reference the IMU alone cannot provide) ---
        odom_states: list[str] = []
        odom_topic = None
        if kind == "mobile":
            odom_topic = "/wheel/odometry"
            blob = (f"{gene.robot_class or ''} {' '.join(caps)} {task}").lower()
            holonomic = any(w in blob for w in ("mecanum", "omni", "holonomic", "swerve"))
            odom_states = ["vx", "vyaw"] + (["vy"] if holonomic else [])
            notes.append("Wheeled base: odometry from the diff/mecanum-drive controller (wheel encoders) supplies "
                         "body velocity; the IMU supplies heading rate and forward acceleration.")
        elif kind == "legged":
            odom_topic = "/leg_odometry/odom"
            odom_states = ["vx", "vy", "vz", "vyaw"]
            notes.append("Legged body: contact-aided leg odometry (foot-contact kinematics, InEKF-style) supplies "
                         "the body twist; the AHRS supplies absolute roll/pitch and the angular rates. This "
                         "contact-odometry estimator is the piece hand-built per-robot elsewhere.")
        else:  # aerial / aquatic / other moving base with no wheel or leg odometry
            missing.append(f"{kind}: no wheel or leg odometry source, so body velocity is unobservable from the "
                           "IMU alone — position WILL drift. Add GPS, visual-inertial odometry, or a flow sensor.")
            notes.append(f"{kind} base: IMU-only attitude estimate; translation left to a downstream VIO/GPS node.")

        for st in odom_states:
            fused.setdefault(st, []).append("wheel/leg odometry")

        # --- IMU config: what the IMU is trusted for ---
        imu_states: list[str] = []
        if primary_imu:
            imu_topic = "/imu/data" if not needs_madgwick else "/imu/data"   # madgwick republishes onto /imu/data
            if two_d:
                imu_states = ["yaw", "vyaw", "ax"]
                notes.append("2-D mode: only planar heading is estimated; z/roll/pitch are held at zero.")
            else:
                # full 3D: absolute roll+pitch from the AHRS (reliable), angular rates, linear accel.
                # absolute yaw is left UNFUSED (a cheap IMU's yaw drifts without a clean magnetometer); yaw is
                # carried by the odometry's vyaw instead — the standard robot_localization recommendation.
                imu_states = ["roll", "pitch", "vroll", "vpitch", "vyaw", "ax", "ay", "az"]
                notes.append("Absolute yaw is intentionally NOT fused from the IMU (magnetometer drift indoors); "
                             "heading rate comes from the IMU gyro and odometry, integrated by the filter.")
            for st in imu_states:
                fused.setdefault(st, []).append(primary_imu["part"])
        else:
            missing.append("No IMU in the BOM: orientation and angular rate are unobservable — the base pose "
                           "cannot be stabilized. Add an IMU/AHRS.")

        files["config/ekf.yaml"] = _ekf_yaml(
            base_frame=base_frame, two_d=two_d,
            odom_topic=odom_topic, odom_mask=_mask(odom_states) if odom_states else None,
            imu_topic=("/imu/data" if primary_imu else None),
            imu_mask=_mask(imu_states) if imu_states else None)

        if needs_madgwick:
            files["config/imu_filter.yaml"] = _madgwick_yaml(
                has_mag=int((component(primary_imu["part"]).specs or {}).get("dof", 6)) >= 9)
            notes.append(f"{primary_imu['part']} is a raw 6-DOF IMU (no onboard fusion): a Madgwick filter "
                         "estimates orientation from accel+gyro and republishes onto /imu/data before the EKF.")

    files["config/sensors.yaml"] = _sensors_yaml(sensors)
    files["launch/sensor_fusion.launch.py"] = _launch_py(
        kind=kind, sensors=sensors, has_ekf=("config/ekf.yaml" in files), needs_madgwick=needs_madgwick)

    manifest = {
        "kind": kind, "two_d_mode": two_d if kind not in ("manipulator",) else None,
        "sensors": sensors,
        "fused_states": {st: srcs for st, srcs in fused.items()},
        "unobserved_states": [s for s in _STATE_ORDER if s not in fused and kind != "manipulator"],
        "missing": missing, "notes": notes,
        "files": sorted(files.keys()),
        "authored_by": "sensor_fusion_compiler (derived from BOM, no hand-authoring)",
    }
    files["config/fusion_manifest.json"] = json.dumps(manifest, indent=2)
    return {**manifest, "_files_content": files}


def _ekf_yaml(*, base_frame, two_d, odom_topic, odom_mask, imu_topic, imu_mask) -> str:
    def _fmt_mask(mask):
        rows = [mask[0:3], mask[3:6], mask[6:9], mask[9:12], mask[12:15]]
        return "[" + ",\n                   ".join(
            ", ".join("true " if b else "false" for b in row) for row in rows) + "]"
    L = ["# robot_localization EKF -- AUTO-GENERATED from the robot's BOM sensors. Do not hand-edit; regenerate.",
         "# State order: [x y z  roll pitch yaw  vx vy vz  vroll vpitch vyaw  ax ay az]",
         "ekf_filter_node:", "  ros__parameters:", "    frequency: 30.0",
         f"    two_d_mode: {'true' if two_d else 'false'}", "    publish_tf: true",
         "    map_frame: map", "    odom_frame: odom", f"    base_link_frame: {base_frame}",
         "    world_frame: odom"]
    if odom_topic and odom_mask:
        L += [f"    odom0: {odom_topic}", f"    odom0_config: {_fmt_mask(odom_mask)}",
              "    odom0_differential: false", "    odom0_queue_size: 10"]
    if imu_topic and imu_mask:
        L += [f"    imu0: {imu_topic}", f"    imu0_config: {_fmt_mask(imu_mask)}",
              "    imu0_differential: false", "    imu0_remove_gravitational_acceleration: true",
              "    imu0_queue_size: 10"]
    return "\n".join(L) + "\n"


def _madgwick_yaml(*, has_mag: bool) -> str:
    return ("# imu_filter_madgwick — orientation from a raw 6-DOF IMU (accel+gyro%s). AUTO-GENERATED.\n"
            "imu_filter_madgwick_node:\n  ros__parameters:\n"
            f"    use_mag: {'true' if has_mag else 'false'}\n"
            "    world_frame: enu\n    publish_tf: false\n    gain: 0.1\n"
            % (" +mag" if has_mag else ""))


def _sensors_yaml(sensors) -> str:
    L = ["# Sensor frames + topics — AUTO-GENERATED from the BOM. Each entry is a real part on a real link.",
         "sensors:"]
    for s in sensors:
        ox, oy, oz = s["offset_xyz"]
        L += [f"  - part: \"{s['part']}\"", f"    vendor: \"{s['vendor']}\"",
              f"    category: {s['category']}", f"    parent_frame: {s['parent_frame']}",
              f"    child_frame: {s['child_frame']}", f"    topic: {s['topic']}",
              f"    msg_type: {s['msg_type']}", f"    xyz: [{ox}, {oy}, {oz}]", "    rpy: [0.0, 0.0, 0.0]"]
    return "\n".join(L) + "\n"


def _launch_py(*, kind, sensors, has_ekf, needs_madgwick) -> str:
    static_tf = []
    for s in sensors:
        ox, oy, oz = s["offset_xyz"]
        static_tf.append(
            "        Node(package='tf2_ros', executable='static_transform_publisher',\n"
            f"             name='tf_{s['child_frame']}',\n"
            f"             arguments=['{ox}','{oy}','{oz}','0','0','0','{s['parent_frame']}','{s['child_frame']}']),")
    nodes = []
    if needs_madgwick:
        nodes.append("        Node(package='imu_filter_madgwick', executable='imu_filter_madgwick_node',\n"
                     "             name='imu_filter', parameters=[cfg('imu_filter.yaml')],\n"
                     "             remappings=[('imu/data_raw','/imu/data_raw'),('imu/data','/imu/data')]),")
    if has_ekf:
        nodes.append("        Node(package='robot_localization', executable='ekf_node',\n"
                     "             name='ekf_filter_node', parameters=[cfg('ekf.yaml')]),")
    return (
        "# AUTO-GENERATED sensor-fusion bringup for a %s robot. Launches exactly the sensors in the BOM.\n"
        "import os\n"
        "from ament_index_python.packages import get_package_share_directory\n"
        "from launch import LaunchDescription\n"
        "from launch_ros.actions import Node\n\n\n"
        "def cfg(name):\n"
        "    return os.path.join(get_package_share_directory('virturoid_robot'), 'config', name)\n\n\n"
        "def generate_launch_description():\n"
        "    return LaunchDescription([\n"
        "%s\n%s\n    ])\n"
        % (kind, "\n".join(static_tf), "\n".join(nodes) if nodes else "        # (fixed base: joint_states only)")
    )


def write_sensor_fusion(gene, output_dir, *, task: str = "") -> Path | None:
    """Compile the fusion stack and write it under ``output_dir/fusion/``. Returns the manifest path (or None on
    a total failure — fusion is value-add and never blocks a package build)."""
    try:
        out = compile_sensor_fusion(gene, task=task)
        base = Path(output_dir) / "fusion"
        for rel, content in out["_files_content"].items():
            p = base / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        return base / "config" / "fusion_manifest.json"
    except Exception:  # noqa: BLE001
        return None
