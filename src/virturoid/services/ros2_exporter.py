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


def _mimic_joints_from_urdf(urdf: str) -> dict:
    """``{driven_joint: {"joint": driver, "multiplier": m, "offset": o}}`` read from the URDF's own ``<mimic>``.

    Read from the shipped description rather than re-derived from the gene, so this package can only ever claim
    a coupling the file it installs actually declares.

    It matters here and not only in RViz: ``joint_trajectory_controller`` commands every joint in its ``joints``
    list, and a mimic joint has no motor of its own — its position is a consequence of the driver's. Listing it
    as commandable asks the hardware interface for an actuator that does not exist and, on a real machine,
    drives two ends of one transmission against each other.
    """
    import xml.etree.ElementTree as ET

    out: dict = {}
    try:
        root = ET.fromstring(urdf)
    except ET.ParseError:
        return out
    for jnt in root.findall("joint"):
        mim = jnt.find("mimic")
        name = jnt.get("name")
        if mim is None or not name or not mim.get("joint"):
            continue
        try:
            mul = float(mim.get("multiplier", "1"))
            off = float(mim.get("offset", "0"))
        except (TypeError, ValueError):
            continue
        out[name] = {"joint": mim.get("joint"), "multiplier": mul, "offset": off}
    return out


def export_ros2_package(package_dir: Path, package_name: str = "virturoid_robot",
                        actuator_map: dict | None = None, actuator_map_source: str | None = None) -> Path:
    """``actuator_map`` (segment OR joint name -> catalog part -- the namespace is DECIDED below by
    ``_map_key_namespace``, never assumed) lets a caller that has the bill of materials IN HAND pass it in. The
    package build writes the BOM to disk AFTER the URDF/ROS2 export, so reading it off disk here found nothing
    on a fresh build; passing it removes the ordering dependency entirely.

    ``actuator_map`` is honoured whenever it is not ``None`` -- an EMPTY dict means "the caller knows there is
    no parts list for this build", and must NOT silently fall back to a file on disk that may belong to a
    previous build of a different robot. ``actuator_map_source`` is the provenance string the emitted YAML
    prints; the header may only claim the parts came from a bill of materials it can NAME."""
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
    mimic: dict = {}
    _urdf_text = ""
    if _src_urdf.exists():
        (root / "urdf").mkdir(parents=True, exist_ok=True)
        _urdf_text = _copy_urdf_meshes(package_dir, root, package_name, _src_urdf.read_text(encoding="utf-8"))
        (root / "urdf" / "robot.urdf").write_text(_urdf_text, encoding="utf-8")
        mimic = _mimic_joints_from_urdf(_urdf_text)
    # A mimic joint is DRIVEN, not commanded: its position follows the joint it mimics through a fixed gear.
    # Everything below that names a motor or a command interface uses the commandable set.
    commandable = [j for j in joints if j not in mimic]

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
                    "has_controller": has_controller, "policy_type": policy_type, "target_positions": targets,
                    # `joints` stays the robot's full joint list (the node publishes a state for each); these two
                    # say which of them a controller may actually command, and by what relation the rest follow.
                    "commandable_joints": commandable, "mimic_joints": mimic},
                   indent=2),
        encoding="utf-8",
    )
    # ros2_control DEPLOY substrate (§4.7): a controller-manager config + a BOM-keyed hardware-interface map (the
    # bridge to the REAL motors you bought) + the safety filter the node applies before commanding a motor.
    if actuator_map is None:
        actuator_map, disk_src = _read_actuator_map(package_dir)
        amap_source = actuator_map_source or disk_src
    else:
        actuator_map = dict(actuator_map)
        amap_source = actuator_map_source or "handed to the exporter by the caller (source not stated)"
    # A parts list keys its map EITHER by the link a joint drives (build_bom) OR by the joint's own name
    # (build_bom_from_genome), and both land at the same path. Decide which from THIS body's own name sets,
    # once for the whole map, then resolve strictly inside that namespace.
    seg_of = _joint_to_segment(genome, _urdf_text)
    _link_names = [l if isinstance(l, str) else (l or {}).get("name", "") for l in genome.get("links", []) or []]
    ns = _map_key_namespace(actuator_map, commandable, seg_of, _link_names)
    gaps = _unresolved_joints(commandable, actuator_map, seg_of, ns["namespace"])
    (root / "config" / "ros2_control.yaml").write_text(_ros2_control_yaml(commandable, mimic), encoding="utf-8")
    (root / "config" / "hardware_interface.yaml").write_text(
        _hardware_interface_yaml(commandable, actuator_map, mimic, seg_of, amap_source, ns), encoding="utf-8")
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
        "`config/ros2_control.yaml` (controller manager) + `config/hardware_interface.yaml` wire the controller\n"
        f"to ros2_control. {len(commandable) - len(gaps)} of {len(commandable)} commandable joints are mapped to\n"
        f"the part assigned to {'the joint itself' if ns['namespace'] == _NS_JOINT else 'the link they drive'}"
        f" by the parts list this export read ({amap_source})"
        + ("" if not gaps else
           f"; {len(gaps)} are NOT (`" + "`, `".join(j for j, _ in gaps[:6])
           + ("`" if len(gaps) <= 6 else "`, ...")
           + "`) — " + ("the key namespace of that parts list could not be determined, so nothing was resolved\n"
                        "from it; see `KEY NAMESPACE: UNDETERMINED` at the top of `config/hardware_interface.yaml`.\n"
                        "That is NOT a claim that it names no actuator for them"
                        if ns["namespace"] == _NS_UNDETERMINED else
                        "the parts list names no actuator for them, so they carry `actuator: null` and are listed\n"
                        "under `unresolved_joints:`. That is a gap in the parts list, not a field for you to fill in"))
        + ". Set `hardware_plugin` to your motor-bus driver\n"
        "(Dynamixel / ODrive / CAN / EtherCAT). `" + package_name + "/safety_filter.py` clamps every command to\n"
        "joint + rate limits before a motor sees it. Validate the closed loop in sim first via\n"
        "`services/sim_ros_bridge` (MuJoCo as virtual hardware behind the same command/state interface).\n")
    _ctrl_md = (
        " — a quadcopter: flight deploys via the onboard autopilot (see below), not this joint node.\n\n" if aerial
        else (f" — runs the exported {'GaitController (trot gait)' if policy_type == 'trot_cpg_gait' else 'ReachController'}.\n\n"
              if has_controller else " (no controller bundle; node publishes a neutral pose).\n\n"))
    _mimic_md = ""
    if mimic:
        _mimic_md = (
            "\n## Coupled joints (URDF `<mimic>`)\n"
            f"{len(mimic)} of this robot's {len(joints)} joints are DRIVEN through a fixed gear by another joint,\n"
            "not commanded. `urdf/robot.urdf` declares each with `<mimic>`, so robot_state_publisher derives its\n"
            "position; `config/ros2_control.yaml` therefore leaves them out of the trajectory controller and\n"
            "`config/hardware_interface.yaml` lists them under `coupled_joints:` with no actuator. Commanding one\n"
            "independently fights the transmission.\n\n"
            + "".join(f"- `{d}` = {m['multiplier']:.10g} x `{m['joint']}`"
                      + (f" + {m['offset']:.10g}" if m.get("offset") else "") + "\n" for d, m in mimic.items()))
    (root / "README.md").write_text(
        f"# {package_name}\n\nGenerated ROS2 package for `{genome.get('id')}`" + _ctrl_md
        + "```\ncolcon build --packages-select " + package_name + "\nros2 launch " + package_name
        + " evaluate.launch.py\n```\n\n" + _deploy_md + _mimic_md,
        encoding="utf-8",
    )
    return root


def maybe_export_ros2_package(package_dir, package_name: str = "virturoid_robot",
                              actuator_map: dict | None = None, actuator_map_source: str | None = None):
    """Export a ROS2 package if the build has a Robot Genome; never raise into the build pipeline.

    Returns the package root path on success, else None (e.g. genome not written yet, or no joints).
    Lets every trained build emit a runnable ROS2 harness (§24/§30) without making the export a hard
    dependency of the build succeeding.
    """
    package_dir = Path(package_dir)
    if not (package_dir / "robot" / "robot_genome.json").exists():
        return None
    try:
        return export_ros2_package(package_dir, package_name, actuator_map=actuator_map,
                                   actuator_map_source=actuator_map_source)
    except Exception:  # noqa: BLE001 - ROS2 export must never break the core build
        return None


_BOM_SEARCH_PATHS = ("robot/bill_of_materials.json", "reports/bill_of_materials.json",
                     "bom.json", "reports/bom.json", "bom/bom.json")


def _read_actuator_map(package_dir: Path) -> tuple:
    """``(the parts list's ``actuator_map`` as written, the provenance string naming where it came from)``.

    WHICH NAMESPACE THAT MAP IS KEYED IN IS NOT KNOWN HERE and this function does not assume one. ``bom_builder``
    has TWO producers: ``build_bom`` keys by SEGMENT, ``build_bom_from_genome`` keys by JOINT NAME, and both land
    at ``robot/bill_of_materials.json``. ``_map_key_namespace`` below decides which, from the body.

    Empty map + a "no parts list found" provenance when no BOM was written with the package -- and the emitted
    YAML prints that provenance, so the header can never claim a bill of materials the code did not read.

    ``bom.json`` at the package root is in the list because that is where ``agent_design_tools.export_held``
    lands the parts list -- the agent-facing door. Reading only the two ``bill_of_materials.json`` paths meant
    the map was EMPTY on every export that came through that door, so every row fell back to the generic
    string before the key lookup below even ran.
    """
    for rel in _BOM_SEARCH_PATHS:
        p = package_dir / rel
        if p.exists():
            try:
                return (json.loads(p.read_text(encoding="utf-8")).get("actuator_map", {}) or {},
                        f"read from {rel} in this package")
            except (json.JSONDecodeError, OSError) as exc:
                return {}, f"{rel} is present in this package but unreadable ({type(exc).__name__})"
    return {}, ("no bill of materials was found in this package (looked for "
                + ", ".join(_BOM_SEARCH_PATHS) + ")")


def _joint_to_segment(genome: dict, urdf_text: str = "") -> dict:
    """``{urdf_joint_name: the link/segment it drives}`` -- the relation, NOT a claim about how the parts list
    is keyed. ``build_bom`` writes its map under these VALUES; ``build_bom_from_genome`` writes its map under
    these KEYS. Which one a given package carries is decided by ``_map_key_namespace``, not assumed here.

    ``build_bom`` keys ``actuator_map`` by SEGMENT (``leg0_0``); the URDF/genome joint that drives that
    segment is a different string (``leg0_0_joint``). Looking the joint name up in that map therefore missed
    on every joint of every composed body: the hit rate was ZERO for whatever joint count the body has --
    measured 0 of 14 on the composed quadruped used here (the count is that body's, not a constant; an earlier
    note said 0/12 from a different body). A total failure that printed as a to-do on every row.
    The relation is already recorded (the genome joint's ``child_link``, the URDF
    joint's ``<child link=...>``), so it is READ rather than reconstructed by string surgery: an imported
    model whose MJCF joint names do not follow ``<segment>_joint`` still resolves.
    """
    import xml.etree.ElementTree as ET

    out: dict = {}
    for j in genome.get("joints", []) or []:
        if not isinstance(j, dict):
            continue
        name = j.get("name")
        child = j.get("child_link") or j.get("child")
        if isinstance(child, dict):
            child = child.get("link")
        if name and child:
            out[str(name)] = str(child)
    if urdf_text:
        try:
            root = ET.fromstring(urdf_text)
        except ET.ParseError:
            root = None
        if root is not None:
            for jnt in root.findall("joint"):
                name, ch = jnt.get("name"), jnt.find("child")
                if name and ch is not None and ch.get("link"):
                    out.setdefault(str(name), str(ch.get("link")))
    return out


_NS_SEGMENT, _NS_JOINT, _NS_UNDETERMINED, _NS_EMPTY = "segment", "joint", "undetermined", "empty"


def _segment_key(joint: str, seg_of: dict) -> str | None:
    """The SEGMENT name this joint drives, or ``None`` when the package records none.

    Recorded first (genome ``child_link`` / URDF ``<child link=...>``). Only when nothing is recorded does it
    fall back to the ``<segment>_joint`` stem -- a naming convention, not a fact, and used only as a segment
    name. It is never used as a joint-namespace key.
    """
    seg = seg_of.get(joint)
    if seg:
        return str(seg)
    return joint[:-6] if joint.endswith("_joint") and len(joint) > 6 else None


def _map_key_namespace(actuator_map: dict, joints: list, seg_of: dict, link_names=()) -> dict:
    """DECIDE, ONCE PER MAP, which namespace a parts list keys ``actuator_map`` in -- from the body, not per joint.

    ``bom_builder`` has two producers and they key DIFFERENTLY: ``build_bom`` by SEGMENT (``leg0_0``),
    ``build_bom_from_genome`` by JOINT NAME (``base_yaw``). Both write ``robot/bill_of_materials.json``, so a
    consumer reading one off disk cannot tell them apart from the file alone. Assuming either one is how the
    strict-segment rule turned three correct manipulator rows into three nulls, and how the fall-through it
    replaced let a known-but-unlisted segment borrow an unrelated part.

    The evidence is the KEY SET against the body's own two name sets. ``n_seg`` = how many keys are segment
    (link) names of this robot, ``n_joint`` = how many are joint names. Majority wins; a tie -- including a map
    whose keys match NEITHER set, e.g. a previous robot's parts list -- is UNDETERMINED and is DISCLOSED rather
    than guessed at. Majority rather than strict containment because a real segment-keyed list can carry a stray
    joint-shaped key (tests/test_ros2_export.py pins exactly that case), and refusing to resolve there would
    throw away rows the body itself decides.

    Returns the record the emitted YAML prints, so the claim is only ever as strong as the evidence.
    """
    keys = {str(k) for k in actuator_map}
    seg_names = ({str(v) for v in seg_of.values() if v}
                 | {s for s in (_segment_key(j, seg_of) for j in joints) if s}
                 | {str(l) for l in link_names if l})
    joint_names = {str(j) for j in joints} | {str(k) for k in seg_of}
    n_seg, n_joint = len(keys & seg_names), len(keys & joint_names)
    if not keys:
        ns = _NS_EMPTY
    elif n_seg > n_joint:
        ns = _NS_SEGMENT
    elif n_joint > n_seg:
        ns = _NS_JOINT
    else:
        ns = _NS_UNDETERMINED
    return {"namespace": ns, "n_keys": len(keys), "n_seg": n_seg, "n_joint": n_joint}


def _resolve_actuator(joint: str, actuator_map: dict, seg_of: dict, namespace: str) -> tuple:
    """``(part_name_or_None, the key it looked for)`` for one joint, STRICTLY WITHIN ``namespace``.

    Exactly ONE key is tried, and which one is decided by ``_map_key_namespace`` for the whole map before any
    joint is resolved. In the SEGMENT namespace that key is the segment the joint drives; in the JOINT
    namespace it is the joint's own name. Nothing falls through to the other namespace in either direction:
      * segment -> joint fall-through was the original defect (a known-but-unlisted segment silently borrowing
        an unrelated part -- 14 honest nulls became 14 confident wrong rows under a 100%-coverage header), and
      * joint -> segment fall-through is its exact mirror (a joint with no entry of its own borrowing the part
        that a SAME-NAMED segment carries, which belongs to a different joint entirely).
    A genuine absence in the namespace the map is actually in is a GAP, and is reported as one.

    UNDETERMINED (and an empty map) resolve NOTHING: the caller discloses that instead of guessing.
    """
    if namespace == _NS_JOINT:
        part = actuator_map.get(joint)
        return (str(part), joint) if part else (None, joint)
    seg = _segment_key(joint, seg_of)
    if namespace == _NS_SEGMENT:
        if seg is None:
            return None, f"{joint} (this package records no segment for it)"
        part = actuator_map.get(seg)
        return (str(part), seg) if part else (None, seg)
    return None, (seg or joint)


def _unresolved_joints(joints: list, actuator_map: dict, seg_of: dict, namespace: str) -> list:
    """``[(joint, the_key_looked_for)]`` for every commandable joint the BOM names no part for.

    ``namespace`` is REQUIRED on purpose: a default would let a future caller silently re-create the
    strict-segment regression by forgetting to determine it.
    """
    out = []
    for j in joints:
        motor, key = _resolve_actuator(j, actuator_map, seg_of, namespace)
        if motor is None:
            out.append((j, key))
    return out


def _ros2_control_yaml(joints: list, mimic: dict | None = None) -> str:
    jl = "\n".join(f"      - {j}" for j in joints) or "      []"
    head = ""
    if mimic:
        head = ("# COUPLED DOF: the joint(s) below are driven through a fixed gear by another joint (URDF\n"
                "# <mimic>) and are deliberately ABSENT from joints:. They have no motor of their own, and\n"
                "# commanding them independently fights the transmission. robot_state_publisher derives their\n"
                "# positions from the driver's.\n"
                + "".join(f"#   {d} = {m['multiplier']:.10g} * {m['joint']}"
                          + (f" + {m['offset']:.10g}" if m.get("offset") else "") + "\n"
                          for d, m in mimic.items()))
    return head + (
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


def _namespace_note(ns: dict) -> str:
    """The comment block that says WHICH namespace the parts list was found to be keyed in, and on what evidence.

    A reader cannot check a lookup they cannot see. The strict-segment regression was invisible to its author
    because nothing printed said which key namespace the export had assumed.
    """
    n, k, s, j = ns["namespace"], ns["n_keys"], ns["n_seg"], ns["n_joint"]
    if n == _NS_EMPTY:
        return ""
    if n == _NS_SEGMENT:
        return (f"# KEY NAMESPACE: SEGMENT. {s} of this parts list's {k} keys are this robot's segment (link)\n"
                f"# names and {j} are joint names, so each row is looked up under the segment its joint drives\n"
                f"# and NOTHING else is tried -- a segment the list does not carry is a gap, never another part.\n")
    if n == _NS_JOINT:
        return (f"# KEY NAMESPACE: JOINT NAME. {j} of this parts list's {k} keys are this robot's joint names\n"
                f"# and {s} are segment (link) names, so each row is looked up under the joint's OWN name and\n"
                f"# NOTHING else is tried -- a joint the list does not carry is a gap, and a joint never takes\n"
                f"# the part a same-named SEGMENT carries, because that part belongs to a different joint.\n")
    return (f"# KEY NAMESPACE: UNDETERMINED -- NOTHING BELOW WAS RESOLVED FROM THIS PARTS LIST. Its {k} key(s)\n"
            f"# match {s} of this robot's segment (link) names and {j} of its joint names; with no majority\n"
            f"# there is no decidable answer to which namespace it is keyed in, and resolving anyway would mean\n"
            f"# guessing. A list whose keys match NEITHER set is usually a DIFFERENT robot's parts list. Every\n"
            f"# joint below is `actuator: null` for that reason -- it is not a claim that the list is empty.\n")


def _hardware_interface_yaml(joints: list, actuator_map: dict, mimic: dict | None = None,
                             seg_of: dict | None = None, actuator_map_source: str | None = None,
                             namespace: dict | None = None) -> str:
    seg_of = seg_of or {}
    ns = namespace or _map_key_namespace(actuator_map, joints, seg_of)
    nsk = ns["namespace"]
    rows = []
    gaps: list = []
    for j in joints:
        motor, key = _resolve_actuator(j, actuator_map, seg_of, nsk)
        if motor is None:
            # NAMED GAP, not an instruction. The old fallback string ("GENERIC position actuator (set from the
            # BOM)") was phrased as a to-do, so a hardware team read a SILENT TOTAL FAILURE -- every row
            # unresolved -- as a small piece of remaining setup. A null with the joint and the key it looked
            # for cannot be mistaken for a part number, and `unresolved_joints:` below counts them.
            gaps.append((j, key))
            # The row states what actually happened to THIS joint. Under a determined namespace a key was
            # looked up and was absent; under an UNDETERMINED one no key was looked up at all, and saying
            # "no BOM entry ... (looked for X)" would assert both a lookup that never ran and an absence
            # that was never established.
            why = (f"no BOM entry for joint {j} (looked for the part assigned to {key})"
                   if nsk != _NS_UNDETERMINED else
                   f"not resolved: the key namespace of this package's parts list is undetermined, so no key "
                   f"was looked up for {j} (see KEY NAMESPACE in the header)")
            rows.append(
                f"  {j}:\n"
                f"    actuator: null\n"
                f'    unresolved: "{why}"\n'
                f"    command_interface: position\n"
                f"    state_interfaces: [position, velocity]")
        else:
            rows.append(f'  {j}:\n    actuator: "{motor}"\n    bom_key: "{key}"\n'
                        f"    command_interface: position\n"
                        f"    state_interfaces: [position, velocity]")
    body = "\n".join(rows) or "  {}"
    if gaps:
        # The REASON must match what actually happened. Under a determined namespace the list genuinely names
        # no part; under an UNDETERMINED one the list may well name parts and this export simply may not say
        # which joint they belong to. Printing "names no part for them" there would be a false claim about the
        # file sitting beside it -- the exact failure mode the strict-segment rule shipped.
        why = ("# the bill of materials shipped with this package assigns no actuator to them.\n"
               if nsk != _NS_UNDETERMINED else
               "# the key namespace of this package's parts list could not be determined (see the header), so\n"
               "# no key was looked up. The list may well assign each of them a part; this export cannot say\n"
               "# which, and will not guess.\n")
        # `looked_for:` may only name a key that was actually tried.
        detail = ("".join(f"  - joint: {j}\n    looked_for: {k}\n" for j, k in gaps)
                  if nsk != _NS_UNDETERMINED else
                  "".join(f"  - joint: {j}\n    looked_for: null   # namespace undetermined\n" for j, _k in gaps))
        body += ("\n\n# Joints this export could NOT name a real part for. Nothing below is a default motor:\n"
                 + why + "unresolved_joints:\n" + detail)
    tail = ""
    if mimic:
        # These have NO motor to buy or command: they are the far end of a transmission the driver joint turns.
        # Given a command interface each, the export would name an actuator the machine does not have.
        tail = ("# Coupled DOF: driven through a fixed gear by the joint named below, with no actuator and no\n"
                "# command interface of their own. Read their state; never command them.\n"
                "coupled_joints:\n"
                + "".join(f'  {d}:\n    driven_by: "{m["joint"]}"\n    multiplier: {m["multiplier"]:.10g}\n'
                          f'    offset: {m["offset"]:.10g}\n    state_interfaces: [position, velocity]\n'
                          for d, m in mimic.items()))
    n_named = len(joints) - len(gaps)
    # PROVENANCE FIRST. The header used to assert "the bill of materials" without saying WHICH -- and on a
    # rebuild into a reused directory that was the PREVIOUS robot's parts list, so a 100%-coverage claim sat
    # above rows naming motors this package's own bill_of_materials.json does not assign. Name the source, so
    # the claim is only ever as strong as what was actually read.
    src = actuator_map_source or "source not recorded"
    # WHAT the source assigns the part TO depends on the namespace it keys, so the sentence follows the finding
    # rather than asserting one shape of parts list for every package.
    assigned_to = ("the joint itself" if nsk == _NS_JOINT else "the link they drive")
    head = (
        f"# PARTS SOURCE: {src}.\n"
        f"# {n_named} of {len(joints)} commandable joints are mapped to the actuator THAT source assigns to\n"
        f"# {assigned_to}, plus the ros2_control interface each exposes. Set hardware_plugin to\n"
        f"# your bus driver (Dynamixel / ODrive / CAN / EtherCAT) before deploying.\n"
        + _namespace_note(ns))
    if gaps and nsk != _NS_UNDETERMINED:
        head += (f"# {len(gaps)} joint(s) are NOT mapped: this package's bill of materials names no part for them.\n"
                 f"# They carry `actuator: null` and are listed again under `unresolved_joints:` -- that is a GAP\n"
                 f"# in the parts list, not a step left for you to fill in with a motor of your choosing.\n")
    return (
        head
        + 'hardware:\n  hardware_plugin: "REPLACE_WITH_YOUR_DRIVER  # e.g. dynamixel_hardware/DynamixelHardware"\n'
        "joints:\n" + body + "\n" + tail)


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
        # COMMANDABLE, not every joint: a mimic/coupled DOF has no motor -- it is the far end of a transmission
        # the driver joint turns. Publishing a target for it commands two ends of one gear against each other.
        # Falls back to the full list for a config written before that distinction existed.
        self.joints = config.get("commandable_joints") or config["joints"]
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
