"""ROS 2 control-interface extractor — Input Ingestion plan, Phase 3.

The plan calls ros2_control hardware interfaces the highest-value enterprise TRUTH after the model itself: a
robot's real joint command/state interfaces, hardware plugin, and controller update rate live in the
``<ros2_control>`` URDF/xacro tag and the controller_manager YAML, not in mesh geometry. This service statically
parses both (no xacro macro EXECUTION — literal tags only) into the existing :class:`ControllerInterfaceSpec`, so
an imported controller can be mapped to actuators before it ever runs. Standard-library XML; YAML if available.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from virturoid.schemas.policy_import import ControllerInterfaceSpec


def _local(tag: str) -> str:
    """Strip any XML namespace: '{http://...}joint' -> 'joint'."""
    return tag.rsplit("}", 1)[-1]


def _iter(elem, name: str):
    for child in elem:
        if _local(child.tag) == name:
            yield child


def parse_ros2_control(xml_text: str) -> dict:
    """Parse every ``<ros2_control>`` block in a URDF/xacro string.

    Returns ``{blocks: [...], joints, command_interfaces, state_interfaces, sensors, hardware_plugins, warnings}``
    where each joint interface is namespaced ``joint/interface`` (ros2_control's convention). ``${...}`` xacro
    expressions that survive un-expanded are recorded as warnings rather than silently treated as literals.
    """
    warnings: list[str] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        return {"blocks": [], "joints": [], "command_interfaces": [], "state_interfaces": [],
                "sensors": [], "hardware_plugins": [], "warnings": [f"XML parse error: {exc}"]}

    # find ros2_control blocks anywhere in the tree (robot root or nested).
    candidates = [root] if _local(root.tag) == "ros2_control" else []
    candidates += [e for e in root.iter() if _local(e.tag) == "ros2_control"]

    joints: list[str] = []
    command_interfaces: list[str] = []
    state_interfaces: list[str] = []
    safety_limits: dict = {}
    sensors: list[str] = []
    plugins: list[str] = []
    blocks: list[dict] = []

    for block in candidates:
        block_name = block.get("name", f"ros2_control_{len(blocks)}")
        for hw in _iter(block, "hardware"):
            for plug in _iter(hw, "plugin"):
                if plug.text:
                    plugins.append(plug.text.strip())
        block_joints: list[str] = []
        for joint in _iter(block, "joint"):
            jname = joint.get("name", "")
            if not jname or jname.startswith("${"):
                warnings.append(f"joint with unresolved/blank name in '{block_name}' (xacro not expanded).")
                continue
            joints.append(jname)
            block_joints.append(jname)
            for ci in _iter(joint, "command_interface"):
                iname = ci.get("name", "")
                key = f"{jname}/{iname}"
                command_interfaces.append(key)
                limits = {}
                for param in _iter(ci, "param"):
                    pname = param.get("name", "")
                    if param.text and not param.text.strip().startswith("${"):
                        limits[pname] = param.text.strip()
                if limits:
                    safety_limits[key] = limits
            for si in _iter(joint, "state_interface"):
                state_interfaces.append(f"{jname}/{si.get('name', '')}")
        for sensor in _iter(block, "sensor"):
            sname = sensor.get("name", "")
            sensors.append(sname)
            for si in _iter(sensor, "state_interface"):
                state_interfaces.append(f"{sname}/{si.get('name', '')}")
        blocks.append({"name": block_name, "type": block.get("type"), "joints": block_joints})

    return {
        "blocks": blocks, "joints": joints, "command_interfaces": command_interfaces,
        "state_interfaces": state_interfaces, "safety_limits": safety_limits,
        "sensors": sensors, "hardware_plugins": plugins, "warnings": warnings,
    }


def parse_controller_yaml(yaml_text: str) -> dict:
    """Parse a controller_manager YAML into ``{update_rate_hz, controllers: {name: {type, joints, ...}}}``."""
    try:
        import yaml
    except ImportError:
        return {"update_rate_hz": None, "controllers": {}, "warnings": ["PyYAML not installed"]}
    try:
        data = yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError as exc:
        return {"update_rate_hz": None, "controllers": {}, "warnings": [f"YAML parse error: {exc}"]}

    warnings: list[str] = []
    cm = (data.get("controller_manager", {}) or {}).get("ros__parameters", {}) or {}
    update_rate = cm.get("update_rate")
    controllers: dict[str, dict] = {}
    for key, val in cm.items():
        if isinstance(val, dict) and "type" in val:
            controllers[key] = {"type": val["type"]}
    # each controller often has its own top-level block with joints/interfaces.
    for name in list(controllers):
        params = (data.get(name, {}) or {}).get("ros__parameters", {}) or {}
        if params:
            controllers[name].update({
                "joints": params.get("joints", []),
                "command_interfaces": params.get("command_interfaces", []),
                "state_interfaces": params.get("state_interfaces", []),
            })
    return {"update_rate_hz": float(update_rate) if update_rate else None,
            "controllers": controllers, "warnings": warnings}


def controller_interface_from_ros2_control(xml_text: str, *, controller_yaml: str | None = None,
                                           spec_id: str = "controller_iface") -> ControllerInterfaceSpec:
    """Build a :class:`ControllerInterfaceSpec` from a ros2_control block (+ optional controller YAML for rate).

    ``action_keys`` = command interfaces, ``observation_keys`` = state interfaces, ``joint_order`` = declared
    joints, ``safety_limits`` from command-interface min/max params, ``control_frequency_hz`` from the YAML update
    rate. This is the contract a would-be imported policy is checked against (dossier: map actions to actuators).
    """
    parsed = parse_ros2_control(xml_text)
    freq = None
    if controller_yaml:
        freq = parse_controller_yaml(controller_yaml).get("update_rate_hz")
    return ControllerInterfaceSpec(
        id=spec_id,
        observation_keys=parsed["state_interfaces"],
        action_keys=parsed["command_interfaces"],
        joint_order=parsed["joints"],
        control_frequency_hz=freq,
        safety_limits=parsed["safety_limits"],
    )


def parse_ros_package(package_xml_text: str) -> dict:
    """Minimal package.xml read: name + build/exec dependencies (enough to recognize a ROS 2 package)."""
    try:
        root = ET.fromstring(package_xml_text)
    except ET.ParseError as exc:
        return {"name": None, "dependencies": [], "warnings": [f"package.xml parse error: {exc}"]}
    name = next((_.text for _ in root if _local(_.tag) == "name"), None)
    deps = sorted({e.text.strip() for e in root
                   if _local(e.tag) in {"depend", "build_depend", "exec_depend", "run_depend"} and e.text})
    return {"name": (name or "").strip() or None, "dependencies": deps, "warnings": []}
