from __future__ import annotations

from virturoid.schemas.components import Component, ComponentCategory
from virturoid.schemas.part_resolution import PartResolutionReport, ResolvedPart
from virturoid.schemas.requirements import RequirementsRecord
from virturoid.services.compatibility import estimate_required_actuator_torque_nm


def resolve_robot_arm_parts(requirements: RequirementsRecord, components: list[Component]) -> PartResolutionReport:
    prompt = requirements.prompt.lower()
    explicit_matches = _explicit_matches(prompt, components)
    requested_mentions = [component.part_number for component in explicit_matches]
    actuator = _select_actuator(requirements, components, explicit_matches)
    sensor = _select_sensor(requirements, components, explicit_matches)
    gripper = _select_category(ComponentCategory.GRIPPER, components, explicit_matches, fallback_id="cmp_gripper_pg2")
    compute = _select_category(ComponentCategory.COMPUTE, components, explicit_matches, fallback_id="cmp_compute_edge1")

    resolved_parts = [
        _resolved_part("joint_actuator", actuator, _source_for(actuator, explicit_matches), _actuator_reason(requirements, actuator, explicit_matches)),
        _resolved_part("wrist_sensor", sensor, _source_for(sensor, explicit_matches), _sensor_reason(requirements, sensor, explicit_matches)),
        _resolved_part("gripper", gripper, _source_for(gripper, explicit_matches), _default_reason(gripper, explicit_matches)),
        _resolved_part("main_compute", compute, _source_for(compute, explicit_matches), _default_reason(compute, explicit_matches)),
    ]
    warnings = _warnings(requirements, explicit_matches, resolved_parts)
    report = PartResolutionReport(
        id=f"parts_{requirements.id}",
        requirements_id=requirements.id,
        requested_part_mentions=requested_mentions,
        resolved_parts=resolved_parts,
        warnings=warnings,
    )
    validation = report.validate()
    if not validation.ok:
        issues = ", ".join(issue.code for issue in validation.issues)
        raise ValueError(f"{report.id} failed validation: {issues}")
    return report


def _explicit_matches(prompt: str, components: list[Component]) -> list[Component]:
    matches: list[Component] = []
    for component in components:
        tokens = [component.part_number, component.normalized_name, *component.aliases]
        if any(_contains_token(prompt, token) for token in tokens if token):
            matches.append(component)
    return matches


def _contains_token(prompt: str, token: str) -> bool:
    normalized = token.lower()
    compact = normalized.replace("-", "").replace(" ", "")
    prompt_compact = prompt.replace("-", "").replace(" ", "")
    return normalized in prompt or compact in prompt_compact


def _select_actuator(requirements: RequirementsRecord, components: list[Component], explicit_matches: list[Component]) -> Component:
    explicit = next((component for component in explicit_matches if component.category == ComponentCategory.ACTUATOR), None)
    if explicit:
        return explicit
    required_torque = estimate_required_actuator_torque_nm(requirements)
    actuators = [
        component for component in components
        if component.category == ComponentCategory.ACTUATOR and component.actuator and component.actuator.stall_torque_nm is not None
    ]
    capable = [component for component in actuators if (component.actuator and component.actuator.stall_torque_nm >= required_torque)]
    return min(capable or actuators, key=lambda item: item.actuator.stall_torque_nm if item.actuator else 999.0)


def _select_sensor(requirements: RequirementsRecord, components: list[Component], explicit_matches: list[Component]) -> Component:
    explicit = next((component for component in explicit_matches if component.category == ComponentCategory.SENSOR), None)
    if explicit:
        return explicit
    requested = " ".join(requirements.sensor_requirements).lower()
    sensors = [component for component in components if component.category == ComponentCategory.SENSOR and component.sensor]
    for component in sensors:
        if component.sensor and component.sensor.sensor_type.lower() in requested:
            return component
    if "lidar" in requested:
        return _sensor_by_type(components, "lidar", preferred_id="cmp_lidar_planar_l1")
    if "rgb_camera" in requested or "rgb camera" in requested or "normal camera" in requested:
        return _sensor_by_type(components, "rgb_camera", preferred_id="cmp_camera_rgb_mini")
    return _sensor_by_type(components, "rgbd_camera", preferred_id="cmp_camera_rgbd_mini")


def _select_category(
    category: ComponentCategory,
    components: list[Component],
    explicit_matches: list[Component],
    *,
    fallback_id: str,
) -> Component:
    explicit = next((component for component in explicit_matches if component.category == category), None)
    if explicit:
        return explicit
    preferred = next((component for component in components if component.id == fallback_id), None)
    if preferred is not None:
        return preferred
    fallback = next((component for component in components if component.category == category), None)
    if fallback is None:
        raise KeyError(f"Component catalog has no {category.value} part for role resolution.")
    return fallback


def _sensor_by_type(components: list[Component], sensor_type: str, *, preferred_id: str) -> Component:
    preferred = next((component for component in components if component.id == preferred_id), None)
    if preferred is not None and preferred.sensor and preferred.sensor.sensor_type == sensor_type:
        return preferred
    match = next((component for component in components
                  if component.sensor and component.sensor.sensor_type == sensor_type), None)
    if match is None:
        raise KeyError(f"Component catalog has no {sensor_type} sensor.")
    return match


def _component_by_id(components: list[Component], component_id: str) -> Component:
    for component in components:
        if component.id == component_id:
            return component
    raise KeyError(f"Component {component_id} is missing from the library.")


def _source_for(component: Component, explicit_matches: list[Component]) -> str:
    return "explicit_user_request" if component in explicit_matches else "requirement_recommendation"


def _actuator_reason(requirements: RequirementsRecord, component: Component, explicit_matches: list[Component]) -> str:
    if component in explicit_matches:
        return f"User prompt explicitly matched {component.part_number}."
    required_torque = estimate_required_actuator_torque_nm(requirements)
    stall_torque = component.actuator.stall_torque_nm if component.actuator else None
    return f"Selected lowest available actuator that clears estimated {required_torque:.2f} Nm torque; part provides {stall_torque:.2f} Nm."


def _sensor_reason(requirements: RequirementsRecord, component: Component, explicit_matches: list[Component]) -> str:
    if component in explicit_matches:
        return f"User prompt explicitly matched {component.part_number}."
    return f"Selected to satisfy sensor requirements: {', '.join(requirements.sensor_requirements)}."


def _default_reason(component: Component, explicit_matches: list[Component]) -> str:
    if component in explicit_matches:
        return f"User prompt explicitly matched {component.part_number}."
    return "Selected as the current reference MVP component for this role."


def _resolved_part(role: str, component: Component, source: str, reason: str) -> ResolvedPart:
    return ResolvedPart(
        role=role,
        component_id=component.id,
        source=source,
        reason=reason,
        constraints_used=_constraints_for(component),
        technical_limits=_technical_limits(component),
        cad_assets=list(component.cad_assets),
    )


def _constraints_for(component: Component) -> list[str]:
    constraints = []
    if component.voltage_range:
        constraints.append(f"voltage_range:{component.voltage_range[0]}-{component.voltage_range[1]}V")
    if component.max_current_a is not None:
        constraints.append(f"max_current:{component.max_current_a}A")
    if component.mass_kg is not None:
        constraints.append(f"mass:{component.mass_kg}kg")
    if component.actuator and component.actuator.stall_torque_nm is not None:
        constraints.append(f"stall_torque:{component.actuator.stall_torque_nm}Nm")
    if component.sensor and component.sensor.sensor_type:
        constraints.append(f"sensor_type:{component.sensor.sensor_type}")
    return constraints


def _technical_limits(component: Component) -> dict[str, str | int | float | None]:
    limits: dict[str, str | int | float | None] = {
        "mass_kg": component.mass_kg,
        "max_current_a": component.max_current_a,
    }
    if component.voltage_range:
        limits["min_voltage_v"] = component.voltage_range[0]
        limits["max_voltage_v"] = component.voltage_range[1]
    if component.actuator:
        limits.update(
            {
                "actuator_type": component.actuator.actuator_type,
                "stall_torque_nm": component.actuator.stall_torque_nm,
                "no_load_speed_rpm": component.actuator.no_load_speed_rpm,
                "rated_torque_nm": component.actuator.rated_torque_nm,
                "rated_speed_rpm": component.actuator.rated_speed_rpm,
                "control_frequency_hz": component.actuator.control_frequency_hz,
                "communication_protocol": component.actuator.communication_protocol,
            }
        )
    if component.sensor:
        limits.update(
            {
                "sensor_type": component.sensor.sensor_type,
                "frame_rate_hz": component.sensor.frame_rate_hz,
                "ros_message_type": component.sensor.ros_message_type,
            }
        )
        if component.sensor.resolution:
            limits["resolution_width_px"] = component.sensor.resolution[0]
            limits["resolution_height_px"] = component.sensor.resolution[1]
        if component.sensor.field_of_view_deg:
            limits["horizontal_fov_deg"] = component.sensor.field_of_view_deg[0]
            limits["vertical_fov_deg"] = component.sensor.field_of_view_deg[1]
        if component.sensor.range_m:
            limits["min_range_m"] = component.sensor.range_m[0]
            limits["max_range_m"] = component.sensor.range_m[1]
    return limits


def _warnings(
    requirements: RequirementsRecord,
    explicit_matches: list[Component],
    resolved_parts: list[ResolvedPart],
) -> list[str]:
    warnings = []
    selected_ids = {part.component_id for part in resolved_parts}
    unused_explicit = [component.part_number for component in explicit_matches if component.id not in selected_ids]
    if unused_explicit:
        warnings.append(f"Explicitly mentioned parts were recognized but not needed for this robot arm role set: {', '.join(unused_explicit)}.")
    requested = " ".join(requirements.sensor_requirements).lower()
    sensor_part = next(part for part in resolved_parts if part.role == "wrist_sensor")
    if "lidar" in requested and "lidar" not in str(sensor_part.technical_limits.get("sensor_type", "")).lower():
        warnings.append("LiDAR was requested but no matching LiDAR sensor was selected.")
    return warnings
