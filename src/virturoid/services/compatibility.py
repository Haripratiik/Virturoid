from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.components import BillOfMaterials, Component, ComponentCategory
from virturoid.schemas.requirements import RequirementsRecord


@dataclass
class CompatibilityCheck:
    check: str
    status: str
    reason: str
    suggestions: list[str] = field(default_factory=list)


@dataclass
class CompatibilityReport:
    requirements_id: str
    bom_id: str
    checks: list[CompatibilityCheck]

    @property
    def ok(self) -> bool:
        return all(check.status != "fail" for check in self.checks)


def check_bom_compatibility(
    requirements: RequirementsRecord,
    bom: BillOfMaterials,
    components: list[Component],
) -> CompatibilityReport:
    by_id = {component.id: component for component in components}
    checks: list[CompatibilityCheck] = []
    total_peak_current = 0.0
    voltage_ranges: list[tuple[float, float]] = []

    for item in bom.items:
        component = by_id.get(item.component_id)
        if component is None:
            checks.append(
                CompatibilityCheck(
                    check=f"component_exists:{item.role}",
                    status="fail",
                    reason=f"Component {item.component_id} is not available in the component library.",
                    suggestions=["Select an available component or add this part to the database."],
                )
            )
            continue

        if component.max_current_a is not None:
            total_peak_current += component.max_current_a * item.quantity
        if component.voltage_range is not None:
            voltage_ranges.append(component.voltage_range)

        if component.category == ComponentCategory.ACTUATOR and component.actuator:
            required_torque = estimate_required_actuator_torque_nm(requirements)
            stall_torque = component.actuator.stall_torque_nm
            if stall_torque is None:
                checks.append(
                    CompatibilityCheck(
                        check=f"actuator_torque:{item.role}",
                        status="warning",
                        reason=f"{component.normalized_name} has no verified stall torque.",
                        suggestions=["Verify datasheet torque before using this actuator."],
                    )
                )
            elif stall_torque < required_torque:
                checks.append(
                    CompatibilityCheck(
                        check=f"actuator_torque:{item.role}",
                        status="fail",
                        reason=(
                            f"{component.normalized_name} provides {stall_torque:.2f} Nm, "
                            f"estimated requirement is {required_torque:.2f} Nm."
                        ),
                        suggestions=["Choose a higher torque actuator.", "Reduce payload.", "Reduce arm reach."],
                    )
                )
            else:
                checks.append(
                    CompatibilityCheck(
                        check=f"actuator_torque:{item.role}",
                        status="pass",
                        reason=(
                            f"{component.normalized_name} provides {stall_torque:.2f} Nm, "
                            f"above estimated {required_torque:.2f} Nm requirement."
                        ),
                    )
                )

        if component.category == ComponentCategory.SENSOR and component.sensor:
            sensor_requirements = " ".join(requirements.sensor_requirements)
            if "camera" in component.sensor.sensor_type and "camera" not in sensor_requirements:
                checks.append(
                    CompatibilityCheck(
                        check=f"sensor_requirement:{item.role}",
                        status="warning",
                        reason=f"{component.normalized_name} is included but no camera requirement was declared.",
                        suggestions=["Confirm this sensor is needed or remove it from the BOM."],
                    )
                )
            checks.extend(_camera_checks(requirements, component, item.role))

    checks.append(_peak_current_check(total_peak_current))
    checks.append(_voltage_overlap_check(voltage_ranges))

    return CompatibilityReport(requirements_id=requirements.id, bom_id=bom.id, checks=checks)


def estimate_required_actuator_torque_nm(requirements: RequirementsRecord) -> float:
    payload = requirements.payload_kg if requirements.payload_kg is not None else 0.25
    reach = requirements.reach_m if requirements.reach_m is not None else 0.5
    safety_factor = 2.0
    return payload * 9.81 * reach * safety_factor


def _camera_checks(requirements: RequirementsRecord, component: Component, role: str) -> list[CompatibilityCheck]:
    if not component.sensor or "camera" not in component.sensor.sensor_type:
        return []

    checks: list[CompatibilityCheck] = []
    fov = component.sensor.field_of_view_deg
    if fov is None:
        checks.append(
            CompatibilityCheck(
                check=f"camera_fov:{role}",
                status="warning",
                reason=f"{component.normalized_name} has no verified field-of-view data.",
                suggestions=["Verify camera FOV before using this sensor for training scenes."],
            )
        )
    else:
        horizontal, vertical = fov
        if horizontal < 55 or vertical < 40:
            checks.append(
                CompatibilityCheck(
                    check=f"camera_fov:{role}",
                    status="warning",
                    reason=f"{component.normalized_name} FOV {horizontal:.1f}x{vertical:.1f} deg may be narrow for tabletop manipulation.",
                    suggestions=["Use a wider-FOV wrist camera or add an overhead camera."],
                )
            )
        else:
            checks.append(
                CompatibilityCheck(
                    check=f"camera_fov:{role}",
                    status="pass",
                    reason=f"{component.normalized_name} FOV {horizontal:.1f}x{vertical:.1f} deg is acceptable for MVP manipulation scenes.",
                )
            )

    sensor_range = component.sensor.range_m
    reach = requirements.reach_m if requirements.reach_m is not None else 0.65
    if sensor_range is None:
        checks.append(
            CompatibilityCheck(
                check=f"camera_range:{role}",
                status="warning",
                reason=f"{component.normalized_name} has no verified operating range.",
                suggestions=["Verify depth/range before relying on this camera for training observations."],
            )
        )
    else:
        min_range, max_range = sensor_range
        if max_range < reach:
            checks.append(
                CompatibilityCheck(
                    check=f"camera_range:{role}",
                    status="fail",
                    reason=f"{component.normalized_name} max range {max_range:.2f} m is below robot reach {reach:.2f} m.",
                    suggestions=["Use a longer-range camera or reduce required reach."],
                )
            )
        elif min_range > 0.35:
            checks.append(
                CompatibilityCheck(
                    check=f"camera_range:{role}",
                    status="warning",
                    reason=f"{component.normalized_name} minimum range {min_range:.2f} m may miss near wrist objects.",
                    suggestions=["Move the camera mount back or select a closer-focus depth camera."],
                )
            )
        else:
            checks.append(
                CompatibilityCheck(
                    check=f"camera_range:{role}",
                    status="pass",
                    reason=f"{component.normalized_name} range {min_range:.2f}-{max_range:.2f} m covers MVP reach {reach:.2f} m.",
                )
            )
    return checks


def _peak_current_check(total_peak_current: float) -> CompatibilityCheck:
    if total_peak_current <= 0:
        return CompatibilityCheck(
            check="power_peak_current",
            status="warning",
            reason="No peak current data available for selected components.",
            suggestions=["Add current specs to component records."],
        )
    status = "pass" if total_peak_current <= 20.0 else "warning"
    reason = f"Estimated selected-component peak current is {total_peak_current:.1f} A."
    suggestions = [] if status == "pass" else ["Review regulator/battery sizing before hardware deployment."]
    return CompatibilityCheck("power_peak_current", status, reason, suggestions)


def _voltage_overlap_check(voltage_ranges: list[tuple[float, float]]) -> CompatibilityCheck:
    if not voltage_ranges:
        return CompatibilityCheck(
            check="power_voltage_overlap",
            status="warning",
            reason="No voltage ranges available for selected components.",
            suggestions=["Add voltage specs to component records."],
        )
    min_required = max(value[0] for value in voltage_ranges)
    max_allowed = min(value[1] for value in voltage_ranges)
    if min_required <= max_allowed:
        return CompatibilityCheck(
            check="power_voltage_overlap",
            status="pass",
            reason=f"Selected components share a direct voltage overlap of {min_required:.1f}-{max_allowed:.1f} V.",
        )
    return CompatibilityCheck(
        check="power_voltage_overlap",
        status="warning",
        reason=(
            f"Selected components do not share one direct voltage rail; actuator/compute/sensor rails span "
            f"{min_required:.1f} V minimum to {max_allowed:.1f} V maximum."
        ),
        suggestions=["Use separate regulated power rails for actuators, compute, and sensors."],
    )
