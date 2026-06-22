from __future__ import annotations

from collections import defaultdict

from virturoid.schemas.components import BillOfMaterials, Component, ComponentCategory
from virturoid.schemas.power import BatteryEstimate, PowerArchitecture, PowerRail


def build_power_architecture(bom: BillOfMaterials, components: list[Component]) -> PowerArchitecture:
    by_id = {component.id: component for component in components}
    grouped_roles: dict[str, list[str]] = defaultdict(list)
    grouped_current: dict[str, float] = defaultdict(float)

    for item in bom.items:
        component = by_id.get(item.component_id)
        if component is None:
            continue
        rail_name = _rail_name(component)
        grouped_roles[rail_name].append(item.role)
        grouped_current[rail_name] += (component.max_current_a or 0.0) * item.quantity

    rails = [
        _build_power_rail(rail_name, grouped_current[rail_name], roles)
        for rail_name, roles in sorted(grouped_roles.items())
    ]
    battery = _build_battery_estimate(rails)

    warnings = []
    if len(rails) > 1:
        warnings.append("Multiple power rails recommended; do not assume one shared supply for actuators, compute, and sensors.")
    if sum(rail.estimated_peak_current_a for rail in rails) > 20:
        warnings.append("Total peak current is high for an MVP reference design; size regulators and wiring carefully.")

    return PowerArchitecture(
        id=f"power_{bom.robot_design_id}",
        robot_design_id=bom.robot_design_id,
        rails=rails,
        battery=battery,
        warnings=warnings,
    )


def _build_power_rail(rail_name: str, peak_current: float, roles: list[str]) -> PowerRail:
    rounded_peak = round(peak_current, 3)
    regulator_current = _round_up_current(rounded_peak * 1.5)
    headroom = 0.0 if rounded_peak == 0 else round(((regulator_current - rounded_peak) / rounded_peak) * 100, 1)
    return PowerRail(
        name=rail_name,
        nominal_voltage_v=_nominal_voltage_for_rail(rail_name),
        estimated_peak_current_a=rounded_peak,
        recommended_regulator_current_a=regulator_current,
        headroom_percent=headroom,
        component_roles=roles,
        notes=_notes_for_rail(rail_name),
    )


def _build_battery_estimate(rails: list[PowerRail]) -> BatteryEstimate:
    peak_power_w = sum(rail.nominal_voltage_v * rail.estimated_peak_current_a for rail in rails)
    average_power_w = max(1.0, round(peak_power_w * 0.35, 2))
    capacity_wh = 99.0
    usable_capacity_wh = round(capacity_wh * 0.8, 2)
    runtime_minutes = round((usable_capacity_wh / average_power_w) * 60, 1)
    return BatteryEstimate(
        nominal_voltage_v=24.0,
        capacity_wh=capacity_wh,
        usable_capacity_wh=usable_capacity_wh,
        estimated_average_power_w=average_power_w,
        estimated_runtime_minutes=runtime_minutes,
        notes=[
            "MVP estimate assumes average draw is 35 percent of selected-component peak power.",
            "Use real duty-cycle measurements before hardware deployment.",
        ],
    )


def _round_up_current(value: float) -> float:
    if value <= 1:
        return 1.0
    if value <= 2:
        return 2.0
    if value <= 5:
        return 5.0
    if value <= 10:
        return 10.0
    if value <= 20:
        return 20.0
    return round(value + 5.0, 1)


def _rail_name(component: Component) -> str:
    if component.category == ComponentCategory.ACTUATOR or component.category == ComponentCategory.GRIPPER:
        return "actuator_24v"
    if component.category == ComponentCategory.COMPUTE:
        return "compute_12v"
    if component.category == ComponentCategory.SENSOR:
        return "sensor_5v"
    return "auxiliary"


def _nominal_voltage_for_rail(rail_name: str) -> float:
    if rail_name == "actuator_24v":
        return 24.0
    if rail_name == "compute_12v":
        return 12.0
    if rail_name == "sensor_5v":
        return 5.0
    return 12.0


def _notes_for_rail(rail_name: str) -> list[str]:
    if rail_name == "actuator_24v":
        return ["Use an isolated or well-filtered actuator rail to reduce motor noise coupling."]
    if rail_name == "compute_12v":
        return ["Use a stable regulator sized for processor peak load."]
    if rail_name == "sensor_5v":
        return ["Keep camera/sensor rail stable to avoid perception dropouts."]
    return ["Review auxiliary load requirements before hardware deployment."]
