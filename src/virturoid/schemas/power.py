from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ValidationResult, VersionedEntity, require_non_empty


@dataclass
class PowerRail:
    name: str
    nominal_voltage_v: float
    estimated_peak_current_a: float
    recommended_regulator_current_a: float
    headroom_percent: float
    component_roles: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class BatteryEstimate:
    nominal_voltage_v: float
    capacity_wh: float
    usable_capacity_wh: float
    estimated_average_power_w: float
    estimated_runtime_minutes: float
    notes: list[str] = field(default_factory=list)


@dataclass
class PowerArchitecture(VersionedEntity):
    robot_design_id: str = ""
    rails: list[PowerRail] = field(default_factory=list)
    battery: BatteryEstimate | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def total_peak_current_a(self) -> float:
        return sum(rail.estimated_peak_current_a for rail in self.rails)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.robot_design_id, "robot_design_id", "Power architecture must reference a robot design.")
        require_non_empty(result, self.rails, "rails", "Power architecture needs at least one rail.")
        for index, rail in enumerate(self.rails):
            if rail.nominal_voltage_v <= 0:
                result.add("invalid_voltage", "Power rail voltage must be positive.", f"rails[{index}].nominal_voltage_v")
            if rail.estimated_peak_current_a < 0:
                result.add("invalid_current", "Power rail current cannot be negative.", f"rails[{index}].estimated_peak_current_a")
            if rail.recommended_regulator_current_a < rail.estimated_peak_current_a:
                result.add("undersized_regulator", "Recommended regulator must cover estimated peak current.", f"rails[{index}].recommended_regulator_current_a")
            if rail.headroom_percent < 0:
                result.add("invalid_headroom", "Rail headroom cannot be negative.", f"rails[{index}].headroom_percent")
            if not rail.component_roles:
                result.add("empty_rail", "Power rail should have at least one component role.", f"rails[{index}].component_roles", "warning")
        if self.battery:
            if self.battery.capacity_wh <= 0:
                result.add("invalid_battery_capacity", "Battery capacity must be positive.", "battery.capacity_wh")
            if self.battery.estimated_runtime_minutes <= 0:
                result.add("invalid_runtime", "Estimated runtime must be positive.", "battery.estimated_runtime_minutes")
        return result
