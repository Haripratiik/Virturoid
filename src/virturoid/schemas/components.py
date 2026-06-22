from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from virturoid.schemas.base import ArtifactRef, ValidationResult, VersionedEntity, require_non_empty


class ComponentCategory(str, Enum):
    ACTUATOR = "actuator"
    SENSOR = "sensor"
    COMPUTE = "compute"
    POWER = "power"
    STRUCTURAL = "structural"
    GRIPPER = "gripper"
    FASTENER = "fastener"
    UNKNOWN = "unknown"


@dataclass
class ActuatorSpecs:
    actuator_type: str
    stall_torque_nm: float | None = None
    no_load_speed_rpm: float | None = None
    rated_torque_nm: float | None = None
    rated_speed_rpm: float | None = None
    control_frequency_hz: float | None = None
    communication_protocol: str | None = None


@dataclass
class SensorSpecs:
    sensor_type: str
    resolution: tuple[int, int] | None = None
    field_of_view_deg: tuple[float, float] | None = None
    frame_rate_hz: float | None = None
    range_m: tuple[float, float] | None = None
    ros_message_type: str | None = None


@dataclass
class Component(VersionedEntity):
    manufacturer: str = ""
    part_number: str = ""
    normalized_name: str = ""
    category: ComponentCategory = ComponentCategory.UNKNOWN
    aliases: list[str] = field(default_factory=list)
    datasheet: ArtifactRef | None = None
    cad_assets: list[ArtifactRef] = field(default_factory=list)
    mass_kg: float | None = None
    voltage_range: tuple[float, float] | None = None
    max_current_a: float | None = None
    actuator: ActuatorSpecs | None = None
    sensor: SensorSpecs | None = None
    spec_confidence: float = 0.0
    source_urls: list[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.normalized_name, "normalized_name", "Component name is required.")
        if self.spec_confidence < 0 or self.spec_confidence > 1:
            result.add("invalid_confidence", "Spec confidence must be between 0 and 1.", "spec_confidence")
        if self.mass_kg is not None and self.mass_kg < 0:
            result.add("invalid_mass", "Mass cannot be negative.", "mass_kg")
        if self.category == ComponentCategory.ACTUATOR and not self.actuator:
            result.add("missing_actuator_specs", "Actuator components need actuator specs.", "actuator")
        if self.category == ComponentCategory.SENSOR and not self.sensor:
            result.add("missing_sensor_specs", "Sensor components need sensor specs.", "sensor")
        return result


@dataclass
class BomItem:
    role: str
    component_id: str
    quantity: int = 1
    required: bool = True
    unit_cost_usd: float | None = None
    notes: str | None = None


@dataclass
class BillOfMaterials(VersionedEntity):
    robot_design_id: str = ""
    items: list[BomItem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.robot_design_id, "robot_design_id", "BOM must point to a robot design.")
        require_non_empty(result, self.items, "items", "BOM must contain at least one item.")
        for index, item in enumerate(self.items):
            if item.quantity <= 0:
                result.add("invalid_quantity", "BOM quantity must be positive.", f"items[{index}].quantity")
            if not item.role:
                result.add("missing_role", "BOM item role is required.", f"items[{index}].role")
            if not item.component_id:
                result.add("missing_component", "BOM item component_id is required.", f"items[{index}].component_id")
        return result

