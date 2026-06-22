from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from virturoid.schemas.base import ArtifactRef, ValidationResult, VersionedEntity, require_non_empty


class CadFormat(str, Enum):
    STEP = "step"
    STL = "stl"
    OBJ = "obj"
    GLTF = "gltf"
    CADQUERY = "cadquery"
    UNKNOWN = "unknown"


@dataclass
class CadModel(VersionedEntity):
    name: str = ""
    source: str = "generated"
    format: CadFormat = CadFormat.UNKNOWN
    exact_geometry_available: bool = False
    mesh_available: bool = False
    artifact: ArtifactRef | None = None
    units: str = "mm"
    bounding_box_mm: tuple[float, float, float] | None = None
    mass_kg: float | None = None
    editable_parameters: dict[str, float | str | bool] = field(default_factory=dict)
    validation_warnings: list[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.name, "name", "CAD model name is required.")
        if not self.exact_geometry_available and not self.mesh_available:
            result.add("missing_geometry", "CAD model needs exact geometry or mesh geometry.")
        if self.mass_kg is not None and self.mass_kg < 0:
            result.add("invalid_mass", "CAD model mass cannot be negative.", "mass_kg")
        return result


@dataclass
class AssemblyInstance:
    instance_id: str
    cad_model_id: str
    role: str
    parent_instance_id: str | None = None
    transform_xyz_rpy: tuple[float, float, float, float, float, float] | None = None


@dataclass
class AssemblyJoint:
    name: str
    joint_type: str
    parent_instance_id: str
    child_instance_id: str
    axis_xyz: tuple[float, float, float] | None = None
    limit: tuple[float, float] | None = None


@dataclass
class CadAssembly(VersionedEntity):
    name: str = ""
    instances: list[AssemblyInstance] = field(default_factory=list)
    joints: list[AssemblyJoint] = field(default_factory=list)
    bom_id: str | None = None
    robot_genome_id: str | None = None

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.name, "name", "Assembly name is required.")
        require_non_empty(result, self.instances, "instances", "Assembly must contain at least one instance.")
        instance_ids = {item.instance_id for item in self.instances}
        for index, joint in enumerate(self.joints):
            if joint.parent_instance_id not in instance_ids:
                result.add("missing_parent_instance", "Joint parent instance not found.", f"joints[{index}].parent_instance_id")
            if joint.child_instance_id not in instance_ids:
                result.add("missing_child_instance", "Joint child instance not found.", f"joints[{index}].child_instance_id")
        return result


@dataclass
class CadDiff(VersionedEntity):
    cad_model_id: str = ""
    summary: str = ""
    changed_parameters: dict[str, float | str | bool] = field(default_factory=dict)
    affected_systems: list[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.cad_model_id, "cad_model_id", "CAD diff must reference a CAD model.")
        require_non_empty(result, self.summary, "summary", "CAD diff summary is required.")
        return result

