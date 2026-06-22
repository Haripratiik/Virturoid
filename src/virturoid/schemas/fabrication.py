from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ArtifactRef, ValidationResult, VersionedEntity, require_non_empty


@dataclass
class PartCadBinding:
    role: str
    component_id: str
    component_cad_assets: list[ArtifactRef] = field(default_factory=list)
    generated_mount_cad_model_id: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class AssemblyOperation:
    step_id: str
    operation: str
    target: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class FabricationCheck:
    check: str
    status: str
    reason: str
    suggested_action: str | None = None


@dataclass
class FabricationBuildPlan(VersionedEntity):
    requirements_id: str = ""
    robot_genome_id: str = ""
    cad_assembly_id: str = ""
    bom_id: str = ""
    parametric_source: ArtifactRef | None = None
    exact_assembly_artifact: ArtifactRef | None = None
    visual_mesh_dir: ArtifactRef | None = None
    part_cad_bindings: list[PartCadBinding] = field(default_factory=list)
    assembly_operations: list[AssemblyOperation] = field(default_factory=list)
    fabrication_checks: list[FabricationCheck] = field(default_factory=list)
    generated_cad_parameters: dict[str, dict[str, float | str | bool]] = field(default_factory=dict)
    estimated_printed_mass_kg: float = 0.0
    manufacturing_notes: list[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.requirements_id, "requirements_id", "Build plan must reference requirements.")
        require_non_empty(result, self.robot_genome_id, "robot_genome_id", "Build plan must reference a robot.")
        require_non_empty(result, self.cad_assembly_id, "cad_assembly_id", "Build plan must reference a CAD assembly.")
        require_non_empty(result, self.bom_id, "bom_id", "Build plan must reference a BOM.")
        require_non_empty(result, self.parametric_source, "parametric_source", "Build plan must reference parametric CAD source.")
        require_non_empty(result, self.exact_assembly_artifact, "exact_assembly_artifact", "Build plan must reference exact assembly output.")
        require_non_empty(result, self.part_cad_bindings, "part_cad_bindings", "Build plan must bind selected parts to CAD assets.")
        require_non_empty(result, self.assembly_operations, "assembly_operations", "Build plan must include assembly operations.")
        require_non_empty(result, self.fabrication_checks, "fabrication_checks", "Build plan must include fabrication checks.")
        if self.estimated_printed_mass_kg < 0:
            result.add("invalid_mass", "Estimated printed mass cannot be negative.", "estimated_printed_mass_kg")
        for index, operation in enumerate(self.assembly_operations):
            if not operation.step_id:
                result.add("missing_step_id", "Assembly operation step id is required.", f"assembly_operations[{index}].step_id")
            if not operation.operation:
                result.add("missing_operation", "Assembly operation name is required.", f"assembly_operations[{index}].operation")
        return result
