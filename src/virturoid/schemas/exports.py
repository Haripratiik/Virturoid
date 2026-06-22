from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ArtifactRef, ValidationResult, VersionedEntity, require_non_empty


@dataclass
class ExportBundle(VersionedEntity):
    project_id: str = ""
    design_artifacts: list[ArtifactRef] = field(default_factory=list)
    cad_artifacts: list[ArtifactRef] = field(default_factory=list)
    bom_artifacts: list[ArtifactRef] = field(default_factory=list)
    robot_artifacts: list[ArtifactRef] = field(default_factory=list)
    simulation_artifacts: list[ArtifactRef] = field(default_factory=list)
    software_artifacts: list[ArtifactRef] = field(default_factory=list)
    dataset_artifacts: list[ArtifactRef] = field(default_factory=list)
    report_artifacts: list[ArtifactRef] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.project_id, "project_id", "Export bundle must reference a project.")
        artifact_count = sum(
            len(items)
            for items in [
                self.cad_artifacts,
                self.design_artifacts,
                self.bom_artifacts,
                self.robot_artifacts,
                self.simulation_artifacts,
                self.software_artifacts,
                self.dataset_artifacts,
                self.report_artifacts,
            ]
        )
        if artifact_count == 0:
            result.add("empty_export", "Export bundle must contain at least one artifact.")
        return result
