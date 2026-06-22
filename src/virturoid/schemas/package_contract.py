from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ValidationResult, VersionedEntity, require_non_empty


@dataclass
class PackageArtifactCheck:
    key: str
    uri: str
    required: bool = True
    exists: bool = False
    parse_status: str = "not_checked"
    media_type: str | None = None
    detail: str = ""


@dataclass
class RobotPackageContract(VersionedEntity):
    """Common package contract shared by all generated robot classes."""

    package_type: str = ""
    robot_class: str = ""
    species: str = ""
    morphology_template_id: str = ""
    task_type: str = ""
    capabilities: list[str] = field(default_factory=list)
    artifact_checks: list[PackageArtifactCheck] = field(default_factory=list)
    ok: bool = False
    notes: list[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.package_type, "package_type", "Package type is required.")
        require_non_empty(result, self.robot_class, "robot_class", "Robot class is required.")
        require_non_empty(result, self.species, "species", "Species is required.")
        require_non_empty(result, self.morphology_template_id, "morphology_template_id", "Morphology template is required.")
        require_non_empty(result, self.artifact_checks, "artifact_checks", "Package contract must include artifact checks.")
        missing = [item.uri for item in self.artifact_checks if item.required and not item.exists]
        if missing:
            result.add("missing_required_artifacts", f"Missing required artifacts: {', '.join(missing)}")
        parse_failures = [item.uri for item in self.artifact_checks if item.required and item.parse_status == "fail"]
        if parse_failures:
            result.add("invalid_required_artifacts", f"Invalid required artifacts: {', '.join(parse_failures)}")
        if not self.ok:
            result.add("package_contract_not_ok", "Package contract is not OK.")
        return result
