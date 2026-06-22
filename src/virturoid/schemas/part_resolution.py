from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ArtifactRef, ValidationResult, VersionedEntity, require_non_empty


@dataclass
class ResolvedPart:
    role: str
    component_id: str
    source: str
    reason: str
    constraints_used: list[str] = field(default_factory=list)
    technical_limits: dict[str, str | int | float | None] = field(default_factory=dict)
    cad_assets: list[ArtifactRef] = field(default_factory=list)


@dataclass
class PartResolutionReport(VersionedEntity):
    requirements_id: str = ""
    requested_part_mentions: list[str] = field(default_factory=list)
    resolved_parts: list[ResolvedPart] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.requirements_id, "requirements_id", "Part resolution must reference requirements.")
        require_non_empty(result, self.resolved_parts, "resolved_parts", "Part resolution must include selected parts.")
        roles = set()
        for index, part in enumerate(self.resolved_parts):
            if not part.role:
                result.add("missing_role", "Resolved part role is required.", f"resolved_parts[{index}].role")
            if not part.component_id:
                result.add("missing_component", "Resolved part component id is required.", f"resolved_parts[{index}].component_id")
            if not part.source:
                result.add("missing_source", "Resolved part source is required.", f"resolved_parts[{index}].source")
            if part.role in roles:
                result.add("duplicate_role", f"Role {part.role} appears more than once.", f"resolved_parts[{index}].role")
            roles.add(part.role)
        return result
