from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ValidationResult, VersionedEntity, require_non_empty


@dataclass
class BuilderTemplateStatus:
    morphology_template_id: str
    robot_class: str
    builder_service: str
    status: str
    notes: list[str] = field(default_factory=list)


@dataclass
class RobotBuilderDispatchRecord(VersionedEntity):
    selected_morphology_template_id: str = ""
    selected_robot_class: str = ""
    selected_builder_service: str = ""
    dispatch_status: str = ""
    implemented_templates: list[BuilderTemplateStatus] = field(default_factory=list)
    planned_templates: list[BuilderTemplateStatus] = field(default_factory=list)
    rationale: str = ""

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.selected_morphology_template_id, "selected_morphology_template_id", "Dispatch must name a selected morphology template.")
        require_non_empty(result, self.selected_robot_class, "selected_robot_class", "Dispatch must name a selected robot class.")
        require_non_empty(result, self.selected_builder_service, "selected_builder_service", "Dispatch must name the selected builder service.")
        require_non_empty(result, self.dispatch_status, "dispatch_status", "Dispatch must include a status.")
        require_non_empty(result, self.implemented_templates, "implemented_templates", "Dispatch must list implemented templates.")
        require_non_empty(result, self.rationale, "rationale", "Dispatch must include rationale.")
        if self.dispatch_status not in {"implemented", "planned"}:
            result.add("invalid_dispatch_status", "Dispatch status must be implemented or planned.", "dispatch_status")
        for index, item in enumerate(self.implemented_templates + self.planned_templates):
            if item.status not in {"implemented", "planned"}:
                result.add("invalid_template_status", "Template status must be implemented or planned.", f"templates[{index}].status")
        return result
