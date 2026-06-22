from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ValidationResult, VersionedEntity, require_non_empty


@dataclass
class Criterion:
    name: str
    expression: str
    measurable: bool = True


@dataclass
class TaskGraph(VersionedEntity):
    name: str = ""
    prompt: str = ""
    task_type: str = ""
    required_skills: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)
    success_criteria: list[Criterion] = field(default_factory=list)
    failure_criteria: list[Criterion] = field(default_factory=list)
    safety_constraints: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.name, "name", "Task name is required.")
        require_non_empty(result, self.prompt, "prompt", "Task prompt is required.")
        require_non_empty(result, self.task_type, "task_type", "Task type is required.")
        require_non_empty(result, self.success_criteria, "success_criteria", "At least one success criterion is required.")
        require_non_empty(result, self.failure_criteria, "failure_criteria", "At least one failure criterion is required.")
        for index, criterion in enumerate(self.success_criteria + self.failure_criteria):
            if not criterion.measurable:
                result.add("unmeasurable_criterion", "Criteria must be measurable before simulation.", f"criteria[{index}]", "warning")
        return result

