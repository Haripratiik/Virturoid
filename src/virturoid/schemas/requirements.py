from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ValidationResult, VersionedEntity, require_non_empty


@dataclass
class RobotRequirement:
    name: str
    value: str | int | float | bool
    unit: str | None = None
    priority: str = "must"


@dataclass
class RequirementsRecord(VersionedEntity):
    prompt: str = ""
    task_summary: str = ""
    environment: str | None = None
    payload_kg: float | None = None
    reach_m: float | None = None
    precision_m: float | None = None
    budget_usd: float | None = None
    manufacturing_constraints: list[str] = field(default_factory=list)
    sensor_requirements: list[str] = field(default_factory=list)
    additional_requirements: list[RobotRequirement] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.prompt, "prompt", "Original user prompt is required.")
        require_non_empty(result, self.task_summary, "task_summary", "Task summary is required.")
        if self.payload_kg is not None and self.payload_kg < 0:
            result.add("invalid_payload", "Payload cannot be negative.", "payload_kg")
        if self.reach_m is not None and self.reach_m <= 0:
            result.add("invalid_reach", "Reach must be positive.", "reach_m")
        return result

