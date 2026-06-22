from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ValidationResult, VersionedEntity, require_non_empty


@dataclass
class ImprovementMetric:
    name: str
    before: float
    after: float
    delta: float
    interpretation: str


@dataclass
class ImprovementReport(VersionedEntity):
    source_dry_run_id: str = ""
    revised_dry_run_id: str = ""
    comparison_basis: str = ""
    outcome: str = "unknown"
    metrics: list[ImprovementMetric] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    source_artifacts: list[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.source_dry_run_id, "source_dry_run_id", "Improvement report must reference the source dry-run.")
        require_non_empty(result, self.revised_dry_run_id, "revised_dry_run_id", "Improvement report must reference the revised dry-run.")
        require_non_empty(result, self.comparison_basis, "comparison_basis", "Improvement report must describe its comparison basis.")
        require_non_empty(result, self.metrics, "metrics", "Improvement report must include metrics.")
        require_non_empty(result, self.source_artifacts, "source_artifacts", "Improvement report must list source artifacts.")
        for index, metric in enumerate(self.metrics):
            if not metric.name:
                result.add("missing_metric_name", "Improvement metric must have a name.", f"metrics[{index}].name")
            if not metric.interpretation:
                result.add("missing_metric_interpretation", "Improvement metric must explain its interpretation.", f"metrics[{index}].interpretation")
        return result
