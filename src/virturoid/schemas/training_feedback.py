from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ValidationResult, VersionedEntity, require_non_empty


@dataclass
class RedesignRecommendation:
    id: str
    target: str
    priority: str
    summary: str
    rationale: str
    suggested_actions: list[str] = field(default_factory=list)
    source_artifacts: list[str] = field(default_factory=list)


@dataclass
class TrainingFeedbackReport(VersionedEntity):
    training_run_config_id: str = ""
    dry_run_result_id: str = ""
    evaluation_run_id: str = ""
    overall_status: str = "needs_iteration"
    recommendations: list[RedesignRecommendation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.training_run_config_id, "training_run_config_id", "Feedback must reference a training run config.")
        require_non_empty(result, self.dry_run_result_id, "dry_run_result_id", "Feedback must reference a dry-run result.")
        require_non_empty(result, self.evaluation_run_id, "evaluation_run_id", "Feedback must reference an evaluation run.")
        require_non_empty(result, self.recommendations, "recommendations", "Feedback must include redesign recommendations.")
        for index, recommendation in enumerate(self.recommendations):
            if not recommendation.id:
                result.add("missing_recommendation_id", "Recommendation must have an id.", f"recommendations[{index}].id")
            if not recommendation.target:
                result.add("missing_recommendation_target", "Recommendation must name a target.", f"recommendations[{index}].target")
            if not recommendation.suggested_actions:
                result.add("missing_recommendation_actions", "Recommendation must include suggested actions.", f"recommendations[{index}].suggested_actions")
        return result
