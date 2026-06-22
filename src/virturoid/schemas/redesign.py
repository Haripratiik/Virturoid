from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ArtifactRef, ValidationResult, VersionedEntity, require_non_empty


@dataclass
class AppliedRecommendation:
    recommendation_id: str
    target: str
    changes: list[str] = field(default_factory=list)


@dataclass
class RedesignRevision(VersionedEntity):
    feedback_report_id: str = ""
    source_policy_artifact: ArtifactRef | None = None
    revised_policy_artifact: ArtifactRef | None = None
    source_scene_artifact: ArtifactRef | None = None
    revised_scene_artifact: ArtifactRef | None = None
    applied_recommendations: list[AppliedRecommendation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.feedback_report_id, "feedback_report_id", "Revision must reference a feedback report.")
        require_non_empty(result, self.source_policy_artifact, "source_policy_artifact", "Revision must reference the source policy.")
        require_non_empty(result, self.revised_policy_artifact, "revised_policy_artifact", "Revision must reference the revised policy.")
        require_non_empty(result, self.source_scene_artifact, "source_scene_artifact", "Revision must reference the source scene set.")
        require_non_empty(result, self.revised_scene_artifact, "revised_scene_artifact", "Revision must reference the revised scene set.")
        require_non_empty(result, self.applied_recommendations, "applied_recommendations", "Revision must apply at least one recommendation.")
        for index, item in enumerate(self.applied_recommendations):
            if not item.recommendation_id:
                result.add("missing_recommendation_id", "Applied recommendation must reference a recommendation.", f"applied_recommendations[{index}].recommendation_id")
            if not item.changes:
                result.add("missing_recommendation_changes", "Applied recommendation must list concrete changes.", f"applied_recommendations[{index}].changes")
        return result
