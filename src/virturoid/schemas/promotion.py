from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ArtifactRef, ValidationResult, VersionedEntity, require_non_empty


@dataclass
class ActiveTrainingInputs(VersionedEntity):
    decision_id: str = ""
    training_config: ArtifactRef | None = None
    policy_plan: ArtifactRef | None = None
    scene_set: ArtifactRef | None = None
    reason: str = ""

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.decision_id, "decision_id", "Active inputs must reference a promotion decision.")
        require_non_empty(result, self.training_config, "training_config", "Active inputs must reference a training config.")
        require_non_empty(result, self.policy_plan, "policy_plan", "Active inputs must reference a policy plan.")
        require_non_empty(result, self.scene_set, "scene_set", "Active inputs must reference a scene set.")
        require_non_empty(result, self.reason, "reason", "Active inputs must include a reason.")
        return result


@dataclass
class PromotionDecision(VersionedEntity):
    improvement_report_id: str = ""
    decision: str = "hold"
    selected_training_config: ArtifactRef | None = None
    selected_policy_plan: ArtifactRef | None = None
    selected_scene_set: ArtifactRef | None = None
    active_inputs_artifact: ArtifactRef | None = None
    rationale: str = ""
    gates: dict[str, bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.improvement_report_id, "improvement_report_id", "Promotion must reference an improvement report.")
        require_non_empty(result, self.decision, "decision", "Promotion must include a decision.")
        require_non_empty(result, self.selected_training_config, "selected_training_config", "Promotion must select a training config.")
        require_non_empty(result, self.selected_policy_plan, "selected_policy_plan", "Promotion must select a policy plan.")
        require_non_empty(result, self.selected_scene_set, "selected_scene_set", "Promotion must select a scene set.")
        require_non_empty(result, self.active_inputs_artifact, "active_inputs_artifact", "Promotion must reference active inputs.")
        require_non_empty(result, self.rationale, "rationale", "Promotion must include rationale.")
        require_non_empty(result, self.gates, "gates", "Promotion must include decision gates.")
        return result
