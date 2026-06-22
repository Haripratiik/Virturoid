from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ArtifactRef, ValidationResult, VersionedEntity, require_non_empty


@dataclass
class TrainingRunConfig(VersionedEntity):
    robot_genome_id: str = ""
    task_graph_id: str = ""
    policy_id: str = ""
    backend: str = "mujoco"
    robot_model: ArtifactRef | None = None
    policy_artifact: ArtifactRef | None = None
    perception_artifact: ArtifactRef | None = None
    objective_artifact: ArtifactRef | None = None
    curriculum_artifact: ArtifactRef | None = None
    compiled_scene_artifact: ArtifactRef | None = None
    scene_artifacts: list[ArtifactRef] = field(default_factory=list)
    training_manifest: ArtifactRef | None = None
    output_dir: str = "runs/mvp_training"
    episode_multiplier_overrides: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.robot_genome_id, "robot_genome_id", "Training run config must reference a robot genome.")
        require_non_empty(result, self.task_graph_id, "task_graph_id", "Training run config must reference a task graph.")
        require_non_empty(result, self.policy_id, "policy_id", "Training run config must reference a policy.")
        require_non_empty(result, self.backend, "backend", "Training run config must define a backend.")
        require_non_empty(result, self.robot_model, "robot_model", "Training run config must reference a robot model artifact.")
        require_non_empty(result, self.policy_artifact, "policy_artifact", "Training run config must reference a policy artifact.")
        require_non_empty(result, self.perception_artifact, "perception_artifact", "Training run config must reference a perception artifact.")
        require_non_empty(result, self.objective_artifact, "objective_artifact", "Training run config must reference a training objective.")
        require_non_empty(result, self.curriculum_artifact, "curriculum_artifact", "Training run config must reference a training curriculum.")
        require_non_empty(result, self.compiled_scene_artifact, "compiled_scene_artifact", "Training run config must reference compiled simulator scenes.")
        require_non_empty(result, self.scene_artifacts, "scene_artifacts", "Training run config must reference scene artifacts.")
        require_non_empty(result, self.training_manifest, "training_manifest", "Training run config must reference a training manifest.")
        return result
