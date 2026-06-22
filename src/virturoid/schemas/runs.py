from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ArtifactRef, ValidationResult, VersionedEntity, require_non_empty


@dataclass
class PolicyRecord(VersionedEntity):
    name: str = ""
    policy_type: str = "scripted_controller"
    compatible_robot_ids: list[str] = field(default_factory=list)
    action_space: str = ""
    observation_space: list[str] = field(default_factory=list)
    artifact: ArtifactRef | None = None


@dataclass
class EpisodeRecord(VersionedEntity):
    run_id: str = ""
    robot_genome_id: str = ""
    task_graph_id: str = ""
    scene_graph_id: str = ""
    policy_id: str = ""
    backend: str = ""
    seed: int = 0
    result: str = "unknown"
    metrics: dict[str, float | int | bool] = field(default_factory=dict)
    safety_events: list[str] = field(default_factory=list)
    replay: ArtifactRef | None = None
    failure_labels: list[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        for field_name in ["run_id", "robot_genome_id", "task_graph_id", "scene_graph_id", "policy_id", "backend"]:
            require_non_empty(result, getattr(self, field_name), field_name, f"{field_name} is required.")
        return result


@dataclass
class FailureRecord(VersionedEntity):
    episode_id: str = ""
    failure_type: str = ""
    severity: str = "medium"
    summary: str = ""
    likely_causes: list[str] = field(default_factory=list)
    suggested_fixes: list[str] = field(default_factory=list)
    regression_scene_id: str | None = None

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.episode_id, "episode_id", "Failure must reference an episode.")
        require_non_empty(result, self.failure_type, "failure_type", "Failure type is required.")
        require_non_empty(result, self.summary, "summary", "Failure summary is required.")
        return result


@dataclass
class EvaluationRun(VersionedEntity):
    robot_genome_id: str = ""
    task_graph_id: str = ""
    scene_set_id: str = ""
    policy_id: str = ""
    backend: str = ""
    episodes: list[EpisodeRecord] = field(default_factory=list)
    failures: list[FailureRecord] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        for field_name in ["robot_genome_id", "task_graph_id", "scene_set_id", "policy_id", "backend"]:
            require_non_empty(result, getattr(self, field_name), field_name, f"{field_name} is required.")
        for episode in self.episodes:
            result.extend(episode.validate())
        for failure in self.failures:
            result.extend(failure.validate())
        return result

