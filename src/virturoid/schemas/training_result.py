from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ValidationResult, VersionedEntity, require_non_empty


@dataclass
class TrainingGroupResult:
    purpose: str
    scene_artifact: str
    scene_count: int
    planned_episodes: int
    simulated_success_rate: float
    first_episode_id: str = ""
    last_episode_id: str = ""


@dataclass
class TrainingEpisodeTrace:
    episode_id: str
    backend: str
    adapter_name: str
    purpose: str
    scene_id: str
    scene_artifact: str
    policy_id: str
    status: str
    reward: float
    success_probability: float
    compiled_scene_xml_uri: str = ""
    observation_keys: list[str] = field(default_factory=list)
    action_trace: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    metrics: dict[str, float | int | bool] = field(default_factory=dict)
    rollout_uri: str = ""


@dataclass
class DryRunReplayIndex(VersionedEntity):
    training_run_config_id: str = ""
    backend: str = "mujoco"
    adapter_name: str = ""
    dry_run_result_uri: str = ""
    episode_trace_uri: str = ""
    episode_count: int = 0
    scene_artifacts: list[str] = field(default_factory=list)
    compiled_scene_index_uri: str = ""
    compiled_scene_xml_uris: list[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.training_run_config_id, "training_run_config_id", "Replay index must reference a training run config.")
        require_non_empty(result, self.backend, "backend", "Replay index must name a backend.")
        require_non_empty(result, self.adapter_name, "adapter_name", "Replay index must name the simulator adapter.")
        require_non_empty(result, self.dry_run_result_uri, "dry_run_result_uri", "Replay index must reference the dry-run result.")
        require_non_empty(result, self.episode_trace_uri, "episode_trace_uri", "Replay index must reference an episode trace.")
        require_non_empty(result, self.scene_artifacts, "scene_artifacts", "Replay index must reference scene artifacts.")
        require_non_empty(result, self.compiled_scene_index_uri, "compiled_scene_index_uri", "Replay index must reference compiled simulator scenes.")
        require_non_empty(result, self.compiled_scene_xml_uris, "compiled_scene_xml_uris", "Replay index must reference compiled simulator scene XML files.")
        if self.episode_count <= 0:
            result.add("invalid_episode_count", "Replay index episode count must be positive.", "episode_count")
        return result


@dataclass
class DryRunTrainingResult(VersionedEntity):
    training_run_config_id: str = ""
    backend: str = "mujoco"
    adapter_name: str = ""
    robot_model_uri: str = ""
    perception_artifact_uri: str = ""
    objective_artifact_uri: str = ""
    curriculum_artifact_uri: str = ""
    compiled_scene_index_uri: str = ""
    reward_terms: list[str] = field(default_factory=list)
    observation_streams: list[str] = field(default_factory=list)
    total_scenes: int = 0
    total_planned_episodes: int = 0
    estimated_success_rate: float = 0.0
    replay_index_uri: str = ""
    episode_trace_uri: str = ""
    group_results: list[TrainingGroupResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.training_run_config_id, "training_run_config_id", "Dry-run result must reference a training run config.")
        require_non_empty(result, self.backend, "backend", "Dry-run result must name a backend.")
        require_non_empty(result, self.adapter_name, "adapter_name", "Dry-run result must name the simulator adapter.")
        require_non_empty(result, self.robot_model_uri, "robot_model_uri", "Dry-run result must reference a robot model.")
        require_non_empty(result, self.perception_artifact_uri, "perception_artifact_uri", "Dry-run result must reference a perception config.")
        require_non_empty(result, self.objective_artifact_uri, "objective_artifact_uri", "Dry-run result must reference a training objective.")
        require_non_empty(result, self.curriculum_artifact_uri, "curriculum_artifact_uri", "Dry-run result must reference a training curriculum.")
        require_non_empty(result, self.compiled_scene_index_uri, "compiled_scene_index_uri", "Dry-run result must reference compiled simulator scenes.")
        require_non_empty(result, self.reward_terms, "reward_terms", "Dry-run result must name reward terms.")
        require_non_empty(result, self.observation_streams, "observation_streams", "Dry-run result must include observation streams.")
        require_non_empty(result, self.replay_index_uri, "replay_index_uri", "Dry-run result must reference a replay index.")
        require_non_empty(result, self.episode_trace_uri, "episode_trace_uri", "Dry-run result must reference an episode trace.")
        require_non_empty(result, self.group_results, "group_results", "Dry-run result must include group results.")
        if self.total_scenes <= 0:
            result.add("invalid_total_scenes", "Dry-run total scenes must be positive.", "total_scenes")
        if self.total_planned_episodes <= 0:
            result.add("invalid_total_episodes", "Dry-run planned episodes must be positive.", "total_planned_episodes")
        if self.estimated_success_rate < 0 or self.estimated_success_rate > 1:
            result.add("invalid_success_rate", "Estimated success rate must be between 0 and 1.", "estimated_success_rate")
        return result
