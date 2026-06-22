from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ValidationResult, VersionedEntity, require_non_empty


@dataclass
class PhysicsEpisodeResult:
    """Outcome of one real MuJoCo physics rollout."""

    episode_id: str
    purpose: str
    scene_id: str
    compiled_scene_xml_uri: str
    seed: int
    status: str
    reward: float
    success_probability: float
    steps_simulated: int
    sim_seconds: float
    metrics: dict[str, float | int | bool] = field(default_factory=dict)
    safety_events: list[str] = field(default_factory=list)
    rollout_uri: str = ""


@dataclass
class PhysicsGroupResult:
    purpose: str
    scene_artifact: str
    scene_count: int
    episodes: int
    success_rate: float


@dataclass
class PhysicsRunResult(VersionedEntity):
    """Summary of a real physics run produced by the MuJoCo adapter.

    Unlike DryRunTrainingResult, every number here comes from stepping a real
    MuJoCo model, so this artifact proves the generated package is simulatable.
    """

    training_run_config_id: str = ""
    backend: str = "mujoco"
    adapter_name: str = ""
    robot_model_uri: str = ""
    compiled_scene_index_uri: str = ""
    control_horizon: int = 0
    timestep_s: float = 0.0
    total_scenes: int = 0
    total_episodes: int = 0
    measured_success_rate: float = 0.0
    stable_episode_rate: float = 0.0
    episode_trace_uri: str = ""
    rollout_dir: str = ""
    group_results: list[PhysicsGroupResult] = field(default_factory=list)
    episodes: list[PhysicsEpisodeResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.training_run_config_id, "training_run_config_id", "Physics result must reference a training run config.")
        require_non_empty(result, self.backend, "backend", "Physics result must name a backend.")
        require_non_empty(result, self.adapter_name, "adapter_name", "Physics result must name the simulator adapter.")
        require_non_empty(result, self.robot_model_uri, "robot_model_uri", "Physics result must reference a robot model.")
        require_non_empty(result, self.compiled_scene_index_uri, "compiled_scene_index_uri", "Physics result must reference compiled simulator scenes.")
        require_non_empty(result, self.episode_trace_uri, "episode_trace_uri", "Physics result must reference an episode trace.")
        require_non_empty(result, self.group_results, "group_results", "Physics result must include group results.")
        require_non_empty(result, self.episodes, "episodes", "Physics result must include per-episode results.")
        if self.total_scenes <= 0:
            result.add("invalid_total_scenes", "Physics total scenes must be positive.", "total_scenes")
        if self.total_episodes <= 0:
            result.add("invalid_total_episodes", "Physics episodes must be positive.", "total_episodes")
        for name in ("measured_success_rate", "stable_episode_rate"):
            value = getattr(self, name)
            if value < 0 or value > 1:
                result.add("invalid_rate", f"{name} must be between 0 and 1.", name)
        return result
