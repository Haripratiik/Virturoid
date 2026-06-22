from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ValidationResult, VersionedEntity, require_non_empty


@dataclass
class PickPlaceEpisode:
    episode_id: str
    scene_id: str
    purpose: str
    status: str
    failure_label: str | None
    placed_count: int
    block_count: int


@dataclass
class FailureCluster:
    failure_label: str
    count: int
    example_scene_ids: list[str] = field(default_factory=list)
    suggested_regression: str = ""


@dataclass
class PhysicsEvaluationReport(VersionedEntity):
    """Task-grounded evaluation from real MuJoCo pick-and-place rollouts.

    Unlike the synthetic mock evaluation, every outcome here comes from the arm
    actually attempting to sort blocks into matching bins under real physics, so
    success rates and failure clusters reflect real task performance.
    """

    robot_genome_id: str = ""
    task_graph_id: str = ""
    backend: str = "mujoco"
    controller: str = "scripted_pick_place"
    # Honest grasp disclosure: the scripted controller PINS the block to the gripper on contact (an idealized
    # engagement), NOT a learned closed-finger friction grasp. The readiness ledger reads this so a build is
    # never certified export-ready over an idealized grasp. Set to a real model only when a contact grasp runs.
    grasp_model: str = "idealized_pin"
    total_episodes: int = 0
    success_count: int = 0
    success_rate: float = 0.0
    blocks_placed: int = 0
    blocks_total: int = 0
    success_rate_by_purpose: dict = field(default_factory=dict)
    failure_clusters: list[FailureCluster] = field(default_factory=list)
    episodes: list[PickPlaceEpisode] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.robot_genome_id, "robot_genome_id", "Report must reference a robot.")
        require_non_empty(result, self.episodes, "episodes", "Report must include episodes.")
        if self.total_episodes <= 0:
            result.add("invalid_total_episodes", "Evaluation must run at least one episode.", "total_episodes")
        if self.success_rate < 0 or self.success_rate > 1:
            result.add("invalid_success_rate", "Success rate must be between 0 and 1.", "success_rate")
        return result
