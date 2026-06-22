"""Demonstration & dataset schemas (startup plan §41 / §32.3).

A demonstration dataset turns competent-controller rollouts into learning-ready expert trajectories so
imitation learning (§12.1's first learning type) and offline RL have something to train on. Each
``DemonstrationEpisode`` carries the per-step trajectory shape + provenance; the ``DatasetCard`` (§41.3)
is the honest summary that travels with the dataset into Memory/Export (robot, task, source, episode
count, success rate, observation modalities, action space, biases, randomization, permissions).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ValidationResult, VersionedEntity, require_non_empty


@dataclass
class DemonstrationEpisode:
    """One expert trajectory's metadata (§32.3 / §41.1). Arrays live in the dataset file by ``uri``."""
    episode_id: str
    length: int                                  # number of timesteps
    success: bool
    obs_dim: int
    action_dim: int
    task_language: str = ""
    scene_params: dict = field(default_factory=dict)
    robot_genome_id: str = ""
    source: str = "simulation"                   # simulation | real_robot | teleop | scripted_expert
    safety_events: list[str] = field(default_factory=list)
    uri: str = ""                                # path to the per-episode array file (npz)
    return_: float = 0.0


@dataclass
class DatasetCard(VersionedEntity):
    """§41.3 dataset card — the honest summary that ships with a demonstration dataset."""
    robot: str = ""
    task: str = ""
    sources: list[str] = field(default_factory=list)
    episodes: int = 0
    success_rate: float = 0.0
    observation_modalities: list[str] = field(default_factory=list)
    action_space: str = ""
    obs_dim: int = 0
    action_dim: int = 0
    total_transitions: int = 0
    known_biases: list[str] = field(default_factory=list)
    domain_randomization: dict = field(default_factory=dict)
    license_or_visibility: str = "private"
    training_permission: str = "full_opt_in"
    formats: list[str] = field(default_factory=list)     # e.g. ["virturoid_npz", "robomimic_hdf5"]
    robomimic_compatible: bool = False

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.robot, "robot", "Dataset card must name the robot.")
        require_non_empty(result, self.task, "task", "Dataset card must name the task.")
        require_non_empty(result, self.sources, "sources", "Dataset card must state its source(s).")
        if self.episodes <= 0:
            result.add("no_episodes", "Dataset has no episodes.", "episodes")
        if not 0.0 <= self.success_rate <= 1.0:
            result.add("invalid_success_rate", "success_rate must be in [0,1].", "success_rate")
        return result
