"""DatasetImportSpec — the typed result of importing a demonstration/log dataset (Input plan, Phase 5).

The plan treats logs and demonstrations as first-class INPUT, not just training output: a robotics team brings
LeRobot datasets, robomimic HDF5 demos, MCAP/ROS bags, or prior npz episodes, and Virturoid should recover
episode counts, rates, modalities, and candidate observation/action specs to seed training. This schema is that
result; :mod:`virturoid.services.dataset_importer` produces it. Standard-library only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ValidationResult, VersionedEntity


@dataclass
class DatasetImportSpec(VersionedEntity):
    source: str = ""
    format: str = "unknown"          # virturoid_npz | robomimic_hdf5 | lerobot | mcap | rosbag | ros2_db3
    episodes: int = 0
    total_frames: int = 0
    rate_hz: float | None = None
    modalities: list[str] = field(default_factory=list)
    candidate_observation_keys: list[str] = field(default_factory=list)
    candidate_action_dim: int | None = None
    topics: list[dict] = field(default_factory=list)          # logs: [{name, type, count}]
    quality_issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        if self.format == "unknown":
            result.add("unknown_format", "Dataset format could not be identified.", "format", severity="warning")
        if self.episodes < 0 or self.total_frames < 0:
            result.add("negative_counts", "Episode/frame counts cannot be negative.", "episodes")
        return result
