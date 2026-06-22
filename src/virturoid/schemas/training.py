from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ValidationResult, VersionedEntity, require_non_empty


@dataclass
class TrainingSceneGroup:
    purpose: str
    scene_set_id: str
    scene_count: int
    usage: str


@dataclass
class TrainingManifest(VersionedEntity):
    task_graph_id: str = ""
    robot_genome_id: str = ""
    backend: str = "mujoco"
    scene_groups: list[TrainingSceneGroup] = field(default_factory=list)
    curriculum_notes: list[str] = field(default_factory=list)

    @property
    def total_scene_count(self) -> int:
        return sum(group.scene_count for group in self.scene_groups)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.task_graph_id, "task_graph_id", "Training manifest must reference a task graph.")
        require_non_empty(result, self.robot_genome_id, "robot_genome_id", "Training manifest must reference a robot genome.")
        require_non_empty(result, self.backend, "backend", "Training manifest must define a backend.")
        require_non_empty(result, self.scene_groups, "scene_groups", "Training manifest must include scene groups.")
        for index, group in enumerate(self.scene_groups):
            if group.scene_count < 0:
                result.add("invalid_scene_count", "Scene count cannot be negative.", f"scene_groups[{index}].scene_count")
            if not group.scene_set_id:
                result.add("missing_scene_set", "Scene group must reference a scene set.", f"scene_groups[{index}].scene_set_id")
        return result

