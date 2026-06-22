from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ValidationResult, VersionedEntity, require_non_empty


@dataclass
class SceneObject:
    name: str
    object_type: str
    pose_xyz_rpy: tuple[float, float, float, float, float, float]
    mass_kg: float | None = None
    material: str | None = None
    friction: float | None = None
    scale: float | None = None


@dataclass
class SceneGraph(VersionedEntity):
    name: str = ""
    backend_targets: list[str] = field(default_factory=list)
    robot_spawn_xyz_rpy: tuple[float, float, float, float, float, float] | None = None
    objects: list[SceneObject] = field(default_factory=list)
    variation_parameters: dict[str, str | float | int | bool] = field(default_factory=dict)
    requirement_trace: list[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.name, "name", "Scene name is required.")
        require_non_empty(result, self.backend_targets, "backend_targets", "Scene must target at least one backend.")
        require_non_empty(result, self.robot_spawn_xyz_rpy, "robot_spawn_xyz_rpy", "Robot spawn pose is required.")
        for index, item in enumerate(self.objects):
            if item.mass_kg is not None and item.mass_kg < 0:
                result.add("invalid_object_mass", "Object mass cannot be negative.", f"objects[{index}].mass_kg")
        return result


@dataclass
class SceneSet(VersionedEntity):
    task_graph_id: str = ""
    purpose: str = "baseline"
    scenes: list[SceneGraph] = field(default_factory=list)
    generation_rules: list[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.task_graph_id, "task_graph_id", "Scene set must reference a task graph.")
        require_non_empty(result, self.scenes, "scenes", "Scene set must contain scenes.")
        for scene in self.scenes:
            result.extend(scene.validate())
        return result

