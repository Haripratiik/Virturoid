from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ValidationResult, VersionedEntity, require_non_empty


@dataclass
class CurriculumSceneBinding:
    scene_id: str
    purpose: str
    scene_set_uri: str
    compiled_scene_xml_uri: str
    episode_multiplier: int
    randomization_profile: str


@dataclass
class CurriculumStage:
    name: str
    purpose: str
    usage: str
    scene_count: int
    planned_episodes: int
    scene_bindings: list[CurriculumSceneBinding] = field(default_factory=list)


@dataclass
class TrainingCurriculum(VersionedEntity):
    robot_genome_id: str = ""
    task_graph_id: str = ""
    backend: str = "mujoco"
    compiled_scene_index_uri: str = ""
    stages: list[CurriculumStage] = field(default_factory=list)
    success_checks: list[str] = field(default_factory=list)
    safety_checks: list[str] = field(default_factory=list)
    randomization_envelopes: dict[str, list[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def total_planned_episodes(self) -> int:
        return sum(stage.planned_episodes for stage in self.stages)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.robot_genome_id, "robot_genome_id", "Curriculum must reference a robot genome.")
        require_non_empty(result, self.task_graph_id, "task_graph_id", "Curriculum must reference a task graph.")
        require_non_empty(result, self.backend, "backend", "Curriculum must define a backend.")
        require_non_empty(result, self.compiled_scene_index_uri, "compiled_scene_index_uri", "Curriculum must reference compiled simulator scenes.")
        require_non_empty(result, self.stages, "stages", "Curriculum must include stages.")
        require_non_empty(result, self.success_checks, "success_checks", "Curriculum must include success checks.")
        require_non_empty(result, self.safety_checks, "safety_checks", "Curriculum must include safety checks.")
        for stage_index, stage in enumerate(self.stages):
            if stage.scene_count <= 0:
                result.add("invalid_stage_scene_count", "Curriculum stage must contain scenes.", f"stages[{stage_index}].scene_count")
            if stage.planned_episodes <= 0:
                result.add("invalid_stage_episode_count", "Curriculum stage must plan episodes.", f"stages[{stage_index}].planned_episodes")
            require_non_empty(result, stage.scene_bindings, f"stages[{stage_index}].scene_bindings", "Curriculum stage must bind scenes.")
            for binding_index, binding in enumerate(stage.scene_bindings):
                field = f"stages[{stage_index}].scene_bindings[{binding_index}]"
                require_non_empty(result, binding.scene_id, f"{field}.scene_id", "Scene binding must name a scene.")
                require_non_empty(result, binding.scene_set_uri, f"{field}.scene_set_uri", "Scene binding must reference a scene set.")
                require_non_empty(result, binding.compiled_scene_xml_uri, f"{field}.compiled_scene_xml_uri", "Scene binding must reference compiled XML.")
        return result
