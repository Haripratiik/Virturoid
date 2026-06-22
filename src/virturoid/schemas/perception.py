from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ValidationResult, VersionedEntity, require_non_empty


@dataclass
class SensorStream:
    name: str
    sensor_component_id: str
    sensor_type: str
    parent_link: str
    observation_key: str
    output_modalities: list[str] = field(default_factory=list)
    simulated_outputs: list[str] = field(default_factory=list)
    technical_limits: dict[str, str | int | float | None] = field(default_factory=dict)


@dataclass
class TextureRandomizationRule:
    target: str
    variants: list[str] = field(default_factory=list)
    purpose: str = "domain_randomization"


@dataclass
class VisionAnnotationSpec:
    name: str
    target_object_types: list[str] = field(default_factory=list)
    annotation_modalities: list[str] = field(default_factory=list)
    label_space: list[str] = field(default_factory=list)
    output_pattern: str = ""


@dataclass
class SyntheticDatasetSpec:
    output_root: str
    renderer_backends: list[str] = field(default_factory=list)
    splits: dict[str, int] = field(default_factory=dict)
    annotation_formats: list[str] = field(default_factory=list)
    domain_randomization_sources: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class PerceptionConfig(VersionedEntity):
    robot_genome_id: str = ""
    task_graph_id: str = ""
    backend: str = "mujoco"
    sensor_streams: list[SensorStream] = field(default_factory=list)
    vision_annotations: list[VisionAnnotationSpec] = field(default_factory=list)
    synthetic_dataset: SyntheticDatasetSpec | None = None
    texture_randomization: list[TextureRandomizationRule] = field(default_factory=list)
    lighting_randomization: list[str] = field(default_factory=list)
    scene_bindings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.robot_genome_id, "robot_genome_id", "Perception config must reference a robot.")
        require_non_empty(result, self.task_graph_id, "task_graph_id", "Perception config must reference a task.")
        require_non_empty(result, self.backend, "backend", "Perception config must define a backend.")
        require_non_empty(result, self.sensor_streams, "sensor_streams", "Perception config must include sensor streams.")
        require_non_empty(result, self.vision_annotations, "vision_annotations", "Perception config must declare CV annotations.")
        require_non_empty(result, self.synthetic_dataset, "synthetic_dataset", "Perception config must include a synthetic dataset spec.")
        require_non_empty(result, self.scene_bindings, "scene_bindings", "Perception config must bind to scene sets.")
        for index, stream in enumerate(self.sensor_streams):
            if not stream.name:
                result.add("missing_stream_name", "Sensor stream name is required.", f"sensor_streams[{index}].name")
            if not stream.sensor_component_id:
                result.add("missing_sensor_component", "Sensor component id is required.", f"sensor_streams[{index}].sensor_component_id")
            if not stream.output_modalities:
                result.add("missing_modalities", "Sensor stream must declare output modalities.", f"sensor_streams[{index}].output_modalities")
        for index, annotation in enumerate(self.vision_annotations):
            if not annotation.annotation_modalities:
                result.add("missing_annotation_modalities", "Vision annotation must declare modalities.", f"vision_annotations[{index}].annotation_modalities")
            if not annotation.label_space:
                result.add("missing_label_space", "Vision annotation must declare labels.", f"vision_annotations[{index}].label_space")
        return result
