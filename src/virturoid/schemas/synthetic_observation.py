from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ValidationResult, VersionedEntity, require_non_empty


@dataclass
class SyntheticObjectAnnotation:
    name: str
    label: str
    object_type: str
    pose_xyz_rpy: tuple[float, float, float, float, float, float]
    bbox_xywh: tuple[int, int, int, int] | None = None
    depth_m: float | None = None
    segmentation_id: int | None = None
    physical_parameters: dict[str, float] = field(default_factory=dict)


@dataclass
class SyntheticObservationFrame(VersionedEntity):
    scene_id: str = ""
    scene_set_id: str = ""
    purpose: str = ""
    sensor_name: str = ""
    sensor_type: str = ""
    modalities: list[str] = field(default_factory=list)
    image_size_px: tuple[int, int] | None = None
    expected_outputs: dict[str, str] = field(default_factory=dict)
    annotations: list[SyntheticObjectAnnotation] = field(default_factory=list)
    world_state_uri: str = ""

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.scene_id, "scene_id", "Observation frame must reference a scene.")
        require_non_empty(result, self.sensor_name, "sensor_name", "Observation frame must name a sensor.")
        require_non_empty(result, self.modalities, "modalities", "Observation frame must declare modalities.")
        require_non_empty(result, self.expected_outputs, "expected_outputs", "Observation frame must declare expected outputs.")
        require_non_empty(result, self.annotations, "annotations", "Observation frame must include object annotations.")
        require_non_empty(result, self.world_state_uri, "world_state_uri", "Observation frame must reference world state.")
        return result


@dataclass
class SyntheticObservationIndex(VersionedEntity):
    world_model_contract_uri: str = ""
    perception_config_uri: str = ""
    frame_count: int = 0
    annotation_formats: list[str] = field(default_factory=list)
    frames: list[dict[str, str | int]] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.world_model_contract_uri, "world_model_contract_uri", "Index must reference world model.")
        require_non_empty(result, self.perception_config_uri, "perception_config_uri", "Index must reference perception config.")
        require_non_empty(result, self.annotation_formats, "annotation_formats", "Index must declare annotation formats.")
        require_non_empty(result, self.frames, "frames", "Index must include frame records.")
        if self.frame_count != len(self.frames):
            result.add("frame_count_mismatch", "Frame count must match number of frame records.", "frame_count")
        return result
