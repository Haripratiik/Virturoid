from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ValidationResult, VersionedEntity, require_non_empty


@dataclass
class ObservableEntity:
    name: str
    object_type: str
    source_scene_ids: list[str] = field(default_factory=list)
    observation_keys: list[str] = field(default_factory=list)
    annotation_labels: list[str] = field(default_factory=list)
    state_variables: list[str] = field(default_factory=list)
    physical_parameters: list[str] = field(default_factory=list)


@dataclass
class WorldStateVariable:
    key: str
    source: str
    units: str | None = None
    used_by: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class PhysicalParameterTarget:
    name: str
    target: str
    initial_source: str
    learning_signal: str
    optimization_role: str


@dataclass
class WorldModelContract(VersionedEntity):
    robot_genome_id: str = ""
    task_graph_id: str = ""
    perception_config_id: str = ""
    scene_set_ids: list[str] = field(default_factory=list)
    observable_entities: list[ObservableEntity] = field(default_factory=list)
    state_variables: list[WorldStateVariable] = field(default_factory=list)
    physical_parameter_targets: list[PhysicalParameterTarget] = field(default_factory=list)
    synthetic_dataset_manifest_uri: str = ""
    policy_observation_keys: list[str] = field(default_factory=list)
    memory_key: str = ""
    notes: list[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.robot_genome_id, "robot_genome_id", "World model must reference a robot.")
        require_non_empty(result, self.task_graph_id, "task_graph_id", "World model must reference a task.")
        require_non_empty(result, self.perception_config_id, "perception_config_id", "World model must reference perception config.")
        require_non_empty(result, self.scene_set_ids, "scene_set_ids", "World model must bind to scene sets.")
        require_non_empty(result, self.observable_entities, "observable_entities", "World model must include observable entities.")
        require_non_empty(result, self.state_variables, "state_variables", "World model must declare state variables.")
        require_non_empty(
            result,
            self.physical_parameter_targets,
            "physical_parameter_targets",
            "World model must declare physical parameters for Physical AI improvement.",
        )
        require_non_empty(
            result,
            self.synthetic_dataset_manifest_uri,
            "synthetic_dataset_manifest_uri",
            "World model must reference a synthetic observation dataset manifest.",
        )
        return result
