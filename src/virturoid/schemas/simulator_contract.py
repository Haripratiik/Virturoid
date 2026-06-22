from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ArtifactRef, ValidationResult, VersionedEntity, require_non_empty


@dataclass
class SimulatorBackendCapability:
    backend: str
    adapter_name: str
    status: str
    available: bool
    reason: str = ""


@dataclass
class SimulatorSceneContract:
    scene_id: str
    purpose: str
    compiled_scene_xml_uri: str
    xml_parse_status: str
    object_count: int = 0


@dataclass
class SimulatorRunContract(VersionedEntity):
    backend: str = "mujoco"
    robot_genome_id: str = ""
    training_run_config_uri: str = ""
    compiled_scene_index_uri: str = ""
    required_inputs: list[ArtifactRef] = field(default_factory=list)
    backend_capabilities: list[SimulatorBackendCapability] = field(default_factory=list)
    scene_contracts: list[SimulatorSceneContract] = field(default_factory=list)
    runner_requirements: list[str] = field(default_factory=list)
    blocking_gaps: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def real_backend_available(self) -> bool:
        return any(item.available and item.status == "real_backend_available" for item in self.backend_capabilities)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.backend, "backend", "Simulator contract must define a backend.")
        require_non_empty(result, self.robot_genome_id, "robot_genome_id", "Simulator contract must reference a robot.")
        require_non_empty(result, self.training_run_config_uri, "training_run_config_uri", "Simulator contract must reference a training config.")
        require_non_empty(result, self.compiled_scene_index_uri, "compiled_scene_index_uri", "Simulator contract must reference compiled scenes.")
        require_non_empty(result, self.required_inputs, "required_inputs", "Simulator contract must list required inputs.")
        require_non_empty(result, self.backend_capabilities, "backend_capabilities", "Simulator contract must list backend capabilities.")
        require_non_empty(result, self.scene_contracts, "scene_contracts", "Simulator contract must include scene contracts.")
        require_non_empty(result, self.runner_requirements, "runner_requirements", "Simulator contract must list runner requirements.")
        for index, scene in enumerate(self.scene_contracts):
            if scene.xml_parse_status != "pass":
                result.add("invalid_compiled_scene_xml", "Compiled scene XML must parse.", f"scene_contracts[{index}].xml_parse_status")
        return result
