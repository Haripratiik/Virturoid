from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ValidationResult, VersionedEntity, require_non_empty


@dataclass
class JointLimit:
    lower: float | None = None
    upper: float | None = None
    velocity: float | None = None
    effort: float | None = None


@dataclass
class RobotJoint:
    name: str
    joint_type: str
    parent_link: str
    child_link: str
    axis_xyz: tuple[float, float, float] | None = None
    limit: JointLimit | None = None
    actuator_component_id: str | None = None


@dataclass
class RobotSensor:
    name: str
    sensor_component_id: str
    parent_link: str
    transform_xyz_rpy: tuple[float, float, float, float, float, float]


@dataclass
class SpeciesNode(VersionedEntity):
    name: str = ""
    parent_species: str | None = None
    morphology_tags: list[str] = field(default_factory=list)
    common_tasks: list[str] = field(default_factory=list)
    common_failures: list[str] = field(default_factory=list)


@dataclass
class RobotGenome(VersionedEntity):
    name: str = ""
    species: str = ""
    morphology_template_id: str | None = None
    source_cad_assembly_id: str | None = None
    links: list[str] = field(default_factory=list)
    joints: list[RobotJoint] = field(default_factory=list)
    sensors: list[RobotSensor] = field(default_factory=list)
    end_effectors: list[str] = field(default_factory=list)
    supported_backends: dict[str, str] = field(default_factory=dict)
    known_skills: list[str] = field(default_factory=list)
    known_failures: list[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.name, "name", "Robot name is required.")
        require_non_empty(result, self.species, "species", "Robot species is required.")
        require_non_empty(result, self.links, "links", "Robot must have links.")
        link_set = set(self.links)
        for index, joint in enumerate(self.joints):
            if joint.parent_link not in link_set:
                result.add("missing_parent_link", "Joint parent link not found.", f"joints[{index}].parent_link")
            if joint.child_link not in link_set:
                result.add("missing_child_link", "Joint child link not found.", f"joints[{index}].child_link")
        for index, sensor in enumerate(self.sensors):
            if sensor.parent_link not in link_set:
                result.add("missing_sensor_link", "Sensor parent link not found.", f"sensors[{index}].parent_link")
        return result
