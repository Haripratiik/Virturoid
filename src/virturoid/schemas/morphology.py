from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ValidationResult, VersionedEntity, require_non_empty


@dataclass
class MorphologyModule:
    role: str
    description: str
    required: bool = True
    compatible_component_categories: list[str] = field(default_factory=list)


@dataclass
class MorphologyTemplate(VersionedEntity):
    robot_class: str = ""
    species_pattern: str = ""
    parent_species: str | None = None
    morphology_tags: list[str] = field(default_factory=list)
    supported_task_types: list[str] = field(default_factory=list)
    required_modules: list[MorphologyModule] = field(default_factory=list)
    default_links: list[str] = field(default_factory=list)
    default_skills: list[str] = field(default_factory=list)
    builder_service: str = ""
    notes: list[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.robot_class, "robot_class", "Morphology template must name a robot class.")
        require_non_empty(result, self.species_pattern, "species_pattern", "Morphology template must define a species pattern.")
        require_non_empty(result, self.morphology_tags, "morphology_tags", "Morphology template must include morphology tags.")
        require_non_empty(result, self.supported_task_types, "supported_task_types", "Morphology template must list supported task types.")
        require_non_empty(result, self.required_modules, "required_modules", "Morphology template must define required modules.")
        require_non_empty(result, self.builder_service, "builder_service", "Morphology template must name a builder service.")
        for index, module in enumerate(self.required_modules):
            require_non_empty(result, module.role, f"required_modules[{index}].role", "Module must name a role.")
            require_non_empty(result, module.description, f"required_modules[{index}].description", "Module must include a description.")
        return result
