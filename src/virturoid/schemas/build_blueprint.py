from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ArtifactRef, ValidationResult, VersionedEntity, require_non_empty


@dataclass
class BlueprintDecision:
    area: str
    decision: str
    rationale: str
    source_refs: list[str] = field(default_factory=list)


@dataclass
class BlueprintHandoff:
    stage: str
    description: str
    inputs: list[ArtifactRef] = field(default_factory=list)
    outputs: list[ArtifactRef] = field(default_factory=list)


@dataclass
class RobotBuildBlueprint(VersionedEntity):
    requirements_id: str = ""
    task_graph_id: str = ""
    robot_genome_id: str = ""
    morphology_template_id: str = ""
    robot_class: str = ""
    morphology: str = ""
    candidate_morphology_template_ids: list[str] = field(default_factory=list)
    autonomy_scope: list[str] = field(default_factory=list)
    decisions: list[BlueprintDecision] = field(default_factory=list)
    handoffs: list[BlueprintHandoff] = field(default_factory=list)
    required_builder_capabilities: list[str] = field(default_factory=list)
    open_limitations: list[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.requirements_id, "requirements_id", "Blueprint must reference requirements.")
        require_non_empty(result, self.task_graph_id, "task_graph_id", "Blueprint must reference a task graph.")
        require_non_empty(result, self.robot_genome_id, "robot_genome_id", "Blueprint must reference a robot genome.")
        require_non_empty(result, self.morphology_template_id, "morphology_template_id", "Blueprint must reference a morphology template.")
        require_non_empty(result, self.robot_class, "robot_class", "Blueprint must name a robot class.")
        require_non_empty(result, self.morphology, "morphology", "Blueprint must name the morphology.")
        require_non_empty(result, self.candidate_morphology_template_ids, "candidate_morphology_template_ids", "Blueprint must include candidate morphology templates.")
        require_non_empty(result, self.autonomy_scope, "autonomy_scope", "Blueprint must describe autonomous scope.")
        require_non_empty(result, self.decisions, "decisions", "Blueprint must include decisions.")
        require_non_empty(result, self.handoffs, "handoffs", "Blueprint must include builder handoffs.")
        for index, decision in enumerate(self.decisions):
            require_non_empty(result, decision.area, f"decisions[{index}].area", "Decision must name an area.")
            require_non_empty(result, decision.decision, f"decisions[{index}].decision", "Decision must describe the choice.")
            require_non_empty(result, decision.rationale, f"decisions[{index}].rationale", "Decision must include rationale.")
        for index, handoff in enumerate(self.handoffs):
            require_non_empty(result, handoff.stage, f"handoffs[{index}].stage", "Handoff must name a stage.")
            require_non_empty(result, handoff.outputs, f"handoffs[{index}].outputs", "Handoff must name outputs.")
        return result


RobotArmBlueprint = RobotBuildBlueprint
