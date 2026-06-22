from __future__ import annotations

from dataclasses import dataclass

from virturoid.schemas.components import Component
from virturoid.schemas.morphology import MorphologyTemplate
from virturoid.schemas.requirements import RequirementsRecord
from virturoid.schemas.robot_builder import BuilderTemplateStatus, RobotBuilderDispatchRecord
from virturoid.services.morphology_selector import MorphologySelection
from virturoid.services.mobile_base_builder import build_differential_mobile_base
from virturoid.services.robot_arm_builder import RobotArmBuild, build_reference_robot_arm


IMPLEMENTED_BUILDERS = {
    "virturoid.services.robot_arm_builder.build_reference_robot_arm": build_reference_robot_arm,
    "virturoid.services.mobile_base_builder.build_differential_mobile_base": build_differential_mobile_base,
}


@dataclass
class RobotBuilderResult:
    robot_build: RobotArmBuild
    dispatch_record: RobotBuilderDispatchRecord


def build_robot_from_selection(
    requirements: RequirementsRecord,
    component_library: list[Component],
    morphology_selection: MorphologySelection,
    morphology_templates: list[MorphologyTemplate],
) -> RobotBuilderResult:
    template = morphology_selection.selected_template
    dispatch_record = build_dispatch_record(morphology_selection, morphology_templates)
    if dispatch_record.dispatch_status != "implemented":
        raise NotImplementedError(
            f"No implemented builder for morphology template {template.id} ({template.builder_service})."
        )
    builder = IMPLEMENTED_BUILDERS[template.builder_service]
    robot_build = builder(requirements, component_library, morphology_template=template)
    return RobotBuilderResult(robot_build=robot_build, dispatch_record=dispatch_record)


def build_dispatch_record(
    morphology_selection: MorphologySelection,
    morphology_templates: list[MorphologyTemplate],
) -> RobotBuilderDispatchRecord:
    selected = morphology_selection.selected_template
    template_statuses = [
        BuilderTemplateStatus(
            morphology_template_id=template.id,
            robot_class=template.robot_class,
            builder_service=template.builder_service,
            status="implemented" if template.builder_service in IMPLEMENTED_BUILDERS else "planned",
            notes=list(template.notes),
        )
        for template in morphology_templates
    ]
    implemented = [item for item in template_statuses if item.status == "implemented"]
    planned = [item for item in template_statuses if item.status == "planned"]
    dispatch_status = "implemented" if selected.builder_service in IMPLEMENTED_BUILDERS else "planned"
    record = RobotBuilderDispatchRecord(
        id=f"builder_dispatch_{selected.id}",
        selected_morphology_template_id=selected.id,
        selected_robot_class=selected.robot_class,
        selected_builder_service=selected.builder_service,
        dispatch_status=dispatch_status,
        implemented_templates=implemented,
        planned_templates=planned,
        rationale=(
            f"Morphology selector chose {selected.id}. "
            f"Builder service {selected.builder_service} is {dispatch_status}; "
            f"{len(implemented)} template(s) implemented and {len(planned)} planned."
        ),
    )
    validation = record.validate()
    if not validation.ok:
        issues = ", ".join(issue.code for issue in validation.issues)
        raise ValueError(f"{record.id} failed validation: {issues}")
    return record
