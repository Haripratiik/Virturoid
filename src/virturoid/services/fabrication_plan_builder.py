from __future__ import annotations

from virturoid.schemas.base import ArtifactRef
from virturoid.schemas.cad import CadAssembly, CadModel
from virturoid.schemas.components import BillOfMaterials, Component
from virturoid.schemas.fabrication import AssemblyOperation, FabricationBuildPlan, FabricationCheck, PartCadBinding
from virturoid.schemas.part_resolution import PartResolutionReport
from virturoid.schemas.requirements import RequirementsRecord
from virturoid.schemas.robot import RobotGenome


def build_fabrication_plan(
    requirements: RequirementsRecord,
    robot: RobotGenome,
    bom: BillOfMaterials,
    components: list[Component],
    part_resolution: PartResolutionReport,
    cad_models: list[CadModel],
    cad_assembly: CadAssembly,
) -> FabricationBuildPlan:
    plan = FabricationBuildPlan(
        id=f"fabrication_{requirements.id}",
        requirements_id=requirements.id,
        robot_genome_id=robot.id,
        cad_assembly_id=cad_assembly.id,
        bom_id=bom.id,
        parametric_source=ArtifactRef(uri="cad/parametric/robot_arm.py", media_type="text/x-python"),
        exact_assembly_artifact=ArtifactRef(uri="cad/exact/robot_assembly.step", media_type="model/step"),
        visual_mesh_dir=ArtifactRef(uri="cad/mesh/visual", media_type="model/stl"),
        part_cad_bindings=_part_cad_bindings(part_resolution, components),
        assembly_operations=_assembly_operations(cad_assembly, robot),
        fabrication_checks=_fabrication_checks(requirements, components, cad_models, cad_assembly),
        generated_cad_parameters={
            model.id: dict(model.editable_parameters)
            for model in cad_models
        },
        estimated_printed_mass_kg=round(sum(model.mass_kg or 0.0 for model in cad_models), 3),
        manufacturing_notes=[
            "MVP CAD is parametric-source plus placeholder STEP/STL until a CAD kernel is connected.",
            "Use selected component CAD assets to replace simplified mounts before hardware fabrication.",
            "Keep actuator axes aligned with RobotGenome joint axes when generating exact CAD.",
        ],
    )
    validation = plan.validate()
    if not validation.ok:
        issues = ", ".join(issue.code for issue in validation.issues)
        raise ValueError(f"{plan.id} failed validation: {issues}")
    return plan


def _part_cad_bindings(part_resolution: PartResolutionReport, components: list[Component]) -> list[PartCadBinding]:
    by_id = {component.id: component for component in components}
    mount_by_role = {
        "joint_actuator": "cad_arm_base",
        "wrist_sensor": "cad_wrist_camera_mount",
        "gripper": "cad_gripper_mount",
        "main_compute": "cad_arm_base",
    }
    bindings = []
    for part in part_resolution.resolved_parts:
        component = by_id.get(part.component_id)
        bindings.append(
            PartCadBinding(
                role=part.role,
                component_id=part.component_id,
                component_cad_assets=list(component.cad_assets) if component else [],
                generated_mount_cad_model_id=mount_by_role.get(part.role),
                notes=[
                    "Exact supplier CAD should be inserted at this binding during CAD-kernel export.",
                    f"Selection source: {part.source}.",
                ],
            )
        )
    return bindings


def _assembly_operations(cad_assembly: CadAssembly, robot: RobotGenome) -> list[AssemblyOperation]:
    operations = [
        AssemblyOperation(
            step_id="generate_parametric_cad",
            operation="generate_cad_source",
            target="cad/parametric/robot_arm.py",
            inputs=[cad_assembly.id],
            outputs=["cad/exact/*.step", "cad/mesh/visual/*.stl"],
            notes=["Generate link and mount solids from editable CAD parameters."],
        ),
    ]
    for instance in cad_assembly.instances:
        operations.append(
            AssemblyOperation(
                step_id=f"place_{instance.instance_id}",
                operation="place_assembly_instance",
                target=instance.instance_id,
                inputs=[instance.cad_model_id],
                outputs=[instance.role],
                notes=[f"Parent instance: {instance.parent_instance_id or 'world'}."],
            )
        )
    for joint in cad_assembly.joints:
        matching_robot_joint = next((item for item in robot.joints if item.name == joint.name), None)
        operations.append(
            AssemblyOperation(
                step_id=f"align_{joint.name}",
                operation="align_joint_axis",
                target=joint.name,
                inputs=[joint.parent_instance_id, joint.child_instance_id],
                outputs=[matching_robot_joint.child_link if matching_robot_joint else joint.child_instance_id],
                notes=[
                    f"Axis: {joint.axis_xyz}.",
                    f"Limit: {joint.limit}.",
                ],
            )
        )
    return operations


def _fabrication_checks(
    requirements: RequirementsRecord,
    components: list[Component],
    cad_models: list[CadModel],
    cad_assembly: CadAssembly,
) -> list[FabricationCheck]:
    reach_m = requirements.reach_m or 0.65
    link_length_m = sum(
        (model.bounding_box_mm[0] / 1000.0)
        for model in cad_models
        if model.id in {"cad_upper_link", "cad_forearm_link"} and model.bounding_box_mm
    )
    checks = [
        FabricationCheck(
            check="reach_from_generated_links",
            status="pass" if link_length_m >= reach_m * 0.75 else "warning",
            reason=f"Generated upper/forearm link length totals {link_length_m:.3f} m for requested reach {reach_m:.3f} m.",
            suggested_action=None if link_length_m >= reach_m * 0.75 else "Increase generated link length parameters.",
        ),
        FabricationCheck(
            check="assembly_instance_count",
            status="pass" if len(cad_assembly.instances) >= 5 else "warning",
            reason=f"Assembly contains {len(cad_assembly.instances)} generated instances.",
        ),
        FabricationCheck(
            check="component_cad_asset_coverage",
            status="pass" if any(component.cad_assets for component in components) else "warning",
            reason=f"{sum(1 for component in components if component.cad_assets)} selected components include CAD asset references.",
            suggested_action="Add supplier STEP files for all selected parts." if any(not component.cad_assets for component in components) else None,
        ),
        FabricationCheck(
            check="off_the_shelf_part_interfaces",
            status="warning",
            reason="MVP mount geometry is simplified and has not yet fitted real supplier CAD.",
            suggested_action="Run collision/clearance checks after CAD-kernel integration.",
        ),
    ]
    return checks
