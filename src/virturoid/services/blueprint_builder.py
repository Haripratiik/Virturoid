from __future__ import annotations

from virturoid.schemas.base import ArtifactRef
from virturoid.schemas.build_blueprint import BlueprintDecision, BlueprintHandoff, RobotBuildBlueprint
from virturoid.schemas.cad import CadAssembly, CadModel
from virturoid.schemas.components import BillOfMaterials, Component
from virturoid.schemas.fabrication import FabricationBuildPlan
from virturoid.schemas.morphology import MorphologyTemplate
from virturoid.schemas.part_resolution import PartResolutionReport
from virturoid.schemas.requirements import RequirementsRecord
from virturoid.schemas.robot import RobotGenome
from virturoid.schemas.scenes import SceneSet
from virturoid.schemas.tasks import TaskGraph
from virturoid.services.morphology_selector import MorphologySelection


def build_robot_build_blueprint(
    requirements: RequirementsRecord,
    task: TaskGraph,
    robot: RobotGenome,
    morphology_selection: MorphologySelection,
    bom: BillOfMaterials,
    components: list[Component],
    part_resolution: PartResolutionReport,
    cad_models: list[CadModel],
    cad_assembly: CadAssembly,
    fabrication_plan: FabricationBuildPlan,
    scene_sets: list[SceneSet],
) -> RobotBuildBlueprint:
    template = morphology_selection.selected_template
    blueprint = RobotBuildBlueprint(
        id=f"blueprint_{requirements.id}",
        requirements_id=requirements.id,
        task_graph_id=task.id,
        robot_genome_id=robot.id,
        morphology_template_id=template.id,
        robot_class=template.robot_class,
        morphology=robot.species,
        candidate_morphology_template_ids=list(morphology_selection.candidate_template_ids),
        autonomy_scope=[
            "parse user prompt into structured requirements",
            "select a robot morphology template from an extensible template catalog",
            "select parts from curated component database",
            "generate robot genome and CAD handoff",
            "generate baseline/variation/regression/holdout/revision scene sets",
            "compile generated scenes into simulator XML placeholders",
            "emit simulator-ready curriculum and dry-run traces",
        ],
        decisions=[
            _morphology_decision(requirements, task, robot, template, morphology_selection),
            _part_decision(part_resolution, components),
            _cad_decision(cad_models, cad_assembly, fabrication_plan),
            _scene_decision(task, scene_sets),
            _training_decision(task),
        ],
        handoffs=[
            BlueprintHandoff(
                stage="requirements_to_parts",
                description="Resolve named or recommended components with technical limits and CAD asset references.",
                inputs=[ArtifactRef(uri="project.json", media_type="application/json")],
                outputs=[
                    ArtifactRef(uri="bom/bom.json", media_type="application/json"),
                    ArtifactRef(uri="bom/part_resolution_report.json", media_type="application/json"),
                ],
            ),
            BlueprintHandoff(
                stage="parts_to_robot",
                description="Build structured robot genome, parametric CAD source, exact placeholder assembly, and visual meshes.",
                inputs=[
                    ArtifactRef(uri="bom/bom.json", media_type="application/json"),
                    ArtifactRef(uri="cad/build_plan.json", media_type="application/json"),
                ],
                outputs=[
                    ArtifactRef(uri="robot/robot_genome.json", media_type="application/json"),
                    ArtifactRef(uri="robot/robot.urdf", media_type="application/xml"),
                    ArtifactRef(uri="cad/parametric/robot_arm.py", media_type="text/x-python"),
                    ArtifactRef(uri="cad/exact/robot_assembly.step", media_type="model/step"),
                ],
            ),
            BlueprintHandoff(
                stage="task_to_scene",
                description="Generate train/evaluate scene families from task criteria, failures, and holdout coverage.",
                inputs=[ArtifactRef(uri="training/training_objective.json", media_type="application/json")],
                outputs=[
                    ArtifactRef(uri="simulation/scene_set.json", media_type="application/json"),
                    ArtifactRef(uri="simulation/mujoco/compiled_scene_index.json", media_type="application/json"),
                    ArtifactRef(uri="training/training_curriculum.json", media_type="application/json"),
                ],
            ),
            BlueprintHandoff(
                stage="scene_to_training",
                description="Run a deterministic simulator-adapter dry run over generated scenes and policy steps.",
                inputs=[
                    ArtifactRef(uri="training/training_run_config.json", media_type="application/json"),
                    ArtifactRef(uri="software/policy_plan.json", media_type="application/json"),
                ],
                outputs=[
                    ArtifactRef(uri="runs/mvp_training/dry_run_result.json", media_type="application/json"),
                    ArtifactRef(uri="runs/mvp_training/logs/episode_trace.jsonl", media_type="application/x-ndjson"),
                ],
            ),
        ],
        required_builder_capabilities=[
            "component_database_lookup",
            "parametric_cad_generation",
            "urdf_export",
            "mjcf_scene_export",
            "curriculum_generation",
            "simulator_adapter_execution",
            "feedback_revision_loop",
        ],
        open_limitations=[
            "CAD and MJCF are structured placeholders, not physics-validated production geometry yet.",
            "Dry-run training validates package contracts but does not execute real dynamics.",
            "Part database is curated fixture data; live datasheet ingestion is still future work.",
        ],
    )
    validation = blueprint.validate()
    if not validation.ok:
        issues = ", ".join(issue.code for issue in validation.issues)
        raise ValueError(f"{blueprint.id} failed validation: {issues}")
    return blueprint


def build_robot_arm_blueprint(*args, **kwargs) -> RobotBuildBlueprint:
    return build_robot_build_blueprint(*args, **kwargs)


def _morphology_decision(
    requirements: RequirementsRecord,
    task: TaskGraph,
    robot: RobotGenome,
    template: MorphologyTemplate,
    selection: MorphologySelection,
) -> BlueprintDecision:
    return BlueprintDecision(
        area="robot_morphology",
        decision=f"Use {template.id} ({robot.species}) with {len(robot.joints)} actuated joints.",
        rationale=(
            f"{selection.rationale} This template supports task_type={task.task_type} and keeps the MVP bounded "
            "while preserving a catalog route for mobile bases, humanoids, and future robot classes."
        ),
        source_refs=[requirements.id, robot.id, template.id],
    )


def _part_decision(part_resolution: PartResolutionReport, components: list[Component]) -> BlueprintDecision:
    names = {component.id: component.normalized_name for component in components}
    selected = [
        f"{part.role}:{names.get(part.component_id, part.component_id)}"
        for part in part_resolution.resolved_parts
    ]
    return BlueprintDecision(
        area="part_selection",
        decision="Select " + ", ".join(selected),
        rationale="Parts satisfy named user requests when present, otherwise use requirement-aware recommendations.",
        source_refs=["bom/part_resolution_report.json"],
    )


def _cad_decision(
    cad_models: list[CadModel],
    cad_assembly: CadAssembly,
    fabrication_plan: FabricationBuildPlan,
) -> BlueprintDecision:
    return BlueprintDecision(
        area="cad_handoff",
        decision=f"Generate {len(cad_models)} CAD model records and {len(cad_assembly.instances)} assembly instances.",
        rationale=(
            "The MVP keeps exact geometry, parametric source, visual meshes, and assembly operations as separate "
            f"handoffs; estimated printed mass is {fabrication_plan.estimated_printed_mass_kg} kg."
        ),
        source_refs=["cad/build_plan.json", "cad/parametric/robot_arm.py"],
    )


def _scene_decision(task: TaskGraph, scene_sets: list[SceneSet]) -> BlueprintDecision:
    scene_summary = ", ".join(f"{scene_set.purpose}:{len(scene_set.scenes)}" for scene_set in scene_sets)
    return BlueprintDecision(
        area="scene_generation",
        decision=f"Generate scene families for {task.task_type}: {scene_summary}.",
        rationale="Scene families separate smoke tests, training variation, failure replay, and holdout generalization.",
        source_refs=["simulation/scene_set.json", "simulation/regression_scene_set.json"],
    )


def _training_decision(task: TaskGraph) -> BlueprintDecision:
    return BlueprintDecision(
        area="training_handoff",
        decision=f"Use measurable task criteria as reward, termination, curriculum, and trace contracts for {task.task_type}.",
        rationale="A real simulator runner can replace the dry-run adapter while keeping the same artifact inputs.",
        source_refs=["training/training_objective.json", "training/training_curriculum.json"],
    )
