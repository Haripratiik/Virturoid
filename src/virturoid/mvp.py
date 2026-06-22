from __future__ import annotations

import argparse
import json
from pathlib import Path

from virturoid.fixtures.components import curated_component_library
from virturoid.fixtures.morphologies import morphology_template_catalog
from virturoid.schemas.exports import ExportBundle
from virturoid.schemas.runs import PolicyRecord
from virturoid.services.blueprint_builder import build_robot_build_blueprint
from virturoid.services.compatibility import check_bom_compatibility
from virturoid.services.cad_placeholder import write_placeholder_cad_files
from virturoid.services.design_validator import validate_design, write_design_validation_report
from virturoid.services.dry_run_trainer import run_dry_training_from_export
from virturoid.services.evaluator import run_mock_evaluation, run_physics_evaluation_in_memory
from virturoid.services.export_writer import write_export_bundle
from virturoid.services.fabrication_plan_builder import build_fabrication_plan
from virturoid.services.improvement_report import write_improvement_report_from_export
from virturoid.services.mujoco_exporter import write_mujoco_scene_collection, write_mujoco_xml
from virturoid.services.morphology_selector import select_morphology_template
from virturoid.services.next_run_config import write_next_training_run_config_from_export
from virturoid.services.package_validator import validate_mvp_export_package, write_validation_report
from virturoid.services.perception_config_builder import build_perception_config
from virturoid.services.policy_plan_builder import build_policy_plan
from virturoid.services.power_architecture import build_power_architecture
from virturoid.services.promotion_decision import write_promotion_decision_from_export
from virturoid.services.report_generator import write_mvp_html_report, write_mvp_summary_report
from virturoid.services.requirements_builder import build_requirements_from_prompt
from virturoid.services.redesign_revision import apply_training_feedback_revision_from_export
from virturoid.services.robot_builder_registry import build_robot_from_selection
from virturoid.services.scene_generator import generate_regression_scene_set, generate_scene_set
from virturoid.services.scene_preview import write_scene_previews
from virturoid.services.simulator_contract_builder import write_simulator_contract_from_export
from virturoid.services.task_builder import build_task_graph
from virturoid.services.training_feedback import analyze_training_feedback_from_export
from virturoid.services.training_curriculum_builder import write_training_curriculum_from_export
from virturoid.services.training_manifest import build_training_manifest
from virturoid.services.training_objective_builder import build_training_objective
from virturoid.services.training_run_config import build_revised_training_run_config, build_training_run_config
from virturoid.services.urdf_exporter import write_robot_urdf
from virturoid.services.world_model_builder import (
    SYNTHETIC_OBSERVATION_INDEX_URI,
    SYNTHETIC_OBSERVATION_MANIFEST_URI,
    WORLD_MODEL_CONTRACT_URI,
    WORLD_STATE_INDEX_URI,
    build_world_model_contract,
    write_world_model_artifacts,
)
from virturoid.schemas.base import ArtifactRef
from virturoid.schemas.requirements import RequirementsRecord
from virturoid.schemas.scenes import SceneSet


def build_mvp_robot_arm_project(requirements: RequirementsRecord | None = None) -> dict:
    """Build the current autonomous MVP artifact graph.

    This function is the narrow first version of:
    requirements -> parts -> BOM -> CAD -> robot genome -> task -> scenes -> eval -> export.
    """
    requirements = requirements or RequirementsRecord(
        id="req_mvp_sorting_arm",
        prompt="Build a tabletop robot arm that sorts red and blue blocks into matching bins.",
        task_summary="Autonomously build and train/evaluate a tabletop block-sorting robot arm.",
        environment="tabletop",
        payload_kg=0.25,
        reach_m=0.65,
        precision_m=0.005,
        manufacturing_constraints=["3d_printable_mounts", "off_the_shelf_actuators"],
        sensor_requirements=["rgbd_camera"],
    )

    component_library = curated_component_library()
    morphology_templates = morphology_template_catalog()
    task = build_task_graph(requirements)
    morphology_selection = select_morphology_template(requirements, task, morphology_templates)
    builder_result = build_robot_from_selection(
        requirements,
        component_library,
        morphology_selection,
        morphology_templates,
    )
    robot_build = builder_result.robot_build
    fabrication_plan = build_fabrication_plan(
        requirements,
        robot_build.robot_genome,
        robot_build.bom,
        robot_build.components,
        robot_build.part_resolution,
        robot_build.cad_models,
        robot_build.cad_assembly,
    )
    compatibility_report = check_bom_compatibility(requirements, robot_build.bom, robot_build.components)
    power_architecture = build_power_architecture(robot_build.bom, robot_build.components)
    baseline_scene_set = generate_scene_set(task, count=2, purpose="baseline")
    scene_set = generate_scene_set(task, count=10, purpose="variation")
    # Activate the (previously dead) edge_case difficulty so the package meets the startup-plan's >=20-scene
    # bar across distinct purposes: 2 baseline + 10 variation + 3 edge_case + 3 holdout + regression = >=20.
    edge_case_scene_set = generate_scene_set(task, count=3, purpose="edge_case")
    policy = PolicyRecord(
        id="policy_mvp_scripted_sort",
        name="MVP scripted sorting baseline",
        compatible_robot_ids=[robot_build.robot_genome.id],
        action_space="discrete_skill_sequence",
        observation_space=["object_poses", "joint_positions", robot_build.robot_genome.sensors[0].name],
    )
    policy.artifact = ArtifactRef(uri="software/policy_plan.json", media_type="application/json")
    policy_plan = build_policy_plan(robot_build.robot_genome, task, policy)
    # REAL physics evaluation (plan §32.5): run the scripted controller under MuJoCo on the variation scenes PLUS
    # a STRESS tier (objects past the reliable grasp envelope + heavy) so genuine failures surface — then the
    # failure-conditioned regression set is built from REAL failures, not synthetic ones. Falls back to a
    # clearly-labelled 'mock' run only when MuJoCo is unavailable (so the package never silently fakes success).
    stress_scene_set = generate_scene_set(task, count=3, purpose="stress")
    eval_scene_set = SceneSet(id=f"scenes_{task.id}_evaluation", task_graph_id=task.id, purpose="variation",
                              scenes=list(scene_set.scenes) + list(stress_scene_set.scenes),
                              generation_rules=["variation + stress tier for honest pass/fail material"])
    evaluation_run = run_physics_evaluation_in_memory(robot_build.robot_genome, task, eval_scene_set, policy)
    regression_scene_set = generate_regression_scene_set(task, evaluation_run.failures, eval_scene_set)
    if not regression_scene_set.scenes:
        # No real failures (a robust robot, or a task the scripted controller doesn't exercise): keep the
        # regression stage meaningful by RE-TESTING the known-hard stress scenes — the cases the robot must
        # keep passing. Keeps the package's redesign/curriculum structure valid without faking failures.
        regression_scene_set = SceneSet(
            id=f"scenes_{task.id}_regression", task_graph_id=task.id, purpose="regression",
            scenes=list(stress_scene_set.scenes),
            generation_rules=["no failures surfaced; regression re-tests the known-hard stress scenes"])
    holdout_scene_set = generate_scene_set(task, count=3, purpose="holdout")
    training_manifest = build_training_manifest(
        robot_build.robot_genome,
        task,
        [baseline_scene_set, scene_set, edge_case_scene_set, regression_scene_set, holdout_scene_set],
        backend="mujoco",
    )
    training_objective = build_training_objective(robot_build.robot_genome, task)
    perception_config = build_perception_config(
        robot_build.robot_genome,
        task,
        robot_build.components,
        [baseline_scene_set, scene_set, edge_case_scene_set, regression_scene_set, holdout_scene_set],
        backend="mujoco",
    )
    world_model_contract = build_world_model_contract(
        robot_build.robot_genome,
        task,
        perception_config,
        [baseline_scene_set, scene_set, edge_case_scene_set, regression_scene_set, holdout_scene_set],
    )
    blueprint = build_robot_build_blueprint(
        requirements,
        task,
        robot_build.robot_genome,
        morphology_selection,
        robot_build.bom,
        robot_build.components,
        robot_build.part_resolution,
        robot_build.cad_models,
        robot_build.cad_assembly,
        fabrication_plan,
        [baseline_scene_set, scene_set, edge_case_scene_set, regression_scene_set, holdout_scene_set],
    )
    training_run_config = build_training_run_config(
        robot_build.robot_genome,
        task,
        policy,
        training_manifest,
        [baseline_scene_set, scene_set, edge_case_scene_set, regression_scene_set, holdout_scene_set],
        backend="mujoco",
    )
    revised_training_run_config = build_revised_training_run_config(
        robot_build.robot_genome,
        task,
        policy,
        backend="mujoco",
    )
    export_bundle = ExportBundle(
        id=f"export_{requirements.id}",
        project_id=f"project_{requirements.id}",
        design_artifacts=[
            ArtifactRef(uri="design/morphology_template.json", media_type="application/json"),
            ArtifactRef(uri="design/builder_dispatch.json", media_type="application/json"),
            ArtifactRef(uri="design/robot_build_blueprint.json", media_type="application/json"),
        ],
        cad_artifacts=[
            ArtifactRef(uri="cad/exact/robot_assembly.step", media_type="model/step"),
            ArtifactRef(uri="cad/parametric/robot_arm.py", media_type="text/x-python"),
            ArtifactRef(uri="cad/build_plan.json", media_type="application/json"),
            *[
                ArtifactRef(uri=f"cad/mesh/visual/{cad_model.id}.stl", media_type="model/stl")
                for cad_model in robot_build.cad_models
            ],
        ],
        bom_artifacts=[
            ArtifactRef(uri="bom/bom.json", media_type="application/json"),
            ArtifactRef(uri="bom/part_resolution_report.json", media_type="application/json"),
        ],
        robot_artifacts=[
            ArtifactRef(uri="power/power_architecture.json", media_type="application/json"),
            ArtifactRef(uri="robot/robot_genome.json", media_type="application/json"),
            ArtifactRef(uri="robot/robot.urdf", media_type="application/xml"),
        ],
        simulation_artifacts=[
            ArtifactRef(uri="simulation/baseline_scene_set.json", media_type="application/json"),
            ArtifactRef(uri="simulation/scene_set.json", media_type="application/json"),
            ArtifactRef(uri="simulation/regression_scene_set.json", media_type="application/json"),
            ArtifactRef(uri="simulation/holdout_scene_set.json", media_type="application/json"),
            ArtifactRef(uri="simulation/revision_scene_set.json", media_type="application/json"),
            ArtifactRef(uri="simulation/perception_config.json", media_type="application/json"),
            ArtifactRef(uri=WORLD_MODEL_CONTRACT_URI, media_type="application/json"),
            ArtifactRef(uri="simulation/simulator_contract.json", media_type="application/json"),
            ArtifactRef(uri="simulation/mujoco/mvp_scene.xml", media_type="application/xml"),
            ArtifactRef(uri="simulation/mujoco/compiled_scene_index.json", media_type="application/json"),
            ArtifactRef(uri="simulation/previews/index.html", media_type="text/html"),
            ArtifactRef(uri="simulation/previews/baseline_scene.svg", media_type="image/svg+xml"),
            ArtifactRef(uri="simulation/previews/variation_scene.svg", media_type="image/svg+xml"),
            ArtifactRef(uri="simulation/previews/regression_scene.svg", media_type="image/svg+xml"),
            ArtifactRef(uri="simulation/previews/holdout_scene.svg", media_type="image/svg+xml"),
        ],
        dataset_artifacts=[
            ArtifactRef(uri="training/training_manifest.json", media_type="application/json"),
            ArtifactRef(uri="training/training_objective.json", media_type="application/json"),
            ArtifactRef(uri="training/training_curriculum.json", media_type="application/json"),
            ArtifactRef(uri="training/training_run_config.json", media_type="application/json"),
            ArtifactRef(uri="training/training_run_config_revised.json", media_type="application/json"),
            ArtifactRef(uri="training/training_run_config_next.json", media_type="application/json"),
            ArtifactRef(uri="training/active_training_inputs.json", media_type="application/json"),
            ArtifactRef(uri=SYNTHETIC_OBSERVATION_MANIFEST_URI, media_type="application/json"),
            ArtifactRef(uri=SYNTHETIC_OBSERVATION_INDEX_URI, media_type="application/json"),
            ArtifactRef(uri=WORLD_STATE_INDEX_URI, media_type="application/json"),
            ArtifactRef(uri="runs/mvp_training/dry_run_result.json", media_type="application/json"),
            ArtifactRef(uri="runs/mvp_training/replay_index.json", media_type="application/json"),
            ArtifactRef(uri="runs/mvp_training/logs/episode_trace.jsonl", media_type="application/x-ndjson"),
            ArtifactRef(uri="runs/mvp_training_revision/dry_run_result.json", media_type="application/json"),
            ArtifactRef(uri="runs/mvp_training_revision/replay_index.json", media_type="application/json"),
            ArtifactRef(uri="runs/mvp_training_revision/logs/episode_trace.jsonl", media_type="application/x-ndjson"),
            ArtifactRef(uri="runs/mvp_training_next/dry_run_result.json", media_type="application/json"),
            ArtifactRef(uri="runs/mvp_training_next/replay_index.json", media_type="application/json"),
            ArtifactRef(uri="runs/mvp_training_next/logs/episode_trace.jsonl", media_type="application/x-ndjson"),
        ],
        software_artifacts=[
            ArtifactRef(uri="software/policy_plan.json", media_type="application/json"),
            ArtifactRef(uri="software/policy_plan_revised.json", media_type="application/json"),
            ArtifactRef(uri="software/ros2_test_harness_placeholder.json", media_type="application/json"),
        ],
        report_artifacts=[
            ArtifactRef(uri="reports/design_validation_report.json", media_type="application/json"),
            ArtifactRef(uri="reports/mvp_evaluation_report.json", media_type="application/json"),
            ArtifactRef(uri="reports/training_feedback.json", media_type="application/json"),
            ArtifactRef(uri="reports/redesign_revision.json", media_type="application/json"),
            ArtifactRef(uri="reports/improvement_report.json", media_type="application/json"),
            ArtifactRef(uri="reports/promotion_decision.json", media_type="application/json"),
            ArtifactRef(uri="reports/package_validation_report.json", media_type="application/json"),
            ArtifactRef(uri="reports/mvp_summary.md", media_type="text/markdown"),
            ArtifactRef(uri="reports/index.html", media_type="text/html"),
        ],
    )

    return {
        "requirements": requirements,
        "component_library": component_library,
        "morphology_templates": morphology_templates,
        "morphology_selection": morphology_selection,
        "morphology_template": morphology_selection.selected_template,
        "builder_dispatch": builder_result.dispatch_record,
        "selected_components": robot_build.components,
        "part_resolution": robot_build.part_resolution,
        "fabrication_plan": fabrication_plan,
        "compatibility_report": compatibility_report,
        "power_architecture": power_architecture,
        "bom": robot_build.bom,
        "cad_models": robot_build.cad_models,
        "cad_assembly": robot_build.cad_assembly,
        "robot_genome": robot_build.robot_genome,
        "task_graph": task,
        "baseline_scene_set": baseline_scene_set,
        "scene_set": scene_set,
        "edge_case_scene_set": edge_case_scene_set,
        "regression_scene_set": regression_scene_set,
        "holdout_scene_set": holdout_scene_set,
        "perception_config": perception_config,
        "world_model_contract": world_model_contract,
        "robot_build_blueprint": blueprint,
        "robot_arm_blueprint": blueprint,
        "training_manifest": training_manifest,
        "training_objective": training_objective,
        "training_run_config": training_run_config,
        "revised_training_run_config": revised_training_run_config,
        "policy": policy,
        "policy_plan": policy_plan,
        "evaluation_run": evaluation_run,
        "export_bundle": export_bundle,
    }


def write_mvp_robot_arm_project(output_dir: Path, requirements: RequirementsRecord | None = None) -> Path:
    project = build_mvp_robot_arm_project(requirements)
    bundle = project["export_bundle"]
    written_dir = write_export_bundle(bundle, project, output_dir)
    write_world_model_artifacts(
        written_dir,
        project["robot_genome"],
        project["task_graph"],
        project["perception_config"],
        [
            project["baseline_scene_set"],
            project["scene_set"],
            project["edge_case_scene_set"],
            project["regression_scene_set"],
            project["holdout_scene_set"],
        ],
    )
    write_placeholder_cad_files(project["cad_models"], project["cad_assembly"], written_dir)
    write_robot_urdf(project["robot_genome"], written_dir, cad_models=project["cad_models"])
    write_mujoco_xml(project["robot_genome"], project["scene_set"], written_dir, cad_models=project["cad_models"])
    write_scene_previews(
        [
            project["baseline_scene_set"],
            project["scene_set"],
            project["edge_case_scene_set"],
            project["regression_scene_set"],
            project["holdout_scene_set"],
        ],
        written_dir,
    )
    write_mujoco_scene_collection(
        project["robot_genome"],
        [
            project["baseline_scene_set"],
            project["scene_set"],
            project["edge_case_scene_set"],
            project["regression_scene_set"],
            project["holdout_scene_set"],
        ],
        written_dir,
        cad_models=project["cad_models"],
    )
    write_simulator_contract_from_export(written_dir)
    write_training_curriculum_from_export(written_dir)
    run_dry_training_from_export(written_dir)
    analyze_training_feedback_from_export(written_dir)
    apply_training_feedback_revision_from_export(written_dir)
    revision_scene_set = _load_scene_set_from_export(written_dir / "simulation" / "revision_scene_set.json")
    write_mujoco_scene_collection(
        project["robot_genome"],
        [
            project["baseline_scene_set"],
            project["scene_set"],
            project["edge_case_scene_set"],
            project["regression_scene_set"],
            project["holdout_scene_set"],
            revision_scene_set,
        ],
        written_dir,
        cad_models=project["cad_models"],
    )
    write_simulator_contract_from_export(written_dir)
    write_training_curriculum_from_export(written_dir)
    run_dry_training_from_export(written_dir, config_uri="training/training_run_config_revised.json")
    write_improvement_report_from_export(written_dir)
    write_promotion_decision_from_export(written_dir)
    write_next_training_run_config_from_export(written_dir)
    run_dry_training_from_export(written_dir, config_uri="training/training_run_config_next.json")
    design_validation = validate_design(
        project["requirements"],
        project["robot_genome"],
        project["cad_models"],
        project["selected_components"],
        [
            project["baseline_scene_set"],
            project["scene_set"],
            project["edge_case_scene_set"],
            project["regression_scene_set"],
            project["holdout_scene_set"],
            revision_scene_set,
        ],
        package_dir=written_dir,
    )
    write_design_validation_report(design_validation, written_dir)
    write_validation_report(written_dir)
    write_mvp_summary_report(project, written_dir)
    write_mvp_html_report(project, written_dir)
    return written_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the current Virturoid MVP robot-arm package.")
    parser.add_argument(
        "--prompt",
        default="Build a tabletop robot arm that sorts red and blue blocks into matching bins.",
        help="User task prompt for the autonomous robot-arm build.",
    )
    parser.add_argument(
        "--output",
        default=str(Path("build") / "mvp_robot_arm"),
        help="Output directory for the generated MVP package.",
    )
    parser.add_argument("--payload-kg", type=float, default=None, help="Optional payload override in kilograms.")
    parser.add_argument("--reach-m", type=float, default=None, help="Optional reach override in meters.")
    parser.add_argument("--sensor", default=None, help="Optional sensor requirement, e.g. rgbd_camera or lidar.")
    parser.add_argument(
        "--physics",
        action="store_true",
        help="After building the package, run real MuJoCo physics rollouts (requires `pip install mujoco`).",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train a reach controller against real MuJoCo rollouts and export a controller bundle (requires `pip install mujoco`).",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run a real MuJoCo pick-and-place sorting evaluation and cluster task failures (requires `pip install mujoco`).",
    )
    parser.add_argument(
        "--co-optimize",
        dest="co_optimize",
        action="store_true",
        help="Co-optimize the robot hardware (link lengths, actuator torque) against real task success (requires `pip install mujoco`).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    requirements = build_requirements_from_prompt(
        args.prompt,
        payload_kg=args.payload_kg,
        reach_m=args.reach_m,
        sensor=args.sensor,
    )
    write_mvp_robot_arm_project(output_dir, requirements)
    physics_summary = _maybe_run_physics(output_dir) if args.physics else None
    training_summary = _maybe_train_controller(output_dir) if args.train else None
    evaluation_summary = _maybe_evaluate_pick_place(output_dir) if args.evaluate else None
    codesign_summary = _maybe_co_optimize(output_dir, args.prompt) if args.co_optimize else None
    summary = {
        "output_dir": str(output_dir),
        "project_json": str(output_dir / "project.json"),
        "manifest_json": str(output_dir / "manifest.json"),
        "part_resolution_report_json": str(output_dir / "bom" / "part_resolution_report.json"),
        "fabrication_build_plan_json": str(output_dir / "cad" / "build_plan.json"),
        "parametric_cad_source": str(output_dir / "cad" / "parametric" / "robot_arm.py"),
        "power_architecture_json": str(output_dir / "power" / "power_architecture.json"),
        "morphology_template_json": str(output_dir / "design" / "morphology_template.json"),
        "builder_dispatch_json": str(output_dir / "design" / "builder_dispatch.json"),
        "robot_build_blueprint_json": str(output_dir / "design" / "robot_build_blueprint.json"),
        "baseline_scene_set_json": str(output_dir / "simulation" / "baseline_scene_set.json"),
        "scene_set_json": str(output_dir / "simulation" / "scene_set.json"),
        "regression_scene_set_json": str(output_dir / "simulation" / "regression_scene_set.json"),
        "holdout_scene_set_json": str(output_dir / "simulation" / "holdout_scene_set.json"),
        "revision_scene_set_json": str(output_dir / "simulation" / "revision_scene_set.json"),
        "perception_config_json": str(output_dir / "simulation" / "perception_config.json"),
        "world_model_contract_json": str(output_dir / WORLD_MODEL_CONTRACT_URI),
        "synthetic_observation_manifest_json": str(output_dir / SYNTHETIC_OBSERVATION_MANIFEST_URI),
        "synthetic_observation_index_json": str(output_dir / SYNTHETIC_OBSERVATION_INDEX_URI),
        "world_state_index_json": str(output_dir / WORLD_STATE_INDEX_URI),
        "simulator_contract_json": str(output_dir / "simulation" / "simulator_contract.json"),
        "training_manifest_json": str(output_dir / "training" / "training_manifest.json"),
        "training_objective_json": str(output_dir / "training" / "training_objective.json"),
        "training_curriculum_json": str(output_dir / "training" / "training_curriculum.json"),
        "training_run_config_json": str(output_dir / "training" / "training_run_config.json"),
        "revised_training_run_config_json": str(output_dir / "training" / "training_run_config_revised.json"),
        "next_training_run_config_json": str(output_dir / "training" / "training_run_config_next.json"),
        "active_training_inputs_json": str(output_dir / "training" / "active_training_inputs.json"),
        "policy_plan_json": str(output_dir / "software" / "policy_plan.json"),
        "revised_policy_plan_json": str(output_dir / "software" / "policy_plan_revised.json"),
        "dry_run_result_json": str(output_dir / "runs" / "mvp_training" / "dry_run_result.json"),
        "replay_index_json": str(output_dir / "runs" / "mvp_training" / "replay_index.json"),
        "episode_trace_jsonl": str(output_dir / "runs" / "mvp_training" / "logs" / "episode_trace.jsonl"),
        "revised_dry_run_result_json": str(output_dir / "runs" / "mvp_training_revision" / "dry_run_result.json"),
        "revised_replay_index_json": str(output_dir / "runs" / "mvp_training_revision" / "replay_index.json"),
        "revised_episode_trace_jsonl": str(output_dir / "runs" / "mvp_training_revision" / "logs" / "episode_trace.jsonl"),
        "next_dry_run_result_json": str(output_dir / "runs" / "mvp_training_next" / "dry_run_result.json"),
        "next_replay_index_json": str(output_dir / "runs" / "mvp_training_next" / "replay_index.json"),
        "next_episode_trace_jsonl": str(output_dir / "runs" / "mvp_training_next" / "logs" / "episode_trace.jsonl"),
        "robot_urdf": str(output_dir / "robot" / "robot.urdf"),
        "mujoco_xml": str(output_dir / "simulation" / "mujoco" / "mvp_scene.xml"),
        "compiled_mujoco_scene_index_json": str(output_dir / "simulation" / "mujoco" / "compiled_scene_index.json"),
        "scene_previews": str(output_dir / "simulation" / "previews" / "index.html"),
        "assembly_step": str(output_dir / "cad" / "exact" / "robot_assembly.step"),
        "visual_mesh_dir": str(output_dir / "cad" / "mesh" / "visual"),
        "validation_report_json": str(output_dir / "reports" / "package_validation_report.json"),
        "training_feedback_json": str(output_dir / "reports" / "training_feedback.json"),
        "redesign_revision_json": str(output_dir / "reports" / "redesign_revision.json"),
        "improvement_report_json": str(output_dir / "reports" / "improvement_report.json"),
        "promotion_decision_json": str(output_dir / "reports" / "promotion_decision.json"),
        "summary_report_md": str(output_dir / "reports" / "mvp_summary.md"),
        "html_report": str(output_dir / "reports" / "index.html"),
        "package_valid": validate_mvp_export_package(output_dir).ok,
    }
    if physics_summary is not None:
        summary["physics"] = physics_summary
    if training_summary is not None:
        summary["training"] = training_summary
    if evaluation_summary is not None:
        summary["pick_place_evaluation"] = evaluation_summary
    if codesign_summary is not None:
        summary["hardware_codesign"] = codesign_summary
    print(json.dumps(summary, indent=2))


def _maybe_co_optimize(output_dir: Path, prompt: str) -> dict:
    from virturoid.services.codesign_optimizer import co_optimize_hardware, write_codesign_report
    from virturoid.services.mujoco_runner import mujoco_available

    if not mujoco_available():
        return {"ran": False, "reason": "MuJoCo not installed; run `pip install mujoco` to co-optimize hardware."}
    report = co_optimize_hardware(prompt=prompt)
    write_codesign_report(report, output_dir)
    return {
        "ran": True,
        "baseline_success_rate": report.baseline.success_rate,
        "best_success_rate": report.best.success_rate,
        "success_improvement": report.success_improvement,
        "candidates_evaluated": report.candidates_evaluated,
        "changed_parameters": report.changed_parameters,
        "codesign_report_json": str(output_dir / "reports" / "codesign_optimization_report.json"),
    }


def _maybe_evaluate_pick_place(output_dir: Path) -> dict:
    from virturoid.services.mujoco_runner import mujoco_available
    from virturoid.services.physics_evaluator import run_physics_pick_place_evaluation

    if not mujoco_available():
        return {
            "ran": False,
            "reason": "MuJoCo Python package not installed; run `pip install mujoco` to evaluate pick-and-place.",
        }
    report = run_physics_pick_place_evaluation(output_dir)
    return {
        "ran": True,
        "controller": report.controller,
        "total_episodes": report.total_episodes,
        "success_rate": report.success_rate,
        "blocks_placed": report.blocks_placed,
        "blocks_total": report.blocks_total,
        "success_rate_by_purpose": report.success_rate_by_purpose,
        "failure_clusters": [
            {"label": c.failure_label, "count": c.count, "suggested_regression": c.suggested_regression}
            for c in report.failure_clusters
        ],
        "physics_evaluation_report_json": str(output_dir / "reports" / "physics_evaluation_report.json"),
    }


def _maybe_train_controller(output_dir: Path) -> dict:
    from virturoid.services.controller_exporter import export_controller_bundle
    from virturoid.services.mujoco_runner import mujoco_available
    from virturoid.services.policy_trainer import train_arm_controller_from_export

    if not mujoco_available():
        return {
            "ran": False,
            "reason": "MuJoCo Python package not installed; run `pip install mujoco` to train a controller.",
        }
    policy = train_arm_controller_from_export(output_dir)
    bundle_path = export_controller_bundle(output_dir, policy)
    from virturoid.services.ros2_exporter import maybe_export_ros2_package
    maybe_export_ros2_package(output_dir)   # runnable ROS2 harness for the exported controller (§24)
    return {
        "ran": True,
        "policy_id": policy.id,
        "method": policy.training.method,
        "initial_reward": policy.training.initial_reward,
        "best_reward": policy.training.best_reward,
        "eval_mean_reach_distance_m": policy.evaluation.mean_reach_distance_m,
        "eval_success_rate": policy.evaluation.success_rate,
        "trained_policy_json": str(output_dir / "software" / "controller" / "trained_policy.json"),
        "controller_bundle_json": str(bundle_path),
        "controller_entrypoint": str(output_dir / "software" / "controller" / "controller.py"),
        "eval_episode_trace": str(output_dir / "runs" / "mvp_policy_eval" / "logs" / "episode_trace.jsonl"),
    }


def _maybe_run_physics(output_dir: Path) -> dict:
    from virturoid.services.mujoco_runner import mujoco_available, run_physics_rollout_from_export

    if not mujoco_available():
        return {
            "ran": False,
            "reason": "MuJoCo Python package not installed; run `pip install mujoco` to enable real rollouts.",
        }
    result = run_physics_rollout_from_export(output_dir)
    return {
        "ran": True,
        "adapter_name": result.adapter_name,
        "total_episodes": result.total_episodes,
        "measured_success_rate": result.measured_success_rate,
        "stable_episode_rate": result.stable_episode_rate,
        "physics_run_result_json": str(output_dir / "runs" / "mvp_physics" / "physics_run_result.json"),
        "episode_trace_jsonl": str(output_dir / result.episode_trace_uri),
        "rollout_dir": str(output_dir / result.rollout_dir),
    }


def _load_scene_set_from_export(path: Path):
    from virturoid.schemas.scenes import SceneGraph, SceneObject, SceneSet

    data = json.loads(path.read_text(encoding="utf-8"))
    return SceneSet(
        id=data["id"],
        version=data.get("version", "0.1.0"),
        task_graph_id=data["task_graph_id"],
        purpose=data["purpose"],
        scenes=[
            SceneGraph(
                id=scene["id"],
                version=scene.get("version", "0.1.0"),
                name=scene["name"],
                backend_targets=scene["backend_targets"],
                robot_spawn_xyz_rpy=tuple(scene["robot_spawn_xyz_rpy"]),
                objects=[
                    SceneObject(
                        name=item["name"],
                        object_type=item["object_type"],
                        pose_xyz_rpy=tuple(item["pose_xyz_rpy"]),
                        mass_kg=item.get("mass_kg"),
                        material=item.get("material"),
                        friction=item.get("friction"),
                        scale=item.get("scale"),
                    )
                    for item in scene["objects"]
                ],
                variation_parameters=scene.get("variation_parameters", {}),
                requirement_trace=scene.get("requirement_trace", []),
            )
            for scene in data["scenes"]
        ],
        generation_rules=data.get("generation_rules", []),
    )


if __name__ == "__main__":
    main()
