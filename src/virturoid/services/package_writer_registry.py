from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from virturoid.schemas.morphology import MorphologyTemplate
from virturoid.schemas.requirements import RequirementsRecord
from virturoid.schemas.tasks import TaskGraph
from virturoid.services.package_validator import validate_mvp_export_package


@dataclass
class PackageWriterStatus:
    robot_class: str
    status: str
    writer_service: str
    package_type: str
    capabilities: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class PackageWriterResult:
    written_dir: Path
    package_type: str
    package_valid: bool
    artifacts: dict[str, str]
    training: dict | None
    notes: list[str]


IMPLEMENTED_PACKAGE_WRITERS = {
    "manipulator": PackageWriterStatus(
        robot_class="manipulator",
        status="implemented",
        writer_service="virturoid.mvp.write_mvp_robot_arm_project",
        package_type="mvp_manipulator_training_package",
        capabilities=[
            "cad_export",
            "robot_genome",
            "scene_generation",
            "compiled_simulation_scenes",
            "training_run_config",
            "dry_run_training",
            "policy_training_when_requested",
            "controller_export_when_requested",
        ],
        notes=["Full MVP route for the first trainable robot class."],
    ),
    "mobile_base": PackageWriterStatus(
        robot_class="mobile_base",
        status="implemented",
        writer_service="virturoid.mobile_base.build_mobile_base_project",
        package_type="mobile_base_foundation_package",
        capabilities=[
            "robot_genome",
            "robot_model_export",
            "navigation_scene_generation",
            "compiled_simulation_scenes",
        ],
        notes=["Foundation package route proving product builds are not arm-only."],
    ),
}


def package_writer_statuses(morphology_templates: list[MorphologyTemplate]) -> list[PackageWriterStatus]:
    statuses = []
    for template in morphology_templates:
        writer = IMPLEMENTED_PACKAGE_WRITERS.get(template.robot_class)
        if writer is not None:
            statuses.append(writer)
            continue
        statuses.append(
            PackageWriterStatus(
                robot_class=template.robot_class,
                status="planned",
                writer_service=f"planned.package_writer.{template.robot_class}",
                package_type=f"{template.robot_class}_planned_package",
                capabilities=["planned_package_contract"],
                notes=[f"No package writer has been implemented yet for {template.robot_class}."],
            )
        )
    return statuses


def write_package_for_template(
    *,
    requirements: RequirementsRecord,
    task: TaskGraph,
    template: MorphologyTemplate,
    output_dir: Path,
    train: bool,
) -> PackageWriterResult:
    if template.robot_class == "manipulator":
        return _write_manipulator_package(requirements, output_dir, train)
    if template.robot_class == "mobile_base":
        return _write_mobile_base_package(requirements, output_dir, train)
    raise NotImplementedError(
        f"Selected {template.id}, but robot class {template.robot_class} has no package writer yet."
    )


def _write_manipulator_package(
    requirements: RequirementsRecord,
    output_dir: Path,
    train: bool,
) -> PackageWriterResult:
    from virturoid.mvp import write_mvp_robot_arm_project

    written_dir = write_mvp_robot_arm_project(output_dir, requirements)
    package_valid = validate_mvp_export_package(written_dir).ok
    # G1 (gap-closure): the template path shipped NO bill of materials. Every package must carry a real parts
    # list — derived from the genome's own joint effort limits + the class/task sensor/compute/power suite.
    from virturoid.services.bom_builder import emit_genome_bom
    emit_genome_bom(written_dir, task=getattr(requirements, "prompt", "") or "")
    artifacts = {
        "project": "project.json",
        "bom": "robot/bill_of_materials.json",
        "robot_genome": "robot/robot_genome.json",
        "robot_urdf": "robot/robot.urdf",
        "scene_set": "simulation/scene_set.json",
        "compiled_scene_index": "simulation/mujoco/compiled_scene_index.json",
        "simulator_contract": "simulation/simulator_contract.json",
        "perception_config": "simulation/perception_config.json",
        "world_model_contract": "simulation/world_model_contract.json",
        "synthetic_observation_manifest": "datasets/synthetic_observation_manifest.json",
        "synthetic_observation_index": "datasets/synthetic_observations/index.json",
        "world_state_index": "simulation/world_state_index.json",
        "training_run_config": "training/training_run_config.json",
        "package_validation_report": "reports/package_validation_report.json",
        "html_report": "reports/index.html",
    }
    training = _maybe_train_manipulator(written_dir) if train else None
    if training and training.get("ran"):
        artifacts.update(
            {
                "trained_policy": "software/controller/trained_policy.json",
                "controller_bundle": "software/controller/controller_bundle.json",
                "controller_entrypoint": "software/controller/controller.py",
                "training_acceptance_report": "reports/training_acceptance_report.json",
                "manipulation_training_acceptance_report": "reports/manipulation_training_acceptance_report.json",
                "contact_grasp_skill": "software/skills/contact_grasp_skill.json",
                "policy_eval_trace": "runs/mvp_policy_eval/logs/episode_trace.jsonl",
            }
        )
    return PackageWriterResult(
        written_dir=written_dir,
        package_type="mvp_manipulator_training_package",
        package_valid=package_valid,
        artifacts=artifacts,
        training=training,
        notes=[
            "Routed to the full manipulator MVP pipeline.",
            "Package includes CAD artifacts, Robot Genome, URDF, scene sets, MuJoCo XML, dry-run traces, and validation reports.",
        ],
    )


def _write_mobile_base_package(
    requirements: RequirementsRecord,
    output_dir: Path,
    train: bool,
) -> PackageWriterResult:
    from virturoid.mobile_base import build_mobile_base_project

    project = build_mobile_base_project(requirements.prompt, output_dir)
    written_dir = project["output_dir"]
    from virturoid.services.bom_builder import emit_genome_bom   # G1: every package carries a real parts list
    emit_genome_bom(written_dir, task=getattr(requirements, "prompt", "") or "")
    training = {"ran": False, "reason": "Learned controller export is currently implemented for the manipulator route only."} if train else None
    return PackageWriterResult(
        written_dir=written_dir,
        package_type="mobile_base_foundation_package",
        package_valid=_mobile_base_package_valid(written_dir),
        artifacts={
            "bom": "robot/bill_of_materials.json",
            "robot_genome": "robot/robot_genome.json",
            "robot_urdf": "robot/robot.urdf",
            "mujoco_xml": "simulation/mujoco/mobile_base_scene.xml",
            "scene_set": "simulation/scene_set.json",
            "perception_config": "simulation/perception_config.json",
            "world_model_contract": "simulation/world_model_contract.json",
            "synthetic_observation_manifest": "datasets/synthetic_observation_manifest.json",
            "synthetic_observation_index": "datasets/synthetic_observations/index.json",
            "world_state_index": "simulation/world_state_index.json",
            "compiled_scene_index": "simulation/mujoco/compiled_scene_index.json",
            "bom": "bom/bom.json",
        },
        training=training,
        notes=[
            "Routed through the same morphology selector and builder registry as the arm.",
            "Mobile-base export is intentionally smaller than the manipulator MVP but proves the package boundary is not arm-specific.",
        ],
    )


def _mobile_base_package_valid(output_dir: Path) -> bool:
    required = [
        output_dir / "robot" / "robot_genome.json",
        output_dir / "robot" / "robot.urdf",
        output_dir / "simulation" / "mujoco" / "mobile_base_scene.xml",
        output_dir / "simulation" / "scene_set.json",
        output_dir / "simulation" / "perception_config.json",
        output_dir / "simulation" / "world_model_contract.json",
        output_dir / "datasets" / "synthetic_observation_manifest.json",
        output_dir / "datasets" / "synthetic_observations" / "index.json",
        output_dir / "simulation" / "world_state_index.json",
        output_dir / "simulation" / "mujoco" / "compiled_scene_index.json",
        output_dir / "bom" / "bom.json",
    ]
    return all(path.exists() for path in required)


def _maybe_train_manipulator(output_dir: Path) -> dict:
    from virturoid.services.controller_exporter import export_controller_bundle, write_training_acceptance_report
    from virturoid.services.manipulation_skill_trainer import train_contact_grasp_skill_from_export
    from virturoid.services.mujoco_runner import mujoco_available
    from virturoid.services.policy_trainer import train_arm_controller_from_export

    if not mujoco_available():
        return {
            "ran": False,
            "reason": "MuJoCo Python package not installed; run `pip install mujoco` to train a controller.",
        }
    policy = train_arm_controller_from_export(output_dir)
    bundle_path = export_controller_bundle(output_dir, policy)
    acceptance_path = write_training_acceptance_report(output_dir, policy, bundle_path)
    manipulation_report = train_contact_grasp_skill_from_export(output_dir)
    from virturoid.services.ros2_exporter import maybe_export_ros2_package
    ros2_root = maybe_export_ros2_package(output_dir)   # runnable ROS2 harness for the exported controller (§24)
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
        "training_acceptance_report_json": str(acceptance_path),
        "manipulation_training_accepted": manipulation_report["accepted"],
        "manipulation_training_acceptance_report_json": str(output_dir / "reports" / "manipulation_training_acceptance_report.json"),
        "contact_grasp_skill_json": str(output_dir / "software" / "skills" / "contact_grasp_skill.json"),
        "controller_entrypoint": str(output_dir / "software" / "controller" / "controller.py"),
        "eval_episode_trace": str(output_dir / "runs" / "mvp_policy_eval" / "logs" / "episode_trace.jsonl"),
        "ros2_package": str(ros2_root) if ros2_root else None,
    }
