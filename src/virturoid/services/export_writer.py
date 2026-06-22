from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from virturoid.schemas.exports import ExportBundle


def write_export_bundle(bundle: ExportBundle, project: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.json"
    project_path = output_dir / "project.json"

    manifest_path.write_text(json.dumps(bundle.to_dict(), indent=2), encoding="utf-8")
    project_path.write_text(json.dumps(_plain(project), indent=2), encoding="utf-8")
    _write_known_artifact_files(bundle, project, output_dir)

    return output_dir


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


def _write_known_artifact_files(bundle: ExportBundle, project: dict[str, Any], output_dir: Path) -> None:
    writers = {
        "bom/bom.json": project.get("bom"),
        "bom/part_resolution_report.json": project.get("part_resolution"),
        "design/morphology_template.json": project.get("morphology_template"),
        "design/builder_dispatch.json": project.get("builder_dispatch"),
        "design/robot_build_blueprint.json": project.get("robot_build_blueprint"),
        "cad/build_plan.json": project.get("fabrication_plan"),
        "power/power_architecture.json": project.get("power_architecture"),
        "robot/robot_genome.json": project.get("robot_genome"),
        "simulation/baseline_scene_set.json": project.get("baseline_scene_set"),
        "simulation/scene_set.json": project.get("scene_set"),
        "simulation/edge_case_scene_set.json": project.get("edge_case_scene_set"),
        "simulation/regression_scene_set.json": project.get("regression_scene_set"),
        "simulation/holdout_scene_set.json": project.get("holdout_scene_set"),
        "simulation/perception_config.json": project.get("perception_config"),
        "simulation/world_model_contract.json": project.get("world_model_contract"),
        "training/training_manifest.json": project.get("training_manifest"),
        "training/training_objective.json": project.get("training_objective"),
        "training/training_run_config.json": project.get("training_run_config"),
        "training/training_run_config_revised.json": project.get("revised_training_run_config"),
        "software/policy_plan.json": project.get("policy_plan"),
        "reports/mvp_evaluation_report.json": project.get("evaluation_run"),
    }
    for relative_path, value in writers.items():
        if value is None:
            continue
        path = output_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_plain(value), indent=2), encoding="utf-8")

    # HONEST ROS2 status: a real ament_python package is written to export/ros2/virturoid_robot ONLY when a
    # controller is trained (--train). Report the status from REALITY (does that package exist on disk?) rather
    # than unconditionally claiming 'exported_on_train' — that hardcoded string was a product-truth lie.
    ros2_pkg = output_dir / "export" / "ros2" / "virturoid_robot"
    exported = ros2_pkg.exists()
    status_path = output_dir / "software" / "ros2_export_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(
            {
                "status": "exported" if exported else "not_exported",
                "detail": ("A runnable ament_python ROS2 package is present." if exported else
                           "No ROS2 package yet — run the build with --train to generate "
                           "export/ros2/virturoid_robot (its node runs the exported controller)."),
                "ros2_package": "export/ros2/virturoid_robot" if exported else None,
                "robot_genome_id": getattr(project.get("robot_genome"), "id", None),
                "task_graph_id": getattr(project.get("task_graph"), "id", None),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # Keep the legacy filename as a pointer so older validators/tests that look for it still resolve (its
    # content now tells the honest status instead of asserting a train-time export happened).
    (output_dir / "software" / "ros2_test_harness_placeholder.json").write_text(
        json.dumps({"status": "see ros2_export_status.json", "ros2_export_status": "software/ros2_export_status.json",
                    "exported": exported}, indent=2),
        encoding="utf-8",
    )
