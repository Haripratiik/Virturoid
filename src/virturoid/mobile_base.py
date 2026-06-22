"""Build a differential-drive mobile base through the shared robot pipeline.

This entrypoint proves the morphology -> registry -> builder -> genome -> export
pipeline is not arm-specific. It selects the mobile-base morphology, dispatches
through the same registry the arm uses, writes the genome/URDF/MJCF, and (with
MuJoCo) runs a real drive rollout showing the base moves and turns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from virturoid.fixtures.components import curated_component_library
from virturoid.fixtures.morphologies import morphology_template_catalog
from virturoid.services.mobile_base_exporter import drive_rollout, write_mobile_base_files, write_mobile_base_scene_collection
from virturoid.services.morphology_selector import select_morphology_template
from virturoid.services.perception_config_builder import build_perception_config
from virturoid.services.requirements_builder import build_requirements_from_prompt
from virturoid.services.robot_builder_registry import build_robot_from_selection
from virturoid.services.scene_generator import generate_scene_set
from virturoid.services.task_builder import build_task_graph
from virturoid.services.world_model_builder import write_world_model_artifacts


def build_mobile_base_project(prompt: str, output_dir: Path) -> dict:
    requirements = build_requirements_from_prompt(prompt)
    task = build_task_graph(requirements)
    templates = morphology_template_catalog()
    selection = select_morphology_template(requirements, task, templates)
    result = build_robot_from_selection(requirements, curated_component_library(), selection, templates)
    robot = result.robot_build.robot_genome
    scene_set = generate_scene_set(task, count=4, purpose="navigation")
    perception_config = build_perception_config(robot, task, result.robot_build.components, [scene_set])

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "robot").mkdir(parents=True, exist_ok=True)
    (output_dir / "robot" / "robot_genome.json").write_text(json.dumps(robot.to_dict(), indent=2), encoding="utf-8")
    (output_dir / "bom").mkdir(parents=True, exist_ok=True)
    (output_dir / "bom" / "bom.json").write_text(json.dumps(result.robot_build.bom.to_dict(), indent=2), encoding="utf-8")
    (output_dir / "simulation").mkdir(parents=True, exist_ok=True)
    (output_dir / "simulation" / "scene_set.json").write_text(json.dumps(scene_set.to_dict(), indent=2), encoding="utf-8")
    (output_dir / "simulation" / "perception_config.json").write_text(json.dumps(perception_config.to_dict(), indent=2), encoding="utf-8")
    files = write_mobile_base_files(robot, output_dir)
    compiled_index = write_mobile_base_scene_collection(robot, scene_set, output_dir)
    world_model_contract = write_world_model_artifacts(output_dir, robot, task, perception_config, [scene_set])

    return {
        "requirements": requirements,
        "task_graph": task,
        "selection": selection,
        "dispatch_record": result.dispatch_record,
        "robot_genome": robot,
        "scene_set": scene_set,
        "perception_config": perception_config,
        "world_model_contract": world_model_contract,
        "files": files,
        "compiled_scene_index": compiled_index,
        "output_dir": output_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a differential-drive mobile base.")
    parser.add_argument(
        "--prompt",
        default="Build a mobile robot base that can drive and navigate indoors to deliver parts.",
        help="Task prompt (should describe a mobile/drive/navigation robot).",
    )
    parser.add_argument("--output", default=str(Path("build") / "mobile_base"))
    parser.add_argument("--drive", action="store_true", help="Run a real MuJoCo drive rollout (requires mujoco).")
    args = parser.parse_args()

    output_dir = Path(args.output)
    project = build_mobile_base_project(args.prompt, output_dir)
    summary = {
        "output_dir": str(output_dir),
        "robot_species": project["robot_genome"].species,
        "robot_class": project["selection"].selected_template.robot_class,
        "task_type": project["task_graph"].task_type,
        "dispatch_status": project["dispatch_record"].dispatch_status,
        "selected_builder": project["dispatch_record"].selected_builder_service,
        "wheel_joints": [j.name for j in project["robot_genome"].joints],
        "scene_count": len(project["scene_set"].scenes),
        "robot_genome_json": str(output_dir / "robot" / "robot_genome.json"),
        "robot_urdf": str(project["files"]["urdf"]),
        "mujoco_xml": str(project["files"]["mjcf"]),
        "scene_set_json": str(output_dir / "simulation" / "scene_set.json"),
        "perception_config_json": str(output_dir / "simulation" / "perception_config.json"),
        "world_model_contract_json": str(output_dir / "simulation" / "world_model_contract.json"),
        "synthetic_observation_manifest_json": str(output_dir / "datasets" / "synthetic_observation_manifest.json"),
        "synthetic_observation_index_json": str(output_dir / "datasets" / "synthetic_observations" / "index.json"),
        "world_state_index_json": str(output_dir / "simulation" / "world_state_index.json"),
        "compiled_mujoco_scene_index_json": str(project["compiled_scene_index"]),
    }

    if args.drive:
        from virturoid.services.mujoco_runner import mujoco_available

        if not mujoco_available():
            summary["drive"] = {"ran": False, "reason": "MuJoCo not installed."}
        else:
            forward = drive_rollout(project["files"]["mjcf"], left_torque=2.0, right_torque=2.0)
            turn = drive_rollout(project["files"]["mjcf"], left_torque=2.5, right_torque=-2.5)
            drive_result = {"ran": True, "forward": forward, "turn_in_place": turn}
            (output_dir / "runs").mkdir(parents=True, exist_ok=True)
            (output_dir / "runs" / "drive_result.json").write_text(json.dumps(drive_result, indent=2), encoding="utf-8")
            summary["drive"] = drive_result

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
