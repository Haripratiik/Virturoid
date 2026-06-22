from __future__ import annotations

import json
from pathlib import Path

from virturoid.schemas.training_curriculum import (
    CurriculumSceneBinding,
    CurriculumStage,
    TrainingCurriculum,
)

CURRICULUM_URI = "training/training_curriculum.json"
COMPILED_SCENE_INDEX_URI = "simulation/mujoco/compiled_scene_index.json"


def write_training_curriculum_from_export(package_dir: Path) -> Path:
    curriculum = build_training_curriculum_from_export(package_dir)
    validation = curriculum.validate()
    if not validation.ok:
        issues = ", ".join(issue.code for issue in validation.issues)
        raise ValueError(f"{curriculum.id} failed validation: {issues}")

    path = package_dir / CURRICULUM_URI
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(curriculum.to_dict(), indent=2), encoding="utf-8")
    return path


def build_training_curriculum_from_export(package_dir: Path) -> TrainingCurriculum:
    manifest = _read_json(package_dir / "training" / "training_manifest.json")
    objective = _read_json(package_dir / "training" / "training_objective.json")
    compiled_index = _read_json(package_dir / COMPILED_SCENE_INDEX_URI)

    scene_set_by_purpose = {
        "baseline": "simulation/baseline_scene_set.json",
        "variation": "simulation/scene_set.json",
        "edge_case": "simulation/edge_case_scene_set.json",
        "regression": "simulation/regression_scene_set.json",
        "holdout": "simulation/holdout_scene_set.json",
        "revision": "simulation/revision_scene_set.json",
    }
    compiled_by_scene_id = {
        entry["scene_id"]: entry["mujoco_xml"]
        for entry in compiled_index.get("scenes", [])
    }

    stages = []
    for group in manifest.get("scene_groups", []):
        scene_set_uri = scene_set_by_purpose[group["purpose"]]
        scene_set = _read_json(package_dir / scene_set_uri)
        bindings = [
            CurriculumSceneBinding(
                scene_id=scene["id"],
                purpose=group["purpose"],
                scene_set_uri=scene_set_uri,
                compiled_scene_xml_uri=compiled_by_scene_id.get(scene["id"], ""),
                episode_multiplier=_episode_multiplier(group["purpose"]),
                randomization_profile=_randomization_profile(scene),
            )
            for scene in scene_set.get("scenes", [])
        ]
        stages.append(
            CurriculumStage(
                name=f"{group['purpose']}_curriculum",
                purpose=group["purpose"],
                usage=group["usage"],
                scene_count=len(bindings),
                planned_episodes=len(bindings) * _episode_multiplier(group["purpose"]),
                scene_bindings=bindings,
            )
        )

    if (package_dir / "simulation" / "revision_scene_set.json").exists():
        revision_set = _read_json(package_dir / "simulation" / "revision_scene_set.json")
        revision_bindings = [
            CurriculumSceneBinding(
                scene_id=scene["id"],
                purpose="revision",
                scene_set_uri="simulation/revision_scene_set.json",
                compiled_scene_xml_uri=compiled_by_scene_id.get(scene["id"], ""),
                episode_multiplier=1,
                randomization_profile=_randomization_profile(scene),
            )
            for scene in revision_set.get("scenes", [])
            if scene["id"] in compiled_by_scene_id
        ]
        if revision_bindings:
            stages.append(
                CurriculumStage(
                    name="revision_curriculum",
                    purpose="revision",
                    usage="feedback_redesign_validation",
                    scene_count=len(revision_bindings),
                    planned_episodes=len(revision_bindings),
                    scene_bindings=revision_bindings,
                )
            )

    success_checks = [
        rule["expression"]
        for rule in objective.get("termination_rules", [])
        if rule.get("terminal_status") == "success"
    ]
    safety_checks = [
        rule["expression"]
        for rule in objective.get("termination_rules", [])
        if rule.get("terminal_status") == "failure"
    ]

    return TrainingCurriculum(
        id=f"curriculum_{manifest['id']}",
        robot_genome_id=manifest["robot_genome_id"],
        task_graph_id=manifest["task_graph_id"],
        backend=manifest["backend"],
        compiled_scene_index_uri=COMPILED_SCENE_INDEX_URI,
        stages=stages,
        success_checks=success_checks,
        safety_checks=safety_checks,
        randomization_envelopes=_randomization_envelopes(stages),
        notes=[
            "Curriculum is generated from exported scene sets and compiled simulator scene XML.",
            "A real training runner can consume these stages in order or sample by purpose.",
        ],
    )


def _episode_multiplier(purpose: str) -> int:
    return {
        "baseline": 3,
        "variation": 5,
        "regression": 4,
        "holdout": 2,
    }.get(purpose, 1)


def _randomization_profile(scene: dict) -> str:
    parameters = scene.get("variation_parameters", {})
    if parameters.get("failure_type"):
        return f"failure_replay:{parameters['failure_type']}"
    if parameters.get("clutter"):
        return "pose_jitter_with_clutter"
    if parameters.get("environment") == "warehouse":
        return "warehouse_pose_jitter"
    return "nominal_pose_jitter"


def _randomization_envelopes(stages: list[CurriculumStage]) -> dict[str, list[str]]:
    envelopes: dict[str, set[str]] = {}
    for stage in stages:
        profiles = envelopes.setdefault(stage.purpose, set())
        for binding in stage.scene_bindings:
            profiles.add(binding.randomization_profile)
    return {purpose: sorted(profiles) for purpose, profiles in envelopes.items()}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
