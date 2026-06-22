from __future__ import annotations

import json
from pathlib import Path

from virturoid.schemas.base import ArtifactRef
from virturoid.schemas.training_run import TrainingRunConfig


NEXT_TRAINING_CONFIG_URI = "training/training_run_config_next.json"


def write_next_training_run_config_from_export(package_dir: Path, revision_episode_multiplier: int = 4) -> TrainingRunConfig:
    active_inputs = _read_json(package_dir / "training" / "active_training_inputs.json")
    selected_config = _read_json(package_dir / active_inputs["training_config"]["uri"])

    config = TrainingRunConfig(
        id=f"{selected_config['id']}_next_001",
        robot_genome_id=selected_config["robot_genome_id"],
        task_graph_id=selected_config["task_graph_id"],
        policy_id=selected_config["policy_id"],
        backend=selected_config["backend"],
        robot_model=_artifact_ref(selected_config["robot_model"]),
        policy_artifact=_artifact_ref(active_inputs["policy_plan"]),
        perception_artifact=_artifact_ref(selected_config["perception_artifact"]),
        objective_artifact=_artifact_ref(selected_config["objective_artifact"]),
        curriculum_artifact=_artifact_ref(selected_config["curriculum_artifact"]),
        compiled_scene_artifact=_artifact_ref(selected_config["compiled_scene_artifact"]),
        scene_artifacts=[_artifact_ref(active_inputs["scene_set"])],
        training_manifest=_artifact_ref(selected_config["training_manifest"]),
        output_dir="runs/mvp_training_next",
        episode_multiplier_overrides={"revision": revision_episode_multiplier},
        notes=[
            f"Generated from {active_inputs['id']} after promotion decision {active_inputs['decision_id']}.",
            "Uses promoted active policy and scene inputs for the next training pass.",
            "Expands revision-scene episode coverage before another promotion decision.",
        ],
    )
    _raise_if_invalid(config.validate(), config.id)
    _write_json(package_dir / NEXT_TRAINING_CONFIG_URI, config.to_dict())
    return config


def _artifact_ref(payload: dict) -> ArtifactRef:
    return ArtifactRef(
        uri=payload["uri"],
        media_type=payload.get("media_type"),
        sha256=payload.get("sha256"),
        description=payload.get("description"),
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _raise_if_invalid(validation_result, entity_id: str) -> None:
    if validation_result.ok:
        return
    issues = ", ".join(issue.code for issue in validation_result.issues)
    raise ValueError(f"{entity_id} failed validation: {issues}")
