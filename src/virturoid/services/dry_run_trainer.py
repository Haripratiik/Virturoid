from __future__ import annotations

import json
from pathlib import Path

from virturoid.schemas.training_result import (
    DryRunReplayIndex,
    DryRunTrainingResult,
    TrainingEpisodeTrace,
    TrainingGroupResult,
)
from virturoid.services.simulator_adapter import EpisodeBatchRequest, SimulatorAdapter, default_simulator_adapter


def run_dry_training_from_export(
    package_dir: Path,
    adapter: SimulatorAdapter | None = None,
    config_uri: str = "training/training_run_config.json",
) -> DryRunTrainingResult:
    """Consume an exported MVP package and write deterministic dry-run training results.

    This is not a simulator. It is a contract test for the training package:
    the config must be readable, scene artifacts must exist, and the runner can
    produce a stable result artifact from those inputs.
    """
    config_path = package_dir / config_uri
    config = json.loads(config_path.read_text(encoding="utf-8"))
    adapter = adapter or default_simulator_adapter(config["backend"])
    output_dir = config["output_dir"]
    result_uri = f"{output_dir}/dry_run_result.json"
    replay_index_uri = f"{output_dir}/replay_index.json"
    episode_trace_uri = f"{output_dir}/logs/episode_trace.jsonl"
    policy_plan = _load_policy_plan(package_dir, config)
    perception_config = _load_perception_config(package_dir, config)
    training_objective = _load_training_objective(package_dir, config)
    training_curriculum = _load_training_curriculum(package_dir, config)
    compiled_scene_index = _load_compiled_scene_index(package_dir, config)
    compiled_scene_xml_by_scene_id = {
        entry["scene_id"]: entry["mujoco_xml"]
        for entry in compiled_scene_index.get("scenes", [])
    }
    action_trace = [step["command"] for step in policy_plan["steps"]]
    perception_observation_keys = [
        stream["observation_key"] for stream in perception_config.get("sensor_streams", [])
    ]
    observation_keys = sorted(set(policy_plan["observation_keys"] + perception_observation_keys))
    reward_terms = [term["name"] for term in training_objective.get("reward_terms", [])]

    group_results: list[TrainingGroupResult] = []
    episode_traces: list[TrainingEpisodeTrace] = []
    for scene_artifact in config["scene_artifacts"]:
        uri = scene_artifact["uri"]
        scene_set = json.loads((package_dir / uri).read_text(encoding="utf-8"))
        purpose = scene_set["purpose"]
        scene_count = len(scene_set["scenes"])
        planned_episodes = _episodes_for_purpose(
            purpose,
            scene_count,
            config.get("episode_multiplier_overrides", {}),
        )
        first_episode_id = _episode_id(purpose, len(episode_traces) + 1)
        group_episode_traces = adapter.run_episode_batch(
            EpisodeBatchRequest(
                backend=config["backend"],
                adapter_name=adapter.name,
                purpose=purpose,
                scene_artifact=uri,
                scenes=scene_set["scenes"],
                planned_episodes=planned_episodes,
                policy_id=config["policy_id"],
                start_index=len(episode_traces) + 1,
                target_success_rate=_success_rate_for_purpose(purpose),
                observation_keys=observation_keys,
                action_trace=action_trace,
                compiled_scene_xml_by_scene_id=compiled_scene_xml_by_scene_id,
            )
        )
        episode_traces.extend(group_episode_traces)
        group_results.append(
            TrainingGroupResult(
                purpose=purpose,
                scene_artifact=uri,
                scene_count=scene_count,
                planned_episodes=planned_episodes,
                simulated_success_rate=_success_rate_for_purpose(purpose),
                first_episode_id=first_episode_id,
                last_episode_id=group_episode_traces[-1].episode_id,
            )
        )

    total_scenes = sum(group.scene_count for group in group_results)
    total_episodes = sum(group.planned_episodes for group in group_results)
    weighted_success = sum(group.simulated_success_rate * group.planned_episodes for group in group_results) / total_episodes
    result = DryRunTrainingResult(
        id=f"dry_{config['id']}",
        training_run_config_id=config["id"],
        backend=config["backend"],
        adapter_name=adapter.name,
        robot_model_uri=config["robot_model"]["uri"],
        perception_artifact_uri=config["perception_artifact"]["uri"],
        objective_artifact_uri=config["objective_artifact"]["uri"],
        curriculum_artifact_uri=config["curriculum_artifact"]["uri"],
        compiled_scene_index_uri=config["compiled_scene_artifact"]["uri"],
        reward_terms=reward_terms,
        observation_streams=perception_observation_keys,
        total_scenes=total_scenes,
        total_planned_episodes=total_episodes,
        estimated_success_rate=round(weighted_success, 3),
        replay_index_uri=replay_index_uri,
        episode_trace_uri=episode_trace_uri,
        group_results=group_results,
        notes=[
            "Dry-run only: no physics simulator was executed.",
            "This verifies the training package can be consumed by a runner-shaped service.",
            f"Perception streams: {', '.join(perception_observation_keys)}.",
            f"Reward terms: {', '.join(reward_terms)}.",
            f"Curriculum stages: {', '.join(stage['name'] for stage in training_curriculum.get('stages', []))}.",
        ],
    )
    replay_index = DryRunReplayIndex(
        id=f"replay_{config['id']}",
        training_run_config_id=config["id"],
        backend=config["backend"],
        adapter_name=adapter.name,
        dry_run_result_uri=result_uri,
        episode_trace_uri=episode_trace_uri,
        episode_count=len(episode_traces),
        scene_artifacts=[artifact["uri"] for artifact in config["scene_artifacts"]],
        compiled_scene_index_uri=config["compiled_scene_artifact"]["uri"],
        compiled_scene_xml_uris=sorted({trace.compiled_scene_xml_uri for trace in episode_traces if trace.compiled_scene_xml_uri}),
    )
    _raise_if_invalid(result.validate(), result.id)
    _raise_if_invalid(replay_index.validate(), replay_index.id)
    _write_episode_trace(package_dir, episode_trace_uri, episode_traces)
    _write_replay_index(package_dir, replay_index_uri, replay_index)
    _write_result(package_dir, result_uri, result)
    return result


def _episodes_for_purpose(purpose: str, scene_count: int, overrides: dict[str, int] | None = None) -> int:
    overrides = overrides or {}
    multiplier = {
        "baseline": 3,
        "variation": 5,
        "regression": 4,
        "holdout": 2,
    }.get(purpose, 1)
    multiplier = overrides.get(purpose, multiplier)
    return scene_count * multiplier


def _success_rate_for_purpose(purpose: str) -> float:
    return {
        "baseline": 0.92,
        "variation": 0.78,
        "regression": 0.62,
        "holdout": 0.74,
    }.get(purpose, 0.7)


def _load_policy_plan(package_dir: Path, config: dict) -> dict:
    policy_artifact = config.get("policy_artifact")
    if not policy_artifact:
        return {
            "observation_keys": ["object_poses", "joint_positions", "wrist_rgbd"],
            "steps": [
                {"command": "observe_scene"},
                {"command": "plan_grasp"},
                {"command": "move_to_target"},
                {"command": "close_gripper"},
                {"command": "place_object"},
            ],
        }
    return json.loads((package_dir / policy_artifact["uri"]).read_text(encoding="utf-8"))


def _load_perception_config(package_dir: Path, config: dict) -> dict:
    perception_artifact = config.get("perception_artifact")
    if not perception_artifact:
        return {"sensor_streams": []}
    return json.loads((package_dir / perception_artifact["uri"]).read_text(encoding="utf-8"))


def _load_training_objective(package_dir: Path, config: dict) -> dict:
    objective_artifact = config.get("objective_artifact")
    if not objective_artifact:
        return {"reward_terms": []}
    return json.loads((package_dir / objective_artifact["uri"]).read_text(encoding="utf-8"))


def _load_training_curriculum(package_dir: Path, config: dict) -> dict:
    curriculum_artifact = config.get("curriculum_artifact")
    if not curriculum_artifact:
        return {"stages": []}
    return json.loads((package_dir / curriculum_artifact["uri"]).read_text(encoding="utf-8"))


def _load_compiled_scene_index(package_dir: Path, config: dict) -> dict:
    compiled_scene_artifact = config.get("compiled_scene_artifact")
    if not compiled_scene_artifact:
        return {"scenes": []}
    return json.loads((package_dir / compiled_scene_artifact["uri"]).read_text(encoding="utf-8"))


def _episode_id(purpose: str, index: int) -> str:
    return f"episode_{purpose}_{index:04d}"


def _write_result(package_dir: Path, uri: str, result: DryRunTrainingResult) -> Path:
    path = package_dir / uri
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return path


def _write_replay_index(package_dir: Path, uri: str, replay_index: DryRunReplayIndex) -> Path:
    path = package_dir / uri
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(replay_index.to_dict(), indent=2), encoding="utf-8")
    return path


def _write_episode_trace(package_dir: Path, uri: str, episode_traces: list[TrainingEpisodeTrace]) -> Path:
    path = package_dir / uri
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(trace.__dict__, sort_keys=True) for trace in episode_traces]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _raise_if_invalid(validation_result, entity_id: str) -> None:
    if validation_result.ok:
        return
    issues = ", ".join(issue.code for issue in validation_result.issues)
    raise ValueError(f"{entity_id} failed validation: {issues}")
