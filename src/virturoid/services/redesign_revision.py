from __future__ import annotations

import json
from pathlib import Path

from virturoid.schemas.base import ArtifactRef
from virturoid.schemas.policy_plan import PolicyPlan, PolicyStep
from virturoid.schemas.redesign import AppliedRecommendation, RedesignRevision
from virturoid.schemas.scenes import SceneGraph, SceneObject, SceneSet


SOURCE_POLICY_URI = "software/policy_plan.json"
REVISED_POLICY_URI = "software/policy_plan_revised.json"
SOURCE_SCENE_URI = "simulation/regression_scene_set.json"
REVISED_SCENE_URI = "simulation/revision_scene_set.json"
REVISION_REPORT_URI = "reports/redesign_revision.json"


def apply_training_feedback_revision_from_export(package_dir: Path) -> RedesignRevision:
    feedback = _read_json(package_dir / "reports" / "training_feedback.json")
    source_policy = _load_policy_plan(package_dir, SOURCE_POLICY_URI)
    source_scene_set = _load_scene_set(package_dir, SOURCE_SCENE_URI)

    revised_policy = _revise_policy_plan(source_policy, feedback)
    revised_scene_set = _revise_scene_set(source_scene_set, feedback)
    applied = _applied_recommendations(feedback)

    revision = RedesignRevision(
        id=f"revision_{feedback['id']}",
        feedback_report_id=feedback["id"],
        source_policy_artifact=ArtifactRef(uri=SOURCE_POLICY_URI, media_type="application/json"),
        revised_policy_artifact=ArtifactRef(uri=REVISED_POLICY_URI, media_type="application/json"),
        source_scene_artifact=ArtifactRef(uri=SOURCE_SCENE_URI, media_type="application/json"),
        revised_scene_artifact=ArtifactRef(uri=REVISED_SCENE_URI, media_type="application/json"),
        applied_recommendations=applied,
        notes=[
            "First autonomous redesign pass generated from training feedback.",
            "The revised policy and scene set are intended inputs for the next training run.",
        ],
    )

    _raise_if_invalid(revised_policy.validate(), revised_policy.id)
    _raise_if_invalid(revised_scene_set.validate(), revised_scene_set.id)
    _raise_if_invalid(revision.validate(), revision.id)
    _write_json(package_dir / REVISED_POLICY_URI, revised_policy.to_dict())
    _write_json(package_dir / REVISED_SCENE_URI, revised_scene_set.to_dict())
    _write_json(package_dir / REVISION_REPORT_URI, revision.to_dict())
    return revision


def _revise_policy_plan(source: PolicyPlan, feedback: dict) -> PolicyPlan:
    steps = [PolicyStep(**step.__dict__) for step in source.steps]
    if not any(step.step_id == "recovery_waypoint" for step in steps):
        insert_at = _index_after(steps, "execute_pick")
        steps.insert(
            insert_at,
            PolicyStep(
                step_id="recovery_waypoint",
                skill="motion_planning",
                command="move_to_safe_lift_waypoint",
                target_selector="held_object",
                expected_observations=["joint_positions", "gripper_state"],
                success_condition="held object clears bins, containers, and distractors",
                fallback="lower object to safe staging zone",
            ),
        )
    if not any(step.step_id == "pre_place_clearance" for step in steps):
        insert_at = _index_before(steps, "execute_place")
        steps.insert(
            insert_at,
            PolicyStep(
                step_id="pre_place_clearance",
                skill="motion_planning",
                command="move_to_pre_place_clearance_pose",
                target_selector="destination_container",
                expected_observations=["bin_pose", "joint_positions"],
                success_condition="end effector is above container rim before descent",
                fallback="replan with wider approach arc",
            ),
        )

    return PolicyPlan(
        id=f"{source.id}_revision_001",
        policy_id=source.policy_id,
        task_graph_id=source.task_graph_id,
        robot_genome_id=source.robot_genome_id,
        policy_type=source.policy_type,
        observation_keys=list(source.observation_keys),
        action_space=source.action_space,
        steps=steps,
        notes=[
            *source.notes,
            f"Revision generated from {feedback['id']}.",
            "Added recovery and pre-place clearance steps from feedback recommendations.",
        ],
    )


def _revise_scene_set(source: SceneSet, feedback: dict) -> SceneSet:
    revised_scenes = []
    for scene in source.scenes:
        revised_objects = [_revise_scene_object(item) for item in scene.objects]
        revised_scenes.append(
            SceneGraph(
                id=f"revision_{scene.id}",
                name=f"Revised {scene.name}",
                backend_targets=list(scene.backend_targets),
                robot_spawn_xyz_rpy=scene.robot_spawn_xyz_rpy,
                objects=revised_objects,
                variation_parameters={
                    **scene.variation_parameters,
                    "revision": True,
                    "clearance_margin_m": 0.06,
                    "bin_spacing_adjustment": "wider_plus_clearance",
                },
                requirement_trace=[
                    *scene.requirement_trace,
                    "revision_from_training_feedback",
                    feedback["id"],
                ],
            )
        )

    return SceneSet(
        id=f"{source.id}_revision_001",
        task_graph_id=source.task_graph_id,
        purpose="revision",
        scenes=revised_scenes,
        generation_rules=[
            *source.generation_rules,
            "applied training feedback recommendations",
            "increase container spacing and clearance margin",
            "pair with revised policy recovery waypoints",
        ],
    )


def _revise_scene_object(item: SceneObject) -> SceneObject:
    x, y, z, roll, pitch, yaw = item.pose_xyz_rpy
    if item.object_type == "container":
        y = y + 0.05 if y >= 0 else y - 0.05
    elif "distractor" in item.name:
        x += 0.04
    return SceneObject(
        name=item.name,
        object_type=item.object_type,
        pose_xyz_rpy=(round(x, 4), round(y, 4), z, roll, pitch, yaw),
        mass_kg=item.mass_kg,
        material=item.material,
        friction=item.friction,
        scale=item.scale,
    )


def _applied_recommendations(feedback: dict) -> list[AppliedRecommendation]:
    applied = []
    for item in feedback.get("recommendations", []):
        changes = []
        if item["target"] in {"policy", "scene_and_policy"}:
            changes.extend(["added recovery waypoint", "added pre-place clearance step"])
        if item["target"] in {"scene", "scene_and_policy"}:
            changes.extend(["widened container spacing", "increased clearance margin"])
        if not changes:
            changes.append("captured recommendation for next revision")
        applied.append(
            AppliedRecommendation(
                recommendation_id=item["id"],
                target=item["target"],
                changes=changes,
            )
        )
    return applied


def _load_policy_plan(path_root: Path, uri: str) -> PolicyPlan:
    data = _read_json(path_root / uri)
    return PolicyPlan(
        id=data["id"],
        version=data.get("version", "0.1.0"),
        policy_id=data["policy_id"],
        task_graph_id=data["task_graph_id"],
        robot_genome_id=data["robot_genome_id"],
        policy_type=data["policy_type"],
        observation_keys=data["observation_keys"],
        action_space=data["action_space"],
        steps=[PolicyStep(**item) for item in data["steps"]],
        notes=data.get("notes", []),
    )


def _load_scene_set(path_root: Path, uri: str) -> SceneSet:
    data = _read_json(path_root / uri)
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


def _index_after(steps: list[PolicyStep], step_id: str) -> int:
    for index, step in enumerate(steps):
        if step.step_id == step_id:
            return index + 1
    return len(steps)


def _index_before(steps: list[PolicyStep], step_id: str) -> int:
    for index, step in enumerate(steps):
        if step.step_id == step_id:
            return index
    return len(steps)


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
