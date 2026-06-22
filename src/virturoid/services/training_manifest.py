from __future__ import annotations

from virturoid.schemas.robot import RobotGenome
from virturoid.schemas.scenes import SceneSet
from virturoid.schemas.tasks import TaskGraph
from virturoid.schemas.training import TrainingManifest, TrainingSceneGroup


def build_training_manifest(
    robot: RobotGenome,
    task: TaskGraph,
    scene_sets: list[SceneSet],
    backend: str = "mujoco",
) -> TrainingManifest:
    groups = [
        TrainingSceneGroup(
            purpose=scene_set.purpose,
            scene_set_id=scene_set.id,
            scene_count=len(scene_set.scenes),
            usage=_usage_for_purpose(scene_set.purpose),
        )
        for scene_set in scene_sets
    ]
    return TrainingManifest(
        id=f"training_{task.id}_{backend}",
        task_graph_id=task.id,
        robot_genome_id=robot.id,
        backend=backend,
        scene_groups=groups,
        curriculum_notes=[
            "Start with variation scenes for basic robustness.",
            "Replay regression scenes after every policy or scene-generation change.",
            "Keep holdout scenes separate once a real simulator backend is connected.",
        ],
    )


def _usage_for_purpose(purpose: str) -> str:
    if purpose == "baseline":
        return "smoke_test"
    if purpose == "variation":
        return "training_and_evaluation"
    if purpose == "regression":
        return "failure_reproduction"
    if purpose == "holdout":
        return "generalization_check"
    return "auxiliary"

