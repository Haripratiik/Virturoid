from __future__ import annotations

from virturoid.schemas.base import ArtifactRef
from virturoid.schemas.robot import RobotGenome
from virturoid.schemas.runs import PolicyRecord
from virturoid.schemas.scenes import SceneSet
from virturoid.schemas.tasks import TaskGraph
from virturoid.schemas.training import TrainingManifest
from virturoid.schemas.training_run import TrainingRunConfig


def build_training_run_config(
    robot: RobotGenome,
    task: TaskGraph,
    policy: PolicyRecord,
    training_manifest: TrainingManifest,
    scene_sets: list[SceneSet],
    backend: str = "mujoco",
) -> TrainingRunConfig:
    return TrainingRunConfig(
        id=f"training_run_{task.id}_{backend}",
        robot_genome_id=robot.id,
        task_graph_id=task.id,
        policy_id=policy.id,
        backend=backend,
        robot_model=ArtifactRef(uri="simulation/mujoco/mvp_scene.xml", media_type="application/xml"),
        policy_artifact=ArtifactRef(uri="software/policy_plan.json", media_type="application/json"),
        perception_artifact=ArtifactRef(uri="simulation/perception_config.json", media_type="application/json"),
        objective_artifact=ArtifactRef(uri="training/training_objective.json", media_type="application/json"),
        curriculum_artifact=ArtifactRef(uri="training/training_curriculum.json", media_type="application/json"),
        compiled_scene_artifact=ArtifactRef(uri="simulation/mujoco/compiled_scene_index.json", media_type="application/json"),
        scene_artifacts=[
            ArtifactRef(uri=_scene_artifact_uri(scene_set.purpose), media_type="application/json")
            for scene_set in scene_sets
        ],
        training_manifest=ArtifactRef(uri="training/training_manifest.json", media_type="application/json"),
        output_dir="runs/mvp_training",
        notes=[
            "Placeholder MuJoCo XML is generated from RobotGenome and first variation scene.",
            "A real runner should consume scene sets according to training_manifest.json.",
        ],
    )


def build_revised_training_run_config(
    robot: RobotGenome,
    task: TaskGraph,
    policy: PolicyRecord,
    backend: str = "mujoco",
) -> TrainingRunConfig:
    return TrainingRunConfig(
        id=f"training_run_{task.id}_{backend}_revision_001",
        robot_genome_id=robot.id,
        task_graph_id=task.id,
        policy_id=policy.id,
        backend=backend,
        robot_model=ArtifactRef(uri="simulation/mujoco/mvp_scene.xml", media_type="application/xml"),
        policy_artifact=ArtifactRef(uri="software/policy_plan_revised.json", media_type="application/json"),
        perception_artifact=ArtifactRef(uri="simulation/perception_config.json", media_type="application/json"),
        objective_artifact=ArtifactRef(uri="training/training_objective.json", media_type="application/json"),
        curriculum_artifact=ArtifactRef(uri="training/training_curriculum.json", media_type="application/json"),
        compiled_scene_artifact=ArtifactRef(uri="simulation/mujoco/compiled_scene_index.json", media_type="application/json"),
        scene_artifacts=[
            ArtifactRef(uri="simulation/revision_scene_set.json", media_type="application/json"),
        ],
        training_manifest=ArtifactRef(uri="training/training_manifest.json", media_type="application/json"),
        output_dir="runs/mvp_training_revision",
        notes=[
            "Second dry-run pass generated after applying training feedback.",
            "Uses the revised policy plan and revision scene set.",
        ],
    )


def _scene_artifact_uri(purpose: str) -> str:
    if purpose == "baseline":
        return "simulation/baseline_scene_set.json"
    if purpose == "variation":
        return "simulation/scene_set.json"
    return f"simulation/{purpose}_scene_set.json"
