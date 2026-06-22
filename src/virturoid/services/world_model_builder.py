from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from virturoid.schemas.perception import PerceptionConfig
from virturoid.schemas.robot import RobotGenome
from virturoid.schemas.scenes import SceneSet
from virturoid.schemas.synthetic_observation import (
    SyntheticObjectAnnotation,
    SyntheticObservationFrame,
    SyntheticObservationIndex,
)
from virturoid.schemas.tasks import TaskGraph
from virturoid.schemas.world_model import (
    ObservableEntity,
    PhysicalParameterTarget,
    WorldModelContract,
    WorldStateVariable,
)

WORLD_MODEL_CONTRACT_URI = "simulation/world_model_contract.json"
SYNTHETIC_OBSERVATION_MANIFEST_URI = "datasets/synthetic_observation_manifest.json"
SYNTHETIC_OBSERVATION_INDEX_URI = "datasets/synthetic_observations/index.json"
WORLD_STATE_INDEX_URI = "simulation/world_state_index.json"


def build_world_model_contract(
    robot: RobotGenome,
    task: TaskGraph,
    perception: PerceptionConfig,
    scene_sets: list[SceneSet],
) -> WorldModelContract:
    contract = WorldModelContract(
        id=f"world_model_{task.id}",
        robot_genome_id=robot.id,
        task_graph_id=task.id,
        perception_config_id=perception.id,
        scene_set_ids=[scene_set.id for scene_set in scene_sets],
        observable_entities=_observable_entities(perception, scene_sets),
        state_variables=_state_variables(robot, task, perception),
        physical_parameter_targets=_physical_parameters(robot, task, scene_sets),
        synthetic_dataset_manifest_uri=SYNTHETIC_OBSERVATION_MANIFEST_URI,
        policy_observation_keys=_policy_observation_keys(robot, task, perception),
        memory_key=f"{_robot_class(robot)}:{robot.species}:{task.task_type}",
        notes=[
            "World model contract maps synthetic CV observations into policy-ready state.",
            "Physical parameters listed here are the first-class knobs for Physical AI learning and co-design.",
        ],
    )
    validation = contract.validate()
    if not validation.ok:
        issues = ", ".join(issue.code for issue in validation.issues)
        raise ValueError(f"{contract.id} failed validation: {issues}")
    return contract


def write_world_model_artifacts(
    package_dir: Path,
    robot: RobotGenome,
    task: TaskGraph,
    perception: PerceptionConfig,
    scene_sets: list[SceneSet],
) -> WorldModelContract:
    contract = build_world_model_contract(robot, task, perception, scene_sets)
    package_dir = Path(package_dir)
    _write_json(package_dir / WORLD_MODEL_CONTRACT_URI, contract.to_dict())
    _write_json(package_dir / SYNTHETIC_OBSERVATION_MANIFEST_URI, _synthetic_manifest(contract, perception, scene_sets))
    observation_index, world_state_index = _write_synthetic_observation_dataset(
        package_dir,
        contract,
        perception,
        scene_sets,
    )
    _write_json(package_dir / SYNTHETIC_OBSERVATION_INDEX_URI, observation_index.to_dict())
    _write_json(package_dir / WORLD_STATE_INDEX_URI, world_state_index)
    return contract


def _observable_entities(perception: PerceptionConfig, scene_sets: list[SceneSet]) -> list[ObservableEntity]:
    by_name: dict[str, dict[str, Any]] = {}
    annotation_labels = sorted(
        {
            label
            for annotation in perception.vision_annotations
            for label in annotation.label_space
        }
    )
    observation_keys = [stream.observation_key for stream in perception.sensor_streams]
    for scene_set in scene_sets:
        for scene in scene_set.scenes:
            for item in scene.objects:
                record = by_name.setdefault(
                    item.name,
                    {
                        "object_type": item.object_type,
                        "source_scene_ids": [],
                        "state_variables": set(),
                        "physical_parameters": set(),
                    },
                )
                record["source_scene_ids"].append(scene.id)
                record["state_variables"].update([f"{item.name}.pose_xyz_rpy", f"{item.name}.visible"])
                if item.mass_kg is not None:
                    record["physical_parameters"].add(f"{item.name}.mass_kg")
                if item.friction is not None:
                    record["physical_parameters"].add(f"{item.name}.friction")
                if item.scale is not None:
                    record["physical_parameters"].add(f"{item.name}.scale")
    return [
        ObservableEntity(
            name=name,
            object_type=record["object_type"],
            source_scene_ids=sorted(set(record["source_scene_ids"])),
            observation_keys=observation_keys,
            annotation_labels=[label for label in annotation_labels if label in {name, record["object_type"]} or record["object_type"] in label],
            state_variables=sorted(record["state_variables"]),
            physical_parameters=sorted(record["physical_parameters"]),
        )
        for name, record in sorted(by_name.items())
    ]


def _state_variables(robot: RobotGenome, task: TaskGraph, perception: PerceptionConfig) -> list[WorldStateVariable]:
    variables = [
        WorldStateVariable(
            key="robot.joint_positions",
            source="robot_state",
            units="radians_or_meters",
            used_by=["controller", "simulator", "policy_training"],
            notes="Joint state is the low-level control context.",
        ),
        WorldStateVariable(
            key="robot.end_effector_pose" if _robot_class(robot) == "manipulator" else "robot.base_pose",
            source="kinematics",
            units="meters_radians",
            used_by=["world_model", "policy_training"],
            notes="Primary body pose used to reason about action feasibility.",
        ),
    ]
    for stream in perception.sensor_streams:
        variables.append(
            WorldStateVariable(
                key=f"perception.{stream.observation_key}",
                source=stream.sensor_type,
                used_by=["cv_model", "world_model"],
                notes=f"Modalities: {', '.join(stream.output_modalities)}.",
            )
        )
    if task.task_type == "navigation":
        variables.extend(
            [
                WorldStateVariable("world.goal_vector", "scene_graph", "meters", ["controller", "planner"]),
                WorldStateVariable("world.obstacle_map", "cv_model", "meters", ["controller", "planner"]),
            ]
        )
    else:
        variables.extend(
            [
                WorldStateVariable("world.object_poses", "cv_model", "meters_radians", ["controller", "reward_model"]),
                WorldStateVariable("world.container_poses", "cv_model", "meters_radians", ["controller", "reward_model"]),
            ]
        )
    return variables


def _physical_parameters(robot: RobotGenome, task: TaskGraph, scene_sets: list[SceneSet]) -> list[PhysicalParameterTarget]:
    targets = [
        PhysicalParameterTarget(
            name="joint_actuator_response",
            target="robot.joints",
            initial_source="bom/part_resolution_report.json",
            learning_signal="rollout tracking error and control saturation",
            optimization_role="controller_tuning_and_actuator_selection",
        ),
        PhysicalParameterTarget(
            name="link_geometry",
            target="robot.links",
            initial_source="cad/build_plan.json",
            learning_signal="task success, collisions, reach margin",
            optimization_role="hardware_codesign",
        ),
        PhysicalParameterTarget(
            name="sensor_noise_model",
            target="perception.sensor_streams",
            initial_source="simulation/perception_config.json",
            learning_signal="synthetic-to-sim observation mismatch",
            optimization_role="cv_domain_randomization",
        ),
    ]
    object_types = sorted({item.object_type for scene_set in scene_sets for scene in scene_set.scenes for item in scene.objects})
    for object_type in object_types:
        targets.append(
            PhysicalParameterTarget(
                name=f"{object_type}_contact_model",
                target=f"scene.objects.{object_type}",
                initial_source="simulation/scene_set.json",
                learning_signal="grasp slip, collision impulse, trajectory deviation",
                optimization_role="world_model_physics_estimation",
            )
        )
    if task.task_type == "navigation":
        targets.append(
            PhysicalParameterTarget(
                name="wheel_ground_contact",
                target="robot.mobile_base",
                initial_source="simulation/mujoco/compiled_scene_index.json",
                learning_signal="odometry drift and obstacle contact",
                optimization_role="locomotion_physical_ai",
            )
        )
    return targets


def _policy_observation_keys(robot: RobotGenome, task: TaskGraph, perception: PerceptionConfig) -> list[str]:
    keys = ["robot.joint_positions"]
    if _robot_class(robot) == "manipulator":
        keys.extend(["robot.end_effector_pose", "world.object_poses"])
    else:
        keys.extend(["robot.base_pose", "world.goal_vector", "world.obstacle_map"])
    keys.extend(f"perception.{stream.observation_key}" for stream in perception.sensor_streams)
    keys.extend(task.metrics)
    return sorted(set(keys))


def _robot_class(robot: RobotGenome) -> str:
    if robot.species.startswith("mobile_base") or robot.morphology_template_id == "morph_mobile_base_differential":
        return "mobile_base"
    if "arm" in robot.species or robot.morphology_template_id == "morph_fixed_arm_three_dof_tabletop":
        return "manipulator"
    return robot.species.split(".")[0] if robot.species else "unknown"


def _synthetic_manifest(
    contract: WorldModelContract,
    perception: PerceptionConfig,
    scene_sets: list[SceneSet],
) -> dict[str, Any]:
    scenes = [
        {"scene_set_id": scene_set.id, "purpose": scene_set.purpose, "scene_ids": [scene.id for scene in scene_set.scenes]}
        for scene_set in scene_sets
    ]
    return {
        "id": f"synthetic_observations_{contract.task_graph_id}",
        "version": "0.1.0",
        "world_model_contract_uri": WORLD_MODEL_CONTRACT_URI,
        "perception_config_id": perception.id,
        "synthetic_observation_index_uri": SYNTHETIC_OBSERVATION_INDEX_URI,
        "world_state_index_uri": WORLD_STATE_INDEX_URI,
        "output_root": perception.synthetic_dataset.output_root if perception.synthetic_dataset else "datasets/synthetic_observations",
        "renderer_backends": perception.synthetic_dataset.renderer_backends if perception.synthetic_dataset else [],
        "splits": perception.synthetic_dataset.splits if perception.synthetic_dataset else {},
        "annotation_formats": perception.synthetic_dataset.annotation_formats if perception.synthetic_dataset else [],
        "vision_annotations": [annotation.__dict__ for annotation in perception.vision_annotations],
        "scene_sets": scenes,
        "observation_streams": [stream.observation_key for stream in perception.sensor_streams],
        "physical_parameter_targets": [target.__dict__ for target in contract.physical_parameter_targets],
        "notes": [
            "This manifest is a renderer-ready dataset contract; image files are generated by a future renderer adapter.",
            "World Labs or a similar world-generation tool can satisfy texture and photorealism fields without changing downstream consumers.",
        ],
    }


def _write_synthetic_observation_dataset(
    package_dir: Path,
    contract: WorldModelContract,
    perception: PerceptionConfig,
    scene_sets: list[SceneSet],
) -> tuple[SyntheticObservationIndex, dict[str, Any]]:
    frame_records: list[dict[str, str | int]] = []
    world_state_records: list[dict[str, str | int]] = []
    frame_counter = 0
    for scene_set in scene_sets:
        for scene in scene_set.scenes:
            world_state_uri = _world_state_uri(scene.id)
            world_state = _world_state(scene_set.id, scene_set.purpose, scene, contract)
            _write_json(package_dir / world_state_uri, world_state)
            world_state_records.append(
                {
                    "scene_id": scene.id,
                    "scene_set_id": scene_set.id,
                    "purpose": scene_set.purpose,
                    "world_state_uri": world_state_uri,
                    "object_count": len(scene.objects),
                }
            )
            for stream in perception.sensor_streams:
                frame = _observation_frame(contract, perception, scene_set.id, scene_set.purpose, scene, stream, world_state_uri)
                validation = frame.validate()
                if not validation.ok:
                    issues = ", ".join(issue.code for issue in validation.issues)
                    raise ValueError(f"{frame.id} failed validation: {issues}")
                annotation_uri = _annotation_uri(scene.id, stream.name)
                _write_json(package_dir / annotation_uri, frame.to_dict())
                frame_records.append(
                    {
                        "frame_index": frame_counter,
                        "scene_id": scene.id,
                        "scene_set_id": scene_set.id,
                        "purpose": scene_set.purpose,
                        "sensor_name": stream.name,
                        "annotation_uri": annotation_uri,
                        "world_state_uri": world_state_uri,
                    }
                )
                frame_counter += 1

    index = SyntheticObservationIndex(
        id=f"synthetic_observation_index_{contract.task_graph_id}",
        world_model_contract_uri=WORLD_MODEL_CONTRACT_URI,
        perception_config_uri="simulation/perception_config.json",
        frame_count=len(frame_records),
        annotation_formats=["frame_json", "coco_json_compatible_fields", "pose_json", "physical_parameter_labels"],
        frames=frame_records,
    )
    validation = index.validate()
    if not validation.ok:
        issues = ", ".join(issue.code for issue in validation.issues)
        raise ValueError(f"{index.id} failed validation: {issues}")
    world_state_index = {
        "id": f"world_state_index_{contract.task_graph_id}",
        "version": "0.1.0",
        "world_model_contract_uri": WORLD_MODEL_CONTRACT_URI,
        "state_variable_count": len(contract.state_variables),
        "physical_parameter_target_count": len(contract.physical_parameter_targets),
        "scene_state_count": len(world_state_records),
        "scene_states": world_state_records,
        "notes": [
            "Each world-state snapshot is generated directly from the scene graph.",
            "Policy training and CV model training can consume these snapshots as ground truth.",
        ],
    }
    return index, world_state_index


def _observation_frame(
    contract: WorldModelContract,
    perception: PerceptionConfig,
    scene_set_id: str,
    purpose: str,
    scene,
    stream,
    world_state_uri: str,
) -> SyntheticObservationFrame:
    width, height = _image_size(stream)
    outputs = _expected_outputs(scene.id, stream)
    annotations = [
        _object_annotation(item, index + 1, width, height)
        for index, item in enumerate(scene.objects)
    ]
    return SyntheticObservationFrame(
        id=f"frame_{scene.id}_{stream.name}",
        scene_id=scene.id,
        scene_set_id=scene_set_id,
        purpose=purpose,
        sensor_name=stream.name,
        sensor_type=stream.sensor_type,
        modalities=list(stream.output_modalities),
        image_size_px=(width, height),
        expected_outputs=outputs,
        annotations=annotations,
        world_state_uri=world_state_uri,
    )


def _object_annotation(item, segmentation_id: int, width: int, height: int) -> SyntheticObjectAnnotation:
    x, y, z, roll, pitch, yaw = item.pose_xyz_rpy
    center_x = int(width * (0.5 + x * 0.35))
    center_y = int(height * (0.5 - y * 0.55))
    extent = max(12, int((item.scale or 1.0) * min(width, height) * 0.08))
    bbox = (
        max(0, center_x - extent // 2),
        max(0, center_y - extent // 2),
        min(extent, width),
        min(extent, height),
    )
    physical_parameters = {}
    if item.mass_kg is not None:
        physical_parameters["mass_kg"] = item.mass_kg
    if item.friction is not None:
        physical_parameters["friction"] = item.friction
    if item.scale is not None:
        physical_parameters["scale"] = item.scale
    return SyntheticObjectAnnotation(
        name=item.name,
        label=item.name,
        object_type=item.object_type,
        pose_xyz_rpy=(x, y, z, roll, pitch, yaw),
        bbox_xywh=bbox,
        depth_m=round(max(0.01, (x * x + y * y + z * z) ** 0.5), 4),
        segmentation_id=segmentation_id,
        physical_parameters=physical_parameters,
    )


def _world_state(scene_set_id: str, purpose: str, scene, contract: WorldModelContract) -> dict[str, Any]:
    objects = []
    for item in scene.objects:
        physical_parameters = {}
        if item.mass_kg is not None:
            physical_parameters["mass_kg"] = item.mass_kg
        if item.friction is not None:
            physical_parameters["friction"] = item.friction
        if item.scale is not None:
            physical_parameters["scale"] = item.scale
        objects.append(
            {
                "name": item.name,
                "object_type": item.object_type,
                "pose_xyz_rpy": list(item.pose_xyz_rpy),
                "visible": True,
                "material": item.material,
                "physical_parameters": physical_parameters,
            }
        )
    return {
        "id": f"world_state_{scene.id}",
        "version": "0.1.0",
        "scene_id": scene.id,
        "scene_set_id": scene_set_id,
        "purpose": purpose,
        "world_model_contract_uri": WORLD_MODEL_CONTRACT_URI,
        "robot_spawn_xyz_rpy": list(scene.robot_spawn_xyz_rpy or (0, 0, 0, 0, 0, 0)),
        "state_variables": [item.key for item in contract.state_variables],
        "objects": objects,
        "variation_parameters": dict(scene.variation_parameters),
    }


def _expected_outputs(scene_id: str, stream) -> dict[str, str]:
    root = f"datasets/synthetic_observations/{scene_id}/{stream.name}"
    outputs: dict[str, str] = {}
    if "rgb_image" in stream.output_modalities:
        outputs["rgb_image"] = f"{root}/rgb.png"
    if "depth_image" in stream.output_modalities:
        outputs["depth_image"] = f"{root}/depth.exr"
    if "range_scan" in stream.output_modalities:
        outputs["range_scan"] = f"{root}/scan.json"
    if "point_cloud" in stream.output_modalities:
        outputs["point_cloud"] = f"{root}/points.pcd"
    outputs["annotations"] = _annotation_uri(scene_id, stream.name)
    return outputs


def _image_size(stream) -> tuple[int, int]:
    width = int(stream.technical_limits.get("resolution_width_px", 640) or 640)
    height = int(stream.technical_limits.get("resolution_height_px", 480) or 480)
    return width, height


def _annotation_uri(scene_id: str, stream_name: str) -> str:
    return f"datasets/synthetic_observations/{scene_id}/{stream_name}/annotations.json"


def _world_state_uri(scene_id: str) -> str:
    return f"simulation/world_states/{scene_id}.json"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
