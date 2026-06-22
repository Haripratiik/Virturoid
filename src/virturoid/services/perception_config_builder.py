from __future__ import annotations

from virturoid.schemas.components import Component
from virturoid.schemas.perception import (
    PerceptionConfig,
    SensorStream,
    SyntheticDatasetSpec,
    TextureRandomizationRule,
    VisionAnnotationSpec,
)
from virturoid.schemas.robot import RobotGenome, RobotSensor
from virturoid.schemas.scenes import SceneSet
from virturoid.schemas.tasks import TaskGraph


def build_perception_config(
    robot: RobotGenome,
    task: TaskGraph,
    components: list[Component],
    scene_sets: list[SceneSet],
    backend: str = "mujoco",
) -> PerceptionConfig:
    by_id = {component.id: component for component in components}
    streams = [
        _sensor_stream(sensor, by_id.get(sensor.sensor_component_id))
        for sensor in robot.sensors
    ]
    config = PerceptionConfig(
        id=f"perception_{task.id}_{backend}",
        robot_genome_id=robot.id,
        task_graph_id=task.id,
        backend=backend,
        sensor_streams=streams,
        vision_annotations=_vision_annotations_for_task(task, streams),
        synthetic_dataset=_synthetic_dataset_for_task(task, scene_sets, backend),
        texture_randomization=_texture_rules_for_task(task),
        lighting_randomization=[
            "vary_key_light_intensity:0.65-1.25",
            "vary_key_light_direction:small_tabletop_offsets",
            "add_sensor_noise_profile:per_stream",
        ],
        scene_bindings=[scene_set.id for scene_set in scene_sets],
        notes=[
            "This config is the scene-to-perception contract for simulator training.",
            "A real renderer should emit these streams from the generated scene and selected robot sensors.",
        ],
    )
    validation = config.validate()
    if not validation.ok:
        issues = ", ".join(issue.code for issue in validation.issues)
        raise ValueError(f"{config.id} failed validation: {issues}")
    return config


def _sensor_stream(sensor: RobotSensor, component: Component | None) -> SensorStream:
    sensor_type = component.sensor.sensor_type if component and component.sensor else "unknown"
    return SensorStream(
        name=sensor.name,
        sensor_component_id=sensor.sensor_component_id,
        sensor_type=sensor_type,
        parent_link=sensor.parent_link,
        observation_key=sensor.name,
        output_modalities=_modalities(sensor_type),
        simulated_outputs=_simulated_outputs(sensor.name, sensor_type),
        technical_limits=_technical_limits(component),
    )


def _modalities(sensor_type: str) -> list[str]:
    if sensor_type == "rgbd_camera":
        return ["rgb_image", "depth_image", "camera_intrinsics"]
    if sensor_type == "rgb_camera":
        return ["rgb_image", "camera_intrinsics"]
    if sensor_type == "lidar":
        return ["range_scan", "point_cloud"]
    return ["generic_observation"]


def _simulated_outputs(stream_name: str, sensor_type: str) -> list[str]:
    if sensor_type == "rgbd_camera":
        return [
            f"observations/{stream_name}/rgb.png",
            f"observations/{stream_name}/depth.exr",
            f"observations/{stream_name}/segmentation.png",
        ]
    if sensor_type == "rgb_camera":
        return [
            f"observations/{stream_name}/rgb.png",
            f"observations/{stream_name}/segmentation.png",
        ]
    if sensor_type == "lidar":
        return [
            f"observations/{stream_name}/scan.json",
            f"observations/{stream_name}/points.pcd",
        ]
    return [f"observations/{stream_name}/observation.json"]


def _vision_annotations_for_task(task: TaskGraph, streams: list[SensorStream]) -> list[VisionAnnotationSpec]:
    labels = _label_space_for_task(task)
    output_modalities = {modality for stream in streams for modality in stream.output_modalities}
    annotations = []
    if "rgb_image" in output_modalities:
        annotations.append(
            VisionAnnotationSpec(
                name="rgb_detection_and_segmentation",
                target_object_types=_target_object_types(task),
                annotation_modalities=["2d_bounding_box", "instance_mask", "semantic_mask"],
                label_space=labels,
                output_pattern="observations/{scene_id}/{sensor_name}/annotations_rgb.json",
            )
        )
    if "depth_image" in output_modalities:
        annotations.append(
            VisionAnnotationSpec(
                name="depth_geometry",
                target_object_types=_target_object_types(task),
                annotation_modalities=["depth_map", "surface_normal", "object_pose_6d"],
                label_space=labels,
                output_pattern="observations/{scene_id}/{sensor_name}/annotations_depth.json",
            )
        )
    if "point_cloud" in output_modalities or "range_scan" in output_modalities:
        annotations.append(
            VisionAnnotationSpec(
                name="range_geometry",
                target_object_types=_target_object_types(task),
                annotation_modalities=["range_return", "point_cloud_cluster", "object_pose_2d"],
                label_space=labels,
                output_pattern="observations/{scene_id}/{sensor_name}/annotations_range.json",
            )
        )
    if annotations:
        return annotations
    return [
        VisionAnnotationSpec(
            name="state_observation_labels",
            target_object_types=_target_object_types(task),
            annotation_modalities=["object_state"],
            label_space=labels,
            output_pattern="observations/{scene_id}/state_labels.json",
        )
    ]


def _synthetic_dataset_for_task(
    task: TaskGraph,
    scene_sets: list[SceneSet],
    backend: str,
) -> SyntheticDatasetSpec:
    scene_count = sum(len(scene_set.scenes) for scene_set in scene_sets)
    return SyntheticDatasetSpec(
        output_root="datasets/synthetic_observations",
        renderer_backends=[backend, "worldlabs_or_equivalent_texture_generator"],
        splits={
            "train_scenes": max(1, round(scene_count * 0.7)),
            "validation_scenes": max(1, round(scene_count * 0.2)),
            "holdout_scenes": max(1, scene_count - round(scene_count * 0.9)),
        },
        annotation_formats=["coco_json", "segmentation_png", "pose_json", "depth_exr_or_range_json"],
        domain_randomization_sources=[
            "simulation/scene_set.json",
            "simulation/holdout_scene_set.json",
            "simulation/perception_config.json",
        ],
        notes=[
            f"Dataset spec generated for task type {task.task_type}.",
            "Renderer integration can swap in a photorealistic/world-model tool while preserving this contract.",
        ],
    )


def _technical_limits(component: Component | None) -> dict[str, str | int | float | None]:
    if not component or not component.sensor:
        return {}
    limits: dict[str, str | int | float | None] = {
        "sensor_type": component.sensor.sensor_type,
        "frame_rate_hz": component.sensor.frame_rate_hz,
        "ros_message_type": component.sensor.ros_message_type,
    }
    if component.sensor.resolution:
        limits["resolution_width_px"] = component.sensor.resolution[0]
        limits["resolution_height_px"] = component.sensor.resolution[1]
    if component.sensor.field_of_view_deg:
        limits["horizontal_fov_deg"] = component.sensor.field_of_view_deg[0]
        limits["vertical_fov_deg"] = component.sensor.field_of_view_deg[1]
    if component.sensor.range_m:
        limits["min_range_m"] = component.sensor.range_m[0]
        limits["max_range_m"] = component.sensor.range_m[1]
    return limits


def _texture_rules_for_task(task: TaskGraph) -> list[TextureRandomizationRule]:
    if task.task_type == "navigation":
        return [
            TextureRandomizationRule("floor", ["polished_concrete", "matte_tile", "warehouse_epoxy"]),
            TextureRandomizationRule("obstacle", ["cardboard_box", "black_plastic_crate", "metal_rack"]),
            TextureRandomizationRule("zone", ["painted_floor_marker", "printed_sign", "tape_boundary"]),
        ]
    if task.task_type == "pick_place_box":
        return [
            TextureRandomizationRule("box", ["plain_cardboard", "shipping_label", "slightly_worn_cardboard"]),
            TextureRandomizationRule("container", ["matte_gray", "blue_plastic", "warehouse_yellow"]),
            TextureRandomizationRule("conveyor", ["rubber_black", "ribbed_black", "dusty_rubber"]),
        ]
    return [
        TextureRandomizationRule("cube:red_block", ["matte_red", "gloss_red", "slightly_worn_red"]),
        TextureRandomizationRule("cube:blue_block", ["matte_blue", "gloss_blue", "slightly_worn_blue"]),
        TextureRandomizationRule("container", ["matte_plastic", "semi_gloss_plastic", "printed_label"]),
        TextureRandomizationRule("distractor", ["matte_gray", "wood", "black_plastic"]),
    ]


def _label_space_for_task(task: TaskGraph) -> list[str]:
    labels = []
    for obj in task.objects:
        if isinstance(obj, dict):
            object_type = obj.get("object_type") or obj.get("type") or obj.get("name")
        else:
            object_type = str(obj)
        if object_type and object_type not in labels:
            labels.append(str(object_type))
    if task.task_type == "navigation":
        labels.extend(["start_zone", "goal_zone", "obstacle", "free_space"])
    if not labels:
        labels.extend(["target_object", "container", "distractor"])
    return sorted(set(labels))


def _target_object_types(task: TaskGraph) -> list[str]:
    if task.task_type == "navigation":
        return ["zone", "obstacle", "floor"]
    if task.task_type == "pick_place_box":
        return ["box", "container", "conveyor"]
    return ["cube", "container", "distractor"]
