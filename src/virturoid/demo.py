from __future__ import annotations

import json

from virturoid.schemas.base import ArtifactRef
from virturoid.schemas.cad import AssemblyInstance, CadAssembly, CadFormat, CadModel
from virturoid.schemas.components import (
    ActuatorSpecs,
    BillOfMaterials,
    BomItem,
    Component,
    ComponentCategory,
    SensorSpecs,
)
from virturoid.schemas.exports import ExportBundle
from virturoid.schemas.requirements import RequirementsRecord
from virturoid.schemas.robot import JointLimit, RobotGenome, RobotJoint, RobotSensor
from virturoid.schemas.runs import EpisodeRecord, EvaluationRun, FailureRecord, PolicyRecord
from virturoid.schemas.scenes import SceneGraph, SceneObject, SceneSet
from virturoid.schemas.tasks import Criterion, TaskGraph


def build_demo_entities() -> dict:
    requirements = RequirementsRecord(
        id="req_demo_sort_blocks",
        prompt="Build a robot arm that sorts red and blue blocks into matching bins.",
        task_summary="Sort colored tabletop blocks into matching bins.",
        environment="tabletop",
        payload_kg=0.25,
        reach_m=0.65,
        precision_m=0.005,
        sensor_requirements=["rgb_camera"],
    )

    shoulder_servo = Component(
        id="cmp_servo_demo",
        manufacturer="Example Robotics",
        part_number="XR-10",
        normalized_name="XR-10 Smart Servo",
        category=ComponentCategory.ACTUATOR,
        mass_kg=0.12,
        voltage_range=(12.0, 24.0),
        max_current_a=2.0,
        actuator=ActuatorSpecs(
            actuator_type="smart_servo",
            stall_torque_nm=10.0,
            no_load_speed_rpm=55.0,
            control_frequency_hz=200.0,
            communication_protocol="uart",
        ),
        spec_confidence=0.8,
        source_urls=["local://fixtures/components/xr-10"],
    )
    wrist_camera = Component(
        id="cmp_camera_demo",
        manufacturer="Example Vision",
        part_number="RGB-720",
        normalized_name="RGB-720 Wrist Camera",
        category=ComponentCategory.SENSOR,
        mass_kg=0.04,
        sensor=SensorSpecs(
            sensor_type="rgb_camera",
            resolution=(1280, 720),
            field_of_view_deg=(70.0, 45.0),
            frame_rate_hz=30.0,
            ros_message_type="sensor_msgs/Image",
        ),
        spec_confidence=0.8,
        source_urls=["local://fixtures/components/rgb-720"],
    )
    bom = BillOfMaterials(
        id="bom_demo_arm",
        robot_design_id="design_demo_arm",
        items=[
            BomItem(role="shoulder_actuator", component_id=shoulder_servo.id, quantity=1, unit_cost_usd=49.0),
            BomItem(role="wrist_camera", component_id=wrist_camera.id, quantity=1, unit_cost_usd=39.0),
        ],
    )

    base_link = CadModel(
        id="cad_base_link",
        name="Demo base link",
        source="generated",
        format=CadFormat.STEP,
        exact_geometry_available=True,
        mesh_available=True,
        artifact=ArtifactRef(uri="cad/exact/base_link.step", media_type="model/step"),
        bounding_box_mm=(120.0, 120.0, 20.0),
        mass_kg=0.2,
    )
    forearm_link = CadModel(
        id="cad_forearm_link",
        name="Demo forearm link",
        source="generated",
        format=CadFormat.STEP,
        exact_geometry_available=True,
        mesh_available=True,
        artifact=ArtifactRef(uri="cad/exact/forearm_link.step", media_type="model/step"),
        bounding_box_mm=(280.0, 35.0, 35.0),
        mass_kg=0.18,
    )
    assembly = CadAssembly(
        id="asm_demo_arm",
        name="Demo tabletop sorting arm",
        instances=[
            AssemblyInstance(instance_id="inst_base", cad_model_id=base_link.id, role="base"),
            AssemblyInstance(instance_id="inst_forearm", cad_model_id=forearm_link.id, role="forearm", parent_instance_id="inst_base"),
        ],
        bom_id=bom.id,
    )

    robot = RobotGenome(
        id="robot_demo_arm",
        name="Demo tabletop arm",
        species="fixed_arm.simple_tabletop",
        source_cad_assembly_id=assembly.id,
        links=["base_link", "forearm_link", "wrist_link"],
        joints=[
            RobotJoint(
                name="shoulder_joint",
                joint_type="revolute",
                parent_link="base_link",
                child_link="forearm_link",
                axis_xyz=(0.0, 0.0, 1.0),
                limit=JointLimit(lower=-1.57, upper=1.57, velocity=1.0, effort=10.0),
                actuator_component_id=shoulder_servo.id,
            ),
        ],
        sensors=[
            RobotSensor(
                name="wrist_rgb",
                sensor_component_id=wrist_camera.id,
                parent_link="wrist_link",
                transform_xyz_rpy=(0.0, 0.0, 0.05, 0.0, 0.0, 0.0),
            )
        ],
        end_effectors=["simple_parallel_gripper"],
        supported_backends={"mujoco": "planned", "isaac": "planned"},
        known_skills=["reach", "grasp", "place"],
    )

    task = TaskGraph(
        id="task_sort_blocks",
        name="Sort colored blocks",
        prompt=requirements.prompt,
        task_type="pick_place_sort",
        required_skills=["reach", "grasp", "lift", "place"],
        objects=["red_block", "blue_block", "red_bin", "blue_bin"],
        success_criteria=[
            Criterion(name="red_sorted", expression="red_block inside red_bin"),
            Criterion(name="blue_sorted", expression="blue_block inside blue_bin"),
        ],
        failure_criteria=[
            Criterion(name="dropped_object", expression="any block outside workspace"),
            Criterion(name="timeout", expression="episode_time > 60"),
        ],
        safety_constraints=["no_bin_collision", "no_joint_limit_violation"],
        metrics=["success_rate", "collision_count", "timeout_rate"],
    )

    scene = SceneGraph(
        id="scene_sort_blocks_001",
        name="Simple tabletop sorting scene",
        backend_targets=["mujoco"],
        robot_spawn_xyz_rpy=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        objects=[
            SceneObject(name="red_block", object_type="cube", pose_xyz_rpy=(0.35, -0.1, 0.02, 0.0, 0.0, 0.0), mass_kg=0.05, material="matte_red"),
            SceneObject(name="blue_block", object_type="cube", pose_xyz_rpy=(0.35, 0.1, 0.02, 0.0, 0.0, 0.0), mass_kg=0.05, material="matte_blue"),
            SceneObject(name="red_bin", object_type="container", pose_xyz_rpy=(0.55, -0.2, 0.02, 0.0, 0.0, 0.0)),
            SceneObject(name="blue_bin", object_type="container", pose_xyz_rpy=(0.55, 0.2, 0.02, 0.0, 0.0, 0.0)),
        ],
        variation_parameters={"lighting": "baseline", "clutter": False},
        requirement_trace=["tabletop", "rgb_camera", "sort colored blocks"],
    )
    scene_set = SceneSet(
        id="scenes_sort_blocks_baseline",
        task_graph_id=task.id,
        purpose="baseline",
        scenes=[scene],
        generation_rules=["single red block", "single blue block", "matching bins visible"],
    )

    policy = PolicyRecord(
        id="policy_scripted_pick_place",
        name="Scripted pick/place baseline",
        compatible_robot_ids=[robot.id],
        action_space="discrete_skill_sequence",
        observation_space=["object_poses", "joint_positions"],
    )
    episode = EpisodeRecord(
        id="episode_demo_001",
        run_id="run_demo_eval",
        robot_genome_id=robot.id,
        task_graph_id=task.id,
        scene_graph_id=scene.id,
        policy_id=policy.id,
        backend="mujoco",
        seed=7,
        result="failed",
        metrics={"success": False, "collision_count": 1, "timeout": False},
        safety_events=["collision_with_red_bin"],
        failure_labels=["collision"],
    )
    failure = FailureRecord(
        id="failure_demo_001",
        episode_id=episode.id,
        failure_type="collision",
        severity="medium",
        summary="End-effector path collided with red bin during place motion.",
        likely_causes=["approach angle too steep", "bin too close to block"],
        suggested_fixes=["generate wider-bin-spacing regression scene", "add pre-place waypoint"],
        regression_scene_id="scene_sort_blocks_regression_bin_collision",
    )
    run = EvaluationRun(
        id="run_demo_eval",
        robot_genome_id=robot.id,
        task_graph_id=task.id,
        scene_set_id=scene_set.id,
        policy_id=policy.id,
        backend="mujoco",
        episodes=[episode],
        failures=[failure],
    )

    export = ExportBundle(
        id="export_demo_bundle",
        project_id="project_demo_sorting_arm",
        cad_artifacts=[ArtifactRef(uri="cad/exact/robot_assembly.step", media_type="model/step")],
        bom_artifacts=[ArtifactRef(uri="bom/bom.yaml", media_type="application/x-yaml")],
        robot_artifacts=[ArtifactRef(uri="robot/robot_genome.yaml", media_type="application/x-yaml")],
        simulation_artifacts=[ArtifactRef(uri="simulation/scenes.yaml", media_type="application/x-yaml")],
        report_artifacts=[ArtifactRef(uri="reports/evaluation_report.md", media_type="text/markdown")],
    )

    artifacts = {
        "requirements": requirements,
        "components": [shoulder_servo, wrist_camera],
        "bom": bom,
        "cad_models": [base_link, forearm_link],
        "cad_assembly": assembly,
        "robot_genome": robot,
        "task_graph": task,
        "scene_set": scene_set,
        "policy": policy,
        "evaluation_run": run,
        "export_bundle": export,
    }
    return artifacts


def build_demo_project() -> dict:
    return {key: _serialize(value) for key, value in build_demo_entities().items()}


def validate_demo_project() -> list[str]:
    project = build_demo_entities()
    errors: list[str] = []
    for key, value in project.items():
        values = value if isinstance(value, list) else [value]
        for item in values:
            if hasattr(item, "validate"):
                result = item.validate()
                if not result.ok:
                    errors.extend(f"{key}: {issue.code}: {issue.message}" for issue in result.issues)
    return errors


def _serialize(value):
    if isinstance(value, list):
        return [item.to_dict() if hasattr(item, "to_dict") else item for item in value]
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def main() -> None:
    print(json.dumps(build_demo_project(), indent=2))


if __name__ == "__main__":
    main()
