from __future__ import annotations

from dataclasses import dataclass

from virturoid.schemas.base import ArtifactRef
from virturoid.schemas.cad import AssemblyInstance, AssemblyJoint, CadAssembly, CadFormat, CadModel
from virturoid.schemas.components import BillOfMaterials, BomItem, Component
from virturoid.schemas.morphology import MorphologyTemplate
from virturoid.schemas.part_resolution import PartResolutionReport
from virturoid.schemas.requirements import RequirementsRecord
from virturoid.schemas.robot import JointLimit, RobotGenome, RobotJoint, RobotSensor
from virturoid.services.part_resolver import resolve_robot_arm_parts


@dataclass
class RobotArmBuild:
    components: list[Component]
    part_resolution: PartResolutionReport
    bom: BillOfMaterials
    cad_models: list[CadModel]
    cad_assembly: CadAssembly
    robot_genome: RobotGenome


def build_reference_robot_arm(
    requirements: RequirementsRecord,
    components: list[Component],
    morphology_template: MorphologyTemplate | None = None,
) -> RobotArmBuild:
    """Autonomously build a simple reference robot arm from requirements.

    This is the first narrow "robot builder" slice. It does not generate real
    CAD geometry yet; it creates exact-CAD artifact references and structured
    assembly/robot records that later CAD services can materialize.
    """
    part_resolution = resolve_robot_arm_parts(requirements, components)
    actuator_id = _component_id_for_role(part_resolution, "joint_actuator")
    sensor_id = _component_id_for_role(part_resolution, "wrist_sensor")
    gripper_id = _component_id_for_role(part_resolution, "gripper")
    compute_id = _component_id_for_role(part_resolution, "main_compute")

    selected_components = [component for component in components if component.id in {actuator_id, sensor_id, gripper_id, compute_id}]
    sensor_name = _sensor_name(selected_components, sensor_id)
    species = morphology_template.species_pattern if morphology_template else "fixed_arm.three_dof.tabletop"
    links = morphology_template.default_links if morphology_template else ["base_link", "upper_link", "forearm_link", "wrist_link", "gripper_link"]
    known_skills = morphology_template.default_skills if morphology_template else ["reach", "grasp", "lift", "place", "sort"]
    bom = BillOfMaterials(
        id=f"bom_{requirements.id}",
        robot_design_id=f"design_{requirements.id}",
        items=[
            BomItem(role="base_actuator", component_id=actuator_id, quantity=1),
            BomItem(role="shoulder_actuator", component_id=actuator_id, quantity=1),
            BomItem(role="elbow_actuator", component_id=actuator_id, quantity=1),
            BomItem(role="wrist_sensor", component_id=sensor_id, quantity=1),
            BomItem(role="gripper", component_id=gripper_id, quantity=1),
            BomItem(role="main_compute", component_id=compute_id, quantity=1),
        ],
    )

    cad_models = _build_placeholder_cad_models(requirements)
    cad_assembly = CadAssembly(
        id=f"asm_{requirements.id}",
        name="Autonomous tabletop robot arm assembly",
        instances=[
            AssemblyInstance(instance_id="inst_base", cad_model_id="cad_arm_base", role="base"),
            AssemblyInstance(instance_id="inst_upper_link", cad_model_id="cad_upper_link", role="upper_link", parent_instance_id="inst_base"),
            AssemblyInstance(instance_id="inst_forearm_link", cad_model_id="cad_forearm_link", role="forearm_link", parent_instance_id="inst_upper_link"),
            AssemblyInstance(instance_id="inst_camera_mount", cad_model_id="cad_wrist_camera_mount", role="camera_mount", parent_instance_id="inst_forearm_link"),
            AssemblyInstance(instance_id="inst_gripper_mount", cad_model_id="cad_gripper_mount", role="gripper_mount", parent_instance_id="inst_forearm_link"),
        ],
        joints=[
            AssemblyJoint("base_yaw", "revolute", "inst_base", "inst_upper_link", axis_xyz=(0.0, 0.0, 1.0), limit=(-3.14, 3.14)),
            AssemblyJoint("shoulder_pitch", "revolute", "inst_upper_link", "inst_forearm_link", axis_xyz=(0.0, 1.0, 0.0), limit=(-1.57, 1.57)),
        ],
        bom_id=bom.id,
    )

    robot_genome = RobotGenome(
        id=f"robot_{requirements.id}",
        name="Autonomous reference tabletop arm",
        species=species,
        morphology_template_id=morphology_template.id if morphology_template else None,
        source_cad_assembly_id=cad_assembly.id,
        links=list(links),
        joints=[
            RobotJoint(
                name="base_yaw",
                joint_type="revolute",
                parent_link="base_link",
                child_link="upper_link",
                axis_xyz=(0.0, 0.0, 1.0),
                limit=JointLimit(lower=-3.14, upper=3.14, velocity=1.2, effort=_actuator_effort(selected_components, actuator_id)),
                actuator_component_id=actuator_id,
            ),
            RobotJoint(
                name="shoulder_pitch",
                joint_type="revolute",
                parent_link="upper_link",
                child_link="forearm_link",
                axis_xyz=(0.0, 1.0, 0.0),
                limit=JointLimit(lower=-1.57, upper=1.57, velocity=1.0, effort=_actuator_effort(selected_components, actuator_id)),
                actuator_component_id=actuator_id,
            ),
            RobotJoint(
                name="elbow_pitch",
                joint_type="revolute",
                parent_link="forearm_link",
                child_link="wrist_link",
                axis_xyz=(0.0, 1.0, 0.0),
                limit=JointLimit(lower=-1.57, upper=1.57, velocity=1.0, effort=_actuator_effort(selected_components, actuator_id)),
                actuator_component_id=actuator_id,
            ),
        ],
        sensors=[
            RobotSensor(
                name=sensor_name,
                sensor_component_id=sensor_id,
                parent_link="wrist_link",
                transform_xyz_rpy=(0.03, 0.0, 0.04, 0.0, 0.0, 0.0),
            )
        ],
        end_effectors=["pg2_parallel_gripper"],
        supported_backends={"mujoco": "schema_ready", "isaac": "planned"},
        known_skills=list(known_skills),
    )

    return RobotArmBuild(
        components=selected_components,
        part_resolution=part_resolution,
        bom=bom,
        cad_models=cad_models,
        cad_assembly=cad_assembly,
        robot_genome=robot_genome,
    )


def _component_id_for_role(part_resolution: PartResolutionReport, role: str) -> str:
    for part in part_resolution.resolved_parts:
        if part.role == role:
            return part.component_id
    raise KeyError(f"Part resolution {part_resolution.id} has no {role} role.")


def _sensor_name(components: list[Component], sensor_id: str) -> str:
    for component in components:
        if component.id == sensor_id and component.sensor:
            return f"wrist_{component.sensor.sensor_type}"
    return "wrist_sensor"


def _build_placeholder_cad_models(requirements: RequirementsRecord) -> list[CadModel]:
    reach_mm = int((requirements.reach_m if requirements.reach_m is not None else 0.6) * 1000)
    # The upper link is a short shoulder bracket between the base yaw and the
    # shoulder pitch; the long primary link (forearm) sits after the pitch so it
    # can swing down to the table. This keeps the arm's workspace over the bench.
    upper_link_length = 70
    forearm_length = max(300, int(reach_mm * 0.55))

    return [
        CadModel(
            id="cad_arm_base",
            name="Robot arm base plate",
            source="generated_placeholder",
            format=CadFormat.STEP,
            exact_geometry_available=True,
            mesh_available=True,
            artifact=ArtifactRef(uri="cad/exact/arm_base.step", media_type="model/step"),
            bounding_box_mm=(160.0, 160.0, 16.0),
            mass_kg=0.45,
            editable_parameters={"width_mm": 160, "depth_mm": 160, "thickness_mm": 16},
        ),
        CadModel(
            id="cad_upper_link",
            name="Robot arm upper link",
            source="generated_placeholder",
            format=CadFormat.STEP,
            exact_geometry_available=True,
            mesh_available=True,
            artifact=ArtifactRef(uri="cad/exact/upper_link.step", media_type="model/step"),
            bounding_box_mm=(upper_link_length, 36.0, 36.0),
            mass_kg=0.22,
            editable_parameters={"length_mm": upper_link_length, "width_mm": 36, "height_mm": 36},
        ),
        CadModel(
            id="cad_forearm_link",
            name="Robot arm forearm link",
            source="generated_placeholder",
            format=CadFormat.STEP,
            exact_geometry_available=True,
            mesh_available=True,
            artifact=ArtifactRef(uri="cad/exact/forearm_link.step", media_type="model/step"),
            bounding_box_mm=(forearm_length, 32.0, 32.0),
            mass_kg=0.18,
            editable_parameters={"length_mm": forearm_length, "width_mm": 32, "height_mm": 32},
        ),
        CadModel(
            id="cad_wrist_camera_mount",
            name="Wrist RGBD camera mount",
            source="generated_placeholder",
            format=CadFormat.STEP,
            exact_geometry_available=True,
            mesh_available=True,
            artifact=ArtifactRef(uri="cad/exact/wrist_camera_mount.step", media_type="model/step"),
            bounding_box_mm=(42.0, 28.0, 18.0),
            mass_kg=0.035,
            editable_parameters={"hole_pattern_mm": "20x20", "hole_diameter_mm": 3.2},
        ),
        CadModel(
            id="cad_gripper_mount",
            name="Parallel gripper mount",
            source="generated_placeholder",
            format=CadFormat.STEP,
            exact_geometry_available=True,
            mesh_available=True,
            artifact=ArtifactRef(uri="cad/exact/gripper_mount.step", media_type="model/step"),
            bounding_box_mm=(50.0, 38.0, 12.0),
            mass_kg=0.04,
            editable_parameters={"hole_pattern_mm": "24x24", "hole_diameter_mm": 3.2},
        ),
    ]


def _actuator_effort(components: list[Component], actuator_id: str) -> float | None:
    for component in components:
        if component.id == actuator_id and component.actuator:
            return component.actuator.stall_torque_nm
    return None
