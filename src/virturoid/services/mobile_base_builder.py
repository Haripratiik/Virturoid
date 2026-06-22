"""Builder for a differential-drive mobile base (#5).

This is the second implemented robot class. It proves the morphology ->
registry -> builder -> Robot Genome -> export pipeline is not arm-specific: a
mobile base flows through the same RobotBuild container and dispatch records as
the arm, while producing a wheeled genome with continuous drive joints.
"""

from __future__ import annotations

from virturoid.schemas.base import ArtifactRef
from virturoid.schemas.cad import AssemblyInstance, AssemblyJoint, CadAssembly, CadFormat, CadModel
from virturoid.schemas.components import BillOfMaterials, BomItem, Component
from virturoid.schemas.morphology import MorphologyTemplate
from virturoid.schemas.part_resolution import PartResolutionReport, ResolvedPart
from virturoid.schemas.requirements import RequirementsRecord
from virturoid.schemas.robot import JointLimit, RobotGenome, RobotJoint, RobotSensor
from virturoid.services.robot_arm_builder import RobotArmBuild

WHEEL_RADIUS_M = 0.06
WHEEL_HALF_WIDTH_M = 0.02
TRACK_HALF_M = 0.14


def build_differential_mobile_base(
    requirements: RequirementsRecord,
    components: list[Component],
    morphology_template: MorphologyTemplate | None = None,
) -> RobotArmBuild:
    drive = _find(components, "cmp_actuator_vx20") or _find_category(components, "actuator")
    sensor = _find(components, "cmp_camera_rgbd_mini") or _find_category(components, "sensor")
    compute = _find(components, "cmp_compute_edge1") or _find_category(components, "compute")
    selected = [c for c in [drive, sensor, compute] if c is not None]

    part_resolution = PartResolutionReport(
        id=f"parts_{requirements.id}",
        requirements_id=requirements.id,
        resolved_parts=[
            ResolvedPart(
                role="drive_actuator",
                component_id=drive.id,
                source="requirement_recommendation",
                reason="High-torque servo selected for wheel drive.",
                technical_limits={
                    "stall_torque_nm": drive.actuator.stall_torque_nm if drive.actuator else None,
                    "no_load_speed_rpm": drive.actuator.no_load_speed_rpm if drive.actuator else None,
                },
                cad_assets=list(drive.cad_assets),
            ),
            ResolvedPart(
                role="localization_sensor",
                component_id=sensor.id,
                source="requirement_recommendation",
                reason="Forward perception for navigation.",
                cad_assets=list(sensor.cad_assets),
            ),
            ResolvedPart(
                role="main_compute",
                component_id=compute.id,
                source="requirement_recommendation",
                reason="Onboard navigation and control compute.",
            ),
        ],
    )

    bom = BillOfMaterials(
        id=f"bom_{requirements.id}",
        robot_design_id=f"design_{requirements.id}",
        items=[
            BomItem(role="left_drive", component_id=drive.id, quantity=1),
            BomItem(role="right_drive", component_id=drive.id, quantity=1),
            BomItem(role="localization_sensor", component_id=sensor.id, quantity=1),
            BomItem(role="main_compute", component_id=compute.id, quantity=1),
        ],
    )

    cad_models = _cad_models()
    cad_assembly = CadAssembly(
        id=f"asm_{requirements.id}",
        name="Differential mobile base assembly",
        instances=[
            AssemblyInstance(instance_id="inst_chassis", cad_model_id="cad_chassis", role="chassis"),
            AssemblyInstance(instance_id="inst_left_wheel", cad_model_id="cad_drive_wheel", role="left_wheel", parent_instance_id="inst_chassis"),
            AssemblyInstance(instance_id="inst_right_wheel", cad_model_id="cad_drive_wheel", role="right_wheel", parent_instance_id="inst_chassis"),
            AssemblyInstance(instance_id="inst_sensor_mast", cad_model_id="cad_sensor_mast", role="sensor_mast", parent_instance_id="inst_chassis"),
        ],
        joints=[
            AssemblyJoint("left_wheel", "continuous", "inst_chassis", "inst_left_wheel", axis_xyz=(0.0, 1.0, 0.0)),
            AssemblyJoint("right_wheel", "continuous", "inst_chassis", "inst_right_wheel", axis_xyz=(0.0, 1.0, 0.0)),
        ],
        bom_id=bom.id,
    )

    effort = drive.actuator.stall_torque_nm if drive.actuator else None
    velocity = (drive.actuator.no_load_speed_rpm * 2 * 3.14159 / 60.0) if drive.actuator else None
    species = morphology_template.species_pattern if morphology_template else "mobile_base.differential.indoor"
    skills = morphology_template.default_skills if morphology_template else ["localize", "plan_path", "avoid_obstacle", "dock"]
    sensor_type = sensor.sensor.sensor_type if sensor.sensor else "rgbd_camera"

    robot_genome = RobotGenome(
        id=f"robot_{requirements.id}",
        name="Autonomous reference mobile base",
        species=species,
        morphology_template_id=morphology_template.id if morphology_template else None,
        source_cad_assembly_id=cad_assembly.id,
        links=["base_link", "left_wheel_link", "right_wheel_link", "sensor_mast_link"],
        joints=[
            RobotJoint(
                name="left_wheel",
                joint_type="continuous",
                parent_link="base_link",
                child_link="left_wheel_link",
                axis_xyz=(0.0, 1.0, 0.0),
                limit=JointLimit(velocity=velocity, effort=effort),
                actuator_component_id=drive.id,
            ),
            RobotJoint(
                name="right_wheel",
                joint_type="continuous",
                parent_link="base_link",
                child_link="right_wheel_link",
                axis_xyz=(0.0, 1.0, 0.0),
                limit=JointLimit(velocity=velocity, effort=effort),
                actuator_component_id=drive.id,
            ),
        ],
        sensors=[
            RobotSensor(
                name=f"front_{sensor_type}",
                sensor_component_id=sensor.id,
                parent_link="sensor_mast_link",
                transform_xyz_rpy=(0.16, 0.0, 0.22, 0.0, 0.0, 0.0),
            )
        ],
        end_effectors=[],
        supported_backends={"mujoco": "schema_ready", "isaac": "planned"},
        known_skills=list(skills),
    )

    return RobotArmBuild(
        components=selected,
        part_resolution=part_resolution,
        bom=bom,
        cad_models=cad_models,
        cad_assembly=cad_assembly,
        robot_genome=robot_genome,
    )


def _cad_models() -> list[CadModel]:
    return [
        CadModel(
            id="cad_chassis",
            name="Mobile base chassis",
            source="generated_placeholder",
            format=CadFormat.STEP,
            exact_geometry_available=True,
            mesh_available=True,
            artifact=ArtifactRef(uri="cad/exact/chassis.step", media_type="model/step"),
            bounding_box_mm=(300.0, 240.0, 90.0),
            mass_kg=2.4,
            editable_parameters={"length_mm": 300, "width_mm": 240, "height_mm": 90},
        ),
        CadModel(
            id="cad_drive_wheel",
            name="Drive wheel",
            source="generated_placeholder",
            format=CadFormat.STEP,
            exact_geometry_available=True,
            mesh_available=True,
            artifact=ArtifactRef(uri="cad/exact/drive_wheel.step", media_type="model/step"),
            bounding_box_mm=(120.0, 40.0, 120.0),
            mass_kg=0.35,
            editable_parameters={"radius_mm": 60, "width_mm": 40},
        ),
        CadModel(
            id="cad_sensor_mast",
            name="Sensor mast",
            source="generated_placeholder",
            format=CadFormat.STEP,
            exact_geometry_available=True,
            mesh_available=True,
            artifact=ArtifactRef(uri="cad/exact/sensor_mast.step", media_type="model/step"),
            bounding_box_mm=(40.0, 40.0, 200.0),
            mass_kg=0.2,
            editable_parameters={"height_mm": 200},
        ),
    ]


def _find(components: list[Component], component_id: str) -> Component | None:
    return next((c for c in components if c.id == component_id), None)


def _find_category(components: list[Component], category: str) -> Component | None:
    return next((c for c in components if c.category.value == category), None)
