from __future__ import annotations

from virturoid.schemas.base import ArtifactRef
from virturoid.schemas.components import ActuatorSpecs, Component, ComponentCategory, SensorSpecs


def curated_component_library() -> list[Component]:
    """Return a tiny curated parts library for deterministic MVP builds.

    These are intentionally fictional parts with explicit specs. Real supplier
    integrations should use the same schema later, but this keeps the MVP
    offline, reproducible, and easy for agents to modify.
    """
    return [
        Component(
            id="cmp_actuator_vx12",
            manufacturer="Virturoid Reference Parts",
            part_number="VX-12",
            normalized_name="VX-12 Smart Servo",
            category=ComponentCategory.ACTUATOR,
            aliases=["vx12", "vx-12", "12 nm servo", "smart servo"],
            mass_kg=0.18,
            voltage_range=(12.0, 24.0),
            max_current_a=3.2,
            actuator=ActuatorSpecs(
                actuator_type="smart_servo",
                stall_torque_nm=12.0,
                no_load_speed_rpm=50.0,
                rated_torque_nm=5.5,
                rated_speed_rpm=35.0,
                control_frequency_hz=200.0,
                communication_protocol="uart",
            ),
            cad_assets=[
                ArtifactRef(uri="fixtures/cad/vx12_servo.step", media_type="model/step", description="Reference exact CAD placeholder")
            ],
            spec_confidence=0.9,
            source_urls=["local://fixtures/components/vx12"],
        ),
        Component(
            id="cmp_actuator_vx20",
            manufacturer="Virturoid Reference Parts",
            part_number="VX-20",
            normalized_name="VX-20 High Torque Smart Servo",
            category=ComponentCategory.ACTUATOR,
            aliases=["vx20", "vx-20", "high torque servo", "22 nm servo"],
            mass_kg=0.28,
            voltage_range=(18.0, 30.0),
            max_current_a=5.0,
            actuator=ActuatorSpecs(
                actuator_type="smart_servo",
                stall_torque_nm=22.0,
                no_load_speed_rpm=42.0,
                rated_torque_nm=9.0,
                rated_speed_rpm=28.0,
                control_frequency_hz=200.0,
                communication_protocol="can",
            ),
            cad_assets=[
                ArtifactRef(uri="fixtures/cad/vx20_servo.step", media_type="model/step", description="Reference exact CAD placeholder")
            ],
            spec_confidence=0.9,
            source_urls=["local://fixtures/components/vx20"],
        ),
        Component(
            id="cmp_camera_rgbd_mini",
            manufacturer="Virturoid Reference Parts",
            part_number="RGBD-MINI",
            normalized_name="RGBD Mini Wrist Camera",
            category=ComponentCategory.SENSOR,
            aliases=["rgbd-mini", "rgbd mini", "rgb-d mini", "depth camera", "rgbd camera"],
            mass_kg=0.065,
            voltage_range=(5.0, 5.0),
            max_current_a=0.8,
            sensor=SensorSpecs(
                sensor_type="rgbd_camera",
                resolution=(1280, 720),
                field_of_view_deg=(74.0, 58.0),
                frame_rate_hz=30.0,
                range_m=(0.2, 3.5),
                ros_message_type="sensor_msgs/Image",
            ),
            cad_assets=[
                ArtifactRef(uri="fixtures/cad/rgbd_mini.step", media_type="model/step", description="Reference exact CAD placeholder")
            ],
            spec_confidence=0.85,
            source_urls=["local://fixtures/components/rgbd-mini"],
        ),
        Component(
            id="cmp_camera_rgb_mini",
            manufacturer="Virturoid Reference Parts",
            part_number="RGB-MINI",
            normalized_name="RGB Mini Wrist Camera",
            category=ComponentCategory.SENSOR,
            aliases=["rgb-mini", "rgb mini", "normal camera", "rgb camera", "webcam"],
            mass_kg=0.04,
            voltage_range=(5.0, 5.0),
            max_current_a=0.35,
            sensor=SensorSpecs(
                sensor_type="rgb_camera",
                resolution=(1920, 1080),
                field_of_view_deg=(82.0, 60.0),
                frame_rate_hz=60.0,
                range_m=(0.1, 5.0),
                ros_message_type="sensor_msgs/Image",
            ),
            cad_assets=[
                ArtifactRef(uri="fixtures/cad/rgb_mini.step", media_type="model/step", description="Reference exact CAD placeholder")
            ],
            spec_confidence=0.8,
            source_urls=["local://fixtures/components/rgb-mini"],
        ),
        Component(
            id="cmp_lidar_planar_l1",
            manufacturer="Virturoid Reference Parts",
            part_number="LIDAR-L1",
            normalized_name="LIDAR-L1 Short Range Planar LiDAR",
            category=ComponentCategory.SENSOR,
            aliases=["lidar-l1", "lidar l1", "lidar", "laser scanner"],
            mass_kg=0.11,
            voltage_range=(5.0, 12.0),
            max_current_a=0.9,
            sensor=SensorSpecs(
                sensor_type="lidar",
                resolution=(720, 1),
                field_of_view_deg=(270.0, 1.0),
                frame_rate_hz=20.0,
                range_m=(0.05, 8.0),
                ros_message_type="sensor_msgs/LaserScan",
            ),
            cad_assets=[
                ArtifactRef(uri="fixtures/cad/lidar_l1.step", media_type="model/step", description="Reference exact CAD placeholder")
            ],
            spec_confidence=0.78,
            source_urls=["local://fixtures/components/lidar-l1"],
        ),
        Component(
            id="cmp_gripper_pg2",
            manufacturer="Virturoid Reference Parts",
            part_number="PG-2",
            normalized_name="PG-2 Parallel Gripper",
            category=ComponentCategory.GRIPPER,
            aliases=["pg2", "pg-2", "parallel gripper", "two finger gripper"],
            mass_kg=0.16,
            voltage_range=(12.0, 24.0),
            max_current_a=1.2,
            cad_assets=[
                ArtifactRef(uri="fixtures/cad/pg2_gripper.step", media_type="model/step", description="Reference exact CAD placeholder")
            ],
            spec_confidence=0.85,
            source_urls=["local://fixtures/components/pg2"],
        ),
        Component(
            id="cmp_compute_edge1",
            manufacturer="Virturoid Reference Parts",
            part_number="EDGE-1",
            normalized_name="EDGE-1 Robotics Compute Module",
            category=ComponentCategory.COMPUTE,
            aliases=["edge1", "edge-1", "robotics compute", "edge compute"],
            mass_kg=0.09,
            voltage_range=(9.0, 20.0),
            max_current_a=2.5,
            spec_confidence=0.8,
            source_urls=["local://fixtures/components/edge1"],
        ),
    ]


def extended_component_library() -> list[Component]:
    """A larger, datasheet-style catalog for the AI Part Recommender (plan §8, Phase C).

    Extends the curated MVP set with more actuators, sensors, grippers, compute, and
    power options so the recommender has real choices to reason over. The deterministic
    builders still use ``curated_component_library()`` (unchanged) — this catalog feeds
    the LLM recommender, which is validated by compatibility checks. A production build
    would replace this with a live supplier catalog behind the same Component schema.
    """
    return curated_component_library() + [
        Component(
            id="cmp_actuator_vx08",
            manufacturer="Virturoid Reference Parts",
            part_number="VX-08",
            normalized_name="VX-08 Compact Smart Servo",
            category=ComponentCategory.ACTUATOR,
            aliases=["vx08", "vx-08", "8 nm servo", "compact servo"],
            mass_kg=0.12,
            voltage_range=(12.0, 24.0),
            max_current_a=2.2,
            actuator=ActuatorSpecs(
                actuator_type="smart_servo",
                stall_torque_nm=8.0,
                no_load_speed_rpm=60.0,
                rated_torque_nm=3.5,
                rated_speed_rpm=42.0,
                control_frequency_hz=200.0,
                communication_protocol="uart",
            ),
            spec_confidence=0.9,
            source_urls=["local://fixtures/components/vx08"],
        ),
        Component(
            id="cmp_actuator_vx30",
            manufacturer="Virturoid Reference Parts",
            part_number="VX-30",
            normalized_name="VX-30 Heavy Smart Servo",
            category=ComponentCategory.ACTUATOR,
            aliases=["vx30", "vx-30", "30 nm servo", "heavy servo"],
            mass_kg=0.42,
            voltage_range=(24.0, 48.0),
            max_current_a=7.5,
            actuator=ActuatorSpecs(
                actuator_type="smart_servo",
                stall_torque_nm=30.0,
                no_load_speed_rpm=36.0,
                rated_torque_nm=14.0,
                rated_speed_rpm=22.0,
                control_frequency_hz=200.0,
                communication_protocol="can",
            ),
            spec_confidence=0.88,
            source_urls=["local://fixtures/components/vx30"],
        ),
        Component(
            id="cmp_camera_stereo_s2",
            manufacturer="Virturoid Reference Parts",
            part_number="STEREO-S2",
            normalized_name="Stereo-S2 Global Shutter Stereo Camera",
            category=ComponentCategory.SENSOR,
            aliases=["stereo", "stereo camera", "global shutter", "depth stereo"],
            mass_kg=0.09,
            voltage_range=(5.0, 5.0),
            max_current_a=0.9,
            sensor=SensorSpecs(
                sensor_type="rgbd_camera",
                resolution=(1280, 800),
                field_of_view_deg=(90.0, 65.0),
                frame_rate_hz=60.0,
                range_m=(0.15, 6.0),
                ros_message_type="sensor_msgs/Image",
            ),
            spec_confidence=0.86,
            source_urls=["local://fixtures/components/stereo-s2"],
        ),
        Component(
            id="cmp_gripper_vac1",
            manufacturer="Virturoid Reference Parts",
            part_number="VAC-1",
            normalized_name="VAC-1 Vacuum Suction Gripper",
            category=ComponentCategory.GRIPPER,
            aliases=["vac1", "vac-1", "vacuum gripper", "suction gripper", "suction"],
            mass_kg=0.11,
            voltage_range=(12.0, 24.0),
            max_current_a=1.6,
            spec_confidence=0.82,
            source_urls=["local://fixtures/components/vac1"],
        ),
        Component(
            id="cmp_compute_edge2",
            manufacturer="Virturoid Reference Parts",
            part_number="EDGE-2",
            normalized_name="EDGE-2 GPU Robotics Compute Module",
            category=ComponentCategory.COMPUTE,
            aliases=["edge2", "edge-2", "gpu compute", "vision compute"],
            mass_kg=0.18,
            voltage_range=(12.0, 24.0),
            max_current_a=6.0,
            spec_confidence=0.83,
            source_urls=["local://fixtures/components/edge2"],
        ),
        Component(
            id="cmp_power_batt48",
            manufacturer="Virturoid Reference Parts",
            part_number="BATT-48",
            normalized_name="BATT-48 48V Battery Pack",
            category=ComponentCategory.POWER,
            aliases=["battery", "batt48", "48v battery", "power pack"],
            mass_kg=0.9,
            voltage_range=(40.0, 54.0),
            max_current_a=20.0,
            spec_confidence=0.8,
            source_urls=["local://fixtures/components/batt48"],
        ),
        Component(
            id="cmp_actuator_vx45",
            manufacturer="Virturoid Reference Parts",
            part_number="VX-45",
            normalized_name="VX-45 Heavy Humanoid-Joint Servo",
            category=ComponentCategory.ACTUATOR,
            aliases=["vx45", "vx-45", "45 nm servo", "humanoid servo"],
            mass_kg=0.6,
            voltage_range=(24.0, 48.0),
            max_current_a=9.0,
            actuator=ActuatorSpecs(
                actuator_type="smart_servo",
                stall_torque_nm=45.0,
                no_load_speed_rpm=30.0,
                rated_torque_nm=20.0,
                rated_speed_rpm=18.0,
                control_frequency_hz=200.0,
                communication_protocol="can",
            ),
            spec_confidence=0.85,
            source_urls=["local://fixtures/components/vx45"],
        ),
    ]


def find_component(components: list[Component], component_id: str) -> Component:
    for component in components:
        if component.id == component_id:
            return component
    raise KeyError(f"Unknown component id: {component_id}")
