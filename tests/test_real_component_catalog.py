from virturoid.adapters.real_component_catalog import load_reference_catalog
from virturoid.services.part_resolver import resolve_robot_arm_parts
from virturoid.services.requirements_builder import build_requirements_from_prompt


def test_supplier_snapshot_is_versioned_and_has_no_fixture_sources():
    snapshot = load_reference_catalog()
    assert snapshot.version == "2026.07.28"
    assert snapshot.frozen_at == "2026-07-28"
    assert len(snapshot.components) >= 6
    for component in snapshot.components:
        assert component.version == snapshot.version
        assert component.manufacturer != "Virturoid Reference Parts"
        assert component.datasheet is not None
        assert component.source_urls
        assert all(url.startswith("https://") for url in component.source_urls)


def test_reference_arm_resolves_supplier_parts_with_traceable_limits():
    snapshot = load_reference_catalog()
    req = build_requirements_from_prompt(
        "Build a tabletop arm with RGBD vision to sort blocks.", payload_kg=0.25, reach_m=0.65
    )
    report = resolve_robot_arm_parts(req, list(snapshot.components))
    by_role = {part.role: part for part in report.resolved_parts}
    assert by_role["joint_actuator"].component_id == "cmp_actuator_cubemars_ak70_10"
    assert by_role["wrist_sensor"].component_id == "cmp_camera_intel_realsense_d435i"
    assert by_role["gripper"].component_id == "cmp_gripper_robotiq_2f85"
    assert by_role["main_compute"].component_id == "cmp_compute_jetson_orin_nano_8gb"
    assert by_role["joint_actuator"].technical_limits["stall_torque_nm"] == 24.8
