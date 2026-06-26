import json
import tempfile
from pathlib import Path

from virturoid.services.spec_sheet import build_spec_sheet, write_spec_sheet


def _make_package(d):
    d = Path(d)
    (d / "robot").mkdir(parents=True, exist_ok=True)
    (d / "reports").mkdir(parents=True, exist_ok=True)
    (d / "robot" / "robot_genome.json").write_text(json.dumps({
        "name": "test.quad", "species": "test.quad", "robot_class": "quadruped",
        "joints": [{"name": f"j{i}"} for i in range(12)],
    }), encoding="utf-8")
    (d / "robot" / "bill_of_materials.json").write_text(json.dumps({
        "robot_class": "quadruped", "dof": 12,
        "actuator_map": {"leg0_0": "Unitree GO-M8010-6", "leg0_1": "T-Motor AK80-9"},
        "totals": {"actuators": 12, "mass_kg": 14.9, "price_usd": 4713.2, "est_power_w": 1329.4},
        "lines": [
            {"part": "Unitree GO-M8010-6", "category": "actuator", "qty": 8, "detail": "peak 23.7 Nm @ 24 V"},
            {"part": "Bosch BNO055", "category": "imu", "qty": 1, "detail": "9-DOF IMU"},
            {"part": "Jetson Orin", "category": "compute", "qty": 1, "detail": "compute module"},
        ],
    }), encoding="utf-8")
    (d / "reports" / "gene_evaluation_report.json").write_text(json.dumps({
        "task_type": "locomotion", "success_rate": 0.45, "forward_m": 0.83, "cadence_hz": 1.7, "upright_frac": 0.9,
    }), encoding="utf-8")
    return d


def test_spec_sheet_aggregates_genome_bom_eval():
    with tempfile.TemporaryDirectory() as tmp:
        _make_package(tmp)
        spec = build_spec_sheet(tmp)
    assert spec["robot_class"] == "quadruped"
    assert spec["dof"] == 12
    assert spec["physical"]["mass_kg"] == 14.9
    assert spec["power_and_cost"]["est_parts_cost_usd"] == 4713.2
    assert spec["power_and_cost"]["est_power_draw_w"] == 1329.4
    assert spec["actuation"]["peak_joint_torque_nm"] == 23.7
    assert "Bosch BNO055" in spec["sensing"]
    assert "Jetson Orin" in spec["compute"]
    assert spec["performance"]["task"] == "locomotion"
    assert spec["summary"] and "quadruped" in spec["summary"]


def test_write_spec_sheet_emits_json_and_markdown():
    with tempfile.TemporaryDirectory() as tmp:
        _make_package(tmp)
        out = write_spec_sheet(tmp)
        assert out is not None and out.exists()
        assert (Path(tmp) / "reports" / "spec_sheet.md").exists()
        assert json.loads(out.read_text(encoding="utf-8"))["dof"] == 12


def test_spec_sheet_handles_missing_bom():
    # legacy path with no BOM -> sparse but must not crash, and still summarizes the morphology
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "robot").mkdir()
        (d / "robot" / "robot_genome.json").write_text(json.dumps({
            "species": "arm", "robot_class": "manipulator",
            "joints": [{"name": "j0"}, {"name": "j1"}, {"name": "j2"}],
        }), encoding="utf-8")
        spec = build_spec_sheet(tmp)
    assert spec["robot_class"] == "manipulator"
    assert spec["dof"] == 3
    assert spec["physical"]["mass_kg"] is None
    assert spec["summary"]


def test_deployment_guide_aggregates_artifacts():
    from virturoid.services.deployment_guide import build_deployment_guide, write_deployment_guide

    with tempfile.TemporaryDirectory() as tmp:
        _make_package(tmp)
        write_spec_sheet(tmp)
        path = write_deployment_guide(tmp)
        assert path is not None and path.exists()
        md = build_deployment_guide(tmp)
    assert "deployment guide" in md.lower()
    assert "Order the parts" in md
    assert "Unitree GO-M8010-6" in md            # from the BOM lines
    assert "leg0_0" in md                         # from the actuator_map assembly table
    assert "caveat" in md.lower() or "sim-to-real" in md.lower()
