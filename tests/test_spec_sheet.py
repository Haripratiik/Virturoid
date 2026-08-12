import json
import tempfile
from pathlib import Path

from virturoid.services.spec_sheet import build_spec_sheet, write_spec_sheet

BOM = {
    "robot_class": "quadruped", "dof": 12,
    "actuator_map": {"leg0_0": "Unitree GO-M8010-6", "leg0_1": "T-Motor AK80-9"},
    "totals": {"actuators": 12, "mass_kg": 14.9, "price_usd": 4713.2, "est_power_w": 1329.4},
    "lines": [
        {"part": "Unitree GO-M8010-6", "category": "actuator", "qty": 8, "detail": "peak 23.7 Nm @ 24 V"},
        {"part": "Aluminum 6061-T6", "category": "material", "qty": 13, "mass_kg": 6.4,
         "detail": "13 links (structural)"},
        {"part": "Bosch BNO055", "category": "imu", "qty": 1, "detail": "9-DOF IMU"},
        {"part": "Jetson Orin", "category": "compute", "qty": 1, "detail": "compute module"},
    ],
}

# A minimal but real MuJoCo model, so the bounding box comes from an actual compile rather than a stub.
MJCF = ('<mujoco><worldbody><geom name="floor" type="plane" size="5 5 .1"/>'
        '<body name="trunk" pos="0 0 .3"><freejoint/><geom type="box" size=".2 .1 .05"/></body>'
        '</worldbody></mujoco>')


def _make_package(d, *, bom_rel="robot/bill_of_materials.json"):
    d = Path(d)
    (d / "robot").mkdir(parents=True, exist_ok=True)
    (d / "reports").mkdir(parents=True, exist_ok=True)
    (d / "robot" / "robot_genome.json").write_text(json.dumps({
        "name": "test.quad", "species": "test.quad", "robot_class": "quadruped",
        "joints": [{"name": f"j{i}"} for i in range(12)],
    }), encoding="utf-8")
    bom_path = d / bom_rel
    bom_path.parent.mkdir(parents=True, exist_ok=True)
    bom_path.write_text(json.dumps(BOM), encoding="utf-8")
    (d / "reports" / "gene_evaluation_report.json").write_text(json.dumps({
        "task_type": "locomotion", "success_rate": 0.45, "forward_m": 0.83, "cadence_hz": 1.7, "upright_frac": 0.9,
    }), encoding="utf-8")
    return d


def _certificate(*, motor="Unitree GO-M8010-6", qty=8, peak=23.7, sim_mass=6.4):
    return {
        "identity": {"robot_class": "quadruped", "dof": 12, "species": "test.quad"},
        "verdict": {"verdict": "CREDIBLE WALK", "credible": True, "gait_source": "default_crawl",
                    "checks": {"survived": True, "forward_m": 1.087, "speed_mps": 0.679, "cadence": 29.4,
                               "support_frac": 0.97, "height_ratio": 0.947, "roll_max_deg": 9.7,
                               "pitch_max_deg": 4.7}},
        "model_sanity": {"ok": True, "total_mass_kg": sim_mass},
        "margins": {"available": True, "bom": [{"part": motor, "qty": qty, "peak_nm": peak}]},
    }


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
    # ...and every missing number says WHY, and what to run. A bare null is the bug, not the sparseness.
    why = spec["unavailable"]["physical.mass_kg"]
    assert "bill of materials" in why and "bom" in why


# ---------------------------------------------------------------------------------------------------------
# THE DEFECT: every export_held package shipped a spec sheet whose every field was null, next to a populated
# bom.json. The data was never absent -- the reader looked for exactly one of the three filenames the
# pipeline writes.
# ---------------------------------------------------------------------------------------------------------

def test_spec_sheet_reads_the_bom_wherever_the_pipeline_actually_wrote_it():
    """``export_held`` writes ``bom.json`` at the package root; the MVP/blueprint writers write
    ``bom/bom.json``; only the package builder writes ``robot/bill_of_materials.json``. All three must
    produce the SAME populated sheet -- the numbers are one directory away, not missing."""
    for rel in ("robot/bill_of_materials.json", "bom.json", "bom/bom.json"):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "robot").mkdir(parents=True, exist_ok=True)
            (d / "robot" / "robot_genome.json").write_text(json.dumps({
                "name": "test.quad", "robot_class": "quadruped",
                "joints": [{"name": f"j{i}", "joint_type": "revolute"} for i in range(12)]}), encoding="utf-8")
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / rel).write_text(json.dumps(BOM), encoding="utf-8")
            spec = build_spec_sheet(d)
        assert spec["physical"]["mass_kg"] == 14.9, f"BOM at {rel} was not read"
        assert spec["physical"]["actuators"] == 12, rel
        assert spec["power_and_cost"]["est_power_draw_w"] == 1329.4, rel
        assert spec["power_and_cost"]["est_parts_cost_usd"] == 4713.2, rel
        assert spec["actuation"]["peak_joint_torque_nm"] == 23.7, rel
        assert spec["sensing"] and spec["compute"], rel


def test_spec_sheet_takes_size_from_a_root_robot_xml_and_ignores_the_ros2_manifest():
    """``size_m`` was null on every export because the extent was only looked for under ``simulation/``,
    while ``export_held`` writes ``robot.xml`` at the package root. ``export/ros2/**/package.xml`` sorts
    first and is XML but is not a model -- it must not be mistaken for one."""
    with tempfile.TemporaryDirectory() as tmp:
        d = _make_package(tmp, bom_rel="bom.json")
        (d / "export" / "ros2" / "pkg").mkdir(parents=True)
        (d / "export" / "ros2" / "pkg" / "package.xml").write_text(
            "<package format='3'><name>virturoid_robot</name></package>", encoding="utf-8")
        (d / "robot.xml").write_text(MJCF, encoding="utf-8")
        spec = build_spec_sheet(d)
    size = spec["physical"]["size_m"]
    assert size is not None, spec["unavailable"]
    assert size["length_m"] > 0 and size["width_m"] > 0 and size["height_m"] > 0
    assert "physical.size_m" not in spec["unavailable"]


def test_spec_sheet_reads_the_verification_certificate_when_there_is_no_eval_report():
    """An export package has no ``gene_evaluation_report.json`` -- but it does carry a signed physics
    rollout in ``verification_certificate.json``, which this sheet used to ignore, leaving the task
    ``"quadruped"`` and every gait number absent."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "robot").mkdir(parents=True)
        (d / "robot" / "robot_genome.json").write_text(json.dumps({
            "name": "test.quad", "robot_class": "quadruped",
            "joints": [{"name": f"j{i}", "joint_type": "revolute"} for i in range(12)]}), encoding="utf-8")
        (d / "bom.json").write_text(json.dumps(BOM), encoding="utf-8")
        (d / "verification_certificate.json").write_text(json.dumps(_certificate()), encoding="utf-8")
        spec = build_spec_sheet(d)
    perf = spec["performance"]
    assert perf["task"] == "locomotion"
    assert perf["verdict"] == "CREDIBLE WALK" and perf["credible"] is True
    assert perf["forward_m"] == 1.087 and perf["speed_mps"] == 0.679
    assert perf["gait_source"] == "default_crawl"
    assert "verification_certificate" in perf["measured_by"]
    # success_rate is genuinely NOT in this package -- so say so, actionably, instead of shipping a null.
    assert perf["success_rate"] is None
    why = spec["unavailable"]["performance.success_rate"]
    assert "CREDIBLE WALK" in why and ("evaluate_robot" in why or "build_gene_package" in why)
    # ...and the one-liner must quote the certificate, not the bare unearned word "walks".
    assert "certified credible walk" in spec["summary"]


def test_spec_sheet_never_ships_a_bare_null_or_empty_field():
    """Every null / empty value in the sheet must carry an entry in ``unavailable`` saying what is missing
    and what to run. ``mass_kg: null`` beside a populated BOM is the defect this whole file exists for."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "robot").mkdir(parents=True)
        (d / "robot" / "robot_genome.json").write_text(json.dumps({
            "name": "bare", "robot_class": "quadruped", "joints": [{"name": "j0"}]}), encoding="utf-8")
        spec = build_spec_sheet(d)
    checked = {"physical.mass_kg": spec["physical"]["mass_kg"],
               "physical.actuators": spec["physical"]["actuators"],
               "physical.size_m": spec["physical"]["size_m"],
               "power_and_cost.est_power_draw_w": spec["power_and_cost"]["est_power_draw_w"],
               "power_and_cost.est_parts_cost_usd": spec["power_and_cost"]["est_parts_cost_usd"],
               "actuation.peak_joint_torque_nm": spec["actuation"]["peak_joint_torque_nm"],
               "actuation.actuator_types": spec["actuation"]["actuator_types"],
               "sensing": spec["sensing"], "compute": spec["compute"],
               "performance.success_rate": spec["performance"]["success_rate"]}
    for path, value in checked.items():
        if value in (None, [], {}):
            why = spec["unavailable"].get(path)
            assert why and len(why) > 30, f"{path} is empty with no actionable reason (got {why!r})"


def test_spec_sheet_markdown_prints_the_reason_not_the_word_none():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "robot").mkdir(parents=True)
        (d / "robot" / "robot_genome.json").write_text(json.dumps({
            "name": "bare", "robot_class": "quadruped", "joints": [{"name": "j0"}]}), encoding="utf-8")
        write_spec_sheet(d)
        md = (d / "reports" / "spec_sheet.md").read_text(encoding="utf-8")
    assert "| None kg |" not in md and "$None" not in md and "| None Nm |" not in md
    assert "bill of materials" in md
    assert "## Not in this package" in md


def test_spec_sheet_separates_as_built_mass_from_the_mass_the_verdict_was_measured_at():
    """A package whose parts list is 2.33x its verified body must say so with both numbers.

    This fixture is the DIVERGENT case, which since task #211 means one of two things: an imported robot (the
    simulated body is the customer's own machine and the parts list is that machine plus what the BOM proposes
    adding) or a body whose parts changed without a re-ground. Either way, printing one number implies the
    robot was verified at a mass it will never have. On a body we compose the two now agree; that case is
    covered in ``tests/test_mass_embodiment.py``.
    """
    with tempfile.TemporaryDirectory() as tmp:
        d = _make_package(tmp, bom_rel="bom.json")
        (d / "verification_certificate.json").write_text(json.dumps(_certificate()), encoding="utf-8")
        spec = build_spec_sheet(d)
    mb = spec["physical"]["mass_breakdown"]
    assert mb["as_built_parts_kg"] == 14.9
    assert mb["structure_only_kg"] == 6.4
    assert mb["simulated_body_kg"] == 6.4
    assert mb["as_built_over_simulated"] == 2.33
    # both masses named in the prose, not just in the fields above it
    assert "6.4 kg" in mb["note"] and "14.9 kg" in mb["note"]
    assert "2.33x" in mb["note"]


def test_spec_sheet_flags_a_package_that_ships_two_motor_selections():
    """One package once shipped 8x AK80-64 in bom.json and 8x AK10-9 in the certificate's margins.bom.
    A spec sheet that quietly picks one is worse than no spec sheet -- it must say so, loudly."""
    with tempfile.TemporaryDirectory() as tmp:
        d = _make_package(tmp, bom_rel="bom.json")
        (d / "verification_certificate.json").write_text(
            json.dumps(_certificate(motor="T-Motor AK10-9", qty=8, peak=48.0)), encoding="utf-8")
        write_spec_sheet(d)
        spec = json.loads((d / "reports" / "spec_sheet.json").read_text(encoding="utf-8"))
        md = (d / "reports" / "spec_sheet.md").read_text(encoding="utf-8")
    cons = spec["consistency"]
    assert cons["ok"] is False
    assert any("motor_selection" in c for c in cons["contradictions"]), cons
    # the PART's rating, checked against the certificate's PART rating -- like for like. (The robot's own
    # peak joint torque is a different quantity with its own check; see the Go2 tests below.)
    assert any("actuator_part_peak_nm" in c for c in cons["contradictions"]), cons
    assert "INCONSISTENT PACKAGE" in spec["summary"]
    assert "## INCONSISTENT PACKAGE" in md


def test_spec_sheet_flags_a_class_or_dof_mismatch_between_artifacts():
    """The ingest audit's failure mode: bom.json said `quadruped` for a biped. If the genome, the BOM and
    the certificate disagree about what this robot even IS, the sheet must not paper over it."""
    with tempfile.TemporaryDirectory() as tmp:
        d = _make_package(tmp, bom_rel="bom.json")
        cert = _certificate()
        cert["identity"]["robot_class"] = "biped"
        cert["identity"]["dof"] = 6
        (d / "verification_certificate.json").write_text(json.dumps(cert), encoding="utf-8")
        spec = build_spec_sheet(d)
    bad = " ".join(spec["consistency"]["contradictions"])
    assert "robot_class" in bad and "dof" in bad, spec["consistency"]
    assert spec["consistency"]["ok"] is False


def test_spec_sheet_agrees_with_the_certificate_on_a_healthy_package():
    with tempfile.TemporaryDirectory() as tmp:
        d = _make_package(tmp, bom_rel="bom.json")
        (d / "verification_certificate.json").write_text(json.dumps(_certificate()), encoding="utf-8")
        spec = build_spec_sheet(d)
    cons = spec["consistency"]
    assert cons["ok"] is True and cons["checked"] >= 4, cons
    assert {c["check"] for c in cons["checks"]} >= {"robot_class", "dof", "motor_selection", "structural_mass"}


def test_spec_sheet_survives_an_unparseable_artifact():
    """A truncated json must not blank the whole sheet -- fall through to the next candidate name."""
    with tempfile.TemporaryDirectory() as tmp:
        d = _make_package(tmp, bom_rel="bom.json")
        (d / "verification_certificate.json").write_text("{not json", encoding="utf-8")
        spec = build_spec_sheet(d)
    assert spec["physical"]["mass_kg"] == 14.9
    assert spec["sources"]["verification_certificate"] is False


def test_export_writes_the_spec_sheet_after_every_artifact_it_summarizes():
    """The sheet is a pure read-over-artifacts summary, so it must run LAST in ``export_held``. It used to run
    between 'bom' and 'usd' -- before the certificate block -- so every exported package shipped a sheet with
    no verdict, no forward distance and no motor-selection cross-check, beside a populated certificate.
    Cheap source-order check rather than a multi-minute real export; it pins exactly the regression."""
    import inspect

    from virturoid.services.agent_design_tools import export_held
    src = inspect.getsource(export_held)
    cert_at = src.index("verification_certificate.json")
    spec_at = src.index("write_spec_sheet")
    assert cert_at < spec_at, (
        "export_held writes the spec sheet BEFORE the verification certificate, so the sheet cannot read the "
        "verdict, the gait numbers or the certificate's motor selection -- move the 'spec' block last")
    bom_at = src.index('"bom.json"')
    assert bom_at < spec_at, "export_held writes the spec sheet before the BOM it summarizes"


# ---------------------------------------------------------------------------------------------------------
# THE DEFECT (2026-08-12, real Menagerie Unitree Go2 through ``call_tool`` -> ``export_held``, reading the
# WRITTEN reports/spec_sheet.md):
#
#     | Peak joint torque | 360.0 Nm |          <- the rating of the catalog part the BOM matched
#     robot.xml, three files away in the SAME package: forcerange="-45.43 45.43"
#
# and ``consistency`` printed "5/5 agree", because the only torque check compared the sheet's number with the
# certificate's -- two artifacts agreeing about a PART while the robot's own model said something else.
# ---------------------------------------------------------------------------------------------------------

#: a Go2-shaped package: the customer's own limits in the genome + a BOM whose catalog part is rated 8x that.
_GO2_EFFORTS = [23.7, 23.7, 45.43] * 4


def _go2ish_package(d, *, joint_limits=True, imported=True):
    d = Path(d)
    (d / "robot").mkdir(parents=True, exist_ok=True)
    (d / "robot" / "robot_genome.json").write_text(json.dumps({
        "name": "quadruped.imported", "robot_class": "quadruped",
        "joints": [{"name": f"j{i}", "joint_type": "revolute",
                    "limit": {"lower": -1.0, "upper": 1.0, "effort": eff, "velocity": 10.0}}
                   for i, eff in enumerate(_GO2_EFFORTS)]}), encoding="utf-8")
    bom = {
        "robot_class": "quadruped", "dof": 12,
        "totals": {"actuators": 12, "mass_kg": 18.058, "price_usd": 1134.37, "est_power_w": 385.3,
                   "already_fitted_usd": 6480.0, "catalog_list_price_usd": 7614.37,
                   "already_fitted_note": "a replacement/spares figure, not a bill"},
        "lines": [
            {"part": "Unitree M107 (B2/H1-class)", "category": "actuator", "qty": 4, "mass_kg": 6.4,
             "price_usd": 3600.0, "in_mass_total": not imported, "in_price_total": not imported,
             "detail": "qdd, peak 360 Nm @ 48 V, gear 1:1"},
            {"part": "Aluminum 6061-T6", "category": "material", "qty": 12, "mass_kg": 8.286,
             "price_usd": 196.37, "detail": "12 links (structural)"},
        ],
        "spec_provenance": {"robot_is": "your imported machine" if imported else "a design of ours"},
    }
    if joint_limits:
        bom["joint_limits"] = {
            "peak_joint_torque_nm": 45.43, "peak_joint": "j2", "n_joints": 12,
            "n_declared_by_your_model": 12, "n_sized_by_us": 0, "declared_by_your_model": True,
            "per_joint": {f"j{i}": {"limit": eff, "unit": "N.m", "source": "your model",
                                    "declared_in": "actuator/ctrlrange (force-mode motor) x gear"}
                          for i, eff in enumerate(_GO2_EFFORTS)},
            "note": "THIS IS THE ROBOT'S CAPABILITY, not the rating of the catalog part"}
    (d / "bom.json").write_text(json.dumps(bom), encoding="utf-8")
    return d


def test_peak_joint_torque_is_the_robots_limit_not_the_catalog_parts_rating():
    with tempfile.TemporaryDirectory() as tmp:
        spec = build_spec_sheet(_go2ish_package(tmp))
    act = spec["actuation"]
    assert act["peak_joint_torque_nm"] == 45.43, "the sheet reported the motor's rating as the robot's"
    assert "your own model" in act["peak_joint_torque_source"]
    # the part rating still ships -- on its own field, named for what it is
    assert act["catalog_part_peak_torque_nm"] == 360.0
    assert act["catalog_part"] == "Unitree M107 (B2/H1-class)"
    assert "not this robot's capability" in act["catalog_part_note"]


def test_the_robots_limit_is_read_from_the_genome_when_the_bom_predates_joint_limits():
    """An older package has no ``joint_limits`` block -- but the genome shipping beside it carries the
    customer's own effort limits, and reading those beats regexing a part rating out of a detail string."""
    with tempfile.TemporaryDirectory() as tmp:
        spec = build_spec_sheet(_go2ish_package(tmp, joint_limits=False))
    assert spec["actuation"]["peak_joint_torque_nm"] == 45.43
    assert "genome" in spec["actuation"]["peak_joint_torque_source"]


def test_an_imported_machine_with_no_declared_limits_refuses_instead_of_quoting_the_part():
    """Nothing in this package states the robot's capability, and it is the CUSTOMER'S machine. Quoting the
    part rating would overstate it by construction, so the sheet says what is missing and what to run."""
    with tempfile.TemporaryDirectory() as tmp:
        d = _go2ish_package(tmp, joint_limits=False)
        genome = json.loads((d / "robot" / "robot_genome.json").read_text(encoding="utf-8"))
        for j in genome["joints"]:
            j.pop("limit", None)
        (d / "robot" / "robot_genome.json").write_text(json.dumps(genome), encoding="utf-8")
        spec = build_spec_sheet(d)
        write_spec_sheet(d)
        md = (d / "reports" / "spec_sheet.md").read_text(encoding="utf-8")
    assert spec["actuation"]["peak_joint_torque_nm"] is None
    why = spec["unavailable"]["actuation.peak_joint_torque_nm"]
    assert "overstates" in why and "360" in why
    assert "| 360.0 Nm |" not in md


def test_a_robot_we_designed_may_still_quote_the_fitted_motors_peak():
    """The refusal must be narrow. On a body we composed the joint limit IS the peak of the motor we chose,
    so a legacy BOM with no joint_limits block may still be read -- and it says that is what it did."""
    with tempfile.TemporaryDirectory() as tmp:
        d = _go2ish_package(tmp, joint_limits=False, imported=False)
        genome = json.loads((d / "robot" / "robot_genome.json").read_text(encoding="utf-8"))
        for j in genome["joints"]:
            j.pop("limit", None)
        (d / "robot" / "robot_genome.json").write_text(json.dumps(genome), encoding="utf-8")
        spec = build_spec_sheet(d)
    assert spec["actuation"]["peak_joint_torque_nm"] == 360.0
    assert "we chose the motor" in spec["actuation"]["peak_joint_torque_source"]


def test_consistency_catches_a_sheet_quoting_a_bigger_torque_than_the_robot_declares():
    """THE CHECK THAT DID NOT EXIST. Nothing compared the reported peak torque against the joint efforts in
    the genome travelling in the same package, so 360.0-vs-45.43 shipped under '5/5 agree'."""
    with tempfile.TemporaryDirectory() as tmp:
        d = _go2ish_package(tmp)
        bom = json.loads((d / "bom.json").read_text(encoding="utf-8"))
        bom["joint_limits"]["peak_joint_torque_nm"] = 360.0        # the defect, re-created
        (d / "bom.json").write_text(json.dumps(bom), encoding="utf-8")
        write_spec_sheet(d)
        spec = json.loads((d / "reports" / "spec_sheet.json").read_text(encoding="utf-8"))
        md = (d / "reports" / "spec_sheet.md").read_text(encoding="utf-8")
    cons = spec["consistency"]
    assert cons["ok"] is False
    bad = " ".join(cons["contradictions"])
    assert "peak_joint_torque_nm" in bad and "45.43" in bad, cons
    assert "INCONSISTENT PACKAGE" in spec["summary"]
    assert "## INCONSISTENT PACKAGE" in md


def test_the_headline_cost_is_the_bill_not_the_machine_the_customer_already_owns():
    with tempfile.TemporaryDirectory() as tmp:
        d = _go2ish_package(tmp)
        spec = build_spec_sheet(d)
        write_spec_sheet(d)
        md = (d / "reports" / "spec_sheet.md").read_text(encoding="utf-8")
    pc = spec["power_and_cost"]
    assert pc["est_parts_cost_usd"] == 1134.37
    assert pc["already_fitted_usd"] == 6480.0
    assert pc["catalog_list_price_usd"] == 7614.37
    assert "~$1,134 parts to buy" in spec["summary"]
    assert "+$6,480 already fitted" in spec["summary"]
    assert "already own" in md


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
