"""WS-S / S2: the operational control scripts must be GENERATED from the robot (joints + BOM-sized actuators +
policy obs layout), and every shipped .py must COMPILE and DRY-RUN. The safety filter's torque ceilings are the
load-bearing honesty claim -- they must be the real actuator's peak torque, so a command that would exceed a
motor is caught. These tests pin that.

The audit that backs the claim is pinned in the block at the bottom of this file. It was ADVERTISED and NOT
IMPLEMENTED: `validate_scripts` was py_compile + a subprocess dry-run, `SafetyFilter.audit` had zero production
callers, and a stack declaring ceilings 10x the datasheet passed GREEN with the filter then clamping 41 Nm into
a 4.1 Nm motor.
"""
from __future__ import annotations

import importlib.util
import json

import pytest

_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="composing bodies needs MuJoCo")


def _compose(prompt):
    from virturoid.services.morphology_composer import compose_robot
    return compose_robot(prompt, llm=None)


def test_every_generated_script_compiles_and_dry_runs(tmp_path):
    """The S2 property: every shipped .py compiles AND dry-runs in a subprocess self-test."""
    from virturoid.services.control_script_compiler import write_control_scripts, validate_scripts
    sdir = write_control_scripts(_compose("a robot dog"), tmp_path, task="patrol")
    assert sdir is not None
    report = validate_scripts(sdir)
    assert report["n_scripts"] == 6
    assert report["all_pass"] is True, report
    for name, r in report["scripts"].items():
        assert r["compiled"] and r["dry_run"], f"{name} failed: {r['detail']}"


def test_safety_filter_ceilings_are_the_bom_actuator_peak_torque():
    """The safety filter must clamp to the REAL actuator's peak torque, per joint -- not a guessed constant."""
    from virturoid.services.bom_builder import _DEFAULT_JOINT_TORQUE_NM, select_actuator
    from virturoid.services.control_script_compiler import compile_control_scripts
    g = _compose("a robot dog")
    out = compile_control_scripts(g, task="walk")
    ceils = out["config"]["torque_ceilings_nm"]
    assert ceils, "a legged robot has actuated joints"
    # each ceiling must equal the peak torque of the actuator the BOM would size for that joint
    for s in g.actuated_joints():
        act = select_actuator(getattr(s, "actuator_torque_nm", None) or _DEFAULT_JOINT_TORQUE_NM)
        key = "".join(c if c.isalnum() or c == "_" else "_" for c in s.name)
        assert ceils[key] == pytest.approx(round(float(act.peak_torque_nm), 3))


def test_obs_layout_mirrors_the_policy_kind():
    """A locomotion robot's obs tracks a base twist command; a manipulator's tracks a target pose. The obs_dim
    must equal the sum of the declared blocks (so the assembler and the net agree)."""
    from virturoid.services.control_script_compiler import compile_control_scripts
    dog = compile_control_scripts(_compose("a robot dog"))["config"]
    arm = compile_control_scripts(_compose("a 6-axis robot arm with a gripper"))["config"]
    assert dog["command_block"] == "velocity_command"       # locomotion tracks a twist
    assert arm["command_block"] == "target_pose"            # a manipulator tracks a pose
    for cfg in (dog, arm):
        assert cfg["obs_dim"] == sum(b["size"] for b in cfg["obs_layout"])


def test_safety_filter_actually_clamps_an_over_torque_command(tmp_path):
    """Run the generated safety_filter as a module and confirm an over-limit torque is clipped to the ceiling and
    flagged by audit() -- the script does what its docstring claims."""
    import subprocess
    import sys
    from virturoid.services.control_script_compiler import write_control_scripts
    sdir = write_control_scripts(_compose("a robot dog"), tmp_path, task="walk")
    probe = sdir / "_probe.py"
    probe.write_text(
        "import json\n"
        "from safety_filter import SafetyFilter, CEIL, JOINTS\n"
        "j = JOINTS[0]\n"
        "sf = SafetyFilter()\n"
        "clamped = sf.clamp_torques({j: CEIL[j]*100})\n"
        "assert abs(clamped[j]) <= CEIL[j] + 1e-6\n"
        "assert sf.audit({j: CEIL[j]*100}) == [(j, CEIL[j]*100, CEIL[j])]\n"
        "print('CLAMP_OK')\n", encoding="utf-8")
    proc = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True, cwd=str(sdir), timeout=30)
    assert "CLAMP_OK" in proc.stdout, proc.stderr


def test_state_machine_never_jumps_straight_from_estop_to_active(tmp_path):
    """A robot recovering from E-stop or a fall must pass through STAND, never resume the policy directly."""
    import subprocess
    import sys
    from virturoid.services.control_script_compiler import write_control_scripts
    sdir = write_control_scripts(_compose("a robot dog"), tmp_path, task="walk")
    probe = sdir / "_probe_sm.py"
    probe.write_text(
        "from state_machine import SafetyStateMachine, ESTOP, STAND, ACTIVE\n"
        "sm = SafetyStateMachine()\n"
        "sm.step(estop=True, tilt_rad=0.0, command_active=False)\n"
        "assert sm.state == ESTOP\n"
        "nxt = sm.step(estop=False, tilt_rad=0.0, command_active=True)\n"
        "assert nxt == STAND, f'jumped to {nxt} from ESTOP'\n"
        "print('SM_OK')\n", encoding="utf-8")
    proc = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True, cwd=str(sdir), timeout=30)
    assert "SM_OK" in proc.stdout, proc.stderr


def _inflate_ceilings(sdir, factor=10.0):
    p = sdir / "control_config.json"
    cfg = json.loads(p.read_text())
    cfg["torque_ceilings_nm"] = {k: round(v * factor, 3) for k, v in cfg["torque_ceilings_nm"].items()}
    p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg


# --------------------------------------------------------------------------------------------------------------
# THE DATASHEET-TORQUE AUDIT. The module's docstring promised it for months while `validate_scripts` was
# py_compile + a subprocess dry-run and nothing else, and `SafetyFilter.audit` had ZERO production callers. These
# five tests are the pin: each one goes GREEN only because the audit runs, so disabling it turns them red.
# --------------------------------------------------------------------------------------------------------------

def test_the_torque_audit_actually_ran_on_the_shipped_package(tmp_path):
    """MUTATION GUARD. If the audit is removed, stubbed, or silently downgraded to 'no_reference', this fails --
    the report must show all four checks executed against a real per-joint datasheet."""
    from virturoid.services.control_script_compiler import write_control_scripts
    write_control_scripts(_compose("a robot dog"), tmp_path, task="walk")
    audit = json.loads((tmp_path / "reports" / "script_validation.json").read_text())["torque_audit"]
    assert audit["status"] == "pass", audit
    assert audit["n_joints"] > 0
    assert audit["checks"] == {"every_joint_guarded": True, "declared_ceilings_are_datasheet": True,
                               "filter_clips_to_datasheet": True, "audit_names_the_breach": True}, audit


def test_a_ceiling_above_the_datasheet_peak_fails_validation_with_the_joint_and_margin_named(tmp_path):
    """THE REPRODUCTION, pinned. A stack whose declared ceilings are 10x the BOM actuator's datasheet peak used
    to validate GREEN (all_pass True, no mention of torque anywhere), and the shipped filter then clamped a
    41 Nm command at 41 Nm into a 4.1 Nm motor while `audit()` returned [] -- because `audit()` reads CEIL from
    the same config. It must now FAIL, naming the joint and the margin."""
    from virturoid.services.control_script_compiler import validate_scripts, write_control_scripts
    gene = _compose("a robot dog")
    sdir = write_control_scripts(gene, tmp_path, task="walk")
    _inflate_ceilings(sdir, 10.0)

    report = validate_scripts(sdir, gene=gene)
    assert report["all_pass"] is False, report["torque_audit"]
    assert report["torque_audit"]["status"] == "fail"
    # the CONFIG check catches the inflated number...
    assert report["torque_audit"]["checks"]["declared_ceilings_are_datasheet"] is False
    # ...and independently, EXECUTING the shipped filter catches the over-torque it lets through
    assert report["torque_audit"]["checks"]["filter_clips_to_datasheet"] is False
    joints = json.loads((sdir / "control_config.json").read_text())["joints"]
    blob = " ".join(report["torque_audit"]["violations"])
    assert any(j in blob for j in joints), blob            # the JOINT is named
    assert "10.0x" in blob and "datasheet peak" in blob, blob   # the MARGIN is named


def test_a_filter_that_clamps_only_some_joints_fails_validation(tmp_path):
    """The audit must EXECUTE the emitted filter over EVERY joint, not just read the config -- and it must reach
    where the dry-run cannot. The shipped ``self_test`` only probes ``JOINTS[:1]``, so a filter that clamps the
    first joint and passes the rest straight to the bus compiles, dry-runs, and self-tests GREEN while eleven
    other motors take 10x their datasheet peak. Only executing the filter across all joints finds it."""
    from virturoid.services.control_script_compiler import validate_scripts, write_control_scripts
    gene = _compose("a robot dog")
    sdir = write_control_scripts(gene, tmp_path, task="walk")
    src = (sdir / "safety_filter.py").read_text()
    clamp = "out[j] = t if c is None else max(-c, min(c, t))"
    gutted = src.replace(clamp, "out[j] = (t if c is None else max(-c, min(c, t))) if j == JOINTS[0] else t")
    assert gutted != src, "the clamp line moved -- update this mutation"
    (sdir / "safety_filter.py").write_text(gutted, encoding="utf-8")

    report = validate_scripts(sdir, gene=gene)
    assert all(r["compiled"] and r["dry_run"] for r in report["scripts"].values()), "compile+dry-run stays green"
    assert report["all_pass"] is False and report["torque_audit"]["status"] == "fail"
    assert report["torque_audit"]["checks"]["declared_ceilings_are_datasheet"] is True   # config is honest
    assert report["torque_audit"]["checks"]["filter_clips_to_datasheet"] is False        # behaviour is not
    assert "to the bus" in " ".join(report["torque_audit"]["violations"])


def test_a_script_we_did_not_emit_is_never_counted_as_torque_audited(tmp_path):
    """SCOPE HONESTY. The audit cannot prove hand-written downstream code routes commands through the filter, so
    a .py the compiler did not emit must be listed as unaudited rather than silently blessed."""
    from virturoid.services.control_script_compiler import validate_scripts, write_control_scripts
    gene = _compose("a robot dog")
    sdir = write_control_scripts(gene, tmp_path, task="walk")
    (sdir / "policy_runner.py").write_text(
        "import json\nfrom pathlib import Path\n"
        'CFG = json.loads((Path(__file__).with_name("control_config.json")).read_text())\n'
        'def command(_o): return {j: c * 10.0 for j, c in CFG["torque_ceilings_nm"].items()}\n'
        'if __name__ == "__main__": print("policy_runner self_test: OK")\n', encoding="utf-8")

    audit = validate_scripts(sdir, gene=gene)["torque_audit"]
    assert audit["unaudited_scripts"] == ["policy_runner.py"], audit["unaudited_scripts"]
    assert "does NOT audit" in audit["scope"] or "not audit" in audit["scope"].lower()


def test_a_failed_torque_audit_blocks_safe_to_export(tmp_path):
    """The readiness ladder must not report a green it did not earn: `controller_exported` attained on FILE
    PRESENCE alone while the guide told the customer the datasheet check had happened."""
    from virturoid.services.control_script_compiler import validate_scripts, write_control_scripts
    from virturoid.services.readiness_ledger import build_product_readiness_ledger
    gene = _compose("a robot dog")
    sdir = write_control_scripts(gene, tmp_path, task="walk")
    (tmp_path / "software" / "controller.py").write_text("# trained controller\n", encoding="utf-8")
    _inflate_ceilings(sdir, 10.0)
    (tmp_path / "reports" / "script_validation.json").write_text(
        json.dumps(validate_scripts(sdir, gene=gene), indent=2), encoding="utf-8")

    ledger = build_product_readiness_ledger(tmp_path, robot_class="legged", gene=gene)
    rec = ledger.by_stage["controller_exported"]
    assert rec.status == "below_gate", rec
    assert "datasheet-torque audit" in rec.detail and "Nm" in rec.detail
    assert ledger.safe_to_export is False, ledger.to_dict()["issues"]


def test_a_jointless_body_reads_not_applicable_never_passed(tmp_path):
    """VACUOUS GREEN. Every check in the audit is trivially true over an EMPTY joint set, so a quadcopter --
    rotors, zero actuated joints -- came back `{"status": "pass", "n_joints": 0}` with all four checks true,
    and the deployment guide then printed the whole paragraph in full: "Datasheet-torque audit passed
    (0 joints) ... `safety_filter.py` was **executed** on a command at 10x each joint's datasheet peak:
    nothing above the peak came back out, and `audit()` named every breach." Nothing was executed and there
    was nothing to execute it on. A pass over zero joints must read as NOT APPLICABLE.
    """
    from virturoid.services.control_script_compiler import write_control_scripts
    from virturoid.services.deployment_guide import build_deployment_guide
    gene = _compose("a quadcopter drone that flies")
    assert len(gene.actuated_joints()) == 0, "this body must be jointless for the case to exist"
    write_control_scripts(gene, tmp_path, task="fly")

    audit = json.loads((tmp_path / "reports" / "script_validation.json").read_text())["torque_audit"]
    assert audit["status"] == "not_applicable", audit
    assert audit["n_joints"] == 0
    assert audit["checks"] == {}, "a check that ran over nothing is not a check"
    assert "NOT APPLICABLE" in audit["scope"] and "not a passed safety check" in audit["scope"]

    guide = build_deployment_guide(tmp_path)
    assert "audit passed" not in guide.lower(), guide
    assert "was **executed**" not in guide
    assert "not applicable" in guide.lower()
    assert "NOT a passed safety check" in guide

    # and the readiness ladder must not launder it back into a green either
    from virturoid.services.readiness_ledger import build_product_readiness_ledger
    (tmp_path / "software").mkdir(exist_ok=True)
    (tmp_path / "software" / "controller.py").write_text("# controller\n", encoding="utf-8")
    rec = build_product_readiness_ledger(tmp_path, robot_class="aerial", gene=gene).by_stage[
        "controller_exported"]
    assert rec.evidence.get("torque_audit_passed") is False, rec
    assert "does not apply" in rec.detail, rec.detail


def test_validation_report_is_written_into_the_package(tmp_path):
    """The build must ship an honest reports/script_validation.json -- the compile+dry-run verdict, visible."""
    from virturoid.services.control_script_compiler import write_control_scripts
    write_control_scripts(_compose("a robot dog"), tmp_path, task="patrol")
    rep = tmp_path / "reports" / "script_validation.json"
    assert rep.exists()
    data = json.loads(rep.read_text())
    assert data["all_pass"] is True and data["n_scripts"] == 6
