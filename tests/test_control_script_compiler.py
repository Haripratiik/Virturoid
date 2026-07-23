"""WS-S / S2: the operational control scripts must be GENERATED from the robot (joints + BOM-sized actuators +
policy obs layout), and every shipped .py must COMPILE and DRY-RUN. The safety filter's torque ceilings are the
load-bearing honesty claim -- they must be the real actuator's peak torque, so a command that would exceed a
motor is caught. These tests pin that.
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


def test_validation_report_is_written_into_the_package(tmp_path):
    """The build must ship an honest reports/script_validation.json -- the compile+dry-run verdict, visible."""
    from virturoid.services.control_script_compiler import write_control_scripts
    write_control_scripts(_compose("a robot dog"), tmp_path, task="patrol")
    rep = tmp_path / "reports" / "script_validation.json"
    assert rep.exists()
    data = json.loads(rep.read_text())
    assert data["all_pass"] is True and data["n_scripts"] == 6
