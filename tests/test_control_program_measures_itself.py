"""``software/control_program.json`` must be a measurement of ``software/gait_controller.py``, not of some
other rollout that happened nearby.

MEASURED on a real Menagerie Unitree Go2 (2026-08-10), one export produced three forward distances:
``verify_robot`` -0.102 m, ``verification_certificate.json`` +0.329 m, ``software/control_program.json``
+0.356 m. The third was the worst of the three, because it was not a measurement of the shipped controller at
all. ``_verify_exported_gait`` ran ``recipe_rollout_morph(gene, MorphPolicy(seed=0), cpg=CPG_DEFAULT)`` under a
comment claiming "zero residual ... == the exported bare gait", and that rollout differs from the exported
program in five independent ways:

  1. ``MorphPolicy(seed=0)`` initialises from ``rng.normal(0, 0.3)`` — a RANDOM residual, not zero. Seeds 0/1/2
     gave -0.488 / -0.572 / -0.475 m on the Go2.
  2. Its PD attractor is ``mj_resetData``'s zero pose; the exported ``default_pose`` is read after
     ``_reset_to_rest``. On the Go2 those differ by **1.8 rad** on 8 of 12 joints — the file ships the Unitree
     home crouch, the rollout drove the robot with its legs straight out. That is the sign flip.
  3. No clamp to the joint position limits the shipped ``GaitController.infer`` applies.
  4. Every physics step (500 Hz) against a file declaring 20 Hz.
  5. ``abs()`` on the result, so the Go2 travelling **-0.488 m** shipped ``sim_forward_m: 0.488``.

The crawl branch had a smaller version of the same problem: it quoted the TUNING rollout's distance rather than
the frozen target program's (measured on a generated quadruped: search +0.94 m, shipped program +0.79 m).

These tests do not check a number. They check the relationship: the file's own claim must be reproducible by
running the file.
"""
from __future__ import annotations

import json

import pytest

from virturoid.services import gene_build as GB

mujoco = pytest.importorskip("mujoco")


def _walker():
    from virturoid.services.morphology_composer import compose_robot
    return compose_robot("a simple four-legged walking robot", ensure_walkable=True)


# ---------------------------------------------------------------- the writer measures the shipped controller
@pytest.mark.slow
def test_the_shipped_control_program_reproduces_its_own_number(tmp_path):
    """Load the WRITTEN files, drive that controller, and land on the distance the file prints."""
    gene = _walker()
    GB._write_gait_control_program(gene, {"id": "genome_t"}, tmp_path)
    prog_p = tmp_path / "software" / "control_program.json"
    ctl_p = tmp_path / "software" / "gait_controller.py"
    if not prog_p.is_file():
        pytest.skip("this body exports no gait control program")
    prog = json.loads(prog_p.read_text(encoding="utf-8"))

    # re-run through the writer's own harness, but from the SERIALISED file + the SERIALISED source, so a
    # round-trip that silently dropped or reordered a parameter would show up as a different distance.
    r = GB._run_exported_controller(gene, prog, ctl_p.read_text(encoding="utf-8"),
                                    steps=int(prog["measured_by"]["horizon_steps"]),
                                    control_hz=float(prog["control_frequency_hz"]))
    assert r["forward"] == pytest.approx(prog["sim_forward_m"], abs=0.02), (
        f"control_program.json claims {prog['sim_forward_m']} m but running the file it ships gives "
        f"{r['forward']} m")
    assert r["verdict"] == prog["sim_verdict"]
    assert prog["verified_walk"] is str(prog["sim_verdict"]).startswith("CREDIBLE")


@pytest.mark.slow
def test_the_declared_control_rate_is_the_rate_it_was_measured_at(tmp_path):
    """``control_frequency_hz`` was the literal 20.0 while the number beside it came from a 500 Hz rollout. On
    the Go2 that is not a rounding difference: the same controller travels +0.035 m at 20 Hz and +0.330 m
    continuous, so the file was advertising ~9x the travel its own declared rate delivers."""
    gene = _walker()
    GB._write_gait_control_program(gene, {"id": "genome_t"}, tmp_path)
    prog_p = tmp_path / "software" / "control_program.json"
    if not prog_p.is_file():
        pytest.skip("this body exports no gait control program")
    prog = json.loads(prog_p.read_text(encoding="utf-8"))
    mb = prog["measured_by"]
    assert mb["control_hz"] == prog["control_frequency_hz"], (
        "the declared deployment rate and the rate the distance was measured at must be one number")
    # ...and the decimation is real, not a label: at 50 Hz on a 500 Hz sim the target is held for 10 steps.
    assert mb["control_period_physics_steps"] == max(
        1, round(1.0 / (mb["control_hz"] * mb["physics_timestep_s"])))
    assert mb["control_period_physics_steps"] > 1, "a decimation of 1 means it was NOT sampled at control_hz"
    # the two rates a package can name are named as different things
    assert "sysid" in prog["control_frequency_hz_meaning"] or "identification" in \
        prog["control_frequency_hz_meaning"]


def test_forward_is_signed_so_a_backward_walker_cannot_read_as_forward(monkeypatch, tmp_path):
    """``abs()`` turned the Go2's measured **-0.488 m** into an advertised ``sim_forward_m: 0.488``. A body
    walking away from the goal must not be able to report travel toward it, so the sign is asserted here on a
    stubbed rollout rather than hoped for on a real one."""
    monkeypatch.setattr(GB, "_run_exported_controller",
                        lambda *a, **k: {"forward": -0.488, "verdict": "CROUCH (low/unstable stance)",
                                         "survived": True, "cadence": 4.4, "upright_frac": 0.3,
                                         "support_frac": 0.6, "measured_by": {"control_hz": 50.0}})
    v = GB._verify_exported_gait(object(), {"policy_type": "trot_cpg_gait"}, "")
    assert v["forward_m"] == -0.488, "the writer must not absolute-value a backward walk into forward travel"
    assert v["credible"] is False

    # ...and it reaches the FILE with its sign intact.
    monkeypatch.setattr(GB, "extract_crawl_gait_params", lambda g: None, raising=False)
    import virturoid.services.morph_policy as MP
    monkeypatch.setattr(MP, "extract_crawl_gait_params", lambda g: None)
    monkeypatch.setattr(MP, "extract_gait_params",
                        lambda g: {"policy_type": "trot_cpg_gait", "joint_names": ["j"], "default_pose": [0.0],
                                   "amplitude": [0.1], "phase_offset": [0.0], "frequency_hz": 1.5,
                                   "position_limits": [[-1.0, 1.0]]})
    GB._write_gait_control_program(object(), {"id": "g"}, tmp_path)
    prog = json.loads((tmp_path / "software" / "control_program.json").read_text(encoding="utf-8"))
    assert prog["sim_forward_m"] == -0.488
    assert prog["verified_walk"] is False


# ---------------------------------------------------------------- the stamp the certificate reads
def test_the_stamp_is_cleared_for_a_body_that_ships_no_controller(tmp_path):
    """A stale stamp would make the certificate describe a controller that is not in the package."""
    from virturoid.schemas.gene import GeneSegment, RobotGene
    tip = GeneSegment(name="link1", parent="base", joint_type="revolute", length_m=0.3, radius_m=0.03,
                      mass_kg=0.5, joint_lower=-1.5, joint_upper=1.5)
    tip.is_end_effector = True
    arm = RobotGene(id="A", species="t.A", robot_class="manipulator", base_mount="table",
                    end_effector_type="gripper",
                    segments=[GeneSegment(name="base", parent=None, joint_type=None, length_m=0.2,
                                          radius_m=0.05, mass_kg=1.0), tip])
    GB._stamp_exported_controller(arm, {"policy_type": "trot_cpg_gait", "program_fingerprint": "stale"})
    assert GB.exported_controller_stamp(arm) is not None
    GB._write_gait_control_program(arm, {"id": "genome_a"}, tmp_path)
    assert GB.exported_controller_stamp(arm) is None, (
        "a body that exports no control program must not carry a previous export's stamp")


@pytest.mark.slow
def test_the_stamp_and_the_written_file_agree_on_the_fingerprint(tmp_path):
    gene = _walker()
    GB._write_gait_control_program(gene, {"id": "genome_t"}, tmp_path)
    prog_p = tmp_path / "software" / "control_program.json"
    if not prog_p.is_file():
        pytest.skip("this body exports no gait control program")
    prog = json.loads(prog_p.read_text(encoding="utf-8"))
    stamp = GB.exported_controller_stamp(gene)
    assert stamp is not None
    assert stamp["program_fingerprint"] == prog["program_fingerprint"]
    assert stamp["sim_forward_m"] == prog["sim_forward_m"]
    assert stamp["policy_type"] == prog["policy_type"]
    # the fingerprint names the CONTROLLER, so re-measuring must not move it, but a parameter change must
    assert GB._program_fingerprint({**prog, "sim_forward_m": 999.0}) == prog["program_fingerprint"]
    assert GB._program_fingerprint({**prog, "frequency_hz": 9.9}) != prog["program_fingerprint"]
