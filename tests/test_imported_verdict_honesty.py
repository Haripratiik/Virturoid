"""A verdict is a claim about a BODY UNDER A CONTROLLER — and on a customer's own robot we have the body and
not the controller.

MEASURED DEFECT (2026-08-04, real MuJoCo Menagerie Go2 through ``agent_tools.call_tool``): ingest read the mass
as 15.206 kg — exactly Unitree's — and ``verify_robot`` answered "FELL by YAW-DRIFT (roll 151 / pitch 86 / yaw
174 deg max), survived False". Every number was true and the sentence was useless: a Go2 walks perfectly well
with Unitree's controller. We drove the customer's machine with OUR generic crawl gait, it fell, and we
published the fall in a form an engineer reads as a finding about their robot.

These tests pin the fix and, just as importantly, pin what the fix must NOT do:

  * ``gait_quality.classify`` is untouched and its string is carried through byte-identical. The un-gameable
    verdict is the differentiator; the defect was applying it to a question the customer did not ask.
  * a body WE composed keeps the classify verdict as its headline, because our generic gait IS the shipped
    controller for a body we designed.

They run against REAL Menagerie packages when the cache is present (fixtures have lied in this repo before) and
otherwise skip; the composed-body no-regression test needs nothing but MuJoCo and always runs.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="import + verify need MuJoCo")

_MENAGERIE = os.path.join(os.path.expanduser("~"), ".cache", "robot_descriptions", "mujoco_menagerie")


def _menagerie(pkg: str, model: str) -> str:
    p = os.path.join(_MENAGERIE, pkg, model)
    if not os.path.exists(p):
        pytest.skip(f"MuJoCo Menagerie not cached at {p}")
    return p


def _imported(pkg: str, model: str):
    from virturoid.services.robot_import import import_robot
    g = import_robot(_menagerie(pkg, model))["gene"]
    assert g is not None, f"{pkg} did not import"
    return g


def _verify(gene, label: str, mode: str = "quick") -> dict:
    from virturoid.services import session_state as S
    from virturoid.services.agent_tools import call_tool
    rid = S.put_robot(gene, prompt=label, label=label)
    try:
        env = call_tool("verify_robot", {"robot_id": rid, "mode": mode})
    finally:
        S.forget_robot(rid)
    assert env.get("ok"), env
    return env["result"]


# ---------------------------------------------------------------------------------------- the refusal itself
def test_imported_go2_gets_no_locomotion_verdict_under_our_generic_gait():
    """The headline on the customer's own Go2 must not be a fall verdict."""
    res = _verify(_imported("unitree_go2", "go2.xml"), "customer go2")
    assert res["kind"] == "legged"
    verdict = str(res["verdict"])
    assert verdict.startswith("NO LOCOMOTION VERDICT"), verdict
    assert "we do not have your robot's controller" in verdict
    assert res["locomotion_verdict"] is None
    assert res["credible_walk"] is False              # we never claim a walk we did not measure
    # the un-gameable classifier still ran, and its string is kept verbatim — one level down, scoped
    demoted = res["under_our_generic_gait"]
    assert demoted["verdict"] and "our generic crawl gait" in demoted["gait"]
    assert "NOT as a finding about your machine" in demoted["means"]


def test_the_classify_string_is_carried_through_byte_identical(monkeypatch):
    """The fix reframes; it must not soften, rewrite or suppress the un-gameable verdict.

    Hints are disabled for both calls (the documented hermetic switch): the recalled gait region depends on
    whatever the flywheel has banked, so leaving it on would compare two rollouts run under different gaits.
    """
    monkeypatch.setenv("VIRTUROID_DISABLE_GAIT_HINTS", "1")
    from virturoid.services.ai_native_tools import _honest_gait
    from virturoid.services.gait_quality import classify
    gene = _imported("unitree_go2", "go2.xml")
    res = _verify(gene, "go2 classify parity")
    assert res["gait_source"] == "default_crawl", res["gait_source"]
    demoted = res["under_our_generic_gait"]["verdict"]
    # the same call the tool makes, at the same horizon, must produce the same classify() sentence
    again = _honest_gait(gene, steps=int(res["horizon_steps"]), render=False, tag="test_parity")
    assert demoted == again["verdict"], f"reframe altered the verdict: {demoted!r} vs {again['verdict']!r}"
    assert classify({}) == "CROUCH (low/unstable stance)"          # classifier itself untouched


def test_controller_provenance_names_whose_controller_it_was():
    res = _verify(_imported("unitree_go2", "go2.xml"), "go2 provenance")
    prov = res["controller_provenance"]
    assert prov["yours_was_not_supplied"] is True
    assert prov["verdict_is_about"] == "our controller on your body"
    assert res["imported"]["mass_from_source"] is True             # their mass, read at import
    assert res["imported"]["torque_from_source"] is True           # their torque limits too
    # the refusal is precise: it lists what we DID take, so "we don't have your controller" is checkable
    took = " ; ".join(prov["what_we_took_from_your_model"])
    assert "torque limits" in took and "masses" in took and "rest pose" in took, took
    assert "ZERO ship a locomotion controller" in prov["what_your_model_could_not_supply"]


def test_it_says_what_would_actually_answer_the_question():
    """A refusal that leaves the engineer with nowhere to go is only half honest."""
    res = _verify(_imported("unitree_go2", "go2.xml"), "go2 next steps")
    routes = res["to_get_a_real_verdict"]
    assert len(routes) >= 3
    joined = " ".join(f"{r['if']} {r['do']}" for r in routes)
    for tool in ("import_onnx_policy", "sandbox_policy", "adapt_gait", "train_held", "probe_robot"):
        assert tool in joined, f"{tool} not offered as a route to a real verdict"


# ------------------------------------------------------------- what we CAN state without their controller
def test_a_real_quadruped_stands_at_its_own_declared_home_pose():
    """The positive half of the refusal, measured on four real quadrupeds from three vendors."""
    from virturoid.services.ai_native_tools import _controller_free_facts
    for pkg, model in (("unitree_go2", "go2.xml"), ("boston_dynamics_spot", "spot.xml"),
                       ("anybotics_anymal_c", "anymal_c.xml"), ("unitree_a1", "a1.xml")):
        facts = _controller_free_facts(_imported(pkg, model))
        stand = facts["stands_under_gravity"]
        assert stand["stands"] is True, f"{pkg} did not hold its own declared pose: {stand}"
        assert stand["tilt_from_spawn_deg"] < 15.0, f"{pkg} tilted: {stand}"
        assert facts["joint_limits_respected"]["ok"], f"{pkg} home pose violates its own limits: {facts}"
        assert facts["static_holding_torque"]["ok"], f"{pkg} needs more torque than it declares: {facts}"


def test_tilt_is_relative_to_spawn_not_an_absolute_euler_angle():
    """An imported root carries the rotation that aligns its reconstructed link frame, so the Go2's base
    quaternion reads roll 90.0 / pitch 76.5 deg AT STEP ZERO on a body standing square on four feet. An
    absolute gate would fail every imported robot before physics ran."""
    import mujoco
    import numpy as np

    from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
    gene = _imported("unitree_go2", "go2.xml")
    m = mujoco.MjModel.from_xml_string(
        compile_gene_to_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene)))
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    w, x, y, z = (float(d.qpos[3]), float(d.qpos[4]), float(d.qpos[5]), float(d.qpos[6]))
    abs_roll = abs(float(np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))))
    assert abs_roll > 45.0, "premise gone: the imported root no longer carries a frame rotation"
    from virturoid.services.ai_native_tools import _controller_free_facts
    assert _controller_free_facts(gene)["stands_under_gravity"]["stands"] is True


def test_a_biped_that_goes_down_under_a_posture_hold_is_reported_INCONCLUSIVE():
    """Balancing 2 legs needs an ACTIVE controller — the very thing we said we do not have. Calling that a
    failure of the customer's machine would be a brand-new false statement."""
    from virturoid.services.ai_native_tools import _controller_free_facts
    stand = _controller_free_facts(_imported("unitree_g1", "g1.xml"))["stands_under_gravity"]
    assert stand["stands"] is None and stand["conclusive"] is False, stand
    assert "not statically stable" in stand["why_inconclusive"]
    assert "says nothing about whether your machine balances" in stand["why_inconclusive"]


def test_a_mounted_arm_has_no_stance_to_hold():
    from virturoid.services.ai_native_tools import _controller_free_facts
    facts = _controller_free_facts(_imported("universal_robots_ur5e", "ur5e.xml"))
    assert facts["stands_under_gravity"]["applicable"] is False
    assert facts["joint_limits_respected"]["ok"]


# ------------------------------------------------------------------------------- the manipulator half
def test_an_arm_with_no_gripper_is_told_so_not_scored_zero_at_grasping():
    """A UR5e ships no gripper. "GRASP WEAK (success 0%)" reads as a capability judgement on a hand that does
    not exist."""
    res = _verify(_imported("universal_robots_ur5e", "ur5e.xml"), "customer ur5e", mode="full")
    assert res["kind"] == "manipulator"
    verdict = str(res["verdict"])
    assert "ARTICULATES" in verdict                       # kinematic reach needs no controller — it stays
    assert "NO GRIPPER in your model" in verdict, verdict
    assert "GRASP WEAK" not in verdict, verdict


def test_reach_is_kinematics_so_it_carries_no_controller_caveat():
    """A quick verify on an arm measures REACH — link lengths and joint ranges, no controller anywhere in it.
    Bolting "under our controller, not yours" onto a ruler would be a caveat with nothing behind it."""
    res = _verify(_imported("universal_robots_ur5e", "ur5e.xml"), "ur5e quick", mode="quick")
    assert str(res["verdict"]).startswith("ARTICULATES"), res["verdict"]
    assert "not yours" not in str(res["verdict"])
    assert res["controller_provenance"]["yours_was_not_supplied"] is True   # provenance still disclosed


def test_a_real_gripper_driven_by_our_script_is_scoped_not_judged():
    res = _verify(_imported("franka_emika_panda", "panda.xml"), "customer panda", mode="full")
    verdict = str(res["verdict"])
    assert "ARTICULATES" in verdict and "NO GRASP VERDICT" in verdict, verdict
    assert "we do not have your gripper's controller" in verdict
    assert res["grasp"]["under"] == "our generic grasp-and-lift script"
    assert "NOT as a finding about your gripper" in res["grasp"]["means"]


# ----------------------------------------------------------------------- what must NOT change
def test_a_body_we_composed_keeps_its_classify_verdict_as_the_headline():
    """Our generic gait IS the shipped controller for a body we designed, so the verdict there is exactly the
    claim we are entitled to make. None of the imported-body scoping may leak onto it."""
    from virturoid.services.morphology_composer import compose_robot
    gene = compose_robot("a walking quadruped robot dog", llm=None)
    assert not (getattr(gene, "metadata", None) or {}).get("imported_from")
    res = _verify(gene, "composed dog")
    assert "NO LOCOMOTION VERDICT" not in str(res["verdict"])
    for leaked in ("controller_provenance", "under_our_generic_gait", "measured_without_your_controller",
                   "to_get_a_real_verdict", "imported", "locomotion_verdict"):
        assert leaked not in res, f"imported-body field {leaked!r} leaked onto a composed body"


def test_a_controller_we_fitted_to_their_body_is_a_claim_we_may_make():
    """The refusal is conditional on having no controller FOR THIS BODY — not on the body being imported. Once
    we fit an operating point to their gene, the verdict describes a controller-body pair we measured."""
    from virturoid.services.ai_native_tools import _reframe_for_imported_body
    gene = _imported("unitree_go2", "go2.xml")
    res = {"kind": "legged", "verdict": "CREDIBLE WALK", "gait_source": "tuned_for_this_body", "forward_m": 1.2}
    _reframe_for_imported_body(res, gene, "legged")
    assert res["verdict"] == "CREDIBLE WALK", "a fitted controller's verdict must stand"
    assert res["controller_provenance"]["whose"] == "ours, fitted to this body"
    assert "under_our_generic_gait" not in res


def test_posture_hold_gains_are_clipped_for_integrator_stability():
    """Sizing kp straight off a big torque limit detonates the hold on exactly the heavy bodies it exists for:
    an imported Spot ran at kd*dt/I ~ 1.3 and diverged to 0.594 rad of tracking error."""
    import mujoco

    from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
    from virturoid.services.morph_policy import posture_hold_gains
    gene = _imported("boston_dynamics_spot", "spot.xml")
    m = mujoco.MjModel.from_xml_string(
        compile_gene_to_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene)))
    kp, kd, tau = posture_hold_gains(m)
    assert m.nu > 0 and float(kp.max()) > 0.0
    dt = float(m.opt.timestep)
    M = __import__("numpy").zeros((m.nv, m.nv))
    d = mujoco.MjData(m); mujoco.mj_forward(m, d); mujoco.mj_fullM(m, M, d.qM)
    for u in range(m.nu):
        if int(m.actuator_trntype[u]) != int(mujoco.mjtTrn.mjTRN_JOINT) or kp[u] <= 0:
            continue
        inertia = max(1e-4, float(M[int(m.jnt_dofadr[int(m.actuator_trnid[u, 0])])][
            int(m.jnt_dofadr[int(m.actuator_trnid[u, 0])])]))
        assert kd[u] * dt / inertia <= 0.51, f"actuator {u}: damping above the explicit-integrator limit"
        assert kp[u] * dt * dt / inertia <= 0.26, f"actuator {u}: stiffness above the explicit-integrator limit"
