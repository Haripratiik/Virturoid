"""The customer's declared joint dynamics must reach the MODEL, not just a metadata key -- and the ONE of them
that is not a statement about their machine must not.

``test_imported_actuator_limits`` already pinned that ``robot_import`` READS armature/damping/frictionloss off
the source and records them on the gene. It did not pin that anything downstream uses them, and nothing did:
``gene_compiler._joint_dynamics_prior`` selected a structural guess from the segment's NAME and never looked
at the record. Measured through ``agent_tools.call_tool`` on a real Menagerie Unitree Go2, which declares
``damping=2.0 frictionloss=0.2`` on all 12 leg joints:

    OUR COMPILED TWIN   damping 0.80 (-60%)   frictionloss 0.12 (-40%)

and on a Panda (``damping=1.0``, dry friction declared NOWHERE) 2.0 / 0.45 -- i.e. we invented 0.45 N.m of
Coulomb friction on a joint whose author declared none.

Why this is worse than a fidelity bug, and why the sysid assertions below are the point of the file:
``sysid.fit.fit_parameters`` reads each parameter's baseline straight off the compiled model's ``dof_*``
arrays. So the substitution became the thing the fit "identified": on a pinned synthetic Go2 (one absolute
physical drivetrain, each arm given the delta that reaches it from its own model) the fit reported

    damping       +1.7506 (+218.8%)  before   ->  +0.6312 (+31.6%) after   =  63.9% of it was our own default
    frictionloss  +0.1435 (+119.6%)  before   ->  +0.0616 (+30.8%) after   =  57.1% of it was our own default

Most of the headline correction on both was self-referential -- a measurement of our own substitution sold as
a measurement of the customer's hardware.

AND ARMATURE IS NOT THE THIRD OF THESE, which the first version of this change got wrong and shipped. Carrying
it too moved simulated dynamics across the product and turned eight green gates red; a one-parameter-at-a-time
ablation put armature alone on the wrong side of every one of them (talos coupling residual 0.00338 ->
0.13119 rad, toddlerbot 0.02373 -> 1.88989, Cassie's loop closure stopped tightening, and a real Spot stopped
holding its own home pose). It is not a claim about hardware the way damping and Coulomb friction are: it is
added to the diagonal of ``qM`` to condition the solver of the model it was written in, MuJoCo's default is 0
so a compiled model cannot say whether it was declared at all, and 14 of these 59 packages read 0 on EVERY
joint. So armature is RECORDED, DISCLOSED, and left on our prior -- see
``gene_compiler._declared_joint_dynamics``, and ``test_the_armature_is_recorded_and_deliberately_not_carried``
below, which is the regression that keeps it that way.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None
_MEN = Path(os.path.expanduser("~/.cache/robot_descriptions/mujoco_menagerie"))
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="import needs MuJoCo")

PARAMS = ("damping", "armature", "frictionloss")
#: The two the twin CARRIES from the source. ``armature`` is deliberately absent — see the module docstring.
CARRIED = ("damping", "frictionloss")


def _import(rel: str):
    src = _MEN / rel
    if not src.is_file():
        pytest.skip(f"{rel} is not cached locally (robot_descriptions fetches on demand)")
    from virturoid.services.robot_import import import_robot
    return import_robot(str(src), robot_id="t")


def _dyn(model, joint_name: str) -> dict:
    """``{damping, armature, frictionloss}`` off a compiled model, by joint NAME."""
    import mujoco
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    assert j >= 0, f"no joint {joint_name!r} in this model"
    adr = int(model.jnt_dofadr[j])
    return {p: round(float(getattr(model, f"dof_{p}")[adr]), 6) for p in PARAMS}


def _compiled(gene):
    import mujoco

    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    return mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene, include_floor=False))


def _source(rel: str):
    import mujoco
    return mujoco.MjModel.from_xml_path(str(_MEN / rel))


def _a_leg_joint(gene):
    """A joint the compiler's LEGGED prior branch (0.8 / 0.01 / 0.12) actually selects, or a skip.

    Asserting against that exact triple is only meaningful on a segment whose name reaches that branch, and
    the composer owns its own naming — so a rename must skip this test rather than fail it with a
    ``StopIteration`` that reads like the carry-through broke.
    """
    leg = next((s for s in gene.actuated_joints()
                if any(t in (s.name or "").lower()
                       for t in ("leg", "hip", "knee", "ankle", "thigh", "shin", "calf"))), None)
    if leg is None:
        pytest.skip("composed quadruped has no leg-named joint; the legged prior branch is unreachable here")
    return leg


# --------------------------------------------------------------- the carry-through, end to end
@pytest.mark.parametrize("rel,segment,joint", [
    ("unitree_go2/go2.xml", "FL_calf", "FL_calf_joint"),
    ("unitree_go2/go2.xml", "FL_hip", "FL_hip_joint"),
    ("franka_emika_panda/panda.xml", "link1", "joint1"),
    ("franka_emika_panda/panda.xml", "link5", "joint5"),
])
def test_the_declared_drivetrain_reaches_the_compiled_model(rel, segment, joint):
    """Not "is recorded on the gene" -- is what MuJoCo integrates, on the model the customer verifies."""
    gene = _import(rel)["gene"]
    want = _dyn(_source(rel), joint)
    got = _dyn(_compiled(gene), f"{segment}_joint")
    assert {p: got[p] for p in CARRIED} == pytest.approx({p: want[p] for p in CARRIED}, abs=1e-4), (
        f"{rel}:{segment} — our twin integrates {got} against the source's own {want}")


def test_the_go2s_numbers_are_the_ones_the_defect_report_named():
    """The exact figures, so a regression cannot pass by moving both sides of the comparison.

    The armature reads 0.01 here on BOTH sides and that is a coincidence, not a carry: the legged branch's
    structural prior happens to be the number Unitree declares. ``test_the_armature_is_recorded_and_
    deliberately_not_carried`` uses a body where the two disagree, so the coincidence cannot hide a regression.
    """
    model = _compiled(_import("unitree_go2/go2.xml")["gene"])
    for seg in ("FL_hip", "FL_thigh", "FL_calf", "RR_calf"):
        assert _dyn(model, f"{seg}_joint") == {"damping": 2.0, "armature": 0.01, "frictionloss": 0.2}


# --------------------------------------------------------------- and the one that must NOT reach the model
def test_the_armature_is_recorded_and_deliberately_not_carried():
    """ARMATURE IS A SOLVER SETTING OF THEIR MODEL, NOT A PROPERTY OF THEIR MACHINE, and carrying it broke
    eight gates that had held.

    Measured, one parameter at a time, on the exact gates that moved (prior -> carrying only this parameter):

        pal_talos coupled-joint residual        0.00338 -> 0.13119 rad   (damping 0.00332, friction 0.00285)
        toddlerbot coupled-joint residual       0.02373 -> 1.88989 rad   (damping 0.01370, friction 0.02371)
        cassie loop-closure gap / MuJoCo default  0.495 -> 0.720         (damping 0.496,   friction 0.492)
        boston_dynamics_spot holds its home pose   True -> True*         (*only the THREE together fail it:
                                                   sag 0.038 -> 0.293 m, tilt 2.6 -> 31.8 deg)

    Damping and frictionloss are on the right side of every one; armature is on the wrong side of every one.
    Two reasons, either sufficient: a compiled ``dof_armature`` of 0 cannot be distinguished from "never
    declared" (14 of the 59 packages read 0 on EVERY joint, this Panda's neighbours among them), and armature
    conditions the mass matrix of the model it was written in — ours adds its own equality constraints over
    primitive link inertias, so their margin is not ours to inherit.

    The Panda is the case where the two disagree with no ambiguity: it declares ``armature=0.1`` on joint1 and
    our manipulator prior is 0.14, so this test fails the moment the carry comes back.
    """
    res = _import("franka_emika_panda/panda.xml")
    gene = res["gene"]
    assert gene.metadata["source_joint_dynamics"]["link1"]["armature"] == pytest.approx(0.1), (
        "premise gone: the Panda no longer declares an armature that differs from our prior")
    assert gene.metadata["source_joint_dynamics_carried_params"] == ["damping", "frictionloss"]
    got = _dyn(_compiled(gene), "link1_joint")
    assert got["armature"] == pytest.approx(0.14, abs=1e-4), (
        f"the source's armature reached the model: {got}")
    assert got["damping"] == pytest.approx(1.0, abs=1e-4)          # ...while the two that DO carry, carried
    # And it is disclosed with the customer's own number rather than dropped in silence.
    hit = [w for w in res["warnings"] if "armature" in w and "NOT carried" in w]
    assert len(hit) == 1 and "0.1-0.1" in hit[0], res["warnings"]


def test_a_declared_zero_is_carried_rather_than_replaced_by_an_invented_friction():
    """The Panda declares NO ``frictionloss``, so its simulator integrates 0. Ours invented 0.45 N.m on
    joint1 and 0.24 on joint5 -- a stiction that exists nowhere but in our compiler, on the parameter a bench
    fit is least able to distinguish from real hardware."""
    gene = _import("franka_emika_panda/panda.xml")["gene"]
    model = _compiled(gene)
    declared = gene.metadata["source_joint_dynamics"]
    assert declared, "the Panda's drivetrain record is missing; this test would be vacuous"
    invented = {n: _dyn(model, f"{n}_joint")["frictionloss"] for n in declared
                if declared[n]["frictionloss"] == 0.0}
    assert invented and not any(invented.values()), (
        f"we put dry friction on joint(s) whose author declared none: "
        f"{ {n: v for n, v in invented.items() if v} }")


def test_the_record_still_reaches_the_model_after_a_serialization_round_trip():
    """``metadata`` is free-form and the gene is persisted as JSON between every tool call, so the carry has
    to survive the trip a held robot actually makes."""
    from virturoid.schemas.gene import RobotGene
    gene = _import("unitree_go2/go2.xml")["gene"]
    assert _dyn(_compiled(RobotGene.from_dict(gene.to_dict())), "FL_calf_joint") == {
        "damping": 2.0, "armature": 0.01, "frictionloss": 0.2}


# --------------------------------------------------------------- the self-referential sysid gap
def test_the_sysid_baseline_is_the_customers_number_not_our_default():
    """``fit_parameters`` reads each parameter's ``prior:`` off the compiled model's ``dof_*`` arrays, so the
    baseline a correction is quoted against IS this number. While it was ours, the fit's headline was partly a
    measurement of our own substitution."""
    from virturoid.services.gene_compiler import _joint_dynamics_prior
    gene = _import("unitree_go2/go2.xml")["gene"]
    seg = next(s for s in gene.actuated_joints() if s.name == "FL_calf")
    assert _joint_dynamics_prior(gene, seg) == (2.0, 0.01, 0.2)


def test_the_bench_rig_measures_the_declared_drivetrain_too():
    """``bench_rig.bench_model`` welds the base to a stand, and the prior branches on ``base_mount`` -- so a
    weld used to move every leg joint onto the manipulator prior. A DECLARED value must be immune to that:
    welding a robot to a bench does not change what its file says its joints are."""
    from virturoid.services.sysid.bench_rig import bench_model
    gene = _import("unitree_go2/go2.xml")["gene"]
    model, meta = bench_model(gene)
    assert meta["fixed_base"]
    assert _dyn(model, "FL_calf_joint") == {"damping": 2.0, "armature": 0.01, "frictionloss": 0.2}


def test_a_calibration_still_outranks_the_declared_value():
    """Precedence: a measurement of the customer's REAL hardware > the customer's declared model > our prior.
    The declared value is the baseline a fit corrects, never a floor that blocks one."""
    import dataclasses

    from virturoid.services.sysid.calibration import CALIBRATION_ARTIFACT, CALIBRATION_KEY
    gene = _import("unitree_go2/go2.xml")["gene"]
    rec = {"artifact": CALIBRATION_ARTIFACT, "applied_to_model": True,
           "joints": {"FL_calf": {"damping": {"from": 2.0, "to": 3.25, "delta": 1.25,
                                              "value_interval": [3.0, 3.5], "unit": "N.m.s/rad"}}}}
    calibrated = dataclasses.replace(gene, metadata={**(gene.metadata or {}), CALIBRATION_KEY: rec})
    got = _dyn(_compiled(calibrated), "FL_calf_joint")
    assert got["damping"] == pytest.approx(3.25, abs=1e-4)
    assert got["frictionloss"] == pytest.approx(0.2, abs=1e-4)     # untouched params keep the declaration


# --------------------------------------------------------------- disclosure, in the customer's own numbers
def test_the_import_discloses_the_drivetrain_with_its_numbers():
    """Mass and torque are loudly preserved and disclosed; this was carried by neither label nor warning."""
    res = _import("unitree_go2/go2.xml")
    gene = res["gene"]
    assert gene.metadata["joint_dynamics_source"] == "source_model"
    hit = [w for w in res["warnings"] if "drivetrain" in w and "AUTHORITATIVE" in w]
    assert len(hit) == 1, res["warnings"]
    assert "damping 2-2" in hit[0] and "frictionloss 0.2-0.2" in hit[0]
    # ...and the one we do NOT carry is in the same sentence with the customer's own number, so "we kept your
    # drivetrain" stays a checkable claim rather than a claim about two thirds of it.
    assert "armature (0.01-0.01) is recorded but NOT carried" in hit[0], hit[0]


def test_a_zero_is_disclosed_as_unattributable_rather_than_claimed_as_a_declaration():
    """MuJoCo's default for all three is 0 and a compiled model keeps no record of which attributes the XML
    wrote, so a 0 cannot be attributed. Claiming to know would be the same over-claim as the substitution."""
    res = _import("franka_emika_panda/panda.xml")
    zeros = res["gene"].metadata["source_joint_dynamics_zero_in_source"]
    assert zeros["link1"] == ["frictionloss"]
    assert any("cannot attribute" in w and "0.0 IS carried" in w for w in res["warnings"])


def test_a_declared_spring_is_disclosed_with_its_number_even_though_it_cannot_be_carried():
    """Cassie's leaf springs are 1250 and 1500 N.m/rad of joint ``stiffness``. A ``RobotGene`` has no field
    for it, no bench fit identifies it, and the twin silently became a machine without springs. It still
    cannot be carried -- but the number and the reason are now in the report."""
    res = _import("agility_cassie/cassie.xml")
    nc = res["gene"].metadata["source_joint_attributes_not_carried"]
    assert nc["left-shin"] == {"stiffness": 1500.0}
    assert nc["left-heel-spring"] == {"stiffness": 1250.0}
    # Matched on the phrase specific to THIS disclosure: the armature clause is also a "NOT carried" sentence,
    # and a matcher that cannot tell two honest disclosures apart is not pinning either of them.
    hit = [w for w in res["warnings"] if "attributes a RobotGene has no " in w]
    # The example named is the LARGEST spring, not whichever joint sorts first — a solver setting must not
    # crowd out a 1500 N.m/rad structural member in the one line a reader skims.
    assert len(hit) == 1, hit
    assert "left-shin" in hit[0] and "1500" in hit[0] and "RETURN SPRING" in hit[0], hit[0]


def test_a_model_that_declares_nothing_unusual_reports_nothing_not_carried():
    """The disclosure has to be driven by a DIFFERENCE from MuJoCo's default, or it is noise on every import
    and gets skipped. The Go2 sets no stiffness/margin/solref, so it must produce no such warning."""
    res = _import("unitree_go2/go2.xml")
    assert "source_joint_attributes_not_carried" not in (res["gene"].metadata or {})
    assert not [w for w in res["warnings"] if "attributes a RobotGene has no " in w]
    # The Go2 drives with <motor>, so there is no actuator-side kv to lose either. Both silences are earned.
    assert "source_actuator_velocity_feedback_not_carried" not in (res["gene"].metadata or {})


# --------------------------------------------------------------- the blast radius, and the defensive path
def test_a_composed_body_is_untouched_and_keeps_the_structural_prior():
    """Only ``robot_import`` writes the record, so every generated body — the whole gait bank — must compile
    exactly as before. A change in these numbers silently re-tunes every banked operating point."""
    from virturoid.services.gene_compiler import _joint_dynamics_prior
    from virturoid.services.morphology_composer import compose_robot
    gene = compose_robot("a quadruped walking robot", llm=None)
    assert "source_joint_dynamics" not in (gene.metadata or {})
    leg = _a_leg_joint(gene)
    assert _joint_dynamics_prior(gene, leg) == (0.8, 0.01, 0.12)


@pytest.mark.parametrize("bad", [
    {"damping": "2.0", "armature": 0.01, "frictionloss": 0.2},        # a pasted string
    {"damping": float("nan"), "armature": 0.01, "frictionloss": 0.2},  # a NaN reaching dof_damping
    {"damping": -1.0, "armature": 0.01, "frictionloss": 0.2},         # negative
    {"damping": True, "armature": 0.01, "frictionloss": 0.2},         # bool is an int in Python
    {"damping": 2.0, "armature": 0.01},                               # half a drivetrain
    "not a dict",
    None,
])
def test_a_malformed_record_falls_back_to_the_prior_instead_of_reaching_dof_damping(bad):
    """This runs inside the compile on a free-form dict that survived a JSON round trip. A malformed value
    must degrade to the structural prior, never take the compile down and never reach the model."""
    from virturoid.services.gene_compiler import _joint_dynamics_prior
    from virturoid.services.morphology_composer import compose_robot
    import dataclasses
    gene = compose_robot("a quadruped walking robot", llm=None)
    leg = _a_leg_joint(gene)
    poisoned = dataclasses.replace(
        gene, metadata={**(gene.metadata or {}), "source_joint_dynamics": {leg.name: bad}})
    assert _joint_dynamics_prior(poisoned, leg) == (0.8, 0.01, 0.12)
    # The CONTAINER can be malformed too — a metadata dict that round-tripped as a list has no ``.get``.
    for container in ([{"damping": 2.0}], "source_joint_dynamics", 7):
        wrecked = dataclasses.replace(
            gene, metadata={**(gene.metadata or {}), "source_joint_dynamics": container})
        assert _joint_dynamics_prior(wrecked, leg) == (0.8, 0.01, 0.12)


@pytest.mark.parametrize("rel", ["hello_robot_stretch/stretch.xml", "kinova_gen3/gen3.xml",
                                 "pal_tiago/tiago.xml", "pal_tiago_dual/tiago_dual.xml",
                                 "pndbotics_adam_lite/adam_lite.xml",
                                 "unitree_go2/go2.xml", "franka_emika_panda/panda.xml",
                                 "agility_cassie/cassie.xml"])
def test_carrying_the_declaration_does_not_cost_simulability(rel):
    """A declared ZERO damping/frictionloss is carried, and these models previously received our nonzero prior.

    Swept across the 59 cached Menagerie packages that yield a drivetrain record, carrying damping and
    frictionloss MOVES THE NUMBERS ON ALL 59 — the substitution was universal, not occasional — and costs
    simulability on exactly ONE, pal_tiago_dual, which the disclosed fallback below rescues.

    The other four names here are the ones a WIDER carry broke and this one does not: with armature carried
    too, hello_robot_stretch, kinova_gen3, pal_tiago and pndbotics_adam_lite all went non-finite, and all four
    step again the moment armature stays ours. They stay in this list precisely because they are the evidence
    that the narrower carry is the right one — and because an eight-package spot check once found only one of
    what it believed were five, which is the sampling error this file must not repeat.
    """
    res = _import(rel)
    assert res["simulable"], f"{rel}: {res.get('simulation_check')}"


def test_a_declaration_that_makes_our_twin_unsteppable_falls_back_loudly():
    """The one case where the customer's own numbers cannot be carried, and the only honest way not to.

    TIAGo++ declares 1000 N.s/m of damping on its torso lift and 40 N.m.s/rad across both arms. That is stable
    in ITS model, whose link inertias are real, and goes non-finite in ours at t=1.052 s, whose link inertias
    are primitives. Carrying it anyway would cost the customer every downstream number on a robot that used to
    produce them; keeping our prior QUIETLY is the defect this file exists to remove. So it falls back, keeps
    their numbers verbatim, and says which, why, and what it means for the gap.

    It is the last member of a set that used to have five, and the shrinking is the point: the other four
    (stretch, kinova_gen3, tiago, adam_lite) were broken by carrying ARMATURE, not by carrying the customer's
    damping. A fallback sized against the wider carry was firing on four robots whose declarations our twin
    can integrate perfectly well.
    """
    res = _import("pal_tiago_dual/tiago_dual.xml")
    md = res["gene"].metadata
    assert res["simulable"], res.get("simulation_check")
    assert md["source_joint_dynamics_carried"] is False
    assert md["joint_dynamics_source"] == "our_structural_prior_declaration_not_steppable"
    assert md["source_joint_dynamics"]["torso_lift_link"] == {
        "armature": 0.0, "damping": 1000.0, "frictionloss": 0.0}
    hit = [w for w in res["warnings"] if "NOT CARRIED" in w]
    assert len(hit) == 1, res["warnings"][:3]
    assert "do not quote it" in hit[0] and "faithful" in hit[0], hit[0]
    # And the compiler must AGREE with the flag: a record that is kept but not carried must not reach the
    # model, or the fallback is exactly the same shape of defect as the substitution it replaces.
    from virturoid.services.gene_compiler import _joint_dynamics_prior
    seg = next(s for s in res["gene"].actuated_joints() if s.name == "torso_lift_link")
    assert _joint_dynamics_prior(res["gene"], seg)[0] != pytest.approx(1000.0)


def test_damping_that_lives_in_the_ACTUATOR_is_named_rather_than_read_as_a_zero():
    """A ``dof_damping`` OF ZERO IS NOT ALWAYS AN UNDAMPED MACHINE, and reporting it as one would turn an
    incomplete read into a claim about the customer's robot.

    MuJoCo's ``<position kp kv>`` applies ``-kv*qvel`` from the ACTUATOR, so a model can be heavily damped with
    nothing at the joint at all. Boston Dynamics' Spot is exactly that: 0 damping / 0 armature / 0 frictionloss
    on all 12 leg joints, and ``kv=40`` on every one of its position actuators. 18 of the 59 packages here do
    the same (kuka_iiwa_14 200, ur5e 400, tidybot 50000). We emit ``<motor>`` — pure torque, because the
    verify/train harness computes its own PD — so there is nowhere for that term to go, and the twin's
    drivetrain really is undamped where theirs is not. The number is disclosed instead of implied away.
    """
    res = _import("boston_dynamics_spot/spot.xml")
    md = res["gene"].metadata
    kv = md["source_actuator_velocity_feedback_not_carried"]
    assert len(kv) == 12 and set(kv.values()) == {40.0}, kv
    assert md["source_joint_dynamics"]["fl_hip"] == {"armature": 0.0, "damping": 0.0, "frictionloss": 0.0}
    hit = [w for w in res["warnings"] if "velocity feedback from their ACTUATOR" in w]
    assert len(hit) == 1, res["warnings"]
    assert "kv=40" in hit[0] and "undamped where yours is not" in hit[0], hit[0]


def test_the_fallback_flag_is_what_the_compiler_obeys_not_the_absence_of_a_record():
    """A gene serialized before the flag existed has no flag and must still carry — absent means carried."""
    import dataclasses
    gene = _import("unitree_go2/go2.xml")["gene"]
    assert "source_joint_dynamics_carried" not in (gene.metadata or {})
    assert _dyn(_compiled(gene), "FL_calf_joint")["damping"] == pytest.approx(2.0)
    blocked = dataclasses.replace(
        gene, metadata={**(gene.metadata or {}), "source_joint_dynamics_carried": False})
    assert _dyn(_compiled(blocked), "FL_calf_joint")["damping"] == pytest.approx(0.8)
