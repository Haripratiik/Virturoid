"""A PANDA WHOSE TWO FINGERS MOVE INDEPENDENTLY IS NOT A PANDA.

``franka_emika_panda/hand.xml`` says the fingers are one DOF in a single line::

    <equality><joint joint1="finger_joint1" joint2="finger_joint2" polycoef="0 1 0 0 0"/></equality>

and so do a Robotiq 2F-85's two drivers, a Stretch's four telescoping arm stages, a Talos gripper's six-bar,
and every one of a ToddlerBot's nine geared pairs. The importer read the body tree and the ``<connect>``
constraints, and DISCLOSED that it was dropping these -- honest, but the twin was still wrong.

Two things are measured here, and only the second is a real test of a constraint:

  * the coupling is CARRIED (gene -> compiled ``<equality><joint>``), per package, against the source's own
    count of the ones a one-joint-per-segment gene can express;
  * the coupled joints MOVE TOGETHER WHEN STEPPED. An equality is solved inside ``mj_step``; ``mj_forward``
    does not project qpos, and at the rest pose both joints read 0 whether or not the constraint exists -- so
    a forward-only check passes identically on a twin with no coupling at all. That trap is asserted directly
    in ``test_mj_forward_alone_cannot_tell_the_two_models_apart`` so the next person to "simplify" this into a
    forward pass sees it fail.

The second file also pins the OTHER half of this change: the source's own ``solref``/``solimp`` are carried
onto the constraint instead of silently falling back to MuJoCo's softer default.

Run against the real MuJoCo Menagerie, never a fixture, for the reason this repo learned the hard way: every
Unitree/Spot/ANYmal model once imported as "0 bodies" while the fixtures stayed green.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("VIRTUROID_DISABLE_GAIT_HINTS", "1")

_MUJOCO = importlib.util.find_spec("mujoco") is not None
_MEN = Path(os.path.expanduser("~/.cache/robot_descriptions/mujoco_menagerie"))
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="import + compile need MuJoCo")

# Every Menagerie package that declares an <equality><joint>, found by compiling all 63 packages (2026-08-07).
# Expected counts are derived from the SOURCE at run time, never hardcoded, so a corpus refresh cannot quietly
# weaken the sweep -- the list is only which files to look at.
_COUPLED_PACKAGES = [
    "agilex_piper/piper.xml", "aloha/aloha.xml", "arx_l5/arx_l5.xml",
    "franka_emika_panda/panda.xml", "hello_robot_stretch/stretch.xml",
    "hello_robot_stretch_3/stretch.xml", "i2rt_yam/yam.xml", "pal_talos/talos.xml",
    "robotiq_2f85/2f85.xml", "robotiq_2f85_v4/2f85.xml", "stanford_tidybot/tidybot.xml",
    "tetheria_aero_hand_open/right_hand.xml", "toddlerbot_2xc/toddlerbot_2xc.xml",
    "toddlerbot_2xm/toddlerbot_2xm.xml", "trossen_vx300s/vx300s.xml", "trossen_wx250s/wx250s.xml",
    "ufactory_lite6/lite6_gripper_narrow.xml", "ufactory_xarm7/xarm7.xml",
]


def _src(rel: str) -> str:
    p = _MEN / rel
    if not p.is_file():
        pytest.skip(f"{rel} is not cached locally (robot_descriptions fetches on demand)")
    return str(p)


def _imported(rel: str):
    from virturoid.services.robot_import import import_robot
    out = import_robot(_src(rel), robot_id=f"coupled_{Path(rel).parent.name}")
    assert out["gene"] is not None, f"{rel} did not import at all: {out['warnings']}"
    return out


def _twin(gene, *, coupled: bool = True):
    """Compile the editable twin, optionally stripped of its couplings (i.e. exactly what HEAD produced)."""
    import copy

    import mujoco

    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    g = copy.deepcopy(gene)
    if not coupled:
        g.coupled_joints = []
    return mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(g, include_floor=True))


def _source_expressible_couplings(path: str) -> int:
    """How many of the source's <equality><joint> a ONE-JOINT-PER-SEGMENT gene can express.

    Independently recomputed from the source here rather than read off the importer, so this is a real
    expectation and not a restatement of the code under test.
    """
    from collections import defaultdict

    import mujoco
    m = mujoco.MjModel.from_xml_path(path)
    first = {}
    for j in range(m.njnt):
        first.setdefault(int(m.jnt_bodyid[j]), j)
    n = 0
    for e in range(int(m.neq)):
        if int(m.eq_type[e]) != int(mujoco.mjtEq.mjEQ_JOINT):
            continue
        j1, j2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
        if j2 < 0:
            continue                                             # pins one DOF to a constant, not a coupling
        b1, b2 = int(m.jnt_bodyid[j1]), int(m.jnt_bodyid[j2])
        if b1 == b2 or first.get(b1) != j1 or first.get(b2) != j2:
            continue
        if any(abs(float(v)) > 1e-12 for v in m.eq_data[e][2:5]):
            continue                                             # higher-degree polycoef
        n += 1
    return n


# ------------------------------------------------------------- the coupling is carried, and it compiles
@pytest.mark.parametrize("rel", _COUPLED_PACKAGES)
def test_every_expressible_coupling_survives_into_the_compiled_twin(rel):
    import mujoco

    want = _source_expressible_couplings(_src(rel))
    assert want > 0, f"premise gone: {rel} no longer declares an expressible <equality><joint>"
    out = _imported(rel)
    gene = out["gene"]
    assert len(gene.coupled_joints) == want, (
        f"{rel}: {want} coupling(s) are expressible, the gene carries {len(gene.coupled_joints)}")
    m = _twin(gene)
    got = sum(1 for e in range(int(m.neq)) if int(m.eq_type[e]) == int(mujoco.mjtEq.mjEQ_JOINT))
    assert got == want, f"{rel}: twin carries {got} <equality><joint> against {want} on the gene"
    assert out["valid"], f"{rel}: reading the couplings made the gene invalid: {out['warnings']}"


def test_the_panda_is_the_named_case():
    """One coupling, the two fingers, at ratio 1 with no offset — exactly what hand.xml declares."""
    gene = _imported("franka_emika_panda/panda.xml")["gene"]
    assert len(gene.coupled_joints) == 1, gene.coupled_joints
    cj = gene.coupled_joints[0]
    assert {cj["a"], cj["b"]} == {"left_finger", "right_finger"}, cj
    assert cj["ratio"] == pytest.approx(1.0), cj
    assert cj["offset"] == pytest.approx(0.0), cj


def test_a_non_unit_gear_ratio_is_carried_as_the_source_states_it():
    """A coupling is not always a mirror. ToddlerBot's neck runs through a 0.909 gear and its hip through
    0.857, both NEGATED; a Stretch's gripper fingers run at 10x its slider. Rounding any of those to +/-1
    would be a different transmission."""
    ratios = {round(cj["ratio"], 6) for cj in _imported("toddlerbot_2xc/toddlerbot_2xc.xml")["gene"].coupled_joints}
    assert -0.90909091 == pytest.approx(min(ratios, key=lambda r: abs(r + 0.90909091)), abs=1e-6), ratios
    assert any(r == pytest.approx(-0.85714286, abs=1e-6) for r in ratios), ratios
    stretch = {round(cj["ratio"], 6) for cj in _imported("hello_robot_stretch/stretch.xml")["gene"].coupled_joints}
    assert any(r == pytest.approx(10.0) for r in stretch), stretch


def test_a_model_with_no_joint_equality_gains_no_coupling():
    """The other half: a Go2's joints are independent and must stay independent. An importer that invents a
    coupling is worse than one that drops it."""
    gene = _imported("unitree_go2/go2.xml")["gene"]
    assert gene.coupled_joints == []
    assert int(_twin(gene).neq) == 0


# ------------------------------------------------------------- THE GATE: they move together when STEPPED
def _worst_residual(m, cjs, *, steps: int = 1500):
    """Sweep the DRIVER joint between its own limits with a force-clamped PD law and return
    ``(worst |q_a - offset - ratio*q_b|, driver travel)``.

    A PD sweep and not a constant open-loop torque on purpose: half of peak motor force held on Aloha's 86 g
    finger diverges to NaN with the couplings OFF as readily as ON, which measures our actuator sizing rather
    than whether the two DOF track each other. `travel` is returned so a coupling that "holds" only because
    nothing ever moved cannot pass.
    """
    import mujoco
    import numpy as np

    plan = []
    for cj in cjs:
        ja = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{cj['a']}_joint")
        jb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{cj['b']}_joint")
        assert ja >= 0 and jb >= 0, f"{cj} names a joint the model does not have"
        act = next((a for a in range(m.nu) if int(m.actuator_trnid[a][0]) == jb), None)
        lo, hi = ((float(m.jnt_range[jb][0]), float(m.jnt_range[jb][1]))
                  if int(m.jnt_limited[jb]) else (-0.5, 0.5))
        plan.append((int(m.jnt_qposadr[ja]), int(m.jnt_qposadr[jb]), int(m.jnt_dofadr[jb]),
                     float(cj["offset"]), float(cj["ratio"]), act, lo, hi))

    d = mujoco.MjData(m)
    worst = travel = 0.0
    q0 = None
    for k in range(steps):
        phase = k / (steps - 1)
        for _, qb, dof, _, _, act, lo, hi in plan:
            if act is None:
                continue
            target = lo + (hi - lo) * (2 * phase if phase < 0.5 else 2 * (1 - phase))
            clo, chi = m.actuator_ctrlrange[act]
            d.ctrl[act] = float(np.clip(20.0 * (target - d.qpos[qb]) - 1.0 * d.qvel[dof], clo, chi))
        mujoco.mj_step(m, d)
        if q0 is None:
            q0 = [float(d.qpos[qb]) for _, qb, *_ in plan]
        for n, (qa, qb, _, off, ratio, *_r) in enumerate(plan):
            worst = max(worst, abs(float(d.qpos[qa]) - off - ratio * float(d.qpos[qb])))
            travel = max(travel, abs(float(d.qpos[qb]) - q0[n]))
    return worst, travel


# Aloha is deliberately absent: its twin's TWO ARMS interpenetrate by 0.14 m (left/base_link against
# right/upper_arm_link) and the contact solver diverges under any excitation, with the couplings ON or OFF
# alike. That is a real pre-existing defect in the dual-arm import, not a property of the coupling, and gating
# it here would only hide it behind an unrelated assertion.
@pytest.mark.parametrize("rel", [
    "franka_emika_panda/panda.xml", "robotiq_2f85/2f85.xml", "stanford_tidybot/tidybot.xml",
    "hello_robot_stretch/stretch.xml", "pal_talos/talos.xml", "toddlerbot_2xc/toddlerbot_2xc.xml",
    "ufactory_xarm7/xarm7.xml",
])
def test_the_coupled_joints_MOVE_TOGETHER_when_the_model_is_STEPPED(rel):
    """THE measurement. Drive the driver, step, and watch the driven joint follow it or not follow it.

    MEASURED on this checkout, worst |q_a - offset - ratio*q_b| in rad/m over a 3 s driven sweep,
    couplings OFF -> ON:
        panda 0.05354 -> 0.00069   robotiq_2f85 0.01088 -> 0.00046   tidybot 0.28041 -> 0.00045
        stretch 0.48485 -> 0.00368 talos 0.77116 -> 0.00278          toddlerbot 6.07368 -> 0.01367
        xarm7 0.25465 -> 0.00036

    RE-MEASURED against the customer's own drivetrain. These numbers moved when ``robot_import`` began
    carrying the source's declared damping and frictionloss into the twin instead of substituting our
    structural prior, so both columns are now taken on a model that integrates the damping the customer's file
    declares. The ratios did not move much and the verdict did not move at all; the point of recording them is
    that a future drift can be told apart from this one.

    The same change also carried ARMATURE for a while, and this test is one of the eight gates that caught it:
    talos went 0.00278 -> 0.13119 and toddlerbot 0.01367 -> 1.88989 under armature alone. See
    ``gene_compiler._declared_joint_dynamics`` for why a compiled ``dof_armature`` is not a declaration.
    """
    gene = _imported(rel)["gene"]
    assert gene.coupled_joints, f"premise gone: {rel} carries no coupling"
    free, _ = _worst_residual(_twin(gene, coupled=False), gene.coupled_joints)
    bound, travel = _worst_residual(_twin(gene, coupled=True), gene.coupled_joints)
    assert travel > 1e-3, f"{rel}: the driver barely moved ({travel:.2e}); the residual proves nothing"
    assert free > 1e-3, f"{rel}: premise gone — the joints already tracked within {free:.2e} uncoupled"
    assert bound < 0.05, f"{rel}: the coupling did not hold: {bound:.5f} of residual"
    assert bound < free / 4.0, f"{rel}: coupled {bound:.5f} vs uncoupled {free:.5f} — barely tighter"


def test_mj_forward_alone_cannot_tell_the_two_models_apart():
    """The trap, asserted so it cannot be re-introduced. An equality is solved inside ``mj_step``;
    ``mj_forward`` does not project qpos onto the constraint manifold, and at qpos0 both joints read 0 — so a
    forward-only check reports the coupling working when there is no coupling at all."""
    import mujoco

    gene = _imported("franka_emika_panda/panda.xml")["gene"]
    for coupled in (True, False):
        m = _twin(gene, coupled=coupled)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        for cj in gene.coupled_joints:
            qa = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{cj['a']}_joint")])
            qb = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{cj['b']}_joint")])
            res = abs(float(d.qpos[qa]) - cj["offset"] - cj["ratio"] * float(d.qpos[qb]))
            assert res < 1e-9, f"residual {res} at qpos0 with coupled={coupled} (neq={int(m.neq)})"


# ------------------------------------------------------------- the schema rules on it
def test_a_coupling_onto_a_WELDED_segment_is_refused_with_a_reason():
    """The compiler names joints ``<segment>_joint`` and emits none at all for a weld, so a coupling onto a
    welded segment names a joint MuJoCo does not have. Left to MuJoCo it would fail at XML-compile time with a
    message about a missing joint name — true, but pointing at the emitter rather than at the design. So
    ``validate()`` names the actual fault and ``compile_gene_to_mjcf`` refuses the gene before emitting."""
    import pytest as _pytest

    from virturoid.schemas.gene import GeneSegment, RobotGene
    from virturoid.services.gene_compiler import _equality_xml, compile_gene_to_mjcf

    gene = RobotGene(
        id="weldcouple", species="manipulator.test", robot_class="manipulator",
        segments=[
            GeneSegment(name="base", parent=None, shape="box", length_m=0.1, radius_m=0.05, mass_kg=1.0),
            GeneSegment(name="arm", parent="base", shape="capsule", length_m=0.2, radius_m=0.02,
                        mass_kg=0.5, joint_type="revolute", joint_lower=-1.0, joint_upper=1.0),
            GeneSegment(name="pad", parent="arm", shape="box", length_m=0.05, radius_m=0.02, mass_kg=0.1,
                        joint_type="fixed", is_end_effector=True),
        ],
        coupled_joints=[{"a": "arm", "b": "pad", "ratio": 1.0, "offset": 0.0}],
    )
    issues = gene.validate()
    assert any("no joint to couple" in i for i in issues), issues
    with _pytest.raises(ValueError, match="no joint to couple"):
        compile_gene_to_mjcf(gene, include_floor=True)
    # And the emitter itself holds the line independently, so a caller reaching it another way (the scene path,
    # a future partial build) still cannot produce XML naming a joint that was never written.
    assert "<joint" not in _equality_xml(gene), _equality_xml(gene)


def test_a_zero_ratio_is_refused():
    """ratio 0 pins the driven joint to a constant. That is a lock, not a coupling, and letting it through
    would hold a DOF at a value the design never chose."""
    from virturoid.schemas.gene import GeneSegment, RobotGene

    gene = RobotGene(
        id="zeroratio", species="manipulator.test", robot_class="manipulator",
        segments=[
            GeneSegment(name="base", parent=None, shape="box", length_m=0.1, radius_m=0.05, mass_kg=1.0),
            GeneSegment(name="j1", parent="base", shape="capsule", length_m=0.2, radius_m=0.02, mass_kg=0.5,
                        joint_type="revolute", joint_lower=-1.0, joint_upper=1.0),
            GeneSegment(name="j2", parent="j1", shape="capsule", length_m=0.2, radius_m=0.02, mass_kg=0.5,
                        joint_type="revolute", joint_lower=-1.0, joint_upper=1.0, is_end_effector=True),
        ],
        coupled_joints=[{"a": "j1", "b": "j2", "ratio": 0.0, "offset": 0.0}],
    )
    assert any("finite and non-zero" in i for i in gene.validate()), gene.validate()


def test_the_couplings_survive_storage():
    """The species tree persists genes as dicts; a field that vanishes on ``to_dict`` is a field the flywheel
    silently loses — and losing this one turns the customer's one-DOF gripper back into two."""
    import mujoco

    from virturoid.schemas.gene import RobotGene
    gene = _imported("franka_emika_panda/panda.xml")["gene"]
    back = RobotGene.from_dict(gene.to_dict())
    assert back.coupled_joints == gene.coupled_joints
    m = _twin(back)
    assert sum(1 for e in range(int(m.neq)) if int(m.eq_type[e]) == int(mujoco.mjtEq.mjEQ_JOINT)) == 1


def test_an_AMEND_keeps_the_couplings_it_can_still_satisfy():
    """``design_critic.add_parallel_gripper`` amends an imported arm, and ``amend_gene`` used to rebuild the
    child from ``segments`` alone — so amending a Panda handed back a Panda whose fingers move independently.
    Carried now, and filtered: a constraint naming a segment the amend REMOVED has to go, or every such amend
    would raise on validate() instead of amending."""
    from virturoid.schemas.gene import amend_gene

    gene = _imported("franka_emika_panda/panda.xml")["gene"]
    assert len(gene.coupled_joints) == 1, gene.coupled_joints
    kept = amend_gene(gene, new_id="panda_amended", species="manipulator.imported.amended")
    assert kept.coupled_joints == gene.coupled_joints, kept.coupled_joints
    assert not kept.validate(), kept.validate()

    dropped = amend_gene(gene, new_id="panda_nofinger", species="manipulator.imported.nofinger",
                         remove_segments=["right_finger"])
    assert dropped.coupled_joints == [], (
        "a coupling onto a segment the amend removed must not survive — it would name a joint that is gone")
    assert not any("coupled_joint" in i for i in dropped.validate()), dropped.validate()


def test_the_couplings_are_disclosed_in_the_import_warnings():
    ws = " ".join(_imported("franka_emika_panda/panda.xml")["warnings"])
    assert "coupled (mimic/slaved) joint pair" in ws, ws
    assert "move together, not independently" in ws, ws


# ------------------------------------------------------------- the source's solver reference is carried
def test_the_sources_own_solref_reaches_the_compiled_model():
    """``agility_cassie/cassie.xml`` opens its equality block with ``solref="0.005 1"`` — four times stiffer
    than MuJoCo's 0.02 default, because a four-bar linkage held at the default comes apart. Compiling at the
    default imported the topology and dropped the tolerance."""
    import mujoco

    gene = _imported("agility_cassie/cassie.xml")["gene"]
    assert all(lc.get("solref") == [0.005, 1.0] for lc in gene.loop_closures), gene.loop_closures
    m = _twin(gene)
    for e in range(int(m.neq)):
        assert float(m.eq_solref[e][0]) == pytest.approx(0.005), (
            f"equality {e} compiled at solref {list(m.eq_solref[e])}, not the source's 0.005")


def test_a_source_that_declares_NO_solref_gets_MUJOCOS_default_not_an_invented_one():
    """The other half, and the one that keeps this honest. iit_softfoot's ``<connect>`` uses MuJoCo's default
    solref and a NON-default solimp; the gene must carry the solimp and stay silent about the solref, so the
    compiled model lands on MuJoCo's own number rather than a stiffness we chose for the customer."""
    import mujoco

    gene = _imported("iit_softfoot/softfoot.xml")["gene"]
    assert gene.loop_closures, "premise gone: softfoot declares no <connect>"
    for lc in gene.loop_closures:
        assert "solref" not in lc, f"invented a solref the source never declared: {lc}"
        assert lc.get("solimp") == [0.95, 0.99, 0.001, 0.5, 2.0], lc
    m = _twin(gene)
    for e in range(int(m.neq)):
        assert float(m.eq_solref[e][0]) == pytest.approx(0.02), list(m.eq_solref[e])
        assert float(m.eq_solimp[e][0]) == pytest.approx(0.95), list(m.eq_solimp[e])


def test_the_carried_solref_MEASURABLY_tightens_the_linkage_when_STEPPED():
    """Not a number in an attribute — a number in the trajectory. Cassie's plantar rod against its foot, worst
    separation over 2000 stepped frames: 698.99 mm with no constraint at all, 29.61 mm carrying the constraint
    at MuJoCo's default stiffness, 14.67 mm carrying the source's own ``solref="0.005 1"``.

    RE-MEASURED on the twin that carries Cassie's own declared joint damping (the earlier 701.70 / 32.83 /
    14.82 were taken against our substituted prior). Carrying the source's ARMATURE too put this at 20.90 mm
    against 31.10 mm — a ratio of 0.672, through the 0.6 gate below — which is one of the eight failures that
    identified armature as the parameter that must not be carried.
    """
    import copy

    import mujoco
    import numpy as np

    from virturoid.services.gene_compiler import compile_gene_to_mjcf

    gene = _imported("agility_cassie/cassie.xml")["gene"]
    stripped = copy.deepcopy(gene)
    for lc in stripped.loop_closures:
        lc.pop("solref", None)                                   # exactly what HEAD emitted
    held = _twin(gene)
    soft = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(stripped, include_floor=True))

    names = [(mujoco.mj_id2name(held, mujoco.mjtObj.mjOBJ_BODY, int(held.eq_obj1id[e])),
              mujoco.mj_id2name(held, mujoco.mjtObj.mjOBJ_BODY, int(held.eq_obj2id[e])),
              np.array(held.eq_data[e][:3]), np.array(held.eq_data[e][3:6]))
             for e in range(int(held.neq))]

    def worst_gap(m, steps=2000):
        d = mujoco.MjData(m)
        ids = [(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n1),
                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n2), a1, a2) for n1, n2, a1, a2 in names]
        for _ in range(steps):
            mujoco.mj_step(m, d)
        return max(float(np.linalg.norm((d.xpos[i] + d.xmat[i].reshape(3, 3) @ a1)
                                        - (d.xpos[j] + d.xmat[j].reshape(3, 3) @ a2)))
                   for i, j, a1, a2 in ids)

    at_default = worst_gap(soft)
    at_source = worst_gap(held)
    assert at_source < at_default, (
        f"the source's solref did not tighten anything: {at_source * 1000:.2f} mm against "
        f"{at_default * 1000:.2f} mm at MuJoCo's default")
    assert at_source < 0.6 * at_default, (
        f"only {at_source * 1000:.2f} mm against {at_default * 1000:.2f} mm — expected roughly half")
