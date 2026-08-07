"""THE TWIN OF A CLOSED-LOOP ROBOT WAS A DIFFERENT MACHINE.

Agility's Cassie is a four-bar linkage per leg. ``agility_cassie/cassie.xml`` says so in four lines::

    <connect body1="left-plantar-rod"  body2="left-foot"        anchor="0.35012 0 0"/>
    <connect body1="left-achilles-rod" body2="left-heel-spring" anchor="0.5012 0 0"/>
    ... and the mirrored pair on the right

The importer walked the body TREE and read nothing else, so the twin compiled with ``neq = 0``: the plantar rod
swung free, the linkage did not exist, and we verified, certified and exported that machine as the customer's.
MEASURED at HEAD -- twin ``neq`` 0 against the source's 4, and the rod separating from the foot by 0.70 m
within 2000 steps of plain gravity.

Nothing here was invented for it. ``RobotGene.loop_closures`` already modelled exactly this shape (see
``tests/test_closed_kinematic_loops.py``), ``gene_compiler._equality_xml`` already emitted ``<connect>`` from it
and ``gene_validation`` already checked ``loop_closures_compiled`` against ``m.neq``. This was the READ that was
never wired.

TWO TRAPS THESE TESTS DELIBERATELY PIN:

  * A ``<connect>`` is a SOFT constraint the solver satisfies with forces during ``mj_step``. ``mj_forward``
    does not project ``qpos``, so a forward-only test passes identically whether or not the constraint exists.
    ``test_mj_forward_alone_cannot_tell_the_two_models_apart`` asserts that trap directly, so the next person
    to "simplify" these tests into a forward pass sees it fail.
  * A ``<connect>`` may name SITES rather than bodies -- both ToddlerBots do -- and then ``eq_data`` is all
    zeros and the anchor lives on the site. Reading ``eq_data`` blindly would carry four loops with the anchor
    at the body origin.

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

# Every Menagerie package that declares a body-to-body or site-to-site <connect>, found by compiling all 63
# packages and counting ``eq_type == mjEQ_CONNECT`` (2026-08-06). The expected counts are asserted against the
# SOURCE model at run time rather than hardcoded, so a corpus refresh cannot silently weaken the sweep -- the
# list is only which files to look at.
_CONNECT_PACKAGES = [
    "agility_cassie/cassie.xml",
    "iit_softfoot/softfoot.xml",
    "robotiq_2f85/2f85.xml",
    "robotiq_2f85_v4/2f85.xml",
    "stanford_tidybot/tidybot.xml",
    "toddlerbot_2xc/toddlerbot_2xc.xml",
    "toddlerbot_2xm/toddlerbot_2xm.xml",
    "ufactory_xarm7/hand.xml",
    "ufactory_xarm7/xarm7.xml",
]


def _src(rel: str) -> str:
    p = _MEN / rel
    if not p.is_file():
        pytest.skip(f"{rel} is not cached locally (robot_descriptions fetches on demand)")
    return str(p)


def _source_connect_count(path: str) -> int:
    import mujoco
    m = mujoco.MjModel.from_xml_path(path)
    return sum(1 for e in range(int(m.neq)) if int(m.eq_type[e]) == int(mujoco.mjtEq.mjEQ_CONNECT))


def _imported(rel: str):
    from virturoid.services.robot_import import import_robot
    out = import_robot(_src(rel), robot_id=f"loops_{Path(rel).parent.name}")
    assert out["gene"] is not None, f"{rel} did not import at all: {out['warnings']}"
    return out


def _twin(gene, *, with_loops: bool = True):
    """Compile the editable twin, optionally stripped of its loops (i.e. exactly what HEAD produced)."""
    import copy

    import mujoco

    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    g = copy.deepcopy(gene)
    if not with_loops:
        g.loop_closures = []
    return mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(g, include_floor=True))


# ------------------------------------------------------------------------------- neq: source vs twin
@pytest.mark.parametrize("rel", _CONNECT_PACKAGES)
def test_the_twins_neq_matches_the_source(rel):
    """The whole defect in one assertion, across every Menagerie package that declares a <connect>."""
    want = _source_connect_count(_src(rel))
    assert want > 0, f"premise gone: {rel} no longer declares any <connect>"
    out = _imported(rel)
    gene = out["gene"]
    assert len(gene.loop_closures) == want, (
        f"{rel}: source declares {want} <connect>, the gene carries {len(gene.loop_closures)}")
    assert int(_twin(gene).neq) == want, (
        f"{rel}: the compiled twin carries {int(_twin(gene).neq)} equality constraints against the source's "
        f"{want} — the loops did not survive to the model")
    assert out["valid"], f"{rel}: reading the loops made the gene invalid: {out['warnings']}"


def test_cassie_is_the_named_case():
    """Four loops, and they are the four the file declares — plantar rod to foot, achilles rod to heel spring."""
    gene = _imported("agility_cassie/cassie.xml")["gene"]
    pairs = {(lc["a"], lc["b"]) for lc in gene.loop_closures}
    assert pairs == {
        ("left-plantar-rod", "left-foot"), ("left-achilles-rod", "left-heel-spring"),
        ("right-plantar-rod", "right-foot"), ("right-achilles-rod", "right-heel-spring")}, pairs


def test_a_model_with_no_equalities_gains_no_loops():
    """The other half: a Go2 is a tree and must stay one. An importer that invents a loop is worse than one
    that drops it."""
    gene = _imported("unitree_go2/go2.xml")["gene"]
    assert gene.loop_closures == []
    assert int(_twin(gene).neq) == 0


# ------------------------------------------------------------------------------- frames
def test_the_anchor_is_carried_into_the_LINK_frame_not_left_in_the_body_frame():
    """A segment's +z IS the link axis; MJCF's anchor is in body1's own frame. Cassie's plantar-rod anchor is
    ``0.35012 0 0`` in the body frame and the rod runs along local +x, so in the LINK frame the same physical
    point is ``0 0 0.35012``. Leaving it unrotated would put the joint of a four-bar linkage 0.35 m sideways.
    """
    import numpy as np
    gene = _imported("agility_cassie/cassie.xml")["gene"]
    lc = next(x for x in gene.loop_closures if x["a"] == "left-plantar-rod")
    a = np.asarray(lc["anchor"], dtype=float)
    assert float(np.linalg.norm(a)) == pytest.approx(0.35012, abs=1e-4), "the anchor moved off the rod"
    assert abs(a[2]) > 0.999 * float(np.linalg.norm(a)), f"anchor is not along the link axis: {lc['anchor']}"


def test_a_site_based_connect_is_resolved_to_its_bodies():
    """Both ToddlerBots close their neck loop with site-to-site connects, where ``eq_data`` is all zeros and
    the anchor is the SITE's position. Reading eq_data blindly gives four loops anchored at the body origin."""
    import numpy as np
    gene = _imported("toddlerbot_2xc/toddlerbot_2xc.xml")["gene"]
    assert len(gene.loop_closures) == 4
    names = {s.name for s in gene.segments}
    for lc in gene.loop_closures:
        assert lc["a"] in names and lc["b"] in names, lc
    assert any(float(np.linalg.norm(np.asarray(lc["anchor"], float))) > 1e-4 for lc in gene.loop_closures), \
        "every site anchor came out at the body origin — eq_data was read instead of site_pos"


# ------------------------------------------------------------------------------- the constraint actually holds
def test_the_plantar_rod_is_HELD_to_the_foot_when_the_model_is_STEPPED():
    """THE measurement. Drop Cassie under gravity and watch the loop close or not close.

    MEASURED on this checkout, worst separation between the two anchor points after 2000 steps:
    loops OFF (HEAD) 701.70 mm, loops ON 29.59 mm. The residual is not zero because ``connect`` is a soft
    constraint solved with the rest of the system — expressiveness and stiffness are different questions.
    """
    import mujoco
    import numpy as np

    gene = _imported("agility_cassie/cassie.xml")["gene"]
    held = _twin(gene, with_loops=True)
    assert int(held.neq) == 4

    # The reference pair of material points: MuJoCo resolves the second anchor at qpos0, so read it back.
    ref = [(int(held.eq_obj1id[e]), int(held.eq_obj2id[e]),
            np.array(held.eq_data[e][:3]), np.array(held.eq_data[e][3:6])) for e in range(int(held.neq))]
    names = [(mujoco.mj_id2name(held, mujoco.mjtObj.mjOBJ_BODY, b1),
              mujoco.mj_id2name(held, mujoco.mjtObj.mjOBJ_BODY, b2), a1, a2) for b1, b2, a1, a2 in ref]

    def worst_gap(m, steps: int) -> float:
        d = mujoco.MjData(m)
        ids = [(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n1),
                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n2), a1, a2) for n1, n2, a1, a2 in names]
        for _ in range(steps):
            mujoco.mj_step(m, d)
        return max(float(np.linalg.norm((d.xpos[i] + d.xmat[i].reshape(3, 3) @ a1)
                                        - (d.xpos[j] + d.xmat[j].reshape(3, 3) @ a2)))
                   for i, j, a1, a2 in ids)

    free = worst_gap(_twin(gene, with_loops=False), 2000)
    bound = worst_gap(held, 2000)
    assert free > 0.30, f"premise gone: the unconstrained rod stayed within {free * 1000:.1f} mm of the foot"
    assert bound < 0.10, f"the linkage did not hold: {bound * 1000:.1f} mm of separation"
    assert bound < free / 4.0, f"loops ON {bound * 1000:.1f} mm vs OFF {free * 1000:.1f} mm"


def test_mj_forward_alone_cannot_tell_the_two_models_apart():
    """The recorded trap, asserted so it cannot be re-introduced. ``mj_forward`` does not project qpos onto the
    constraint manifold, and MuJoCo resolves the second anchor AT qpos0 — so at the rest pose both models show
    a zero gap and a forward-only test reports the constraint working when there is no constraint at all."""
    import mujoco
    import numpy as np

    gene = _imported("agility_cassie/cassie.xml")["gene"]
    held = _twin(gene, with_loops=True)
    free = _twin(gene, with_loops=False)
    ref = [(mujoco.mj_id2name(held, mujoco.mjtObj.mjOBJ_BODY, int(held.eq_obj1id[e])),
            mujoco.mj_id2name(held, mujoco.mjtObj.mjOBJ_BODY, int(held.eq_obj2id[e])),
            np.array(held.eq_data[e][:3]), np.array(held.eq_data[e][3:6])) for e in range(int(held.neq))]
    for m in (held, free):
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        for n1, n2, a1, a2 in ref:
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n1)
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n2)
            gap = float(np.linalg.norm((d.xpos[i] + d.xmat[i].reshape(3, 3) @ a1)
                                       - (d.xpos[j] + d.xmat[j].reshape(3, 3) @ a2)))
            assert gap < 1e-6, f"{n1}<->{n2} gap {gap:.6f} m at qpos0 (neq={int(m.neq)})"


def test_the_design_validator_now_rules_on_the_loop():
    """``gene_validation`` has always checked ``loop_closures_meet``; with nothing to check it never ran. On
    Cassie's twin it now runs and passes, which is the check that would catch an anchor carried in the wrong
    frame."""
    from virturoid.services.gene_validation import validate_gene_design
    checks = (validate_gene_design(_imported("agility_cassie/cassie.xml")["gene"]) or {}).get("checks") or {}
    assert checks.get("loop_closures_compiled") is True, checks
    assert checks.get("loop_closures_meet") is True, checks


# ------------------------------------------------------------------------------- honesty about what is lost
def test_the_loops_are_disclosed_in_the_import_warnings():
    ws = " ".join(_imported("agility_cassie/cassie.xml")["warnings"])
    assert "closed kinematic loop" in ws and "left-plantar-rod<->left-foot" in ws, ws


def test_a_joint_equality_is_reported_not_silently_dropped():
    """24 of the 37 Menagerie models that declare equalities declare ONLY ``<equality><joint>`` — a coupled
    (mimic) DOF, which a RobotGene cannot express at all. Dropping it is a loss; dropping it silently is the
    defect. The Panda's hand couples its two fingers exactly this way."""
    ws = " ".join(_imported("franka_emika_panda/panda.xml")["warnings"])
    assert "cannot model" in ws and "<joint>" in ws, ws
    assert "move independently" in ws, ws


def test_a_connect_restating_a_parent_child_edge_is_refused_with_a_reason():
    """``RobotGene.validate`` rejects a loop over an edge the tree already has, on purpose: a parent and child
    are already rigidly related through their joint. Carrying one through would make every such import report
    ``valid: False`` instead of importing."""
    from virturoid.services.robot_import import import_robot
    xml = """
    <mujoco model="pc">
      <worldbody>
        <body name="upper" pos="0 0 1">
          <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03"/>
          <body name="lower" pos="0 0 -0.3">
            <joint name="j" type="hinge" axis="0 1 0" range="-1 1"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03"/>
          </body>
        </body>
      </worldbody>
      <equality><connect body1="lower" body2="upper" anchor="0 0 0"/></equality>
    </mujoco>"""
    out = import_robot(xml, robot_id="pc_loop")
    assert out["gene"].loop_closures == []
    ws = " ".join(out["warnings"])
    assert "already parent and child" in ws, ws
    assert out["valid"], f"the refusal must not make the import invalid: {out['warnings']}"


# ------------------------------------------------------------------------------- the loops survive storage
def test_the_loops_round_trip_through_the_gene_serializer():
    """The species tree persists genes as dicts; a field that vanishes on ``to_dict`` is a field the flywheel
    silently loses — and losing it turns the customer's closed-loop robot back into a different machine."""
    from virturoid.schemas.gene import RobotGene
    gene = _imported("agility_cassie/cassie.xml")["gene"]
    back = RobotGene.from_dict(gene.to_dict())
    assert back.loop_closures == gene.loop_closures
    assert int(_twin(back).neq) == 4
