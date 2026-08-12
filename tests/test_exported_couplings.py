"""AN EXPORTED PANDA WHOSE TWO FINGERS OPEN SEPARATELY IS NOT A PANDA EITHER.

``RobotGene.coupled_joints`` carries 52 of the corpus's 97 couplings and the MJCF emits them as
``<equality><joint>``. The exporters did not: measured at the commit before this one, URDF emitted **0** of 24
carried couplings across the nine constraint-bearing Menagerie packages, and USD 0 of 24. An engineer taking
that URDF to RViz/MoveIt/Gazebo got a gripper that opens one jaw.

URDF has a NATIVE element for exactly this -- ``<mimic joint multiplier offset>``, a degree-1 relation, which is
what all 97 corpus couplings turned out to be -- so it is EMITTED, not disclosed. What each lane can express was
established rather than assumed, and the answers differ per lane and per constraint kind:

    lane   coupled_joints                                 loop_closures
    URDF   EMITTED as <mimic>                             CANNOT (strict tree) -> disclosed in the file
    USD    CANNOT (no UsdPhysics schema) -> disclosed     EMITTED as UsdPhysics.FixedJoint (rigid, disclosed)
    Isaac  inherits USD; handed over as data + README     inherits USD
    ROS 2  inherits the URDF's <mimic>, and drops the driven joints from the command interface

THE VERIFICATION RULE HERE: nothing is checked against the string the exporter just returned. Every URDF
assertion re-parses the file FROM DISK with ElementTree and rebuilds the relation from the parsed attributes;
every USD assertion re-opens the stage with pxr. And the numbers are then tied back to physics --
``test_the_urdfs_mimic_predicts_what_MUJOCO_ACTUALLY_SOLVES`` drives the URDF's own reference joint, STEPS the
model, and requires the URDF's ``multiplier * q_driver + offset`` to reproduce MuJoCo's solved driven angle. An
equality is solved inside ``mj_step`` and ``mj_forward`` does not project onto it, so a forward-only check would
pass on a model with no coupling at all.

Run against the real MuJoCo Menagerie, never a fixture.
"""
from __future__ import annotations

import importlib.util
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("VIRTUROID_DISABLE_GAIT_HINTS", "1")

_MUJOCO = importlib.util.find_spec("mujoco") is not None
_PXR = importlib.util.find_spec("pxr") is not None
_MEN = Path(os.path.expanduser("~/.cache/robot_descriptions/mujoco_menagerie"))
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="the exporters transcribe a compiled MuJoCo model")

# The four named in the brief, plus the two chained bodies and one more mirror pair. Talos and Stretch are the
# interesting ones: their couplings form CHAINS in the source, which URDF consumers do not resolve.
PANDA = "franka_emika_panda/panda.xml"
ROBOTIQ = "robotiq_2f85/2f85.xml"
STRETCH = "hello_robot_stretch/stretch.xml"
TODDLER = "toddlerbot_2xc/toddlerbot_2xc.xml"
TALOS = "pal_talos/talos.xml"
TIDYBOT = "stanford_tidybot/tidybot.xml"
XARM7 = "ufactory_xarm7/xarm7.xml"
COUPLED = [PANDA, ROBOTIQ, STRETCH, TODDLER, TALOS, TIDYBOT, XARM7]

_GENES: dict[str, object] = {}


def _src(rel: str) -> str:
    p = _MEN / rel
    if not p.is_file():
        pytest.skip(f"{rel} is not cached locally (robot_descriptions fetches on demand)")
    return str(p)


def _gene(rel: str):
    """Imported once per process — these are big models and every test below wants the same body."""
    if rel not in _GENES:
        from virturoid.services.robot_import import import_robot
        out = import_robot(_src(rel), robot_id=f"export_{Path(rel).parent.name}")
        assert out["gene"] is not None, f"{rel} did not import: {out['warnings']}"
        _GENES[rel] = out["gene"]
    return _GENES[rel]


# ---------------------------------------------------------------- re-read the URDF, not the string we wrote
def _urdf_root(rel: str, tmp_path: Path):
    """Export, WRITE TO DISK, and parse the file back. The returned tree is the only thing tests may look at."""
    from virturoid.services.gene_urdf import gene_to_urdf
    path = tmp_path / f"{Path(rel).parent.name}.urdf"
    path.write_text(gene_to_urdf(_gene(rel)), encoding="utf-8")
    return ET.parse(str(path)).getroot(), path


def _parsed_mimics(root) -> dict[str, tuple[str, float, float]]:
    """``{driven joint: (driver joint, multiplier, offset)}`` rebuilt from the parsed XML attributes."""
    out = {}
    for jnt in root.findall("joint"):
        mim = jnt.find("mimic")
        if mim is None:
            continue
        assert jnt.get("name"), "a <mimic> on an unnamed joint is unreferenceable"
        out[jnt.get("name")] = (mim.get("joint"), float(mim.get("multiplier")), float(mim.get("offset")))
    return out


def _expected_relations(gene) -> dict[str, tuple[str, float, float]]:
    """What the gene's couplings MEAN, composed independently of the exporter, in URDF joint names.

    Recomputed here from ``gene.coupled_joints`` rather than imported from ``gene_urdf._flatten_couplings``, so
    this is a real expectation and not a restatement of the code under test.
    """
    direct = {cj["a"]: (cj["b"], float(cj["ratio"]), float(cj.get("offset") or 0.0))
              for cj in gene.coupled_joints}
    out = {}
    for a, (b, mul, off) in direct.items():
        seen = {a}
        while b in direct and b not in seen:
            seen.add(b)
            b2, r2, o2 = direct[b]
            b, mul, off = b2, mul * r2, mul * o2 + off
        out[f"{a}_joint"] = (f"{b}_joint", mul, off)
    return out


@pytest.mark.parametrize("rel", COUPLED)
def test_every_carried_coupling_is_a_mimic_in_the_reparsed_urdf(rel, tmp_path):
    gene = _gene(rel)
    assert gene.coupled_joints, f"premise gone: {rel} carries no coupling"
    root, path = _urdf_root(rel, tmp_path)
    got = _parsed_mimics(root)
    want = _expected_relations(gene)
    assert set(got) == set(want), (
        f"{rel}: {len(want)} coupling(s) carried, {len(got)} <mimic> survived the round trip through {path}")
    for driven, (driver, mul, off) in want.items():
        g_driver, g_mul, g_off = got[driven]
        assert g_driver == driver, f"{rel}: {driven} mimics {g_driver}, expected {driver}"
        assert g_mul == pytest.approx(mul, rel=1e-9, abs=1e-12), f"{rel}: {driven} multiplier"
        assert g_off == pytest.approx(off, rel=1e-9, abs=1e-12), f"{rel}: {driven} offset"
    # Every reference must name a joint that EXISTS and is itself independently commanded.
    names = {j.get("name") for j in root.findall("joint")}
    for driven, (driver, _m, _o) in got.items():
        assert driver in names, f"{rel}: {driven} mimics {driver}, which is not a joint in this URDF"
        assert driver not in got, (
            f"{rel}: {driven} mimics {driver}, which is ITSELF a mimic. robot_state_publisher resolves a mimic "
            f"from the joint_states message and a mimic joint never appears there, so the chain reads 0.")


def test_the_gear_ratios_are_the_sources_own_numbers_not_normalised(tmp_path):
    """ToddlerBot's neck runs through -0.90909091 and its hips through -0.85714286, both NEGATIVE; a Stretch's
    gripper fingers run at 10x its slider. Rounding any of those to +/-1 is a different transmission, and a
    negated one is a mechanism that moves the wrong way."""
    tb = _parsed_mimics(_urdf_root(TODDLER, tmp_path)[0])
    muls = sorted(m for _d, m, _o in tb.values())
    assert any(m == pytest.approx(-0.90909091, abs=1e-8) for m in muls), muls
    assert sum(1 for m in muls if m == pytest.approx(-0.85714286, abs=1e-8)) == 2, muls
    assert all(m < 0 for m in muls), f"every ToddlerBot gear pair is negated in the source: {muls}"

    st = _parsed_mimics(_urdf_root(STRETCH, tmp_path)[0])
    assert sum(1 for _d, m, _o in st.values() if m == pytest.approx(10.0)) == 2, st


def test_a_chained_coupling_is_composed_onto_an_independently_driven_joint(tmp_path):
    """A Stretch's four telescoping stages are declared l0<-l1<-l2<-l3 and a Talos gripper's six-bar
    motor_single<-motor_double<-inner_double. MuJoCo solves such a chain jointly; a URDF consumer does not.
    Composition is exact for degree 1, so all three Stretch stages must end up referencing l3 directly."""
    st = _parsed_mimics(_urdf_root(STRETCH, tmp_path)[0])
    arm = {d: v for d, v in st.items() if d.startswith("link_arm_l")}
    assert len(arm) == 3, arm
    assert {v[0] for v in arm.values()} == {"link_arm_l3_joint"}, arm
    assert all(v[1] == pytest.approx(1.0) for v in arm.values()), arm

    tl = _parsed_mimics(_urdf_root(TALOS, tmp_path)[0])
    # motor_single = -1 * motor_double and motor_double = 1 * inner_double  =>  motor_single = -1 * inner_double
    assert tl["gripper_left_motor_single_link_joint"] == ("gripper_left_inner_double_link_joint", -1.0, 0.0), tl
    assert tl["gripper_left_motor_double_link_joint"] == ("gripper_left_inner_double_link_joint", 1.0, 0.0), tl


def test_a_body_with_no_couplings_exports_no_mimic(tmp_path):
    """The other half. A Go2's joints are independent; an exporter that invents a coupling is worse than one
    that drops it."""
    gene = _gene("unitree_go2/go2.xml")
    assert gene.coupled_joints == []
    root, _ = _urdf_root("unitree_go2/go2.xml", tmp_path)
    assert _parsed_mimics(root) == {}


# ---------------------------------------------------------------- THE GATE: the URDF's numbers vs the physics
def _twin(gene, *, coupled: bool = True):
    import copy

    import mujoco

    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    g = copy.deepcopy(gene)
    if not coupled:
        g.coupled_joints = []
    return mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(g, include_floor=True))


def _worst_prediction_error(m, relations, *, steps: int = 1200):
    """Drive each URDF-named DRIVER joint through its range, STEP, and score the URDF's own prediction.

    Returns ``(worst |q_driven - (multiplier*q_driver + offset)|, driver travel)``. ``travel`` is returned so a
    relation that "holds" only because nothing ever moved cannot pass.
    """
    import mujoco
    import numpy as np

    def jid(n):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)

    drivers = {}
    for _driven, driver, _mul, _off in relations:
        j = jid(driver)
        assert j >= 0, f"the URDF names a driver joint {driver!r} the model does not have"
        act = next((a for a in range(m.nu) if int(m.actuator_trnid[a][0]) == j), None)
        lo, hi = ((float(m.jnt_range[j][0]), float(m.jnt_range[j][1]))
                  if int(m.jnt_limited[j]) else (-0.5, 0.5))
        drivers[driver] = (j, int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j]), act, lo, hi)
    plan = []
    for driven, driver, mul, off in relations:
        ja = jid(driven)
        assert ja >= 0, f"the URDF names a driven joint {driven!r} the model does not have"
        plan.append((int(m.jnt_qposadr[ja]), drivers[driver][1], float(mul), float(off)))

    d = mujoco.MjData(m)
    worst = travel = 0.0
    q0 = None
    for k in range(steps):
        phase = k / (steps - 1)
        for _j, qb, dof, act, lo, hi in drivers.values():
            if act is None:
                continue
            target = lo + (hi - lo) * (2 * phase if phase < 0.5 else 2 * (1 - phase))
            clo, chi = m.actuator_ctrlrange[act]
            d.ctrl[act] = float(np.clip(20.0 * (target - d.qpos[qb]) - 1.0 * d.qvel[dof], clo, chi))
        mujoco.mj_step(m, d)
        if q0 is None:
            q0 = {qb: float(d.qpos[qb]) for _j, qb, *_r in drivers.values()}
        for qa, qb, mul, off in plan:
            worst = max(worst, abs(float(d.qpos[qa]) - (mul * float(d.qpos[qb]) + off)))
        travel = max(travel, max(abs(float(d.qpos[qb]) - q0[qb]) for qb in q0))
    return worst, travel


@pytest.mark.parametrize("rel", [PANDA, ROBOTIQ, STRETCH, TODDLER, TALOS])
def test_the_urdfs_mimic_predicts_what_MUJOCO_ACTUALLY_SOLVES(rel, tmp_path):
    """THE measurement, and the reason this file is not an exporter self-test.

    The relation is read out of the WRITTEN URDF, then used as a prediction about a model the URDF had no hand
    in: drive the reference joint the URDF names, ``mj_step``, and ask whether ``multiplier * q_driver + offset``
    is where the driven joint actually ends up. A sign error, a driver/driven inversion, a chain composed the
    wrong way round, or a ratio rounded to +/-1 all fail here and none of them fail a string check.

    MEASURED on this checkout, worst |q_driven - (multiplier*q_driver + offset)| in rad/m over the sweep, with
    the model's couplings OFF (i.e. what the URDF would describe if the exporter were wrong) -> ON:
        panda 0.05308 -> 0.00068   robotiq_2f85 0.01088 -> 0.00046   stretch 0.37677 -> 0.00672
        toddlerbot 6.02917 -> 0.01397   talos 0.79142 -> 0.00360

    RE-MEASURED against the customer's own drivetrain: ``robot_import`` now carries the source's declared joint
    damping and frictionloss into the twin instead of substituting our structural prior, so both columns are
    taken on a model that integrates the numbers the customer's file states. The verdict is unchanged on all
    five. An interim version of that change also carried ARMATURE, which put talos at 0.22868 and toddlerbot at
    1.49290 — through the 0.05 gate below — and this test was one of the eight that caught it; see
    ``gene_compiler._declared_joint_dynamics``.
    """
    gene = _gene(rel)
    root, _ = _urdf_root(rel, tmp_path)
    parsed = _parsed_mimics(root)
    assert parsed, f"premise gone: {rel} exported no <mimic>"
    relations = [(driven, driver, mul, off) for driven, (driver, mul, off) in parsed.items()]

    free, _ = _worst_prediction_error(_twin(gene, coupled=False), relations)
    bound, travel = _worst_prediction_error(_twin(gene, coupled=True), relations)
    assert travel > 1e-3, f"{rel}: the driver barely moved ({travel:.2e}); the residual proves nothing"
    assert free > 1e-3, f"{rel}: premise gone — the joints already tracked within {free:.2e} uncoupled"
    assert bound < 0.05, f"{rel}: the URDF's relation mispredicts the simulated body by {bound:.5f}"
    assert bound < free / 4.0, f"{rel}: coupled {bound:.5f} vs uncoupled {free:.5f} — barely tighter"


def test_mj_forward_alone_cannot_tell_the_two_models_apart(tmp_path):
    """The trap, asserted so it cannot be re-introduced into the test above: at qpos0 every joint reads 0, so
    the URDF's relation is satisfied to machine precision whether or not the constraint exists."""
    import mujoco

    gene = _gene(PANDA)
    relations = [(d, drv, m, o) for d, (drv, m, o) in _parsed_mimics(_urdf_root(PANDA, tmp_path)[0]).items()]
    for coupled in (True, False):
        m = _twin(gene, coupled=coupled)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        for driven, driver, mul, off in relations:
            qa = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, driven)])
            qb = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, driver)])
            assert abs(float(d.qpos[qa]) - (mul * float(d.qpos[qb]) + off)) < 1e-9, (
                f"forward-only residual at qpos0 with coupled={coupled} (neq={int(m.neq)})")


# ---------------------------------------------------------------- loop closures: which lane can, which cannot
@pytest.mark.parametrize("rel", ["agility_cassie/cassie.xml", ROBOTIQ, TIDYBOT])
def test_urdf_genuinely_cannot_carry_a_loop_and_says_so_in_the_file(rel, tmp_path):
    """Disclosure is the RIGHT answer here, and this proves the premise rather than assuming it: a URDF is a
    strict tree, so every link has exactly one parent and the file structurally cannot state a second path. The
    check is on the re-parsed tree (link count vs joint count), not on the presence of a comment."""
    gene = _gene(rel)
    assert gene.loop_closures, f"premise gone: {rel} carries no loop"
    root, path = _urdf_root(rel, tmp_path)
    links = [ln.get("name") for ln in root.findall("link")]
    joints = root.findall("joint")
    children = [j.find("child").get("link") for j in joints]
    assert len(children) == len(set(children)), "a URDF link cannot have two parents; that is the limitation"
    assert len(joints) == len(links) - 1, (
        f"{rel}: {len(joints)} joints over {len(links)} links is not a tree — a loop leaked into the structure")
    # ...and the file says so, in the file, where someone opening it will see it.
    text = path.read_text(encoding="utf-8")
    assert "closed kinematic loop" in text and "NOT represented" in text, text[:400]
    for lc in gene.loop_closures:
        assert f"{lc['a']}<->{lc['b']}" in text, f"{lc} is dropped without being named"


def _usd_loops(path):
    """``[(body0, body1, localPos0), ...]`` read off a freshly-opened stage."""
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(path)
    out = []
    for p in stage.Traverse():
        if not (p.IsA(UsdPhysics.FixedJoint) and p.GetName().startswith("loop_")):
            continue
        j = UsdPhysics.FixedJoint(p)
        b0 = [str(t).rsplit("/", 1)[-1] for t in j.GetBody0Rel().GetTargets()]
        b1 = [str(t).rsplit("/", 1)[-1] for t in j.GetBody1Rel().GetTargets()]
        assert b0 and b1, f"{p.GetPath()} joins nothing"
        out.append((b0[0], b1[0], tuple(round(float(v), 6) for v in j.GetLocalPos0Attr().Get())))
    return out


@pytest.mark.skipif(not _PXR, reason="USD lane needs usd-core")
@pytest.mark.parametrize("rel", ["agility_cassie/cassie.xml", ROBOTIQ, TODDLER])
def test_usd_DOES_carry_the_loops_and_the_reopened_stage_proves_it(rel, tmp_path):
    """USD is a graph, not a tree, so here the loops are EMITTED. Counted by re-opening the written stage with
    pxr and reading each fixed joint's body relationships, not from the writer's own list."""
    from collections import Counter

    from virturoid.services.usd_exporter import _safe, export_usd
    gene = _gene(rel)
    assert gene.loop_closures, f"premise gone: {rel} carries no loop"
    path = str(tmp_path / "loops.usda")
    man = export_usd(gene, path)
    found = _usd_loops(path)
    assert Counter((a, b) for a, b, _p in found) == Counter(
        (_safe(lc["a"]), _safe(lc["b"])) for lc in gene.loop_closures), found
    assert man["constraints_reread"]["loop_fixed_joints"] == len(gene.loop_closures)


@pytest.mark.skipif(not _PXR, reason="USD lane needs usd-core")
def test_two_loops_between_the_same_pair_of_bodies_are_two_prims(tmp_path):
    """ONE PRIM PER CONSTRAINT. Two ``<connect>``s between the same body pair at different anchors is a real
    construction -- it pins a rotation as well as a position -- and ToddlerBot ships four of them. Named on the
    body pair alone, ``FixedJoint.Define`` returned the prim already at that path and the second anchor
    overwrote the first: 4 declared, 2 authored, and the survivor carried the wrong anchor. Nothing in the
    writer noticed; counting the re-read file against the gene did."""
    from virturoid.services.usd_exporter import export_usd
    gene = _gene(TODDLER)
    pairs = [(lc["a"], lc["b"]) for lc in gene.loop_closures]
    assert len(pairs) > len(set(pairs)), f"premise gone: ToddlerBot no longer double-joins a pair: {pairs}"
    path = str(tmp_path / "toddler.usda")
    export_usd(gene, path)
    found = _usd_loops(path)
    assert len(found) == len(gene.loop_closures), (
        f"{len(gene.loop_closures)} loops declared, {len(found)} prims in the file")
    anchors = [p for _a, _b, p in found]
    assert len(set(anchors)) == len(anchors), f"two loop joints share an anchor; one overwrote the other: {found}"


# ---------------------------------------------------------------- USD/Isaac: what they cannot express, said
@pytest.mark.skipif(not _PXR, reason="USD lane needs usd-core")
def test_core_usdphysics_really_has_no_way_to_express_a_coupling():
    """The premise for disclosing rather than emitting, asserted instead of asserted-by-comment. If a future
    usd-core ships a joint-coupling schema (or PhysxSchema becomes importable), this fails and the USD lane
    should start EMITTING instead."""
    import importlib

    from pxr import UsdPhysics
    names = [n for n in dir(UsdPhysics) if not n.startswith("_")]
    assert not [n for n in names if any(k in n.lower() for k in ("mimic", "gear", "rackandpinion", "coupl"))], (
        f"UsdPhysics now offers a coupling schema: {names}")
    assert importlib.util.find_spec("pxr.PhysxSchema") is None, (
        "PhysxSchema is importable now — its mimic/gear joints can be authored AND re-read, so emit them")


@pytest.mark.skipif(not _PXR, reason="USD lane needs usd-core")
@pytest.mark.parametrize("rel", [PANDA, TODDLER])
def test_usd_discloses_every_coupling_in_the_file_and_in_the_manifest(rel, tmp_path):
    """Silence would be the failure. The relation is written onto the driven joint prim as customData and read
    back off a freshly-opened stage — so a record that failed to author is a failure here, not a green export."""
    from pxr import Usd

    from virturoid.services.usd_exporter import export_usd
    gene = _gene(rel)
    path = str(tmp_path / "coupled.usda")
    man = export_usd(gene, path)

    assert man["coupled_joints_expressed"] is False
    assert len(man["coupled_joints"]) == len(gene.coupled_joints)
    assert man["constraints_reread"]["coupling_records"] == len(gene.coupled_joints)
    assert "no joint-coupling schema" in man["coupled_joints_note"]

    stage = Usd.Stage.Open(path)
    on_file = {}
    for p in stage.Traverse():
        rec = p.GetCustomDataByKey("virturoid_coupledJoint")
        if rec:
            on_file[p.GetName()] = (str(rec["driverJoint"]).rsplit("/", 1)[-1],
                                    float(rec["multiplier"]), float(rec["offset"]))
    want = {f"{cj['a']}_joint": (f"{cj['b']}_joint", float(cj["ratio"]), float(cj.get("offset") or 0.0))
            for cj in gene.coupled_joints}
    assert on_file == want, f"{rel}: the .usda's own record disagrees with the gene\n{on_file}\n{want}"


@pytest.mark.skipif(not _PXR, reason="the Isaac Lab package is built around the USD")
def test_the_isaac_package_hands_the_couplings_over_instead_of_dropping_them(tmp_path):
    """Isaac Lab has no coupling representation of its own (an ArticulationCfg maps joint-name regexes to drive
    gains and nothing more), and the PhysX layer that does is the layer this box cannot run. So the relations
    are handed over as data the engineer can act on — in generated Python that BYTE-COMPILES, and in the README
    — rather than as unverified generated constraints."""
    import py_compile

    from virturoid.services.isaac_lab_exporter import export_isaac_lab
    gene = _gene(PANDA)
    man = export_isaac_lab(gene, str(tmp_path / "isaac"), robot_name="panda")
    cfg = Path(man["files"]["cfg"])
    py_compile.compile(str(cfg), doraise=True)

    ns: dict = {}
    body = cfg.read_text(encoding="utf-8")
    exec(compile(body[body.index("# --- CONSTRAINTS"):], str(cfg), "exec"), ns)   # noqa: S102 - our own output
    assert ns["COUPLED_JOINTS"] == [
        {"driven": "left_finger_joint", "driver": "right_finger_joint", "multiplier": 1.0, "offset": 0.0}]
    assert ns["LOOP_CLOSURES"] == []
    assert man["coupled_joints_expressed"] is False
    readme = (tmp_path / "isaac" / "README.md").read_text(encoding="utf-8")
    assert "NOT ENFORCED by the USD" in readme and "left_finger_joint" in readme, readme


# ---------------------------------------------------------------- the composer's own rules
def test_the_composer_refuses_what_urdf_cannot_state():
    """Two couplings onto one joint (URDF permits one <mimic> per joint) and a closed cycle of couplings (a
    <mimic> needs an independently-commanded reference and a cycle has none) are both dropped WITH A REASON, not
    silently mis-emitted."""
    from virturoid.services.gene_urdf import _flatten_couplings

    class _G:
        def __init__(self, cjs):
            self.coupled_joints = cjs

    res, notes = _flatten_couplings(_G([{"a": "x", "b": "y", "ratio": 2.0, "offset": 0.1},
                                        {"a": "x", "b": "z", "ratio": 3.0, "offset": 0.0}]))
    assert res == {"x": ("y", 2.0, 0.1)}
    assert any("one <mimic> per joint" in n for n in notes), notes

    res, notes = _flatten_couplings(_G([{"a": "p", "b": "q", "ratio": 2.0, "offset": 0.0},
                                        {"a": "q", "b": "p", "ratio": 0.5, "offset": 0.0}]))
    assert res == {}, res
    assert all("closes a cycle" in n for n in notes) and len(notes) == 2, notes

    # and the offset composes with the ratio, which is the easy half to get wrong
    res, _n = _flatten_couplings(_G([{"a": "a", "b": "b", "ratio": 2.0, "offset": 0.5},
                                     {"a": "b", "b": "c", "ratio": 3.0, "offset": 0.25}]))
    assert res["a"] == ("c", 6.0, 1.0), res      # a = 2*(3c + 0.25) + 0.5 = 6c + 1.0
    assert res["b"] == ("c", 3.0, 0.25), res


def test_the_ros2_package_does_not_command_a_joint_with_no_motor():
    """The URDF carries into the ROS 2 package verbatim, so ``<mimic>`` arrives for free. What does NOT come for
    free: ``joint_trajectory_controller`` commands every joint in its ``joints`` list, and a mimic joint has no
    actuator — its position is a consequence of the driver's. Listing it asks the hardware interface for a motor
    that does not exist."""
    import json

    from virturoid.services.ros2_exporter import _mimic_joints_from_urdf, _ros2_control_yaml

    urdf = ('<robot name="r">'
            '<link name="a"/><link name="b"/><link name="c"/>'
            '<joint name="drv" type="revolute"><parent link="a"/><child link="b"/>'
            '<axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>'
            '<joint name="slave" type="revolute"><parent link="b"/><child link="c"/>'
            '<axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/>'
            '<mimic joint="drv" multiplier="-0.90909091" offset="0.25"/></joint>'
            "</robot>")
    mimic = _mimic_joints_from_urdf(urdf)
    assert mimic == {"slave": {"joint": "drv", "multiplier": -0.90909091, "offset": 0.25}}
    yaml = _ros2_control_yaml([j for j in ["drv", "slave"] if j not in mimic], mimic)
    assert "      - drv" in yaml
    assert "      - slave" not in yaml, yaml
    assert "-0.90909091" in yaml, "the excluded joint's relation must still be stated"
    assert json.dumps(mimic)                                    # serialisable into config/robot.yaml
