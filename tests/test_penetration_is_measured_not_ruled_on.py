"""The simulability gate MEASURES self-penetration and RULES ON NOTHING. This is the spec for the rule it owes.

``robot_import._simulability_probe`` (9fbdcdc) compiles the editable twin, settles it, drives every actuator
across its own ``ctrlrange``, and fails the import if MuJoCo raises BADQACC/BADQVEL/BADQPOS/BADCTRL. That is a
COMPILE-AND-DIVERGENCE detector, and it is a good one. It is not a geometry-correctness detector, and it was
described as though it were.

MEASURED 2026-08-07 by re-breaking each of the four multi-root twins the way the multi-root bug broke them
(every reparented root's ``mount_offset`` forced to (0,0,0)) and re-running the probe:

    package            correct pen.   re-broken pen.   probe rejects the broken twin?
    aloha                 0.05464          0.14172      YES
    trossen_wxai          0.03132          0.10000      no
    shadow_dexee          0.04894          0.07468      no
    apptronik_apollo      0.06844          0.09959      no

One of four. ALOHA is caught because stacking two 0.94 m-apart arms drives the solver to a NaN; the other three
stack into a geometry that is wrong but numerically survivable, and the probe has nothing to say about wrong.
The number that would say it -- ``max_self_penetration_m`` -- is computed on every run, reported in the payload,
and compared to nothing anywhere in the codebase.

WHY WE DID NOT SIMPLY ADD A THRESHOLD. A capsule twin overlaps a little at every joint by construction, so some
penetration is normal and a naive budget rejects good robots. Measured across all 63 MuJoCo Menagerie packages
(62 twins; flybody does not compile at all):

    correct twins   34/62 penetrate ZERO   p50 0.00000   p75 0.03017   p90 0.05418   p95 0.05712   max 0.06844
    re-broken (4)   0.07468 / 0.09959 / 0.10000 / 0.14172

A global threshold must therefore live in (0.06844, 0.07468] -- a 9.1% window, with a single observation at
each edge, on a corpus of 62 robots. Two things make that worse, not better:

  * NORMALISING BY LINK RADIUS SHRINKS THE WINDOW. depth / thinnest-half-extent-of-the-thinner-geom gives
    worst-correct 3.2627 (shadow_dexee, CORRECT) against weakest-broken 3.4227 (trossen_wxai): a 4.9% window.
  * "PENETRATION BETWEEN BODIES THAT SHARE NO KINEMATIC PATH" DOES NOT SEPARATE THEM EITHER, because in a
    correct twin it already happens. ``gene_compiler._self_collision_excludes_xml`` excludes every
    ancestor/descendant pair, so ALL 62 twins report exactly 0.0 m of lineal penetration and every contact that
    survives is already cross-branch. The deepest contact in the CORRECT Apollo twin is
    ``l_wrist_pitch_link`` inside ``l_hip_fe_link`` -- 13 hops apart, lowest common ancestor ``base_link``,
    0.06844 m. That is "two limbs inside each other" by any structural definition, on a twin that is right.
    Restricting to hops >= 6 makes correct Apollo and BROKEN Apollo tie at 0.06844 exactly.

The repo's one existing penetration rule, ``structural_hygiene.DEFAULT_PENETRATION_BUDGET_M = 0.02`` (zero
production callers), would reject 22 of the 62 correct twins -- a Franka Panda, a PAL Talos, an Agility Cassie --
and 4 of the 63 CUSTOMERS' OWN UNMODIFIED MODELS, which self-penetrate up to 0.07026 m (Talos) as shipped.

WHAT DOES SEPARATE THEM is not a penetration threshold at all: it is whether the twin PUT THE CUSTOMER'S LINKS
WHERE THE CUSTOMER PUT THEM. Penetration has two causes -- our collider approximation (legitimate, unbounded by
any constant) and a placement error (a defect) -- and only the second is our bug. The discriminator is measured
in ``test_the_signal_that_separates_them_is_placement_not_a_threshold`` below: a correct twin reproduces every
pairwise body-origin distance in the source to ~1e-5 m; a re-broken one is off by half a metre. That is five
orders of magnitude of margin instead of nine percent -- and the information it needs (the source model) is in
``import_robot``'s hand and is never passed to the probe, which sees only a gene.

See docs/what_the_simulability_gate_does_not_catch.md.
"""
from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("VIRTUROID_DISABLE_GAIT_HINTS", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="importing a robot needs MuJoCo")

_MEN = Path(os.path.expanduser("~/.cache/robot_descriptions/mujoco_menagerie"))

# The four packages that declare more than one body under <worldbody>, and the penetration each one's twin
# reaches when correct / when re-broken. Measured 2026-08-07; quoted in the asserts so a drift is legible.
MULTI_ROOT = [
    ("aloha", "aloha/aloha.xml", 0.05464, 0.14172),
    ("trossen_wxai", "trossen_wxai/trossen_ai_bimanual.xml", 0.03132, 0.10000),
    ("shadow_dexee", "shadow_dexee/shadow_dexee.xml", 0.04894, 0.07468),
    ("apptronik_apollo", "apptronik_apollo/apptronik_apollo.xml", 0.06844, 0.09959),
]

WORST_CORRECT_M = 0.06844      # apptronik_apollo, the deepest self-penetration in any of the 62 correct twins
WEAKEST_BROKEN_M = 0.07468     # shadow_dexee re-broken, the shallowest of the four genuine defects


def _import(rel: str, prefix: str):
    from virturoid.services.robot_import import import_robot

    src = _MEN / rel
    if not src.is_file():
        pytest.skip(f"{rel} is not in the local Menagerie cache")
    return import_robot(str(src), robot_id=f"{prefix}_{Path(rel).stem}")


def _collapse_roots(gene):
    """The PRE-9fbdcdc twin, rebuilt: every reparented root loses its source world pose and stacks on the base."""
    g = copy.deepcopy(gene)
    root = g.root()
    for s in g.segments:
        if s.parent == root.name:
            s.mount_offset = (0.0, 0.0, 0.0)
    return g


def _zero_pose_xpos(model):
    """Body-origin positions at the ZERO configuration -- the pose ``body_pos``/``body_quat`` alone determine,
    so the numbers read the static tree geometry and nothing about joint angles or spawn height."""
    import mujoco

    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)
    mujoco.mj_forward(model, d)
    return {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b): d.xpos[b].copy()
            for b in range(1, model.nbody)}


def _twin_model(gene):
    import mujoco

    from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
    return mujoco.MjModel.from_xml_string(
        compile_gene_to_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene)))


def _max_pairwise_deviation(src_pos, twin_pos):
    """max |d_source(a,b) - d_twin(a,b)| over shared body pairs. Invariant to any rigid re-datum, which an
    import legitimately performs, and sensitive to exactly the thing it may never do: move the links."""
    import numpy as np

    shared = sorted(set(src_pos) & set(twin_pos))
    assert len(shared) >= 2, f"only {len(shared)} shared bodies"
    A = np.asarray([src_pos[n] for n in shared])
    B = np.asarray([twin_pos[n] for n in shared])
    da = np.linalg.norm(A[:, None] - A[None], axis=-1)
    db = np.linalg.norm(B[:, None] - B[None], axis=-1)
    return float(np.abs(da - db).max()), len(shared)


def _model_extent(pos):
    """The model's own longest link-to-link span -- the scale a placement error should be judged against.

    Menagerie spans two orders of magnitude here (shadow_dexee 0.297 m, apptronik_apollo 1.583 m), so any
    bound stated in bare metres means something different on each machine.
    """
    import numpy as np

    P = np.asarray([v for v in pos.values()])
    return float(np.linalg.norm(P[:, None] - P[None], axis=-1).max()) if len(P) >= 2 else 0.0


# --------------------------------------------------------------------------- the spec
@pytest.mark.xfail(reason="THE RULE THE GATE OWES, NOT YET WRITTEN. ``_simulability_probe`` rules only on "
                          "MuJoCo's divergence warnings, so it rejects 1 of the 4 stacked twins (aloha, which "
                          "NaNs) and passes the other 3, which are geometrically wrong but numerically "
                          "survivable. It already computes ``max_self_penetration_m`` and compares it to "
                          "nothing. Landing this needs a rule that separates OUR placement error from OUR "
                          "collider approximation -- see the module docstring: no global threshold does, the "
                          "window is (0.06844, 0.07468], and the discriminator that works needs the SOURCE "
                          "model, which import_robot has and never hands to the probe.", strict=True)
def test_a_twin_whose_limbs_have_been_stacked_on_one_point_is_rejected():
    """Re-break each multi-root twin the way the bug broke it; every one of them must fail the gate."""
    from virturoid.services.robot_import import _simulability_probe

    accepted, evidence = [], []
    for pkg, rel, _ok_pen, want_pen in MULTI_ROOT:
        if not (_MEN / rel).is_file():
            pytest.skip(f"{rel} is not in the local Menagerie cache")
        out = _import(rel, "spec")
        assert out["gene"] is not None, f"{pkg} did not import: {out.get('warnings')}"
        res = _simulability_probe(_collapse_roots(out["gene"]))
        pen = res.get("max_self_penetration_m")
        evidence.append(f"{pkg}: ok={res.get('ok')} pen={pen}")
        assert pen == pytest.approx(want_pen, abs=5e-4), (
            f"{pkg}: the re-broken twin's penetration moved from the measured {want_pen} to {pen}; "
            f"re-derive the window in the module docstring before trusting any threshold built on it")
        if res.get("ok"):
            accepted.append(pkg)
    assert not accepted, (
        "the gate ACCEPTED these stacked twins: " + ", ".join(accepted) + ". Every downstream number "
        "(verdict, certificate, BOM, spec sheet, calibration gap) is computed by stepping a body whose limbs "
        "are inside each other. Evidence: " + "; ".join(evidence))


# --------------------------------------------------------------------------- the guards the fix must not break
@pytest.mark.parametrize("pkg,rel,ok_pen,_bad", MULTI_ROOT, ids=[m[0] for m in MULTI_ROOT])
def test_the_rule_must_not_reject_the_correct_twin_of_the_same_robot(pkg, rel, ok_pen, _bad):
    """The other half of the spec, and the reason a naive budget is not the answer: these four twins are RIGHT,
    and three of them out-penetrate the 0.02 m budget the repo already has lying around."""
    from virturoid.services.robot_import import _simulability_probe

    out = _import(rel, "guard")
    assert out["gene"] is not None, f"{pkg} did not import: {out.get('warnings')}"
    res = _simulability_probe(out["gene"])
    assert res["checked"] and res["ok"], f"{pkg}: the correct twin must pass the gate: {res}"
    assert res["max_self_penetration_m"] == pytest.approx(ok_pen, abs=5e-4), (
        f"{pkg}: a correct twin's self-penetration moved from the measured {ok_pen} to "
        f"{res['max_self_penetration_m']}. Every threshold in this file is calibrated against that number.")


def test_no_global_penetration_threshold_separates_the_two_populations():
    """The window, as an executable fact. A threshold must reject shadow_dexee's stacked twin (0.07468) and
    accept Apollo's correct one (0.06844) -- 9.1% apart, one observation at each edge."""
    from virturoid.services.robot_import import _simulability_probe

    apollo = _import("apptronik_apollo/apptronik_apollo.xml", "window")
    dexee = _import("shadow_dexee/shadow_dexee.xml", "window")
    assert apollo["gene"] is not None and dexee["gene"] is not None

    worst_correct = _simulability_probe(apollo["gene"])["max_self_penetration_m"]
    weakest_broken = _simulability_probe(_collapse_roots(dexee["gene"]))["max_self_penetration_m"]
    assert worst_correct == pytest.approx(WORST_CORRECT_M, abs=5e-4)
    assert weakest_broken == pytest.approx(WEAKEST_BROKEN_M, abs=5e-4)
    assert weakest_broken > worst_correct, (
        f"a correct twin now out-penetrates a broken one ({worst_correct} vs {weakest_broken}); NO global "
        f"threshold exists at all, and the module docstring's conclusion is now stronger, not weaker")
    assert weakest_broken / worst_correct < 1.10, (
        f"the whole margin available to a global threshold is {weakest_broken / worst_correct:.3f}x. If this "
        f"ever exceeds 1.10 the corpus has changed and the 'not defensible' claim should be re-argued.")


def test_the_repos_only_penetration_budget_would_reject_real_robots():
    """``structural_hygiene.DEFAULT_PENETRATION_BUDGET_M`` is 0.02 m and has no production callers. Wiring it
    into the import path is the obvious fix and it is wrong: 22 of the 62 correct Menagerie twins exceed it."""
    from virturoid.services.robot_import import _simulability_probe
    from virturoid.services.structural_hygiene import DEFAULT_PENETRATION_BUDGET_M

    out = _import("apptronik_apollo/apptronik_apollo.xml", "budget")
    pen = _simulability_probe(out["gene"])["max_self_penetration_m"]
    assert pen > DEFAULT_PENETRATION_BUDGET_M, (
        f"Apollo's CORRECT twin penetrates {pen} m against a {DEFAULT_PENETRATION_BUDGET_M} m budget. If this "
        f"is now false the compiler's colliders changed and the whole distribution needs re-measuring.")


def test_every_penetration_a_twin_reports_is_already_cross_branch():
    """Why "penetration between bodies that share no kinematic path" cannot be the rule: it is the only kind
    there is. The compiler excludes every ancestor/descendant pair, so a lineal overlap is unobservable, and
    the deepest contact in a CORRECT humanoid twin is a wrist 13 hops from the hip it sits inside."""
    import mujoco

    out = _import("apptronik_apollo/apptronik_apollo.xml", "branch")
    m = _twin_model(out["gene"])
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    floor = {g for g in range(m.ngeom) if int(m.geom_type[g]) == mujoco.mjtGeom.mjGEOM_PLANE}

    def ancestors(b):
        out_, x = [], int(b)
        while x != 0:
            out_.append(x)
            x = int(m.body_parentid[x])
        return out_

    worst, worst_hops, worst_pair = 0.0, 0, None
    for c in d.contact[:d.ncon]:
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 in floor or g2 in floor or c.dist >= 0:
            continue
        b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
        a1, a2 = ancestors(b1), ancestors(b2)
        assert b1 not in a2 and b2 not in a1, (
            f"an ancestor/descendant pair penetrates: "
            f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b1)} / "
            f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b2)}. The compiler is meant to exclude those, "
            f"so this changes what the penetration number means.")
        shared = set(a1) & set(a2)
        hops = min(a1.index(v) + a2.index(v) for v in shared) if shared else len(a1) + len(a2)
        if -c.dist > worst:
            worst, worst_hops = float(-c.dist), hops
            worst_pair = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b1),
                          mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b2))
    assert worst == pytest.approx(WORST_CORRECT_M, abs=5e-4), (worst, worst_pair)
    assert worst_hops >= 6, (
        f"the deepest overlap in a CORRECT twin is {worst_hops} hops apart ({worst_pair}) -- the measured case "
        f"is 13. A structural 'they share no kinematic path' filter would flag it exactly like a real defect.")


# --------------------------------------------------------------------------- what a fix should measure instead
@pytest.mark.parametrize("pkg,rel,_ok,_bad", MULTI_ROOT, ids=[m[0] for m in MULTI_ROOT])
def test_the_signal_that_separates_them_is_placement_not_a_threshold(pkg, rel, _ok, _bad):
    """Penetration has two causes and only one of them is a bug. The one that IS a bug -- our twin moving the
    customer's links -- is measurable against the customer's own model, with a margin four orders of magnitude
    wider than the 9% a penetration threshold gets. This is evidence, not a proposal: it is measured here.

    Note what it costs: the SOURCE model. ``_simulability_probe`` takes a gene and cannot do this;
    ``import_robot`` is holding the source when it calls the probe and does not pass it.
    """
    import mujoco

    src = _MEN / rel
    if not src.is_file():
        pytest.skip(f"{rel} is not in the local Menagerie cache")
    src_pos = _zero_pose_xpos(mujoco.MjModel.from_xml_path(str(src)))

    out = _import(rel, "place")
    assert out["gene"] is not None
    good_dev, n_shared = _max_pairwise_deviation(src_pos, _zero_pose_xpos(_twin_model(out["gene"])))
    bad_dev, _ = _max_pairwise_deviation(
        src_pos, _zero_pose_xpos(_twin_model(_collapse_roots(out["gene"]))))

    # "A limb's worth" is a fraction of THE MACHINE, not a fixed number of metres. The floor used to be a flat
    # 0.1 m, which is a limb on a 1.5 m humanoid and a THIRD OF THE WHOLE ROBOT on a 0.30 m hand -- so it read
    # as a real bound on three packages and as an accident on the fourth. Measured over the four, stacked
    # deviation against the source model's own longest link-to-link span:
    #     aloha 0.938/0.977 = 0.96   trossen_wxai 0.915/1.403 = 0.65
    #     apollo 0.376/1.583 = 0.24  shadow_dexee 0.088/0.297 = 0.30
    # so a fifth of the body is a floor every one of them clears with margin, and it does not quietly weaken
    # on a small machine. (shadow_dexee reads 0.088 m rather than the old 0.1+ because its three <attach>
    # placement frames are no longer emitted as segments -- see `_frame_only_bodies`; the finger LINKS still
    # move by a finger's length, which is what this measures.)
    _extent = _model_extent(src_pos)
    assert good_dev < 0.001, (
        f"{pkg}: the CORRECT twin already deviates {good_dev:.6f} m from the customer's own link placement "
        f"over {n_shared} shared bodies; the discriminator below is only as good as this number is small")
    assert bad_dev > 0.2 * _extent, (
        f"{pkg}: the STACKED twin deviates only {bad_dev:.6f} m from the source, {bad_dev / _extent:.2f} of "
        f"the robot's own {_extent:.3f} m span -- expected a limb's worth")
    assert bad_dev / max(good_dev, 1e-9) > 1000.0, (
        f"{pkg}: placement separates the two populations by {bad_dev / max(good_dev, 1e-9):.0f}x "
        f"({good_dev:.6f} m correct vs {bad_dev:.4f} m broken), against the 1.09x a penetration threshold gets")
