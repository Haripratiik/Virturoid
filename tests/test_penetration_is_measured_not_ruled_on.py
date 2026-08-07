"""The gate MEASURED self-penetration and RULED ON NOTHING. This was the spec for the rule it owed; it is now met.

WHAT SHIPPED (2026-08-07): NOT a penetration threshold. ``robot_import._source_link_placement`` records where
the customer's own model puts every link, ``_placement_fidelity`` checks the compiled twin reproduces those
pairwise distances, and ``_simulability_probe`` fails the import when it does not. All four re-broken twins are
now rejected. All 63 packages were driven through ``agent_tools.call_tool('ingest_project')``: ZERO false
rejections, and the tightest of the 59 that reach the check sits 385x under its own budget. The measurement
that ruled the obvious fix out is preserved below, because it is the argument for the rule that did ship.


--------------------------------------------------------------------------- WHAT THE GATE WAS, AND WHY

``robot_import._simulability_probe`` (9fbdcdc) compiled the editable twin, settled it, drove every actuator
across its own ``ctrlrange``, and failed the import if MuJoCo raised BADQACC/BADQVEL/BADQPOS/BADCTRL. That is a
COMPILE-AND-DIVERGENCE detector, and it is a good one. It is not a geometry-correctness detector, and it was
described as though it were. It is still there -- it is rule 2 -- and this is what it alone could do.

MEASURED 2026-08-07 by re-breaking each of the four multi-root twins the way the multi-root bug broke them
(every reparented root's ``mount_offset`` forced to (0,0,0)) and re-running the probe as it then was:

    package            correct pen.   re-broken pen.   divergence alone rejects the broken twin?
    aloha                 0.05464          0.14172      YES
    trossen_wxai          0.03132          0.10000      no
    shadow_dexee          0.04894          0.07468      no
    apptronik_apollo      0.06844          0.23964      no      (0.09959 as first written; see MULTI_ROOT)

One of four. ALOHA is caught because stacking two 0.94 m-apart arms drives the solver to a NaN; the other three
stack into a geometry that is wrong but numerically survivable, and a divergence detector has nothing to say
about wrong. The number that looked like it would say it -- ``max_self_penetration_m`` -- was computed on every
run, reported in the payload, and compared to nothing anywhere in the codebase. It still is: see below for the
measurement that says it never can be.

WHY WE DID NOT SIMPLY ADD A THRESHOLD. A capsule twin overlaps a little at every joint by construction, so some
penetration is normal and a naive budget rejects good robots. Measured across all 63 MuJoCo Menagerie packages
(62 twins; flybody does not compile at all):

    correct twins   32/62 penetrate ZERO   p50 0.00000   p75 0.03017   p90 0.05418   p95 0.05712   max 0.06844
    re-broken (4)   0.07468 / 0.10000 / 0.14172 / 0.23964

(The zero count read 34 when this was first written and measures 32 on the re-sweep. Every percentile and the
max reproduce to the digit on the same 62 models, so that is a counting convention -- these numbers are the
probe's own, i.e. the worst over BOTH start poses -- and not a corpus that moved.)

A global threshold must therefore live in (0.06844, 0.07468] -- a 9.1% window, with a single observation at
each edge, on a corpus of 62 robots. THE HYPOTHESIS WORTH TESTING FIRST -- "penetration between bodies that
share no kinematic path, normalised by link size" -- was swept over the whole corpus on 2026-08-07 in every
combination, and not one of them separates the populations. Each row is worst-correct -> weakest-broken:

    statistic                                       window    correct twins at/above the weakest broken
    max depth                          0.0684 -> 0.0747  1.091x   0 / 62
    depth, >= 2 hops apart             0.0684 -> 0.0747  1.091x   0 / 62   (identical -- see below)
    depth, >= 4 hops apart             0.0684 -> 0.0696  1.017x   0 / 62
    depth, >= 6 hops apart             0.0684 -> 0.0300  0.438x   4 / 62   INVERTED
    depth / thinner geom half-extent   3.2627 -> 3.4227  1.049x   0 / 62
    same, >= 2 hops                    3.2627 -> 3.4227  1.049x   0 / 62
    same, >= 4 hops                    3.2627 -> 3.4227  1.049x   0 / 62
    same, >= 6 hops                    2.4046 -> 2.0000  0.832x   2 / 62   INVERTED

Normalising buys 4.9% instead of 9.1%. Requiring the bodies to be structurally distant buys NOTHING (the >= 2
column is identical to the raw one) and then INVERTS: at >= 6 hops a correct ``iit_softfoot`` out-penetrates a
broken ``shadow_dexee`` and no threshold exists in either direction. The reason the >= 2 column is identical is
the load-bearing one: ``gene_compiler._self_collision_excludes_xml`` already excludes every ancestor/descendant
pair, so ALL 62 twins report exactly 0.0 m of lineal penetration and EVERY contact that survives is already
cross-branch. The deepest contact in the CORRECT Apollo twin is ``l_wrist_pitch_link`` inside ``l_hip_fe_link``
-- 13 hops apart, lowest common ancestor ``base_link``, 0.06844 m. That is "two limbs inside each other" by any
structural definition, on a twin that is right.

The repo's one existing penetration rule, ``structural_hygiene.DEFAULT_PENETRATION_BUDGET_M = 0.02`` (zero
production callers), would reject 22 of the 62 correct twins -- a Franka Panda, a PAL Talos, an Agility Cassie --
and 4 of the 63 CUSTOMERS' OWN UNMODIFIED MODELS, which self-penetrate up to 0.07026 m (Talos, a right gripper
motor inside the right leg) as shipped. Both counts re-measured 2026-08-07 and both reproduce exactly.

WHAT DOES SEPARATE THEM is not a penetration threshold at all: it is whether the twin PUT THE CUSTOMER'S LINKS
WHERE THE CUSTOMER PUT THEM. Penetration has two causes -- our collider approximation (legitimate, unbounded by
any constant) and a placement error (a defect) -- and only the second is our bug. The discriminator is measured
in ``test_the_signal_that_separates_them_is_placement_not_a_threshold`` below: a correct twin reproduces every
pairwise body-origin distance in the source to ~1e-5 m; a re-broken one is off by half a metre.

That is what shipped. The information it needs -- the source model -- was in ``import_robot``'s hand and was
dropped there; it is now recorded ONTO THE GENE at import (``metadata['source_link_placement']``), so the probe
still takes a gene and a gene alone. Measured over all 63 packages through ``call_tool('ingest_project')``:
worst correct twin 0.000017 m (pal_tiago_dual), tightest against its own budget 0.26% of it (385x margin,
tetheria_aero_hand_open); weakest defect 0.0881 m against a 0.0059 m budget (14.9x over). The rule is NOT a
fitted constant -- it is "an imported twin may not move the customer's links" -- and it expires the moment a
link is resized, so an amend is never failed by it.

See docs/what_the_simulability_gate_does_not_catch.md for what it still does not catch.
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
# apptronik_apollo's re-broken figure was written here as 0.09959 and IS NOT REPRODUCIBLE on this checkout; the
# assert below caught it and it is corrected to the measured 0.23964. Apollo is the one of the four that is NOT
# multi-root once its empty `world_link` datum is folded away (see test_multiroot_twin_is_simulable), so it keeps
# its OWN `base_link` as the gene root and `_collapse_roots` flattens five real children onto it -- both torso
# cameras, the torso roll and both hips -- rather than the two mounting frames the multi-root bug moved. That is
# a harsher break than the bug, and it makes the "no threshold separates them" conclusion STRONGER, not weaker:
# the broken population moves further from the correct one. Nothing else in this file rests on it (the window is
# set by shadow_dexee at 0.07468, which reproduces exactly).
MULTI_ROOT = [
    ("aloha", "aloha/aloha.xml", 0.05464, 0.14172),
    ("trossen_wxai", "trossen_wxai/trossen_ai_bimanual.xml", 0.03132, 0.10000),
    ("shadow_dexee", "shadow_dexee/shadow_dexee.xml", 0.04894, 0.07468),
    ("apptronik_apollo", "apptronik_apollo/apptronik_apollo.xml", 0.06844, 0.23964),
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
def test_a_twin_whose_limbs_have_been_stacked_on_one_point_is_rejected():
    """Re-break each multi-root twin the way the bug broke it; every one of them must fail the gate.

    Was a strict xfail until 2026-08-07: the gate rejected only ALOHA, the one of the four that NaNs. The three
    that survive numerically are now rejected on PLACEMENT, not on any number measured from a rollout -- so the
    ``stage`` is asserted too, or a future divergence-only pass would read as this rule working.
    """
    from virturoid.services.robot_import import _simulability_probe

    accepted, evidence = [], []
    for pkg, rel, _ok_pen, want_pen in MULTI_ROOT:
        if not (_MEN / rel).is_file():
            pytest.skip(f"{rel} is not in the local Menagerie cache")
        out = _import(rel, "spec")
        assert out["gene"] is not None, f"{pkg} did not import: {out.get('warnings')}"
        res = _simulability_probe(_collapse_roots(out["gene"]))
        pen = res.get("max_self_penetration_m")
        evidence.append(f"{pkg}: ok={res.get('ok')} stage={res.get('stage')} pen={pen}")
        assert pen == pytest.approx(want_pen, abs=5e-4), (
            f"{pkg}: the re-broken twin's penetration moved from the measured {want_pen} to {pen}; "
            f"re-derive the window in the module docstring before trusting any threshold built on it")
        if res.get("ok"):
            accepted.append(pkg)
        else:
            place = res.get("placement_check") or {}
            assert res.get("stage") == "placement" and place.get("checked") and not place.get("ok"), (
                f"{pkg} was rejected, but not by the placement rule ({res.get('stage')!r}). Three of these four "
                f"twins step perfectly well; a gate that only sees divergence passes them.")
            assert place["max_link_displacement_m"] > 10.0 * place["tolerance_m"], (
                f"{pkg}: the defect is only {place['max_link_displacement_m'] / place['tolerance_m']:.1f}x the "
                f"budget. This rule is worth having because that ratio is large; if it is not, re-measure.")
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
    wider than the 9% a penetration threshold gets. This is the evidence the shipped rule rests on, and it is
    measured HERE, independently: it re-reads the source file itself rather than the record the import wrote,
    so it would still fail if ``_source_link_placement`` recorded the twin's own positions instead of the
    customer's.
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


# --------------------------------------------------------------------------- the rule must not fire elsewhere
# An overfitted gate that rejects a customer's working robot is worse than a gate with known limits. These two
# are the cases where "the twin does not match the source" is not a defect at all, and the rule has to KNOW that
# rather than be tuned around it: there is no source, or the customer moved the links themselves.
def test_a_body_we_composed_here_is_not_ruled_on_at_all():
    """No source model, no claim. A gene we generated has nothing it is supposed to reproduce, and a rule that
    guessed at one would fail bodies that are correct by construction."""
    from virturoid.services.morphology_composer import compose_from_spec, morphology_from_requirements
    from virturoid.services.robot_import import _placement_fidelity, _simulability_probe

    gene = compose_from_spec(morphology_from_requirements(
        0.65, 0.25, prompt="a small four-legged walking robot", robot_class="quadruped"))
    res = _simulability_probe(gene)
    place = res.get("placement_check") or {}
    assert place.get("checked") is False, f"a composed body was ruled on against a source it never had: {place}"
    assert "no record" in str(place.get("reason", "")).lower()
    assert res.get("stage") != "placement"
    # and the helper says the same thing when called directly on a gene with no record at all
    gene.metadata = {}
    assert _placement_fidelity(gene, None, None)["checked"] is False


def test_an_amend_that_resizes_a_link_retires_the_rule_instead_of_failing_the_customer():
    """A child hangs off its parent's TIP, so lengthening a link MOVES everything below it -- correctly. The
    record is stamped with the link lengths it was measured at and expires when they change, which is why an
    ordinary amend cannot be turned into an import rejection."""
    from virturoid.services.edit_operators import apply_op
    from virturoid.services.robot_import import _simulability_probe

    out = _import("unitree_go2/go2.xml", "amend")
    assert out["gene"] is not None
    assert (_simulability_probe(out["gene"])["placement_check"] or {}).get("ok") is True

    longer, _diff = apply_op(out["gene"], "scale_group", {"group": "legs", "dims": "length", "factor": 1.4})
    place = _simulability_probe(longer).get("placement_check") or {}
    assert place.get("checked") is False, (
        "a 1.4x leg-scale amend was still ruled on against the ORIGINAL placement, so a legitimate edit reads "
        f"as an import defect: {place}")
    assert place.get("n_edited_links"), place
