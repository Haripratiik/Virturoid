"""EVERY site that decides "what kind of body is this?" must agree with the body's actual structure.

This is the guard for a bug family that has now been closed FIVE times and kept coming back at a new site
each time -- an imported body judged by the WRONG RUBRIC:

  #214  a fixed-base URDF quadruped was called a manipulator, so verify ran the ARM rubric on it
  #218  legged/arm bodies got the wrong verify rubric ("TIPPED while driving" on a Go2)
  #244  ``_infer_class`` counted branches, so no humanoid could ever classify as one
  80ec693 a SECOND, coarser classifier downstream hard-coded "quadruped" for any class outside
          ("quadruped","legged","hexapod"), so every humanoid became a quadruped and got a quad crawl gait
  this  ``robot_kind`` called Talos a manipulator (hands checked before legs) and a drone a manipulator;
        ``build_appendage_map`` found ZERO legs on a Booster T1, TWO legs on a 2-finger gripper, and a
        SPINE on a UR5e

The reason it kept recurring is that several places each RE-DERIVED body kind with their own heuristic and
nothing ever tested them together. So the test is deliberately cross-site: one real robot, every decision
site, one table. A new site that re-decides on its own will disagree here on the first body that exercises
it, instead of shipping a quadruped crawl gait to somebody's humanoid.

Menagerie models are fetched on demand by ``robot_descriptions``; every body skips cleanly when its model is
not cached, so CI stays hermetic.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None
_MEN = Path(os.path.expanduser("~/.cache/robot_descriptions/mujoco_menagerie"))
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="body-kind derivation needs MuJoCo")

# body -> (menagerie path, expected coarse kind, expected legged family, expected n_legs, expected n_wheels).
# "family" is "" for a body that is not legged at all. These are the STRUCTURAL facts about real robots, not
# our labels for them: a Go2 stands on four legs, a Talos on two, a Panda on none.
CORPUS = [
    ("go2",         "unitree_go2/go2.xml",              "legged",      "quadruped", 4, 0),
    ("anymal_c",    "anybotics_anymal_c/anymal_c.xml",  "legged",      "quadruped", 4, 0),
    ("spot",        "boston_dynamics_spot/spot.xml",    "legged",      "quadruped", 4, 0),
    ("booster_t1",  "booster_t1/t1.xml",                "legged",      "humanoid",  2, 0),
    ("g1",          "unitree_g1/g1.xml",                "legged",      "humanoid",  2, 0),
    ("talos",       "pal_talos/talos.xml",              "legged",      "humanoid",  2, 0),
    ("cassie",      "agility_cassie/cassie.xml",        "legged",      "humanoid",  2, 0),
    ("op3",         "robotis_op3/op3.xml",              "legged",      "humanoid",  2, 0),
    ("panda",       "franka_emika_panda/panda.xml",     "manipulator", "",          0, 0),
    ("ur5e",        "universal_robots_ur5e/ur5e.xml",   "manipulator", "",          0, 0),
    ("shadow_hand", "shadow_hand/right_hand.xml",       "manipulator", "",          0, 0),
    ("leap_hand",   "leap_hand/right_hand.xml",         "manipulator", "",          0, 0),
    ("robotiq",     "robotiq_2f85/2f85.xml",            "manipulator", "",          0, 0),
    ("tiago",       "pal_tiago/tiago.xml",              "mobile",      "",          0, 2),
    ("skydio_x2",   "skydio_x2/x2.xml",                 "mobile",      "",          0, 0),
    ("crazyflie",   "bitcraze_crazyflie_2/cf2.xml",     "mobile",      "",          0, 0),
]
_IDS = [c[0] for c in CORPUS]


def _gene(rel: str):
    src = _MEN / rel
    if not src.is_file():
        pytest.skip(f"{rel} is not cached locally (robot_descriptions fetches on demand)")
    from virturoid.services.robot_import import import_robot
    res = import_robot(str(src), robot_id="agree")
    if res.get("gene") is None:
        pytest.skip(f"{rel} did not import: {res.get('warnings')}")
    return res["gene"], res.get("robot_class")


@pytest.fixture(scope="module")
def _cache():
    return {}


def _row(rel, cache):
    """Every decision site's answer for one body, in one dict -- the table this test is really about."""
    if rel in cache:
        return cache[rel]
    import mujoco

    from virturoid.services.appendage_map import build_appendage_map
    from virturoid.services.body_kind import body_kind, measured_legs
    from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
    from virturoid.services.input_training_tools import _legged_family, _needs_legged_reconciliation
    from virturoid.services.isaac_lab_exporter import _kind_is_legged
    from virturoid.services.task_matched_eval import robot_capabilities, robot_kind

    gene, imported_class = _gene(rel)
    model = mujoco.MjModel.from_xml_string(
        compile_gene_to_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene)))
    amap = build_appendage_map(model)
    bk = body_kind(gene)
    row = {
        "gene": gene, "model": model,
        "infer_class": imported_class,                 # robot_import._infer_class -> gene.robot_class
        "robot_kind": robot_kind(gene),                # the coarse dispatch kind (~40 call sites)
        "capabilities": robot_capabilities(gene),      # task_verifier feasibility
        "body_kind": bk.kind, "family": bk.family,     # the single derivation
        "amap_legs": amap.n_legs, "amap_wheels": amap.n_wheels, "amap_kind": amap.kind(),
        "legged_family": _legged_family(gene),         # the family written back onto an ingest
        "needs_reconcile": _needs_legged_reconciliation(gene),
        "isaac_legged": _kind_is_legged(gene),
        "biped_gate": measured_legs(gene),             # what _honest_biped gates on (== 2 -> biped rubric)
    }
    cache[rel] = row
    return row


@pytest.mark.parametrize("name,rel,kind,family,n_legs,n_wheels", CORPUS, ids=_IDS)
def test_every_site_agrees_with_the_bodys_actual_structure(name, rel, kind, family, n_legs, n_wheels, _cache):
    """One real robot, every rubric-deciding site, one verdict.

    A site that re-derives body kind with its own heuristic disagrees here as soon as a body exercises the
    difference -- which is exactly what #214/#218/#244/80ec693 each shipped."""
    r = _row(rel, _cache)
    table = {k: v for k, v in r.items() if k not in ("gene", "model")}

    # 1. the STRUCTURE the whole system has to agree on
    assert r["amap_legs"] == n_legs, f"{name}: appendage map counts {r['amap_legs']} legs, not {n_legs}\n{table}"
    if n_wheels:
        assert r["amap_wheels"] >= n_wheels, f"{name}: {r['amap_wheels']} wheels, expected >={n_wheels}\n{table}"

    # 2. the coarse dispatch kind -- this picks the VERIFY RUBRIC, the eval task and the training recipe
    assert r["robot_kind"] == kind, f"{name}: robot_kind says {r['robot_kind']!r}, it is {kind!r}\n{table}"
    assert r["body_kind"] == kind, f"{name}: body_kind says {r['body_kind']!r}\n{table}"
    assert kind in r["capabilities"], f"{name}: capabilities {r['capabilities']} omit {kind!r}\n{table}"

    # 3. the legged FAMILY -- this picks the gait, the BOM sensor suite, the spec sheet and the template offer
    assert r["family"] == family, f"{name}: family {r['family']!r} != {family!r}\n{table}"
    if family:
        assert r["infer_class"] in (family, "legged"), (
            f"{name}: the import wrote robot_class={r['infer_class']!r} but it is a {family}\n{table}")
        assert r["legged_family"] == family, f"{name}: _legged_family {r['legged_family']!r}\n{table}"
    else:
        assert r["infer_class"] not in ("quadruped", "hexapod", "humanoid", "biped"), (
            f"{name}: the import called a non-legged body {r['infer_class']!r}\n{table}")

    # 4. the derived gates: legged export env, and the biped honesty block
    assert r["isaac_legged"] is (kind == "legged"), f"{name}: isaac legged gate {r['isaac_legged']}\n{table}"
    assert r["biped_gate"] == r["amap_legs"], (
        f"{name}: _honest_biped gates on {r['biped_gate']} legs while the appendage map sees "
        f"{r['amap_legs']} -- two counters again\n{table}")
    if n_legs == 2:
        assert r["biped_gate"] == 2, (
            f"{name}: the biped honesty block declines a 2-legged body, so it is scored by the multi-leg "
            f"crawl gait that just fells it\n{table}")

    # 5. nothing that already reads as legged gets "reconciled" into another family (80ec693)
    if family:
        assert not r["needs_reconcile"], f"{name}: a {family} was queued for a #214 rename\n{table}"


@pytest.mark.parametrize("name,rel,kind,family,n_legs,n_wheels", CORPUS, ids=_IDS)
def test_no_site_claims_a_capability_the_structure_denies(name, rel, kind, family, n_legs, n_wheels, _cache):
    """The mirror image: a wrong POSITIVE is what routes a body into a rubric it cannot pass.

    A wheeled base offered the ``legged`` capability gets handed a walking task it will fail at 0.0 m; a
    hand offered it gets a crawl gait."""
    r = _row(rel, _cache)
    table = {k: v for k, v in r.items() if k not in ("gene", "model")}
    if kind != "legged":
        assert "legged" not in r["capabilities"], (
            f"{name}: a {kind} body claims the legged capability, so a walk task reads as feasible\n{table}")
        assert r["amap_kind"] != "legged", f"{name}: appendage map calls it legged\n{table}"
    if kind != "mobile":
        assert r["amap_wheels"] == 0, f"{name}: {r['amap_wheels']} wheels on a {kind}\n{table}"


def test_the_one_family_ladder_is_shared_not_re_derived():
    """``_infer_class`` and ``_legged_family`` each used to carry their OWN leg-count ladder, and they
    disagreed at 3 legs. One ladder, imported by both."""
    from virturoid.services import input_training_tools as IT
    from virturoid.services import robot_import as RI
    from virturoid.services.body_kind import family_from_legs

    assert RI.family_from_legs is family_from_legs
    assert IT.family_from_legs is family_from_legs
    assert [family_from_legs(n) for n in (0, 1, 2, 3, 4, 6, 8)] == [
        "", "", "humanoid", "quadruped", "quadruped", "hexapod", "hexapod"]


def test_the_floating_base_class_set_is_shared_not_re_listed():
    """Two lists of "which classes get a free base" had drifted apart: robot_import's was missing
    ``legged``/``biped``/``aerial``/``aquatic``, so an import in one of those families was WELDED TO A TABLE
    and could not translate -- guaranteeing a 0.0 m walk verdict on a body that walks."""
    from virturoid.services import anatomy_compiler as AC
    from virturoid.services import robot_import as RI
    from virturoid.services.body_kind import FLOATING_BASE_CLASSES

    assert RI.FLOATING_BASE_CLASSES is FLOATING_BASE_CLASSES
    assert AC.FLOATING_BASE_CLASSES is FLOATING_BASE_CLASSES
    for cls in ("legged", "biped", "humanoid", "quadruped", "hexapod", "aerial", "aquatic", "mobile_base"):
        assert cls in FLOATING_BASE_CLASSES, f"a {cls} welded to a table cannot move at all"
    for cls in ("manipulator", "arm"):
        assert cls not in FLOATING_BASE_CLASSES, f"a {cls} must stay bolted down or it falls over"


def test_a_fixed_base_body_is_not_read_as_standing_on_what_it_is_bolted_to():
    """The measured cause of three wrong answers at once: on a bench-mounted body every chain hangs near the
    bench, so a proximity-to-lowest-geom test called a 2-finger gripper's fingers LEGS (-> "humanoid") and a
    6-DOF arm's single chain a SNAKE'S SPINE. Ground contact is only evidence of legs when enough chains share
    it to make a support polygon -- the same >=3 threshold ``_infer_class`` measured across the corpus."""
    import mujoco

    from virturoid.services.appendage_map import build_appendage_map
    from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
    for rel, why in (("robotiq_2f85/2f85.xml", "a 2-finger gripper is not a biped"),
                     ("universal_robots_ur5e/ur5e.xml", "a 6-DOF arm is not a snake")):
        gene, _ = _gene(rel)
        amap = build_appendage_map(mujoco.MjModel.from_xml_string(
            compile_gene_to_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene))))
        assert amap.n_legs == 0, f"{why}: got {amap.n_legs} legs"
        assert amap.spine is None, f"{why}: got a spine"
        assert amap.kind() == "manipulator", f"{why}: kind={amap.kind()}"


def test_legs_below_a_shared_waist_joint_are_still_legs():
    """Booster T1 measured ZERO legs. Its two hips hang off a shared ``Waist`` yaw token, and the chain walk
    stopped dead at that branch point, collapsing both legs into one 1-token stub classified "other". Every
    imported robot with a waist/pelvis actuator above the hips lost every leg the same way -- and with
    ``n_legs != 2`` the biped honesty block declined, so a humanoid was scored by the multi-leg crawl gait."""
    import mujoco

    from virturoid.services.appendage_map import build_appendage_map
    from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
    from virturoid.services.ai_native_tools import _honest_biped
    gene, cls = _gene("booster_t1/t1.xml")
    amap = build_appendage_map(mujoco.MjModel.from_xml_string(
        compile_gene_to_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene))))
    assert cls == "humanoid"
    assert amap.n_legs == 2, f"a T1 stands on two legs, appendage map says {amap.n_legs}"
    # ...and the product consequence: the biped honesty block engages instead of handing the body to the crawl
    v = _honest_biped(gene, steps=400)
    assert v is not None and "crawl wave gait" in (v.get("note") or ""), f"biped verdict declined: {v}"


def test_a_biped_with_hands_is_still_judged_a_walker():
    """Talos: 45 segments, floating base, two legs -- and two HANDS, which ``robot_kind`` checked FIRST, so it
    returned "manipulator". An imported Talos skipped the entire legged honesty block and was scored on
    PICK-PLACE, the #218 defect at a new site."""
    from virturoid.services.task_matched_eval import robot_capabilities, robot_kind
    gene, _ = _gene("pal_talos/talos.xml")
    assert gene.end_effector_type in ("gripper", "hand"), "precondition: Talos imports with hands"
    assert robot_kind(gene) == "legged"
    assert {"legged", "manipulator"} <= robot_capabilities(gene), "it walks AND manipulates"


def test_a_body_with_no_joints_at_all_is_not_a_manipulator():
    """A Skydio X2 imports as one rigid free body with zero actuated joints, and fell through every branch to
    the ``return "manipulator"`` at the bottom -- so a drone was scored on REACH and pick-place. A body with
    nothing to articulate cannot manipulate anything; the honest answer is the mobile/free-body rubric."""
    from virturoid.services.task_matched_eval import robot_kind
    gene, cls = _gene("skydio_x2/x2.xml")
    assert not [s for s in gene.segments if s.joint_type == "revolute"], "precondition: no revolute joints"
    assert cls == "mobile_base"
    assert robot_kind(gene) == "mobile"


def test_the_214_fixed_base_quadruped_still_reconciles_to_quadruped():
    """Guard against over-correcting. #214's body really is a fixed base with four one-joint limbs -- the
    exact shape of a bench-mounted 4-fingered hand -- and it must STILL come out legged, by its measured leg
    count of 4, or verify goes back to running the arm rubric on somebody's quadruped."""
    import sys
    import tempfile

    sys.path.insert(0, str(Path(__file__).parent))
    from test_import_verify_honesty import _fixed_base_quad_urdf

    from virturoid.services.body_kind import body_kind
    from virturoid.services.input_training_tools import _legged_family
    from virturoid.services.robot_import import import_robot
    from virturoid.services.task_matched_eval import robot_kind

    tmp = _fixed_base_quad_urdf(tempfile.mkdtemp())
    gene = import_robot(os.path.join(tmp, "q.urdf"))["gene"]
    assert gene.base_mount != "free", "precondition for #214: MuJoCo fused the torso, so the base is fixed"
    assert robot_kind(gene) == "legged"
    assert body_kind(gene).n_legs == 4
    assert _legged_family(gene) == "quadruped"


def test_weak_evidence_returns_the_honest_generic_not_an_invented_family():
    """The other half of not over-correcting (80ec693): when the leg count is inconclusive the derivation says
    the generic ``legged`` rather than inventing "quadruped", because the family drives the BOM, the spec
    sheet, the verify rubric and the walkable-template offer."""
    from virturoid.services.body_kind import family_from_legs
    assert family_from_legs(0) == ""
    assert family_from_legs(1) == ""
    from virturoid.services.input_training_tools import _legged_family
    gene, _ = _gene("franka_emika_panda/panda.xml")
    assert _legged_family(gene) == "legged", "no measured legs -> the honest generic, never a guessed family"
