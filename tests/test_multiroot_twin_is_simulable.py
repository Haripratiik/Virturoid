"""A multi-root robot's twin must be the customer's robot, and it must be STEPPABLE — or the import must say so.

Two defects, one family. Measured on the real MuJoCo Menagerie corpus (2026-08-06), 4 of 63 packages declare more
than one body under ``<worldbody>``: aloha, trossen_wxai (both bi-manual), shadow_dexee (a palm plus three
fingers) and apptronik_apollo. All four produced a broken twin, and none of them said so.

  * ``robot_import`` synthesizes a welded base and reparents the roots onto it, and it DISCARDED their world
    pose while doing so -- every root got ``mount_offset (0,0,0)``. ALOHA's two arms sit at x = -0.469 and
    +0.469 with the right one yawed 180 degrees; both landed on (0, 0, 0.075). They interpenetrated by
    0.14172 m (``left/base_link`` vs ``right/upper_arm_link``) and the model diverged to a bad-QACC NaN.
  * Nothing noticed. ``import_robot`` reported ``valid=True``, because ``RobotGene.validate()`` is a SCHEMA
    check -- one root, parents present, acyclic -- and says nothing about whether the body survives contact.
    Every downstream number (verdict, certificate, BOM, spec sheet, calibration gap) is computed by STEPPING
    this twin, so all of them were being computed on a model that cannot be simulated.

The second test class is the load-bearing one, and it is subtle: **MuJoCo hides the divergence**. ``mj_checkAcc``
reacts to a non-finite ``qacc`` by raising ``mjWARN_BADQACC`` *and calling* ``mj_resetData``, so a post-hoc
``np.isfinite(d.qpos)`` reads True on a model that exploded a millisecond earlier. Any gate written the obvious
way passes the exact case it was written to catch.
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

# (package/model, the roots the SOURCE declares under <worldbody>). The full 63-package sweep found exactly
# these; a fifth package taking the same reparenting branch with ONE root (umi_gripper) is covered too, since
# its root also carries a non-identity source pose.
MULTI_ROOT = [
    ("aloha/aloha.xml", ["left/base_link", "right/base_link"]),
    ("trossen_wxai/trossen_ai_bimanual.xml", ["left/root", "right/root"]),
    ("shadow_dexee/shadow_dexee.xml", ["hand_base", "F0/", "F1/", "F2/"]),
    ("apptronik_apollo/apptronik_apollo.xml", ["base_link", "world_link"]),
]


def _import(rel: str, prefix: str):
    from virturoid.services.robot_import import import_robot

    src = _MEN / rel
    if not src.exists():
        pytest.skip(f"{rel} is not in the local Menagerie cache")
    return import_robot(str(src), robot_id=f"{prefix}_{Path(rel).stem}")


def _twin_xpos(gene):
    """World position of every body in the compiled twin at its REST POSE, keyed by name.

    The keyframe matters and is easy to get wrong: ALOHA at qpos 0 folds both arms into the middle and its two
    grippers overlap by 0.035 m *in the customer's own model*, while at its ``neutral_pose`` keyframe -- which
    the import carries onto the gene -- they are 0.65 m apart and clear. A check run at zero measures the
    source's own pose and blames us for it.
    """
    import mujoco
    from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z

    m = mujoco.MjModel.from_xml_string(
        compile_gene_to_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene)))
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return m, d, {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b): d.xpos[b].copy() for b in range(1, m.nbody)}


def _collapse_roots(gene):
    """The PRE-FIX twin, rebuilt: every reparented root loses its source world pose and stacks on the base."""
    g = copy.deepcopy(gene)
    root = g.root()
    for s in g.segments:
        if s.parent == root.name:
            s.mount_offset = (0.0, 0.0, 0.0)
    return g


# --------------------------------------------------------------------------- the roots keep their world pose
@pytest.mark.parametrize("rel,root_names", MULTI_ROOT, ids=[r.split("/")[0] for r, _ in MULTI_ROOT])
def test_a_multi_root_models_roots_keep_their_source_world_pose(rel, root_names):
    """Distances BETWEEN the roots survive the reparenting.

    Pairwise distance, not position: an import may legitimately re-datum the whole body (and does -- the twin
    spawns at ``standing_spawn_z``). It may not move the customer's two arms onto the same point.

    The source is read at its DEFAULT configuration, which is what ``body_pos``/``body_quat`` encode and what the
    contract preserves. Deliberately not at a keyframe: a root carrying a freejoint (Apollo's ``base_link``) is
    moved by the keyframe's free-base qpos, and a RobotGene has no freejoint to carry that -- the twin welds the
    root at its static pose, and 0.065 m of Apollo's free base is not a reparenting error.
    """
    import mujoco
    import numpy as np

    src = _MEN / rel
    if not src.exists():
        pytest.skip(f"{rel} is not in the local Menagerie cache")
    sm = mujoco.MjModel.from_xml_path(str(src))
    sd = mujoco.MjData(sm)
    mujoco.mj_forward(sm, sd)
    src_pos = {mujoco.mj_id2name(sm, mujoco.mjtObj.mjOBJ_BODY, b): sd.xpos[b].copy() for b in range(1, sm.nbody)}

    out = _import(rel, "mr")
    assert out["gene"] is not None, f"{rel} produced no gene: {out.get('warnings')}"
    _, _, twin_pos = _twin_xpos(out["gene"])

    present = [n for n in root_names if n in src_pos and n in twin_pos]
    assert len(present) >= 2, f"{rel}: only {present} of {root_names} survived the import"
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            a, b = present[i], present[j]
            want = float(np.linalg.norm(src_pos[a] - src_pos[b]))
            got = float(np.linalg.norm(twin_pos[a] - twin_pos[b]))
            assert abs(want - got) < 0.01, (
                f"{rel}: {a} and {b} are {want:.4f} m apart on the customer's robot and {got:.4f} m apart on "
                f"our twin. The synthesized base reparents the roots; it may not throw their world pose away.")


@pytest.mark.parametrize("rel", ["aloha/aloha.xml", "trossen_wxai/trossen_ai_bimanual.xml"],
                         ids=["aloha", "trossen_wxai"])
def test_the_two_arms_of_a_bimanual_robot_do_not_occupy_the_same_space(rel):
    """The reported defect, as its own named gate: -0.14172 m of left-arm-inside-right-arm at rest (aloha),
    -0.100 m (trossen_wxai). Measured post-fix, both are ZERO — not merely 'small'. The arms are ~0.94 m apart
    on both machines, so any left/* against right/* contact at all means they have been stacked again."""
    import mujoco

    out = _import(rel, "pen")
    assert out["gene"] is not None
    m, d, _ = _twin_xpos(out["gene"])
    floor = {g for g in range(m.ngeom) if int(m.geom_type[g]) == mujoco.mjtGeom.mjGEOM_PLANE}

    def name_of(g):
        return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""

    worst, pair = 0.0, None
    for c in d.contact[:d.ncon]:
        if int(c.geom1) in floor or int(c.geom2) in floor or c.dist >= 0:
            continue
        n1, n2 = name_of(int(c.geom1)), name_of(int(c.geom2))
        s1 = n1.split("/")[0] if "/" in n1 else None
        s2 = n2.split("/")[0] if "/" in n2 else None
        if s1 and s2 and s1 != s2 and float(-c.dist) > worst:      # left/* against right/*
            worst, pair = float(-c.dist), (n1, n2)
    assert worst < 0.005, (
        f"{rel}: the two arms interpenetrate by {worst:.5f} m ({pair}) — they have been stacked on one point "
        f"again. The source places their bases roughly 0.94 m apart.")


def test_the_right_arm_keeps_its_180_degree_yaw():
    """A reparented root's ORIENTATION is part of its pose. ALOHA's right arm is quat (0,0,0,1) -- yawed 180
    degrees so the two arms face each other; dropping the quat leaves two identical left arms."""
    import numpy as np

    out = _import("aloha/aloha.xml", "yaw")
    gene = out["gene"]
    root = gene.root()
    kids = {s.name: s for s in gene.segments if s.parent == root.name}
    assert set(kids) == {"left/base_link", "right/base_link"}, sorted(kids)
    yaw_l = float(kids["left/base_link"].mount_euler[2])
    yaw_r = float(kids["right/base_link"].mount_euler[2])
    assert abs(abs(yaw_r - yaw_l) - np.pi) < 0.05, (
        f"the two arms are yawed {yaw_l:.3f} and {yaw_r:.3f} rad — the source yaws the right arm 180 degrees "
        f"relative to the left, and that rotation lives in body_quat, which the reparenting dropped.")


# --------------------------------------------------------------------------- the gate
def test_mujoco_hides_the_divergence_so_a_finiteness_check_on_state_is_vacuous():
    """The reason this bug survived: ``mj_checkAcc`` calls ``mj_resetData`` when it trips.

    A stepped-to-death model therefore reports perfectly finite qpos/qvel afterwards. This test asserts the trap
    itself, so that anybody who "simplifies" the probe back to ``np.isfinite(d.qpos)`` gets a red test explaining
    why that does not work.
    """
    import mujoco
    import numpy as np
    from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z

    out = _import("aloha/aloha.xml", "hide")
    bad = _collapse_roots(out["gene"])
    m = mujoco.MjModel.from_xml_string(
        compile_gene_to_mjcf(bad, include_floor=True, spawn_z=standing_spawn_z(bad)))
    d = mujoco.MjData(m)
    for _ in range(600):
        mujoco.mj_step(m, d)
    diverged = int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
    assert diverged > 0, "the collapsed twin was expected to diverge; it did not"
    assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all(), (
        "MuJoCo used to reset the data on a bad qacc, which is what made a state finiteness check useless. If "
        "this now fails, MuJoCo changed behaviour and the probe can be simplified — check mj_checkAcc first.")


def test_a_twin_that_cannot_be_stepped_fails_the_import_loudly():
    """The gate. A NaN twin must be distinguishable from a working one, to callers and to the customer."""
    from virturoid.services.robot_import import _simulability_probe

    out = _import("aloha/aloha.xml", "gate")
    good = _simulability_probe(out["gene"])
    assert good["ok"] and good["checked"], f"the fixed ALOHA twin should pass the gate: {good}"

    bad = _simulability_probe(_collapse_roots(out["gene"]))
    assert bad["checked"] and not bad["ok"], f"the collapsed twin should FAIL the gate: {bad}"
    assert bad["first_bad_time_s"] > 0, bad
    assert bad["mujoco_warnings"], "the failure must be attributed to a named MuJoCo warning, not a bare bool"
    assert bad["max_self_penetration_m"] > 0.1, (
        f"the probe must also report WHY: {bad['max_self_penetration_m']} m of overlap")
    assert "non-finite" in bad["reason"], bad["reason"]
    # The collapsed twin survives the whole probe from `neutral_pose` and dies at t=0.106 s from zero. That is
    # why both start poses are run; a probe that picked one would have missed this half the time.
    assert bad["stage"] and "zero_pose" in bad["reason"], bad["reason"]
    assert [r["ok"] for r in bad["runs"]] == [True, False], (
        f"expected the rest-keyframe run to survive and the zero-pose run to diverge: "
        f"{[(r['from'], r['ok']) for r in bad['runs']]}")


def test_the_gate_reports_a_named_reason_it_could_not_compile():
    """A twin that does not even compile is the same failure class and must read the same way, naming the link."""
    from virturoid.services.robot_import import _simulability_probe

    out = _import("aloha/aloha.xml", "cmp")
    broken = copy.deepcopy(out["gene"])
    for s in broken.segments:                       # a massless, sizeless link: MuJoCo refuses the whole model
        if s.parent is not None and s.joint_type in ("revolute", "prismatic"):
            s.mass_kg, s.length_m, s.radius_m = 0.0, 0.0, 0.0
            break
    res = _simulability_probe(broken)
    assert not res["ok"] and res["stage"] == "compile", res
    assert "does not compile" in res["reason"], res["reason"]


def test_an_unsimulable_import_is_not_reported_as_valid():
    """``valid`` may not be true for a body nobody can step. flybody is the corpus case: at HEAD it imported
    ``valid=True`` and its gene could not be compiled at all (a zero-inertia link, 'haustellum')."""
    rel = "flybody/fruitfly.xml"
    if not (_MEN / rel).exists():
        pytest.skip("flybody is not in the local Menagerie cache")
    out = _import(rel, "unsim")
    assert out["simulable"] is False, "flybody's twin does not compile; the import must say so"
    assert out["valid"] is False, "an unsimulable twin may not be reported valid"
    assert any("NOT SIMULABLE" in w for w in out["warnings"]), out["warnings"][:3]
    assert out["simulation_check"]["stage"] == "compile"
    assert "haustellum" in out["simulation_check"]["reason"], (
        f"the reason must name the offending link: {out['simulation_check']['reason']}")


def test_every_healthy_menagerie_import_still_passes_the_gate():
    """The gate must not be a new way to fail. A representative slice across every family."""
    for rel in ("unitree_go2/go2.xml", "franka_emika_panda/panda.xml", "unitree_g1/g1.xml",
                "boston_dynamics_spot/spot.xml", "shadow_hand/right_hand.xml",
                "aloha/aloha.xml", "trossen_wxai/trossen_ai_bimanual.xml"):
        if not (_MEN / rel).exists():
            continue
        out = _import(rel, "healthy")
        assert out["simulable"] is True, f"{rel} regressed to unsimulable: {out.get('simulation_check')}"
        assert out["valid"] is True, f"{rel}: {[w for w in out['warnings'] if 'validation' in w]}"


def test_the_verdict_survives_the_import_memo():
    """``VIRTUROID_IMPORT_CACHE`` memoizes imports; a cache may make an import faster, never make it lie. A
    replayed result that dropped ``simulable`` would hand the second caller a twin marked healthy that was
    never simulated."""
    from virturoid.services.robot_import import _restore_import, _snapshot_import

    out = _import("aloha/aloha.xml", "memo")
    again = _restore_import(_snapshot_import(out))
    assert again["simulable"] == out["simulable"]
    assert again["valid"] == out["valid"]
    assert again["simulation_check"]["ok"] == out["simulation_check"]["ok"]


# --------------------------------------------------------------------------- what the sweep turned up alongside
def test_an_end_effector_site_on_the_worldbody_is_not_a_segment():
    """shadow_dexee parks ``tcp`` and ``attachment_site`` on body 0 so a parent scene can mount it. We resolved
    the site to ``body_name[0] == 'world'``, matched no segment, and marked NOTHING as the end effector -- the
    gene then failed validation and the twin could not be compiled at all."""
    out = _import("shadow_dexee/shadow_dexee.xml", "eesite")
    gene = out["gene"]
    assert gene is not None
    ees = [s.name for s in gene.segments if s.is_end_effector]
    assert len(ees) == 1, f"expected exactly one end effector, got {ees}"
    assert ees[0] != "world"
    assert not gene.validate(), gene.validate()
    assert out["simulable"] is True, out.get("simulation_check")


def test_a_link_whose_welded_mesh_exceeds_mujocos_stl_ceiling_still_compiles():
    """A link's baked STL is the WELD of every mesh geom on that body, so it can clear MuJoCo's 200 000-face
    decoder ceiling even when no single source mesh does. Apollo's ``torso_link`` welds to 201 358 triangles,
    and MuJoCo rejects the WHOLE MODEL for it -- a 37-segment humanoid lost to one link."""
    out = _import("apptronik_apollo/apptronik_apollo.xml", "stl")
    gene = out["gene"]
    assert gene is not None
    torso = next((s for s in gene.segments if s.name == "torso_link"), None)
    if torso is None or not (torso.geometry or {}).get("path"):
        pytest.skip("this build did not bake a source mesh for torso_link")
    import struct
    with open(torso.geometry["path"], "rb") as fh:
        fh.read(80)
        n_faces = struct.unpack("<I", fh.read(4))[0]
    assert 0 < n_faces <= 200000, f"torso_link baked {n_faces} faces; MuJoCo's stl_decoder refuses over 200000"
    assert out["simulable"] is True, out.get("simulation_check")
