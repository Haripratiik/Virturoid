"""THE CRAWL GAIT DROVE 4 OF THE UNITREE G1'S 29 JOINTS AND HELD 25 OF THEM RIGID.

Two independent defects in ``morph_policy.crawl_gait_rollout``, both found while diagnosing the biped, both
CPU-fixable, and neither of them a claim that a biped can now balance:

1. **The lift joint was chosen by DEPTH INDEX.** ``knee_k = stride[-1]`` takes the deepest fore-aft joint. On a
   two-link animal leg that is the knee and it is right -- but only because, once the hip is spoken for, there
   is exactly ONE fore-aft joint left and no choice is being made. On a three-joint humanoid leg it is the
   ANKLE PITCH, so the gait drove the ankle and left the knee frozen at its PD default.

2. **Half the probe budget ran byte-identical duplicates.** The direction arm reverses the metachronal wave --
   which does nothing when there is no wave to reverse. Two of every four probe rollouts were the same
   rollout, silently, on every evaluation of such a body.

These tests pin the fix AND its blast radius, because the blast radius is the point: this function produces
every walk verdict the product ships, so the contract is "correct where it was wrong, byte-identical
everywhere else". The identity half is not an assumption -- it is structural, and asserted below on real
robots: a quadruped leg offers ONE candidate, so the choice cannot change.

Menagerie-backed where a real robot is the right witness (fixtures have lied in this repo); the
composed/authored bodies need nothing but MuJoCo and always run.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="the crawl gait needs MuJoCo")

_MENAGERIE = os.path.join(os.path.expanduser("~"), ".cache", "robot_descriptions", "mujoco_menagerie")


# ------------------------------------------------------------------------------------------------ fixtures
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


def _template():
    """The canonical taught template -- the one body the whole shipped operating point is co-tuned with."""
    from virturoid.services.anatomy_compiler import _generic_legged_graph, build_from_anatomy
    return build_from_anatomy(_generic_legged_graph(n_pairs=2, girth=0.22, fan=True))


def _hexapod():
    from virturoid.services.anatomy_compiler import _generic_legged_graph, build_from_anatomy
    return build_from_anatomy(_generic_legged_graph(n_pairs=3))


def _legs(gene):
    """(model, data-at-rest, graph, appendage map) -- the exact objects the rollout builds its leg map from."""
    import mujoco
    from virturoid.services.appendage_map import build_appendage_map
    from virturoid.services.morph_graph import encode_robot
    from virturoid.services.morph_policy import _reset_to_rest, compiled_model, robot_mjcf
    model = compiled_model(robot_mjcf(gene), solver_iterations=20)
    data = mujoco.MjData(model)
    _reset_to_rest(model, data)
    mujoco.mj_forward(model, data)
    return model, data, encode_robot(model), build_appendage_map(model)


def _jname(model, graph, tok):
    import mujoco
    jid = int(model.actuator_trnid[int(graph.act_u[tok]), 0])
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"joint_{jid}"


def _picks(gene):
    """Every leg's (hip, lift) joint NAMES, as the rollout would pick them."""
    from virturoid.services.morph_policy import lift_joint_for_leg
    model, data, graph, amap = _legs(gene)
    qadr = np.asarray(graph.qadr, int)
    act_u = np.asarray(graph.act_u, int)
    out = []
    for lg in amap.legs:
        if not lg.stride_tokens:
            continue
        hip = lg.stride_tokens[0]
        pick, disc = lift_joint_for_leg(model, data, qadr, act_u, lg, hip)
        out.append({"hip": _jname(model, graph, hip), "lift": _jname(model, graph, pick),
                    "n_candidates": len([t for t in lg.stride_tokens if t != hip]),
                    "depth_rule": _jname(model, graph, lg.stride_tokens[-1]) if len(lg.stride_tokens) >= 2
                    else _jname(model, graph, lg.tokens[-1]),
                    "disclosure": disc})
    return out


# -------------------------------------------------------------------- (1) the lift joint, by kinematic role
def test_the_g1_crawl_drives_the_knee_not_the_ankle():
    """THE DEFECT ITSELF, on the customer's actual robot. The G1's leg is hip-pitch / hip-roll / hip-yaw /
    knee / ankle-pitch / ankle-roll; its fore-aft joints are hip-pitch, knee and ankle-pitch, so the DEEPEST
    one -- the shipped pick -- is the ankle, and the knee never moved."""
    picks = _picks(_imported("unitree_g1", "g1.xml"))
    assert len(picks) == 2, f"the G1 should present two legs, got {picks}"
    for leg in picks:
        assert "knee" in leg["lift"], f"the lift joint should be the knee, got {leg['lift']}"
        assert "ankle" in leg["depth_rule"], f"the depth rule should have picked the ankle, got {leg}"
        assert leg["disclosure"]["changed"] is True


def test_every_humanoid_in_the_menagerie_was_driving_its_ankle():
    """Not one robot -- the whole class. Each of these has >=2 fore-aft joints below the hip, so each of them
    was making a choice by depth index, and each of them chose the distal one."""
    for pkg, model, knee in (("unitree_g1", "g1.xml", "knee"), ("unitree_h1", "h1.xml", "knee"),
                             ("robotis_op3", "op3.xml", "knee")):
        picks = _picks(_imported(pkg, model))
        assert picks, f"{pkg} presented no legs"
        for leg in picks:
            assert leg["n_candidates"] >= 2, f"{pkg} leg should offer a choice: {leg}"
            assert knee in leg["lift"], f"{pkg} should lift with the {knee}, got {leg['lift']}"
            assert leg["lift"] != leg["depth_rule"], f"{pkg} should have MOVED off the depth pick: {leg}"


def test_cassie_keeps_its_distal_joint_because_the_measurement_says_so():
    """THE RULE IS A MEASUREMENT, NOT A RENAME, and Cassie is the body that proves it. Cassie offers THREE
    candidates (knee, shin, tarsus) and the new rule still picks the distal ``tarsus`` -- correctly: Cassie's
    foot hangs off a four-bar and the tarsus is the segment that carries it, so the tarsus has nearly twice
    the knee's foot-raise authority. A rule that just moved every biped onto the joint named "knee" would get
    this body wrong."""
    picks = _picks(_imported("agility_cassie", "cassie.xml"))
    assert picks, "Cassie presented no legs"
    for leg in picks:
        assert leg["n_candidates"] == 3, f"Cassie should offer three candidates: {leg}"
        assert "tarsus" in leg["lift"], f"Cassie should still lift with the tarsus, got {leg['lift']}"
        scores = leg["disclosure"]["raise_per_rad"]
        best = max(scores.values())
        assert best == scores[leg["disclosure"]["picked"]]
        assert best > 1.5 * sorted(scores.values())[-2], "the tarsus should win clearly, not by a hair"


def test_a_quadruped_leg_offers_no_choice_so_the_pick_cannot_move():
    """WHY THE BLAST RADIUS IS ZERO ON EVERY FOUR-LEGGED BODY, stated as structure rather than as luck. After
    the hip there is exactly ONE fore-aft joint on a quadruped leg, so the new rule and the depth rule are
    picking from a set of size one and must agree. Asserted on a real Go2, a real ANYmal C and the template."""
    for gene, label in ((_imported("unitree_go2", "go2.xml"), "go2"),
                        (_imported("anybotics_anymal_c", "anymal_c.xml"), "anymal_c"),
                        (_template(), "template")):
        picks = _picks(gene)
        assert len(picks) == 4, f"{label} should present four legs, got {len(picks)}"
        for leg in picks:
            assert leg["n_candidates"] == 1, f"{label} leg offered a choice it should not have: {leg}"
            assert leg["lift"] == leg["depth_rule"], f"{label} pick moved: {leg}"
            assert leg["disclosure"] is None, "no choice was made, so nothing should be disclosed"


def test_a_radial_leg_keeps_its_deepest_token_fallback():
    """A spider/crab leg can read as having ONE fore-aft joint (its knee's axis is world-x), leaving no
    candidate at all. That case must keep the existing deepest-token fallback rather than fall back to the hip
    -- otherwise the leg has no lift joint and the body crouches instead of stepping."""
    from virturoid.services.morphology_composer import compose_robot
    picks = _picks(compose_robot("a spider robot", llm=None))
    assert picks, "the spider presented no legs"
    assert any(leg["n_candidates"] == 0 for leg in picks), \
        "this test is only meaningful while some spider leg has a single fore-aft joint"
    for leg in picks:
        if leg["n_candidates"] == 0:
            assert leg["lift"] == leg["depth_rule"]
            assert leg["lift"] != leg["hip"], "a leg whose lift joint IS its hip cannot pick its foot up"


def test_the_probe_reads_the_foot_and_not_the_tip_bodys_frame_origin():
    """The probe target is load bearing and the obvious choice is wrong. A revolute joint's anchor often sits
    exactly on the frame origin of the body it drives, so ``xpos[tip_body]`` does not move when that joint
    turns. On a real Go2 the calf joint moves the tip body's ORIGIN by 0.000 m and the FOOT by centimetres --
    a ranking built on the origin would be ranking zeros."""
    import mujoco
    from virturoid.services.morph_policy import _foot_geom_id
    model, data, graph, amap = _legs(_imported("unitree_go2", "go2.xml"))
    qadr = np.asarray(graph.qadr, int)
    leg = amap.legs[0]
    calf = leg.stride_tokens[-1]
    assert "calf" in _jname(model, graph, calf)
    fg = _foot_geom_id(model, data, int(leg.tip_body))
    assert fg is not None
    origin0 = np.array(data.xpos[int(leg.tip_body)])
    foot0 = np.array(data.geom_xpos[fg])
    qa = int(qadr[calf]); q0 = float(data.qpos[qa])
    data.qpos[qa] = q0 + 0.3
    mujoco.mj_forward(model, data)
    origin_moved = float(np.linalg.norm(np.array(data.xpos[int(leg.tip_body)]) - origin0))
    foot_moved = float(np.linalg.norm(np.array(data.geom_xpos[fg]) - foot0))
    data.qpos[qa] = q0
    mujoco.mj_forward(model, data)
    assert origin_moved < 1e-9, f"the tip body's origin should be pinned to the joint anchor, moved {origin_moved}"
    assert foot_moved > 0.02, f"the foot itself should sweep, moved {foot_moved}"


def test_the_lift_pick_is_deterministic():
    """A probe that decided a controller's structure differently on two identical calls would make every
    verdict on this body unreproducible."""
    gene = _imported("unitree_g1", "g1.xml")
    first = [leg["lift"] for leg in _picks(gene)]
    second = [leg["lift"] for leg in _picks(gene)]
    assert first == second and first


def test_the_g1_rollout_actually_moves_the_knees():
    """The end-to-end form of the headline: not "the map names the knee" but "the knee travels". Both knees
    must sweep a real angle over the rollout; under the depth rule they sat at their PD default the whole way
    down."""
    import mujoco
    from virturoid.services.morph_graph import encode_robot
    from virturoid.services.morph_policy import compiled_model, crawl_gait_rollout, robot_mjcf
    gene = _imported("unitree_g1", "g1.xml")
    r = crawl_gait_rollout(gene, steps=400, record_qpos=True, frame_every=5)
    model = compiled_model(robot_mjcf(gene), solver_iterations=20)
    graph = encode_robot(model)
    Q = np.asarray(r["qpos_frames"])
    swept = {}
    for t in range(graph.n_tokens):
        jid = int(model.actuator_trnid[int(graph.act_u[t]), 0])
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"joint_{jid}"
        swept[nm] = float(np.ptp(Q[:, int(graph.qadr[t])]))
    knees = {k: v for k, v in swept.items() if "knee" in k}
    assert len(knees) == 2, f"expected two knees, got {sorted(knees)}"
    for k, v in knees.items():
        assert v > 0.05, f"{k} swept only {v:.4f} rad -- it is still being held rigid"


# ---------------------------------------------------------------- (2) the degenerate direction probe arm
def _phases(gene):
    """Re-derive the two wave-direction phase dicts exactly as the rollout does, so the collapse can be shown
    as ARITHMETIC rather than inferred from a disclosure the same code emits."""
    from virturoid.services.gait_engine import select_duty
    _model, _data, _graph, amap = _legs(gene)
    legs_list = amap.legs
    beta = 0.75 if amap.n_legs == 4 else select_duty(amap, _model)
    xs = sorted({round(lg.tip_xy[0], 2) for lg in legs_list})
    rank_of = {x: i for i, x in enumerate(xs)}
    seg = {i: rank_of[round(lg.tip_xy[0], 2)] for i, lg in enumerate(legs_list)}
    right = {i: (1.0 if lg.side < 0 else 0.0) for i, lg in enumerate(legs_list)}
    top = max(seg.values(), default=0)
    fwd = {i: (seg[i] * (1.0 - beta) + 0.5 * right[i]) % 1.0 for i in seg}
    rev = {i: ((top - seg[i]) * (1.0 - beta) + 0.5 * right[i]) % 1.0 for i in seg}
    return fwd, rev, len(xs), beta


def test_a_bipeds_two_wave_directions_are_the_same_wave():
    """ONE fore-aft station -> ``seg`` is 0 for every leg and ``max_seg`` is 0, so the reversal is the identity
    map. This is the arithmetic behind "half the budget ran duplicates"."""
    fwd, rev, stations, _beta = _phases(_imported("unitree_g1", "g1.xml"))
    assert stations == 1
    assert fwd == rev == {0: 0.0, 1: 0.5}


def test_the_hexapod_direction_arm_collapses_too_and_for_a_second_reason():
    """FOUND WHILE MEASURING THIS FIX, and worth more than the biped case because a hexapod is a body we
    ship. A hexapod has THREE stations, so it looks like it has a wave -- but at the tripod duty beta=0.5 the
    reversal shifts every leg by ``max_seg * (1 - beta) = 2 * 0.5 = 1.0`` cycles, i.e. by nothing at all. Any
    body with an ODD number of leg stations at beta=0.5 degenerates the same way."""
    fwd, rev, stations, beta = _phases(_hexapod())
    assert (stations, beta) == (3, 0.5)
    assert fwd == rev, f"the hexapod's wave should reverse onto itself: {fwd} vs {rev}"


def test_a_quadruped_really_does_have_two_distinct_wave_directions():
    """The control that keeps the detection honest: on the template the two directions differ, so nothing
    four-legged loses its direction arm."""
    fwd, rev, stations, _beta = _phases(_template())
    assert stations == 2
    assert fwd != rev


def test_the_collapsed_arm_is_disclosed_and_the_budget_is_spent_not_saved():
    """A result that silently ran half the search it claims to have run is the defect; the fix says which arm
    it dropped, why, and where the freed rollouts went. Four probe rollouts before, four after."""
    from virturoid.services.morph_policy import crawl_gait_rollout
    r = crawl_gait_rollout(_imported("unitree_g1", "g1.xml"), steps=120)
    probe = r.get("direction_probe")
    assert probe, "the G1 collapses its direction arm and must say so"
    assert probe["arms"] == 1
    assert "no wave to reverse" in probe["reason"]
    assert len(probe["frequencies_hz"]) == 4, "1 arm x 4 frequencies = the same 4 rollouts as 2 arms x 2"
    assert probe["arms"] * len(probe["frequencies_hz"]) == 4


def test_the_frequency_ladder_is_a_superset_of_the_shipped_pair():
    """The freed budget must not move the search off the two points it used to try -- it may only ADD. The
    ladder is the shipped (1x, 1.7x) with one more step of the same ratio in each direction."""
    from virturoid.services.morph_policy import crawl_gait_rollout
    r = crawl_gait_rollout(_imported("unitree_g1", "g1.xml"), steps=120, freq=1.5)
    fq = r["direction_probe"]["frequencies_hz"]
    assert pytest.approx(1.5, rel=1e-6) in fq
    assert pytest.approx(1.5 * 1.7, rel=1e-6) in fq
    assert min(fq) < 1.5 < max(fq)


def test_a_quadruped_result_dict_is_untouched_key_for_key():
    """Every existing consumer of a four-legged result must see the dict it always saw. Neither disclosure key
    may appear on a body where neither defect applies."""
    from virturoid.services.morph_policy import crawl_gait_rollout
    r = crawl_gait_rollout(_template(), steps=200)
    assert "direction_probe" not in r
    assert "lift_joints" not in r
