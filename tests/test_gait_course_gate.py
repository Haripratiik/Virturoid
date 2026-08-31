"""A ROBOT THAT WALKS IN A CIRCLE MUST NOT SCORE CREDIBLE WALK.

Found 2026-08-08 by LOOKING AT A RENDER, not at a number (docs/body_vs_controller_ruling.md, D5). The offline
compositor's lynx, at the fragile operating point ``fit_gait_for_body`` adopts on seed 7, leaves the start,
walks a ~2.2 m loop, comes back past its own start line — and reported ``forward 1.582 m`` and **CREDIBLE WALK**.

Two mechanisms combined, and this file pins both shut:

  1. ``morph_policy``'s ``forward`` is world-frame delta-x. That is the honest projection on the heading the body
     set out on, and it is BOUNDED for a loop — but the 0.3 m gate it is tested against is cleared by any circle
     wider than 0.3 m, so "bounded" was never enough on its own.
  2. ``gait_quality.classify`` had no course gate at all. ``orientation_summary`` computed ``yaw_max`` and
     ``classify`` used it in exactly one place: naming the dominant fall mode on a body that had ALREADY failed.
     On a body that passed the scalar gates a 179 deg heading sweep was invisible.

The fix is ``course_summary`` + a two-part gate, and the two parts are not redundant — each catches a body the
other misses, which is why both are here and both are tested:

  * heading deviation (p95, unwrapped, vs the start heading) catches TURNING;
  * straightness (net displacement / ground covered) catches MILLING, which a body can do while its heading
    barely moves (measured: the template quad under a hard steer holds heading dev at 35 deg and straightness
    at 0.37).

THE GATES MUST NOT EAT THE HONEST VERDICTS WE ALREADY HAVE. Re-simulated 2026-08-08 at 800 / 1500 / 6000 steps:
all 83 of the 102 banked locomotion rows whose body could be rebuilt from a durable source. 63 are credible under
the old ruler at some horizon; the course gate removes NONE of them. Their worst heading deviation is 65.5 deg
(gate 90) and their worst straightness 0.725 (gate 0.5). 20 rows DO circle or mill, and every one of them was
already failing for another reason (CROUCH / FELL / LURCHES / FORWARD BUT SHORT) — so the hole was one an
optimiser found, not one the corpus had fallen into. A gait that CURVES is not a gait that LOOPS, and the hexapod
tests below hold that line at 5.6 m of travel through a 60 deg turn.
"""
from __future__ import annotations

import importlib.util
import math
import os

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("MUJOCO_GL", "glfw")
_MUJOCO = importlib.util.find_spec("mujoco") is not None

# The offline lynx's seed-7 operating point, read off `fit_gait_for_body(gene, seed=7, cache=False)` on the
# grounded `design_cassette_v1` body. This is THE episode the defect was found in.
_CIRCLE_OP_POINT = {"freq": 1.9675273580952262, "hip_amp": 0.5247357158524661,
                    "knee_amp": 1.0003326527892034, "kp": 117.07482994076271, "kd": 12.360744964313335}


def _frames(path_xy, yaws):
    """A qpos trace: [x, y, z, qw, qx, qy, qz, ...] per frame, with the base quaternion carrying yaw only."""
    out = []
    for (x, y), yaw in zip(path_xy, yaws):
        out.append([x, y, 0.30, math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0), 0.0])
    return out


def _walker(path_xy, yaws, *, forward=None, steps=2000):
    """A rollout dict whose SCALARS are a textbook credible walk, so only the course can fail it.

    ``steps`` stays below ``_SETTLE_MIN_STEPS`` so the settling gate is off and each test isolates one gate.
    """
    fwd = (path_xy[-1][0] - path_xy[0][0]) if forward is None else forward
    return {"survived": True, "upright_frac": 0.95, "height_ratio": 0.9, "cadence": 2.4,
            "support_frac": 0.8, "forward": round(fwd, 3), "steps": steps, "frame_every": 10,
            "qpos_frames": _frames(path_xy, yaws)}


def _arc(total_deg: float, radius: float = 1.1, n: int = 200):
    """A constant-radius arc starting at the origin heading +x and curving left through ``total_deg``."""
    th = [math.radians(total_deg) * i / (n - 1) for i in range(n)]
    return [(radius * math.sin(t), radius * (1 - math.cos(t))) for t in th], th


# --------------------------------------------------------------------------- the defect, as a unit test
def test_a_closed_circle_is_not_a_credible_walk():
    """The whole point. A full 360 deg loop of radius 1.1 m covers 6.9 m of ground and ends where it started."""
    from virturoid.services.gait_quality import classify, course_summary
    P, Y = _arc(360.0)
    # `forward` is forced to the value a loop can genuinely book: the far side of this circle sits at +x, and
    # delta-x measured mid-loop is what the real defect reported (1.582 m on a 2.2 m loop).
    r = _walker(P, Y, forward=1.582)
    c = course_summary(r)
    assert c["heading_dev_p95_deg"] > 300, c            # it turned right round
    assert c["straightness"] < 0.05, c                  # and got nowhere
    assert not classify(r).startswith("CREDIBLE"), classify(r)
    assert "CIRCLE" in classify(r), classify(r)


def test_the_verdict_string_carries_the_numbers_that_convict_it():
    """D2's lesson: robustness was disclosed in the payload and invisible on the field consumers read. The
    course must not repeat that — ``verdict`` is what ``design_bench._is_credible`` and every chat summary see."""
    from virturoid.services.gait_quality import classify
    P, Y = _arc(360.0)
    v = classify(_walker(P, Y, forward=1.582))
    assert " m net for " in v and " m walked" in v and "heading swung" in v, v


def test_a_half_circle_is_not_a_credible_walk_either():
    """180 deg is the point where the body is travelling back the way it came. Its net displacement (the
    diameter) is real and is also the MOST it can ever be, however long you run."""
    from virturoid.services.gait_quality import classify
    P, Y = _arc(180.0)
    assert not classify(_walker(P, Y)).startswith("CREDIBLE"), classify(_walker(P, Y))


# --------------------------------------------------------------------------- ...and the honest verdicts survive
def test_a_straight_walk_is_still_a_credible_walk():
    from virturoid.services.gait_quality import classify
    P = [(0.02 * i, 0.0) for i in range(200)]
    assert classify(_walker(P, [0.0] * 200)) == "CREDIBLE WALK"


def test_a_gait_that_curves_gently_is_still_a_credible_walk():
    """A gently curving walk is NOT the same failure as one that loops, and must not be scored as one. At 60 deg
    of turn the chord is still 95% of the arc — measured on the real hexapod: 6.2 m of travel, straightness
    0.93. The gate has to let this through, or every asymmetric-but-working body reads as a circler."""
    from virturoid.services.gait_quality import classify, course_summary
    P, Y = _arc(60.0, radius=6.0)
    r = _walker(P, Y)
    c = course_summary(r)
    assert c["heading_dev_p95_deg"] < 90.0 and c["straightness"] > 0.9, c
    assert classify(r) == "CREDIBLE WALK", classify(r)


def test_a_body_that_wanders_without_turning_is_caught_by_straightness_not_heading():
    """The two gates are not redundant. This body's heading never leaves +-20 deg, so a yaw gate alone passes
    it, and it covers 4x its net displacement in ground. Measured equivalent: template quad at turn_bias 0.9,
    heading dev 35 deg, straightness 0.370, and CREDIBLE WALK before this gate existed."""
    from virturoid.services.gait_quality import classify, course_summary
    P, Y = [], []
    x = y = 0.0
    for i in range(200):                                # a tight zig-zag: lots of ground, little progress
        x += 0.012
        y += 0.05 * (1 if (i // 5) % 2 == 0 else -1)
        P.append((x, y)); Y.append(math.radians(18.0 if (i // 5) % 2 == 0 else -18.0))
    r = _walker(P, Y, forward=x, steps=2000)
    c = course_summary(r)
    assert c["heading_dev_p95_deg"] < 90.0, c           # a heading gate ALONE would let this pass
    assert c["straightness"] < 0.5, c
    assert not classify(r).startswith("CREDIBLE"), classify(r)
    assert "MILLS" in classify(r), classify(r)


def test_a_metric_only_rollout_keeps_exactly_the_verdict_it_always_had():
    """No trace and no course fields = the course is unanswerable, and inventing a failure for a question the
    data cannot answer would be a second false verdict (the rule ``settling`` already follows)."""
    from virturoid.services.gait_quality import classify, course_summary
    r = {"survived": True, "upright_frac": 0.95, "height_ratio": 0.9, "cadence": 2.4,
         "support_frac": 0.8, "forward": 1.2}
    assert course_summary(r) is None
    assert classify(r) == "CREDIBLE WALK"


def test_the_rollouts_own_numbers_and_the_trace_agree():
    """``course_summary`` prefers the rollout's per-physics-step accumulation and falls back to the trace, so a
    banked row's replay and a live rollout must not be judged by two different rulers."""
    from virturoid.services.gait_quality import course_summary
    P, Y = _arc(100.0, radius=2.0)
    r = _walker(P, Y)
    from_trace = course_summary(r)
    net = math.hypot(P[-1][0] - P[0][0], P[-1][1] - P[0][1])
    path = sum(math.dist(P[i], P[i + 1]) for i in range(len(P) - 1))
    r_fields = dict(r, path_m=path, net_m=net, heading_dev_p95_deg=from_trace["heading_dev_p95_deg"],
                    heading_dev_max_deg=from_trace["heading_dev_max_deg"])
    from_fields = course_summary(r_fields)
    assert from_fields["source"] == "rollout" and from_trace["source"] == "trace"
    assert abs(from_fields["straightness"] - from_trace["straightness"]) < 0.01


# --------------------------------------------------------------------------- against real physics
@pytest.mark.skipif(not _MUJOCO, reason="the course gate's calibration is a claim about real rollouts")
def test_the_rollout_reports_its_own_course_and_an_unwrapped_heading():
    """``yaw_change`` used to be ``atan2(sin(dyaw), cos(dyaw))`` — the yaw MODULO a full turn, so a body that
    turned 190 deg reported -170 (the wrong way round) and one that turned 360 reported 0. Those are exactly
    the headings a locomotion verdict most needs to tell apart from "went straight"."""
    from virturoid.services.morph_policy import crawl_gait_rollout
    from virturoid.services.morphology_composer import compose_robot
    g = compose_robot("a six-legged walking robot", ensure_walkable=True)
    r = crawl_gait_rollout(g, steps=3000, record_qpos=True, frame_every=20, turn_bias=0.9)
    for k in ("path_m", "net_m", "straightness", "heading_dev_p95_deg", "heading_dev_max_deg"):
        assert k in r, f"the rollout must disclose its own course: missing {k}"
    assert r["path_m"] >= r["net_m"] - 1e-6, r         # a path can never be shorter than the displacement
    # the integrated heading and the trace-derived one are the same quantity
    assert abs(math.degrees(abs(r["yaw_change"])) - r["heading_dev_max_deg"]) < 15.0, r


@pytest.mark.skipif(not _MUJOCO, reason="the course gate's calibration is a claim about real rollouts")
def test_a_straight_hexapod_walk_clears_both_gates_with_room():
    """The anchor. If this ever sits near a gate, the gate moved onto the corpus and not onto the defect."""
    from virturoid.services.gait_quality import _CLEAN_HEADING_DEV, _MIN_STRAIGHTNESS, classify, course_summary
    from virturoid.services.morph_policy import crawl_gait_rollout
    from virturoid.services.morphology_composer import compose_robot
    g = compose_robot("a six-legged walking robot", ensure_walkable=True)
    r = crawl_gait_rollout(g, steps=6000, record_qpos=True, frame_every=20)
    assert classify(r) == "CREDIBLE WALK", classify(r)
    c = course_summary(r)
    assert c["heading_dev_p95_deg"] < 0.5 * _CLEAN_HEADING_DEV, c    # 2x margin, not 2%
    assert c["straightness"] > 1.5 * _MIN_STRAIGHTNESS, c


@pytest.mark.skipif(not _MUJOCO, reason="the course gate's calibration is a claim about real rollouts")
def test_a_steered_body_that_mills_loses_its_credible_walk():
    """Same body, same controller, one input changed — so the verdict flip is about the COURSE and nothing else."""
    from virturoid.services.gait_quality import classify
    from virturoid.services.morph_policy import crawl_gait_rollout
    from virturoid.services.morphology_composer import compose_robot
    g = compose_robot("a six-legged walking robot", ensure_walkable=True)
    r = crawl_gait_rollout(g, steps=6000, record_qpos=True, frame_every=20, turn_bias=0.9)
    v = classify(r)
    assert not v.startswith("CREDIBLE"), v
    assert ("CIRCLE" in v or "MILLS" in v or "TURNS OFF COURSE" in v), v


@pytest.mark.skipif(not _MUJOCO, reason="the course gate's calibration is a claim about real rollouts")
def test_a_steered_body_that_merely_curves_keeps_its_credible_walk():
    """The other half of the same experiment, and the one that proves the gate is not just 'reject anything that
    is not dead straight'. 6 m of travel through a real 60 deg turn stays a walk."""
    from virturoid.services.gait_quality import classify, course_summary
    from virturoid.services.morph_policy import crawl_gait_rollout
    from virturoid.services.morphology_composer import compose_robot
    g = compose_robot("a six-legged walking robot", ensure_walkable=True)
    r = crawl_gait_rollout(g, steps=6000, record_qpos=True, frame_every=20, turn_bias=0.3)
    c = course_summary(r)
    assert c["heading_dev_max_deg"] > 40.0, c          # it really did turn
    assert classify(r) == "CREDIBLE WALK", (classify(r), c)


@pytest.mark.skipif(not _MUJOCO, reason="the course gate's calibration is a claim about real rollouts")
def test_the_offline_lynx_circle_no_longer_reports_a_credible_walk():
    """THE REGRESSION TEST. This exact operating point, on this exact body, reported CREDIBLE WALK 1.582 m."""
    from pathlib import Path

    from virturoid.services.design_cassette import DesignCassette
    from virturoid.services.gait_quality import classify, course_summary
    from virturoid.services.gene_build import ground_and_repair
    from virturoid.services.morph_policy import crawl_gait_rollout
    cass = Path(__file__).resolve().parent / "fixtures" / "design_cassette_v1.json"
    if not cass.exists():
        pytest.skip("offline design cassette not in this checkout")
    g = DesignCassette(cass).get_gene("lynx__appearance")
    ground_and_repair(g)
    r = crawl_gait_rollout(g, steps=6000, record_qpos=True, frame_every=20, **_CIRCLE_OP_POINT)
    c = course_summary(r)
    v = classify(r)
    assert not v.startswith("CREDIBLE"), f"the circle is credible again: {v} / {c}"
    assert c["heading_dev_p95_deg"] >= 90.0 or c["straightness"] < 0.5, c
