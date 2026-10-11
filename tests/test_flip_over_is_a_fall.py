"""A body that went upside down FELL, whatever its trunk height says.

THE DEFECT (measured 2026-10-11, driving the product over MCP). ``create_robot`` "a quadruped inspection robot that
patrols a warehouse aisle carrying a 2 kg sensor payload", then ``edit_robot`` ``set_payload`` 2 kg (which leaves
the gait fitted before the edit in place), then a full ``verify_robot``: "LURCHES (pitch 59 / roll 180 deg
sustained ...)", ``survived: true``, forward 2.639 m over 6000 steps. Replayed, the body rears until its front axis
points straight up at step 4080 and lands on its back, where it stays: 96 of the 300 recorded frames are past
90 deg of tilt, the most 139 deg. Two things combined:

  * ``survived`` was height-only, and the payload box on its top propped the inverted trunk at 0.70 to 0.79 of its
    standing height, so it passed; ``upright_frac >= 0.6`` passed too, at 32% of the rollout upside down.
  * "roll 180" is an Euler artifact. Past 90 deg of tilt the decomposition re-expresses an end-over-end fall as
    roll 180 plus a pitch that comes back down; this body never rolled. It went over backwards.

The pure tests pin the contract on that measured trace (``tests/fixtures/flip_over_payload_quadruped_v1.json``)
and on synthetic attitudes; the MuJoCo ones check that the rollout producers refuse ``survived`` to a body on its
back and leave every other body's result exactly as it was.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import pathlib
import re

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("VIRTUROID_DISABLE_GAIT_HINTS", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None
_FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "flip_over_payload_quadruped_v1.json"
_FLIP_KEYS = ("inverted_at_step", "inverted_frac", "tilt_max_deg", "flipped_over", "survived_by_height")


def _measured() -> dict:
    """The defect's own rollout, as the height-only producer reported it (``survived: true``)."""
    d = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    return {**d["rollout"], "qpos_frames": d["base_qpos_frames"]}


def _quat(axis: str, deg: float) -> tuple:
    h = math.radians(deg) / 2.0
    return {"x": (math.cos(h), math.sin(h), 0.0, 0.0), "y": (math.cos(h), 0.0, math.sin(h), 0.0)}[axis]


def _walk(attitudes, *, steps=6000, travel=2.0, **scalars) -> dict:
    """A metrics dict whose scalars are a textbook credible walk, over a trace with the given per-frame attitude
    ``(axis, deg)``, covering ``travel`` metres along x at a steady rate in a straight line."""
    n = len(attitudes)
    frames = []
    for i, (axis, deg) in enumerate(attitudes):
        w, x, y, z = _quat(axis, deg)
        frames.append([travel * i / (n - 1), 0.0, 0.3, w, x, y, z])
    r = {"survived": True, "upright_frac": 0.95, "height_ratio": 0.9, "cadence": 6.0, "support_frac": 0.6,
         "forward": round(travel, 3), "steps": steps, "frame_every": steps // n, "qpos_frames": frames}
    r.update(scalars)
    return r


# --------------------------------------------------------------------------- the measured defect
def test_the_measured_rollout_reads_fell_by_flip_over():
    from virturoid.services.gait_quality import classify
    v = classify(_measured())
    assert v.startswith("FELL by FLIP-OVER"), v
    assert "reared over backwards at step 4080" in v, v
    assert "upside down for 32% of the rollout" in v and "tilt up to 139 deg" in v, v
    # ...and nothing a reader could take for the Euler artifact the old verdict printed as a sustained roll
    assert "LURCH" not in v and "roll 180" not in v and "sustained" not in v, v


def test_orientation_summary_reports_tilt_beside_the_euler_angles():
    """Roll/pitch stay what they always were (the 180 below IS the artifact, kept so nothing that reads them moves);
    tilt and the inverted share are the numbers that stay true past 90 deg."""
    from virturoid.services.gait_quality import orientation_summary
    o = orientation_summary(_measured()["qpos_frames"])
    assert o["roll_max"] == 180.0 and o["roll_p95"] == 180.0         # what the old verdict quoted
    assert o["tilt_max"] == pytest.approx(139.1, abs=0.2)
    assert o["tilt_p95"] > 90.0
    assert o["inverted_frac"] == pytest.approx(96 / 300, abs=1e-3)


def test_the_rollouts_own_count_needs_no_trace_and_cannot_be_credible():
    """A metric-only dict has no trace, so the level gate cannot see it (``level`` is True when nothing was
    recorded). The producer's own per-step count must be enough on its own, even next to a ``survived`` that a
    producer still computing it by height alone would report."""
    from virturoid.services.gait_quality import classify
    r = {k: v for k, v in _measured().items() if k != "qpos_frames"}
    r.update({"inverted_at_step": 4079, "inverted_frac": 0.32, "tilt_max_deg": 139.2, "flipped_over": "backwards"})
    for survived in (False, True):
        v = classify({**r, "survived": survived})
        assert v.startswith("FELL by FLIP-OVER (reared over backwards at step 4079"), v
    # the same scalars WITHOUT the count read credible: the count is the only thing standing in the way
    clean = {k: v for k, v in r.items() if k not in _FLIP_KEYS}
    clean.update({"heading_dev_p95_deg": 2.0, "heading_dev_max_deg": 4.0, "path_m": 2.7, "net_m": 2.64})
    assert classify(clean) == "CREDIBLE WALK"
    assert not classify({**clean, "inverted_at_step": 4079, "inverted_frac": 0.32, "tilt_max_deg": 139.2,
                         "flipped_over": "backwards"}).startswith("CREDIBLE")


# --------------------------------------------------------------------------- which way it went over
@pytest.mark.parametrize("axis,deg,words", [("y", -130.0, "reared over backwards"),
                                            ("y", 130.0, "pitched over forwards"),
                                            ("x", 130.0, "rolled over sideways"),
                                            ("x", -130.0, "rolled over sideways")])
def test_the_way_it_went_over_is_named_from_the_body_axes(axis, deg, words):
    """Upright for two thirds of the run, then over. An end-over-end fall reads as roll 180 in Euler terms
    (asserted, so the test shows what it guards against); the verdict names the axis it actually went over."""
    from virturoid.services.gait_quality import classify, orientation_summary
    att = [("x", 2.0)] * 200 + [(axis, deg * k / 10.0) for k in range(1, 11)] + [(axis, deg)] * 90
    r = _walk(att)
    v = classify(r)
    assert v.startswith("FELL by FLIP-OVER (" + words), v
    assert "tilt up to 130 deg" in v and not re.search(r"roll \d", v), v
    if axis == "y":
        assert orientation_summary(r["qpos_frames"])["roll_max"] == 180.0     # the artifact the verdict avoids


def test_a_body_seen_only_after_it_was_over_does_not_get_a_direction_invented():
    from virturoid.services.gait_quality import classify, flip_direction
    assert flip_direction(0.0, 1.0, 0.0, 0.0) is None                      # flat on its back: no crossing seen
    v = classify(_walk([("x", 180.0)] * 300))
    assert v.startswith("FELL by FLIP-OVER (went over at step 0 and was upside down for 100%"), v


# --------------------------------------------------------------------------- what must not move
def test_a_level_walk_is_unaffected():
    from virturoid.services.gait_quality import classify, flip_summary, orientation_summary
    r = _walk([("x", 5.0 if i % 2 else -5.0) for i in range(300)])
    assert classify(r) == "CREDIBLE WALK"
    assert flip_summary(r) is None
    o = orientation_summary(r["qpos_frames"])
    assert o["inverted_frac"] == 0.0 and o["tilt_max"] == pytest.approx(5.0, abs=0.01)


def test_a_fall_short_of_90_deg_keeps_its_euler_name():
    """Below 90 deg of tilt the Euler roll is confined to (-90, 90) and means what it says, so a body that tipped
    to 80 deg and fell by height is still named by it."""
    from virturoid.services.gait_quality import classify
    att = [("x", 2.0)] * 200 + [("x", 80.0)] * 100
    v = classify(_walk(att, survived=False, height_ratio=0.4))
    assert v.startswith("FELL by ROLL-OVER (roll 80"), v


def test_a_hard_ceiling_frame_short_of_90_is_still_a_lurch_not_a_flip():
    from virturoid.services.gait_quality import _HARD_PITCH_MAX, classify
    att = [("y", 3.0)] * 296 + [("y", _HARD_PITCH_MAX + 15.0)] * 4
    v = classify(_walk(att))
    assert v.startswith("LURCHES"), v


# --------------------------------------------------------------------------- backward travel
def test_a_backward_walk_is_named_backward_and_stays_non_credible():
    """MEASURED 2026-10-11: the design schema's hexapod example, through submit_design and a full verify, travels
    -1.9 m in 6000 steps and was reported "FORWARD BUT SHORT"."""
    from virturoid.services.gait_quality import classify
    level = [("x", 2.0)] * 300
    v = classify(_walk(level, travel=-1.9))
    assert v.startswith("WALKS BACKWARD (1.90 m backward over 6000 steps"), v
    assert classify(_walk(level, travel=-0.2)) == "FORWARD BUT SHORT / not a credible walk"    # under a walk's 0.3
    assert classify(_walk(level, travel=0.2)) == "FORWARD BUT SHORT / not a credible walk"
    assert classify(_walk(level, travel=0.3)) == "CREDIBLE WALK"                                # the bar is unmoved
    assert classify(_walk(level, travel=-1.9, cadence=0.2)).startswith("SLIDE")                 # earlier names win
    assert classify(_walk(level, travel=-1.9, survived=False)) == "FORWARD BUT SHORT / not a credible walk"


# --------------------------------------------------------------------------- the producers
_ON_ITS_BACK = """
<mujoco model="on_its_back">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="trunk" pos="0 0 0.051" quat="0 1 0 0">
      <freejoint name="root"/>
      <geom name="trunk_g" type="box" size="0.2 0.1 0.05" mass="6"/>
      {legs}
    </body>
  </worldbody>
  <actuator>{motors}</actuator>
</mujoco>"""
_LEG = """
      <body name="hip{i}" pos="{x} {y} -0.05">
        <joint name="hip{i}_j" type="hinge" axis="0 1 0" range="-1 1"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.12" size="0.02" mass="0.3"/>
        <body name="shin{i}" pos="0 0 -0.12">
          <joint name="knee{i}_j" type="hinge" axis="0 1 0" range="-2 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.12" size="0.015" mass="0.2"/>
        </body>
      </body>"""


def _on_its_back_xml() -> str:
    legs = "".join(_LEG.format(i=i, x=x, y=y) for i, (x, y) in enumerate(((0.15, 0.1), (0.15, -0.1),
                                                                            (-0.15, 0.1), (-0.15, -0.1))))
    motors = "".join(f'<motor joint="{j}{i}_j" ctrlrange="-8 8" ctrllimited="true"/>'
                     for i in range(4) for j in ("hip", "knee"))
    return _ON_ITS_BACK.format(legs=legs, motors=motors)


@pytest.mark.skipif(not _MUJOCO, reason="needs MuJoCo")
def test_the_crawl_refuses_survived_to_a_body_lying_on_its_back():
    """A quadruped lying on its back with its legs in the air never drops its trunk, so the height-only rule passed
    it: ``alive`` is the whole horizon and the height ratio stays near 1."""
    from virturoid.services.gait_quality import classify
    from virturoid.services.morph_policy import crawl_gait_rollout
    r = crawl_gait_rollout(_on_its_back_xml(), steps=300, record_qpos=True)
    assert r["alive"] == 300 and r["height_ratio"] > 0.6, r          # what used to make it "survived"
    assert r["survived"] is False and r["survived_by_height"] is True
    assert r["inverted_at_step"] == 0 and r["inverted_frac"] == 1.0 and r["tilt_max_deg"] > 170.0, r
    assert classify(r).startswith("FELL by FLIP-OVER"), classify(r)


@pytest.mark.skipif(not _MUJOCO, reason="needs MuJoCo")
def test_the_gait_search_still_ranks_a_flip_by_the_height_test():
    """Reported and gated: a fall. RANKED by the search: what the height test says, exactly as before, because
    scoring a flip as a fall moves the CEM (measured on the defect's own body: the fit's adopted walk went from
    10.477 m, sturdy to a 0.1 perturbation, to 5.033 m, sturdy to 0.001). That is a search-design change for its own
    A/B; this pins that the verdict fix did not make it by accident."""
    from virturoid.services.gait_search import evaluate_gait, reward_features_from_rollout
    from virturoid.services.morph_policy import crawl_gait_rollout
    op = {"freq": 1.5, "hip_amp": 0.9, "knee_amp": 1.0, "kp": 32.0, "kd": 1.5}
    e = evaluate_gait(_on_its_back_xml(), op, steps=300)
    assert e["survived"] is False and e["credible"] is False
    assert e["verdict"].startswith("FELL by FLIP-OVER"), e["verdict"]
    assert e["fitness"] == pytest.approx(0.3 * e["forward"])       # a non-credible survivor's score, not hr - 1.2
    r = crawl_gait_rollout(_on_its_back_xml(), steps=300, **op)
    assert reward_features_from_rollout(r)["alive"] == 1.0


@pytest.mark.skipif(not _MUJOCO, reason="needs MuJoCo")
def test_the_recipe_rollout_refuses_it_too():
    from virturoid.services.morph_graph import encode_robot
    from virturoid.services.morph_policy import MorphPolicy, compiled_model, recipe_rollout_morph
    xml = _on_its_back_xml()
    p = MorphPolicy(encode_robot(compiled_model(xml, solver_iterations=20)).feature_dim, seed=0)
    r = recipe_rollout_morph(xml, p, steps=300)
    assert r["alive"] == 300 and r["height_ratio"] > 0.6, r
    assert r["survived"] is False and r["inverted_at_step"] == 0, r


@pytest.mark.skipif(not _MUJOCO, reason="needs MuJoCo")
def test_an_upright_walker_gets_exactly_the_dict_it_always_got():
    """The four keys are a disclosure a body has to earn: the canonical template walking at its shipped point
    never goes over, so its result carries none of them and its ``survived`` is the height rule's."""
    from virturoid.services.morph_policy import crawl_gait_rollout
    from virturoid.services.morphology_composer import compose_robot
    g = compose_robot("a robot dog", llm=None, ensure_walkable=True)
    r = crawl_gait_rollout(g, steps=800, record_qpos=True)
    assert not any(k in r for k in _FLIP_KEYS), r.keys()
    assert r["survived"] is bool(r["alive"] >= 800 and r["height_ratio"] > 0.6)
