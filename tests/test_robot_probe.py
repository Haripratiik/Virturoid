"""The agent must be able to MEASURE the robot it is holding.

`get_robot` returns a fixed summary, and `describe_robot`/`diagnose_body` compose a NEW body from a prompt instead
of inspecting the held one. So the questions a designer actually asks were unanswerable: how far is the gripper
from its mount, do these two links pass through each other anywhere in a joint's range, what is the torque margin
here, how much ground clearance is there.

This is the half of the loop the Articraft result says dominates -- raw GPT-5.5 ranked second-to-LAST against
baselines while the same model behind an SDK plus a verification harness ranked FIRST (91.8% retention over
10,909 assets). The model was never the variable.

Every assertion below is checked against physics or an independent computation, not against the probe's own
output, because a measurement surface that agrees with itself is worthless.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("VIRTUROID_DISABLE_GAIT_HINTS", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="probing needs MuJoCo")


@pytest.fixture(scope="module")
def dog():
    from virturoid.services.morphology_composer import compose_robot
    return compose_robot("a four legged robot dog", llm=None)


def test_a_yaw_joint_carries_no_gravity_torque():
    """The sharpest check available, because it has an analytic answer: gravity produces NO moment about a
    VERTICAL axis. An earlier version used the perpendicular distance to the axis instead of the moment about it,
    which invented torque on every yaw joint -- and a torque figure that flatters the design is worse than none,
    since the whole point is that a datasheet gets checked against it."""
    from virturoid.services.anatomy_compiler import build_from_anatomy
    from virturoid.services.robot_probe import probe
    yaw = build_from_anatomy({"robot_class": "manipulator", "name": "yaw", "parts": [
        {"name": "base", "role": "body", "size": 0.2, "girth": 0.06},
        {"name": "post", "role": "arm", "parent": "base", "attach": "front_top", "aim": "up",
         "size": 0.3, "girth": 0.03, "joint": "revolute", "axis": [0, 0, 1]}]})
    t = probe(yaw, {"fields": ["torque"]})["torque"]
    assert t, "no joints measured"
    for name, row in t.items():
        assert row["static_hold_nm"] == pytest.approx(0.0, abs=1e-6), f"{name}: {row}"


def test_mass_and_com_agree_with_mujoco_itself(dog):
    """Cross-check against the engine rather than trusting the probe's own arithmetic."""
    import mujoco
    import numpy as np

    from virturoid.services.morph_policy import compiled_model, robot_mjcf
    from virturoid.services.robot_probe import probe
    rep = probe(dog, {"fields": ["mass"]})["mass"]
    m = compiled_model(robot_mjcf(dog))
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    assert rep["total_kg"] == pytest.approx(float(m.body_mass.sum()), rel=1e-6)
    com = np.sum(m.body_mass[:, None] * d.xipos, axis=0) / float(m.body_mass.sum())
    assert rep["com"] == pytest.approx([float(v) for v in com], abs=1e-4)


def test_a_standing_quadruped_reports_four_contacting_parts_and_a_positive_margin(dog):
    """A body standing on four feet must measure as standing on four feet, with its COM inside them. This is the
    number the mobile-manipulator wheelie turned on (#246), so it is worth being able to ask for directly."""
    from virturoid.services.robot_probe import probe
    mass = probe(dog, {"fields": ["mass"]})["mass"]
    assert mass["n_contacting_parts"] == 4, mass
    assert mass["support_margin_m"] > 0.0, mass


def test_the_probe_measures_the_pose_the_body_SHIPS(dog):
    """A measurement taken in a stance the robot never holds is not useful -- the same principle the drive verdict
    and the imported rest pose both needed."""
    from virturoid.services.robot_probe import probe
    assert probe(dog, {"fields": ["parts"]})["posed_from"] == "rest keyframe"


def test_swept_collision_catches_what_the_rest_pose_cannot():
    """THE check a static one structurally cannot make. A design can be clean at rest and drive one link straight
    through another partway through a joint's travel -- Articraft names this as a top LLM failure mode.

    Built deliberately: a long arm hinged on a short body, with a range wide enough to fold back into the torso.
    At rest it is clear; somewhere in that range it is not."""
    from virturoid.services.anatomy_compiler import build_from_anatomy
    from virturoid.services.robot_probe import probe
    g = build_from_anatomy({"robot_class": "manipulator", "name": "folder", "parts": [
        {"name": "torso", "role": "body", "size": 0.30, "girth": 0.16},
        {"name": "boom", "role": "arm", "parent": "torso", "attach": "front_top", "aim": "forward",
         "size": 0.40, "girth": 0.035, "joint": "revolute", "axis": [0, 1, 0],
         "lower": -3.1, "upper": 3.1},
        {"name": "mast", "role": "sensor_mast", "like": "neck", "parent": "torso",
         "attach": {"along": 0.2, "height": 0.8}, "aim": "up", "size": 0.30, "girth": 0.035}]})
    rep = probe(g, {"fields": ["overlaps", "swept"]})
    assert rep["overlaps"] == [], f"this body is meant to be CLEAN at rest: {rep['overlaps']}"
    pairs = {frozenset(p) for p in rep["swept"]["pairs"]}
    # a check that never fires is not a check: the boom sweeps a full turn through where the mast stands
    assert frozenset(("boom", "mast")) in pairs, (
        f"the swept check found {rep['swept']['pairs']}, but a 0.40 m boom swinging +-3.1 rad MUST pass through a "
        "0.30 m mast on the same torso — if this ever goes quiet the check has become vacuous")
    # exact, so it can say WHERE and HOW DEEP -- an unactionable list is barely better than no list
    det = rep["swept"]["detail"]["boom|mast"]
    assert det["joint"] == "boom_joint" and 0.0 <= det["at_travel"] <= 1.0, det
    assert det["penetration_m"] < 0, f"a reported contact must actually penetrate: {det}"


def test_hitting_the_FLOOR_is_reported_apart_from_hitting_yourself():
    """Both are real defects and they are different ones: a mounted arm fouling its own bench is not the same
    problem as two links passing through each other, and conflating them buries one inside the other."""
    from virturoid.services.anatomy_compiler import build_from_anatomy
    from virturoid.services.robot_probe import probe
    g = build_from_anatomy({"robot_class": "manipulator", "name": "sweeper", "parts": [
        {"name": "torso", "role": "body", "size": 0.30, "girth": 0.16},
        {"name": "boom", "role": "arm", "parent": "torso", "attach": "front_top", "aim": "forward",
         "size": 0.40, "girth": 0.035, "joint": "revolute", "axis": [0, 1, 0],
         "lower": -3.1, "upper": 3.1}]})
    sw = probe(g, {"fields": ["swept"]})["swept"]
    assert "boom" in sw["ground_strikes"], f"a 0.40 m boom swinging down MUST reach the floor: {sw}"
    assert not any("world" in p for p in sw["pairs"]), "the floor must not appear as a self-collision pair"


def test_the_swept_check_is_exact_not_a_box_estimate():
    """It reads MuJoCo's own contacts, so a reported pair IS touching. The AABB version returned 43 pairs on a
    quadruped, most of them two capsules passing near each other diagonally -- and a list you cannot act on is
    barely better than no list."""
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.robot_probe import probe
    sw = probe(compose_robot("a four legged robot dog", llm=None), {"fields": ["swept"]})["swept"]
    assert len(sw["pairs"]) < 43, f"still box-like: {len(sw['pairs'])} pairs"
    for k, v in sw["detail"].items():
        assert v["penetration_m"] < 0, f"{k} reported without real penetration: {v}"
        assert 0.0 <= v["at_travel"] <= 1.0 and v["joint"], v


def test_probing_never_mutates_the_robot(dog):
    """Read-only means read-only: the held gene must be byte-identical afterwards, or an agent that measures its
    design would be silently editing it."""
    from virturoid.services.robot_probe import probe
    before = dog.to_dict()
    probe(dog)
    assert dog.to_dict() == before
