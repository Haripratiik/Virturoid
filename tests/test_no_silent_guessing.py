"""What the compiler does NOT understand, it must say — not quietly pick something.

Two places guessed. Both were found by rendering machines rather than counting their joints, and both produced a
design that compiled, passed every structural check, and was wrong where a customer looks.

The rule is not new: the ROLE vocabulary already refuses to guess, and makes an unfamiliar role declare what it
is `like`. These are the two spots that predated it.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("VIRTUROID_DISABLE_GAIT_HINTS", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="compiling a body needs MuJoCo")


def test_an_aim_the_compiler_does_not_know_is_refused_not_guessed():
    """`right` reads exactly like a direction and is not one of the 18 tokens. It used to become `forward`
    silently, which is how a gantry ended up with its bridge running DOWN the machine instead of ACROSS it —
    compiled, correct DOF, wrong robot."""
    from virturoid.services.anatomy_compiler import build_from_anatomy
    with pytest.raises(ValueError) as e:
        build_from_anatomy({"robot_class": "gantry", "name": "g", "parts": [
            {"name": "bed", "role": "body", "size": 1.0, "girth": 0.3},
            {"name": "beam", "role": "beam", "like": "arm", "parent": "bed", "aim": "right",
             "size": 0.6, "girth": 0.05}]})
    msg = str(e.value)
    assert "right" in msg and "forward" in msg, msg
    assert "attach" in msg or "symmetry" in msg, f"the error must say what to use INSTEAD: {msg}"


def test_an_omitted_aim_still_takes_the_role_default():
    """Refusing an unknown token must not turn every omitted one into an error — saying nothing is a valid way
    to accept the default, and this is the regression surface for every animal ever composed."""
    from virturoid.services.anatomy_compiler import build_from_anatomy
    g = build_from_anatomy({"robot_class": "quadruped", "name": "q", "parts": [
        {"name": "torso", "role": "body", "size": 0.5, "girth": 0.13},
        {"name": "leg", "role": "leg", "parent": "torso", "attach": "front_bottom",
         "size": 0.35, "girth": 0.02, "segments": 4, "symmetry": "left_right", "joint": "revolute"}]})
    assert len(g.segments) > 2


def test_a_part_can_point_anywhere_not_only_where_a_token_names():
    """Every one of the 18 aim tokens has y >= 0. They were written for ANIMALS, whose limbs come in mirrored
    pairs placed by `symmetry: left_right`, so nothing in the vocabulary points a single part at -y — and a
    RADIAL layout (three delta arms at 120 degrees, a radial urchin) is unauthorable however they are combined.

    An explicit [x, y, z] direction removes the ceiling without touching the shorthand, the same way parametric
    `attach` did for the 8 named mount sites."""
    import math

    import mujoco
    import numpy as np
    from virturoid.services.anatomy_compiler import build_from_anatomy
    from virturoid.services.morph_policy import compiled_model, robot_mjcf

    parts = [{"name": "plate", "role": "frame", "like": "body", "size": 0.40, "girth": 0.20}]
    for i in range(3):
        th = math.radians(120 * i)
        parts.append({"name": f"arm{i}", "role": "delta_arm", "like": "arm", "parent": "plate",
                      "attach": {"along": 0.5, "lateral": 0.0, "height": 0.2},
                      "aim": [math.cos(th) * 0.6, math.sin(th) * 0.6, -0.53],
                      "size": 0.34, "girth": 0.028, "joint": "revolute", "axis": [0, 1, 0],
                      "lower": -1.2, "upper": 0.6})
    gene = build_from_anatomy({"robot_class": "delta", "name": "delta", "parts": parts})
    m = compiled_model(robot_mjcf(gene))
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    az = []
    for i in range(3):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"arm{i}")
        gs = [gi for gi in range(m.ngeom) if int(m.geom_bodyid[gi]) == b]
        assert gs, f"arm{i} has no geometry"
        org = np.array(d.xpos[b], dtype=float)
        far = max((np.array(d.geom_xpos[gi], dtype=float) for gi in gs),
                  key=lambda p: float(np.linalg.norm(p - org)))
        v = far - org
        az.append(math.degrees(math.atan2(float(v[1]), float(v[0]))))
    az.sort()
    assert az[1] - az[0] == pytest.approx(120.0, abs=8) and az[2] - az[1] == pytest.approx(120.0, abs=8), (
        f"the three arms sit at {[round(a, 1) for a in az]} degrees, not a 120-degree fan")
    assert min(az) < -30.0, f"no arm reaches negative y ({[round(a, 1) for a in az]}) — the ceiling is still there"


def test_an_aim_vector_with_no_direction_is_refused():
    """A zero vector is not a direction, and silently treating it as one would put the part back on the default
    it was trying to escape."""
    from virturoid.services.anatomy_compiler import build_from_anatomy
    with pytest.raises(ValueError, match="no direction"):
        build_from_anatomy({"robot_class": "manipulator", "name": "z", "parts": [
            {"name": "base", "role": "body", "size": 0.3, "girth": 0.1},
            {"name": "arm", "role": "arm", "parent": "base", "aim": [0, 0, 0],
             "size": 0.3, "girth": 0.03}]})


def test_a_declared_fixed_joint_survives_a_wheel_role():
    """The compiler's own teaching error tells an agent that a TRACK is "like a wheel". The wheel branch ran
    before the explicit-joint check, so a tracked loader declaring two FIXED track units was handed two motors
    it never asked for — and an actuator count is exactly what our gates check, so nothing objected."""
    from virturoid.services.anatomy_compiler import build_from_anatomy
    graph = {"robot_class": "mobile_base", "name": "loader", "parts": [
        {"name": "hull", "role": "body", "size": 0.9, "girth": 0.3},
        {"name": "track", "role": "wheel", "parent": "hull", "attach": "front_bottom", "aim": "forward",
         "size": 0.6, "girth": 0.1, "symmetry": "left_right", "joint": "fixed"}]}
    g = build_from_anatomy(graph)
    tracks = [s for s in g.segments if s.name.startswith("track")]
    assert tracks, [s.name for s in g.segments]
    for s in tracks:
        assert s.joint_type is None, f"{s.name} was declared fixed and came back {s.joint_type!r}"


def test_a_wheel_that_declares_nothing_is_still_a_free_hinge():
    """The other half: the inference is right when nothing is said, and a rover must keep rolling. This is the
    behaviour the fix has to preserve while letting an explicit declaration win."""
    from virturoid.services.anatomy_compiler import build_from_anatomy
    g = build_from_anatomy({"robot_class": "mobile_base", "name": "rover", "parts": [
        {"name": "chassis", "role": "body", "size": 0.8, "girth": 0.34, "aspect": "deck"},
        {"name": "wheel", "role": "wheel", "parent": "chassis", "attach": "front_bottom",
         "size": 0.2, "girth": 0.08, "symmetry": "left_right"}]})
    wheels = [s for s in g.segments if s.name.startswith("wheel")]
    assert wheels and all(s.joint_type == "revolute" for s in wheels), \
        [(s.name, s.joint_type) for s in wheels]
