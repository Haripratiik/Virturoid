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
