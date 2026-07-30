"""A design can declare the pose it RESTS in, not just the range it can move through.

The part language gained joint type, axis and limits, so a SCARA's elbows turn about Z and a gantry's carriage
slides a bounded stroke. What it never gained is the fourth thing on that list: a rest angle. Every joint a
machine has therefore sat at 0, and the effect is only visible if you LOOK at the body rather than count its
degrees of freedom.

Measured 2026-07-30 by rendering all eight non-animal benchmark families (build/inspect/expressivity/): the
excavator passed its DOF check 4/4 and rendered as a pipe lying flat on the ground, because boom, stick and
bucket all rest at 0 and a chain of forward-aimed links at 0 is a straight horizontal line. Declaring a stance
on the same body stood it up, 0.025 -> 1.248 m tall.

That is the failure mode Articraft names -- "poor global shape quality despite passing local structural checks"
-- and it is why the plan's Phase 2 gate says to look at the renders rather than trust the count.

The rest angle belongs to the AUTHOR, not to a per-class table in our code: an excavator parked with its bucket
down and one loading a truck are the same machine in different stances, and only the designer knows which is
meant. Legs keep their derived knee bend, because that one is structural (a body must stand).
"""
from __future__ import annotations

import importlib.util
import os

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("VIRTUROID_DISABLE_GAIT_HINTS", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="compiling a body needs MuJoCo")

_EXCAVATOR = {"robot_class": "manipulator", "name": "excavator", "parts": [
    {"name": "house", "role": "body", "size": 0.9, "girth": 0.30},
    {"name": "slew", "role": "slew_ring", "like": "neck", "parent": "house", "attach": "front_top",
     "aim": "up", "size": 0.16, "girth": 0.20, "joint": "revolute", "axis": [0, 0, 1],
     "lower": -3.1, "upper": 3.1},
    {"name": "boom", "role": "boom", "like": "arm", "parent": "slew", "aim": "forward",
     "size": 0.95, "girth": 0.07, "joint": "revolute", "axis": [0, 1, 0], "lower": -0.2, "upper": 1.2},
    {"name": "stick", "role": "stick", "like": "arm", "parent": "boom", "aim": "forward",
     "size": 0.70, "girth": 0.05, "joint": "revolute", "axis": [0, 1, 0], "lower": -2.2, "upper": 0.2},
    {"name": "bucket", "role": "bucket", "like": "hand", "parent": "stick", "aim": "forward",
     "size": 0.32, "girth": 0.15, "joint": "revolute", "axis": [0, 1, 0], "lower": -2.6, "upper": 0.4}]}


def _staged(**rest):
    import copy
    g = copy.deepcopy(_EXCAVATOR)
    for p in g["parts"]:
        if p["name"] in rest:
            p["rest"] = rest[p["name"]]
    return g


def _posed_height(gene):
    """Height span of the body IN THE POSE IT SHIPS — the keyframe, not qpos0."""
    import mujoco
    from virturoid.services.morph_policy import compiled_model, robot_mjcf
    m = compiled_model(robot_mjcf(gene))
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    zs = [float(d.xpos[b][2]) for b in range(1, m.nbody)]
    return max(zs) - min(zs)


def test_a_declared_rest_angle_reaches_the_compiled_keyframe():
    """The whole chain: graph -> gene metadata -> MJCF <keyframe> -> the pose MuJoCo actually loads."""
    import mujoco
    from virturoid.services.anatomy_compiler import build_from_anatomy
    from virturoid.services.morph_policy import compiled_model, robot_mjcf
    gene = build_from_anatomy(_staged(boom=-0.15, stick=-1.6, bucket=-1.2))
    assert gene.metadata["rest_pose"]["stick_joint"] == pytest.approx(-1.6)
    m = compiled_model(robot_mjcf(gene))
    assert m.nkey, "a declared stance must be emitted as a keyframe"
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "stick_joint")
    assert float(d.qpos[m.jnt_qposadr[j]]) == pytest.approx(-1.6, abs=1e-6)


def test_the_stance_is_what_stands_the_machine_up():
    """THE measurement, and the reason this exists. Same body, same DOF, same joint types — the only difference
    is whether the designer said where it rests."""
    from virturoid.services.anatomy_compiler import build_from_anatomy
    flat = _posed_height(build_from_anatomy(_EXCAVATOR))
    stood = _posed_height(build_from_anatomy(_staged(boom=-0.15, stick=-1.6, bucket=-1.2)))
    assert stood > flat * 1.8, (
        f"declaring a stance changed the body's height from {flat:.3f} m to {stood:.3f} m — a forward-aimed "
        "chain resting at 0 is a straight horizontal line, and that is what a customer sees")


def test_a_rest_angle_outside_the_declared_limits_is_clamped_not_obeyed():
    """A stance MuJoCo cannot hold is worse than none: the solver would snap the joint on the first step and the
    body a customer was shown would not be the body that simulates. Clamp to the range the same design declared."""
    import mujoco
    from virturoid.services.anatomy_compiler import build_from_anatomy
    from virturoid.services.morph_policy import compiled_model, robot_mjcf
    gene = build_from_anatomy(_staged(stick=-9.9))            # declared range is [-2.2, 0.2]
    assert gene.metadata["rest_pose"]["stick_joint"] == pytest.approx(-2.2)
    m = compiled_model(robot_mjcf(gene))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "stick_joint")
    lo, hi = (float(v) for v in m.jnt_range[j])
    assert lo - 1e-6 <= float(d.qpos[m.jnt_qposadr[j]]) <= hi + 1e-6


def test_a_prismatic_stroke_can_declare_its_parked_position():
    """Not only revolute: a gantry ram or a rail carriage parks somewhere along its stroke, and 0 is rarely it."""
    import mujoco
    from virturoid.services.anatomy_compiler import build_from_anatomy
    from virturoid.services.morph_policy import compiled_model, robot_mjcf
    gene = build_from_anatomy({"robot_class": "rail", "name": "rail", "parts": [
        {"name": "rail", "role": "rail", "like": "body", "size": 1.5, "girth": 0.05},
        {"name": "carriage", "role": "carriage", "like": "body", "parent": "rail",
         "attach": {"along": 0.5, "height": 1.0}, "aim": "up", "size": 0.18, "girth": 0.07,
         "joint": "prismatic", "axis": [1, 0, 0], "lower": -0.65, "upper": 0.65, "rest": -0.5}]})
    m = compiled_model(robot_mjcf(gene))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "carriage_joint")
    assert float(d.qpos[m.jnt_qposadr[j]]) == pytest.approx(-0.5, abs=1e-6)


def test_a_leg_keeps_its_derived_knee_bend_when_nothing_is_declared():
    """The derived poses are STRUCTURAL — a body has to stand — so they must survive the new opt-in. Only a part
    that says `rest` gets to override, and the regression surface here is every animal ever composed."""
    from virturoid.services.morphology_composer import compose_robot
    dog = compose_robot("a four legged robot dog", llm=None)
    pose = (dog.metadata or {}).get("rest_pose") or {}
    assert pose, "the composed quadruped lost its rest pose"
    assert any(abs(float(v)) > 1e-6 for v in pose.values()), f"every joint flattened to 0: {pose}"
