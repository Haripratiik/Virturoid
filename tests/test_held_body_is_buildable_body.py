"""Stage 2.1 invariant: the robot we VERIFY is the robot we EXPORT.

Mass/torque grounding used to run only at export/build time, so `create_robot` held an ungrounded body while its
own export shipped a grounded one -- measured, a "robot dog" was held at 3.57 kg (-76% vs a Go2's 15 kg) while
`export_held` shipped 13.50 kg. Every CREDIBLE WALK verdict was therefore earned on a styrofoam twin of the robot
the customer was told to build (a 3.8x mass split), and the walk verdict did not transfer to the shipped package.

Grounding now happens when the body is composed and held, so one robot flows through verify -> train -> evaluate
-> render -> export. These tests pin BOTH halves: the held body is grounded (realistic mass), and re-grounding it
(what export does) is a no-op.
"""
from __future__ import annotations

import copy
import importlib.util
import os

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="composition + verification need MuJoCo")


def _held_mass(gene) -> float:
    return sum(float(getattr(s, "mass_kg", 0.0) or 0.0) for s in gene.segments)


@pytest.mark.parametrize(
    ("prompt", "lo_kg", "hi_kg"),
    [
        # Reference bands, generously wide -- this pins "not a toy", not a specific number.
        # Unitree Go2 is 15 kg; Franka FR3 is 18.3 kg; a warehouse AMR (MiR250) is 94 kg.
        ("a robot dog", 8.0, 25.0),
        ("a 6-axis robot arm with a gripper", 8.0, 30.0),
        ("a four-wheeled warehouse rover", 30.0, 200.0),
    ],
)
def test_held_body_has_a_realistic_mass(prompt, lo_kg, hi_kg, tmp_path, monkeypatch):
    monkeypatch.setenv("VIRTUROID_SESSIONS_DIR", str(tmp_path))
    from virturoid.services import session_state as S
    from virturoid.services.agent_tools import call_tool

    rid = call_tool("create_robot", {"prompt": prompt})["result"]["robot_id"]
    mass = _held_mass(S.get_robot(rid))
    assert lo_kg <= mass <= hi_kg, (
        f"{prompt!r} is held at {mass:.2f} kg, outside the realistic band {lo_kg}-{hi_kg} kg. "
        "A body this far off real mass makes every downstream torque, contact and traction number wrong.")


@pytest.mark.parametrize("prompt", ["a robot dog", "a 6-axis robot arm with a gripper", "a hexapod robot"])
def test_the_held_body_is_already_the_exported_body(prompt, tmp_path, monkeypatch):
    """Re-grounding the held gene (exactly what export_held does) must not change it."""
    monkeypatch.setenv("VIRTUROID_SESSIONS_DIR", str(tmp_path))
    from virturoid.services import session_state as S
    from virturoid.services.agent_tools import call_tool
    from virturoid.services.gene_build import ground_and_repair

    rid = call_tool("create_robot", {"prompt": prompt})["result"]["robot_id"]
    held = S.get_robot(rid)
    held_mass = _held_mass(held)

    exported = copy.deepcopy(held)
    ground_and_repair(exported)                      # the export path's own grounding
    export_mass = _held_mass(exported)

    assert abs(export_mass - held_mass) < 0.05, (
        f"{prompt!r}: verified body is {held_mass:.2f} kg but the exported body is {export_mass:.2f} kg "
        f"({export_mass / max(held_mass, 1e-9):.2f}x) -- the verdict was earned on a different robot.")


@pytest.mark.parametrize(
    "prompt",
    ["a robot dog", "a hexapod robot", "a 6-axis robot arm with a gripper", "a four-wheeled warehouse rover"],
)
def test_every_link_can_house_the_motor_that_drives_it(prompt, tmp_path, monkeypatch):
    """Visual-fidelity defect T1. Actuator housings are drawn at their true datasheet envelope, but nothing
    forced the LIMB to be able to carry that part -- the compiler emitted a 5.6 mm rod mounting a 98 mm-diameter
    T-Motor AK10-9 (housing/limb 8.86x on a hexapod hip, 3.23x on the dog). That is physically impossible and it
    is the dominant "toy" cue in every render: fat drums threaded onto pencil sticks."""
    monkeypatch.setenv("VIRTUROID_SESSIONS_DIR", str(tmp_path))
    from virturoid.services import session_state as S
    from virturoid.services.agent_tools import call_tool
    from virturoid.services.component_geometry import actuator_for_joint

    rid = call_tool("create_robot", {"prompt": prompt})["result"]["robot_id"]
    worst, worst_name = 0.0, ""
    for seg in S.get_robot(rid).segments:
        ds = actuator_for_joint(seg)
        limb_r = float(getattr(seg, "radius_m", 0.0) or 0.0)
        if ds is None or limb_r <= 0.0:
            continue
        env = ds["envelope_m"]
        ratio = (max(float(env[0]), float(env[1])) / 2.0) / limb_r
        if ratio > worst:
            worst, worst_name = ratio, seg.name
    assert worst <= 1.55, (
        f"{prompt!r}: {worst_name} has a housing/limb ratio of {worst:.2f}x -- the motor is far wider than the "
        "link it bolts to, which cannot be built and reads as a toy.")


def test_grounding_the_held_body_did_not_cost_the_walk(tmp_path, monkeypatch):
    """The honest failure mode of grounding would be 'now it's realistic but it can't move'. It isn't:
    measured, the grounded dog walks FURTHER than the styrofoam one (1.70 m vs 0.64 m)."""
    monkeypatch.setenv("VIRTUROID_SESSIONS_DIR", str(tmp_path))
    from virturoid.services.agent_tools import call_tool

    rid = call_tool("create_robot", {"prompt": "a robot dog"})["result"]["robot_id"]
    verdict = call_tool("verify_robot", {"robot_id": rid, "mode": "quick"})["result"]
    assert "CREDIBLE" in str(verdict.get("verdict", "")), (
        f"the grounded dog no longer walks: {verdict.get('verdict')}")
