"""I3 (#220): the manipulator REACH verdict must gravity-compensate. A torque-actuated arm driven open-loop just
sags under gravity, so an AMENDED arm that embodied real motor mass read a dishonest STUCK -- the arm was fine;
the strawman controller ignored dynamics. The verdict now drives a gravity-compensated PD (like any real arm),
so the sweep tests the ARM's articulation, not a controller that can't hold it up.
"""
from __future__ import annotations

import importlib.util

import pytest

_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="the reach verdict needs MuJoCo")


def _compose(prompt):
    from virturoid.services.morphology_composer import compose_robot
    return compose_robot(prompt, llm=None)


def _reach(gene):
    from virturoid.services.ai_native_tools import _honest_reach
    return _honest_reach(gene)


def test_a_composed_arm_articulates_not_stuck():
    r = _reach(_compose("a 6-axis robot arm with a gripper"))
    assert r["kind"] == "manipulator"
    assert r["verdict"].startswith("ARTICULATES"), r["verdict"]
    assert r["reach_span_m"] > 0.02


def test_an_amended_longer_arm_still_articulates():
    """The #220 regression: scaling the arm links longer embodies more mass; the gravity-compensated verdict
    must still read ARTICULATES, never STUCK."""
    from virturoid.services.edit_operators import scale_group
    g = _compose("a 6-axis robot arm with a gripper")
    longer = scale_group(g, group="arms", dims="length", factor=1.5)
    if isinstance(longer, tuple):
        longer = longer[0]
    r = _reach(longer)
    assert r["verdict"].startswith("ARTICULATES"), f"amended arm read {r['verdict']!r} (the #220 dishonest STUCK)"
    assert r["reach_m"] >= 0.87                               # the longer arm reaches at least as far


def test_a_payload_loaded_arm_still_articulates():
    """Embodying a real payload must not flip the verdict to STUCK -- gravity comp carries it."""
    from virturoid.services.edit_operators import set_payload
    g = _compose("a 6-axis robot arm with a gripper")
    loaded = set_payload(g, payload_kg=3.0)
    if isinstance(loaded, tuple):
        loaded = loaded[0]
    assert _reach(loaded)["verdict"].startswith("ARTICULATES")
