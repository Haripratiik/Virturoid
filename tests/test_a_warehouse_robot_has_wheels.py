"""A robot described only by the warehouse it works in is a wheeled base, not the creature default.

MEASURED 2026-10-11: "a warehouse robot" and "a large heavy-duty warehouse robot" named no class and no task, fell
through the router's no-cue default and composed a 20-segment quadruped. The site is read only in that fallback,
so a prompt that does name a body or a task keeps it.
"""
from __future__ import annotations

import pytest


@pytest.mark.parametrize("prompt, cls", [
    ("a warehouse robot", "mobile_base"),
    ("a large heavy-duty warehouse robot", "mobile_base"),
    ("a robot for a logistics centre", "mobile_base"),
    ("a warehouse robot arm to move boxes", "manipulator"),
    ("a four-legged walking robot that patrols a warehouse", "quadruped"),
    ("a giraffe robot", "quadruped"),
])
def test_the_site_decides_only_when_nothing_else_does(prompt, cls):
    from virturoid.services.intent_planner import _heuristic_plan
    assert _heuristic_plan(prompt).robot_class == cls
