"""run_task says what its success means when the goal names a place it did not simulate.

MEASURED 2026-10-11 through the MCP server: "walk to the far end of the warehouse aisle" on a composed quadruped
returned success 1.0 after 1.07 m on open ground. The locomote skill has no goal position and establishes
reached_goal by 0.3 m of upright forward travel, and run_task takes no scene, so a goal naming a place is scored as
a walk. The success stands (the robot did walk); what it was a success AT is now said beside it.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
pytestmark = pytest.mark.skipif(importlib.util.find_spec("mujoco") is None, reason="needs MuJoCo")


@pytest.fixture(scope="module")
def walker():
    from virturoid.services.agent_design_tools import get_design_schema, submit_design
    return submit_design({"graph": get_design_schema({})["examples"]["quadruped"]})["robot_id"]


def _run(rid, goal):
    from virturoid.services.agent_tools import call_tool
    env = call_tool("run_task", {"robot_id": rid, "goal": goal})
    assert env["ok"], env
    return env["result"]


def test_a_goal_naming_a_place_is_told_the_place_was_not_simulated(walker):
    r = _run(walker, "walk to the far end of the warehouse aisle")
    assert r["success"] is True, r
    assert any("did not simulate" in i for i in r["issues"]), r["issues"]
    checked = r["steps"][0]["detail"]["checked"]
    assert "no goal position" in checked and "0.3 m" in checked, checked


def test_a_plain_walk_carries_no_destination_warning(walker):
    r = _run(walker, "walk forward")
    assert not any("did not simulate" in i for i in r["issues"]), r["issues"]


@pytest.mark.parametrize("goal, names_a_place", [
    ("walk to the far end of the warehouse aisle", True), ("go to the door", True), ("reach the shelf", True),
    ("walk towards the exit", True), ("walk forward", False), ("walk 2 m", False), ("trot fast", False)])
def test_destination_phrasing(goal, names_a_place):
    from virturoid.services.agent_design_tools import _names_a_destination
    assert _names_a_destination(goal) is names_a_place
