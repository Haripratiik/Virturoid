"""M11 (2026-07-24 audit): run_task and create_scene must fail honestly on impossible/unknown input instead of
silently scoring a fake success. `run_task(goal="fly to the moon")` returned success 1.0 (planner mapped any
goal to the body's default task); `create_scene(theme="marsbase")` silently built a warehouse labelled marsbase.
"""
from __future__ import annotations

import importlib.util

import pytest

_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="needs MuJoCo to compose a robot")


def _dog():
    from virturoid.services.agent_tools import call_tool
    return call_tool("create_robot", {"prompt": "a robot dog"})["result"]["robot_id"]


def test_impossible_goal_is_honestly_infeasible():
    from virturoid.services.agent_tools import call_tool
    rid = _dog()
    for goal in ("fly to the moon and back", "reach orbit", "teleport to the target", "phase through the wall"):
        r = call_tool("run_task", {"robot_id": rid, "goal": goal})["result"]
        assert r["feasible"] is False and r["success"] is False and r["score"] == 0.0, (goal, r)
        assert r["task"] == "out_of_domain" and r["issues"]


def test_legitimate_goals_still_run():
    from virturoid.services.agent_tools import call_tool
    rid = _dog()
    r = call_tool("run_task", {"robot_id": rid, "goal": "walk forward to the goal"})["result"]
    assert r["feasible"] is True                                  # a real locomotion goal is not over-rejected
    # a charitable go-to phrased as "fly to the target" on a legged body is still reinterpreted as locomote
    r2 = call_tool("run_task", {"robot_id": rid, "goal": "fly to the target zone"})["result"]
    assert r2["feasible"] is True and r2["task"] == "locomote"


def test_unknown_scene_theme_is_rejected():
    from virturoid.services.agent_tools import call_tool
    r = call_tool("create_scene", {"theme": "marsbase", "task": "navigation"})
    res = r.get("result", r)
    assert res.get("ok") is False and "unknown theme" in str(res.get("error", ""))
    ok = call_tool("create_scene", {"theme": "warehouse", "task": "navigation"}).get("result", {})
    assert ok.get("ok") is True and ok.get("scene_id")           # a known theme still builds
