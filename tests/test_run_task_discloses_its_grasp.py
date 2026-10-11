"""run_task carries its own disclosure to the agent: per-step metric and detail, the grasp model, and a success that
never rests on a grasp that was not simulated.

MEASURED DEFECT (2026-10-11, the "bring your own arm" journey through the MCP server): on an ingested Franka
Panda, "pick up the cube in front of you and place it 20 cm to the left" ran the scripted pick-place with the
IDEALIZED pin (the block welded to the hand the moment they touched) and placed 1 block in 4. The executor
knew both facts (``grasp_model``, ``grasp_success_rate``) and each skill reported its own metric and detail; the
MCP tool rebuilt every step as ``{skill, success, detail}`` with ``detail`` always null, and dropped the rest. The
agent saw ``success: false, detail: null``. At 2 blocks in 4 it would have seen a plain ``success: true`` for a
grasp that never happened.

Most of these run on stub skills (no physics): they pin the SHAPE of what reaches the agent. One runs the real
skill on the real Panda when the Menagerie is cached.
"""
from __future__ import annotations

import importlib.util
import math
import os

import pytest

from virturoid.schemas.task_spec import Entity, Predicate, PredicateOp, SkillCall, TaskSpec

_MUJOCO = importlib.util.find_spec("mujoco") is not None
_MENAGERIE = os.path.join(os.path.expanduser("~"), ".cache", "robot_descriptions", "mujoco_menagerie")


def _spec():
    return TaskSpec(name="pp", prompt="pick the cube and place it", entities=[Entity("cube", "object")],
                    steps=[SkillCall("s0", "pick_place", {})], goal=[Predicate(PredicateOp.OBJECTS_PLACED, [])],
                    source="test")


def _stub_pick_place(monkeypatch, *, grasp_model: str, rate: float):
    """Swap pick_place's runner for one that reports what the real skill reports, without the physics."""
    from virturoid.services import capability_registry as C
    import dataclasses

    def runner(gene, params):
        ok = rate >= 0.5
        detail = {"task_type": "place_to_target", "grasp_model": grasp_model, "success_rate": rate,
                  "blocks_placed": int(rate * 4), "blocks_total": 4,
                  "episodes": [{"status": "success" if i < rate * 4 else "failure",
                                "failure_label": None if i < rate * 4 else "missed_grasp",
                                "metrics": {"trace": list(range(500))}} for i in range(4)],
                  "frames": [[0.0] * 64 for _ in range(10)], "not_a_number": float("nan"),
                  "pin_because": "the real contact grasp ran first and placed 0 of 3 blocks"}
        return C.SkillOutcome("pick_place", ok, {PredicateOp.OBJECTS_PLACED} if ok else set(), rate, detail)
    monkeypatch.setitem(C.REGISTRY, "pick_place", dataclasses.replace(C.REGISTRY["pick_place"], runner=runner))


def _gripper_arm():
    from virturoid.schemas.gene import GeneSegment, RobotGene
    segs = [GeneSegment("base", None, "box", 0.1, 0.05, 1.0),
            GeneSegment("l1", "base", "capsule", 0.3, 0.03, 0.5, "revolute", joint_lower=-2.0, joint_upper=2.0),
            GeneSegment("palm", "l1", "box", 0.05, 0.03, 0.2, "revolute", joint_lower=-2.0, joint_upper=2.0,
                        is_end_effector=True),
            GeneSegment("jaw_a", "palm", "box", 0.05, 0.01, 0.03, "prismatic", joint_lower=0.0, joint_upper=0.04),
            GeneSegment("jaw_b", "palm", "box", 0.05, 0.01, 0.03, "prismatic", joint_lower=0.0, joint_upper=0.04)]
    return RobotGene(id="arm", species="test.arm", robot_class="manipulator", segments=segs,
                     end_effector_type="gripper")


# ------------------------------------------------------------------------------------------- the executor
def test_the_executor_keeps_each_steps_metric_and_a_compact_detail(monkeypatch):
    from virturoid.services.task_executor import run_task
    _stub_pick_place(monkeypatch, grasp_model="idealized_pin", rate=0.25)
    r = run_task(_spec(), _gripper_arm())
    step = r["steps"][0]
    assert step["metric"] == 0.25 and step["grasp_model"] == "idealized_pin"
    d = step["detail"]
    assert d["blocks_placed"] == 1 and d["blocks_total"] == 4 and d["success_rate"] == 0.25
    assert d["episodes_n"] == 4 and d["episodes_outcomes"] == "missed_grasp: 3, success: 1"
    assert "frames" not in d and "episodes" not in d, "frames and per-episode records stay with the skill"
    assert d["not_a_number"] is None, "a non-finite number must not reach a JSON client as NaN"
    assert r["idealized_grasp"] is True and r["goal_met_without_idealized_grasp"] == 0


def test_the_compact_detail_keeps_scalars_and_short_strings_only():
    from virturoid.services.task_executor import _compact_detail
    import numpy as np
    d = _compact_detail({"a": np.float64(0.123456), "b": np.int64(3), "c": np.bool_(True), "d": "x" * 500,
                         "e": np.zeros((4, 4)), "f": {"nested": 1}, "g": math.inf, "h": None})
    assert d["a"] == 0.1235 and d["b"] == 3 and d["c"] is True and d["h"] is None and d["g"] is None
    assert len(d["d"]) <= 200 and d["d"].endswith("...")
    assert "e" not in d and "f" not in d


# --------------------------------------------------------------------------------------- the agent's view
def test_a_pinned_placement_is_not_a_success_even_when_the_predicate_passes(monkeypatch):
    """3 in 4 placed with the pin: the goal predicate holds, and the agent must still read success false."""
    from virturoid.services.agent_design_tools import _task_outcome_for_agent
    from virturoid.services.task_executor import run_task
    _stub_pick_place(monkeypatch, grasp_model="idealized_pin", rate=0.75)
    raw = run_task(_spec(), _gripper_arm())
    assert raw["success"] is True                          # the executor still scores the predicate
    out = _task_outcome_for_agent(raw)
    assert out["success"] is False and out["score"] == 0.0 and out["goal_met"] == 0
    assert out["with_idealized_grasp"] == {"success": True, "score": 1.0, "goal_met": 1}
    assert out["grasp_model"] == "idealized_pin" and out["grasp_success_rate"] == 0.75
    assert out["grasp"]["simulated"] is False
    assert out["grasp"]["blocks_placed"] == 3 and out["grasp"]["blocks_total"] == 4
    assert "NOT SIMULATED" in out["grasp"]["means"]
    assert out["grasp"]["why_pinned"].startswith("the real contact grasp ran first")
    assert out["warning"].startswith("IDEALIZED GRASP")
    step = out["steps"][0]
    assert step["success"] is False and step["success_with_idealized_grasp"] is True
    assert step["metric"] == 0.75 and step["detail"]["blocks_placed"] == 3


def test_a_contact_grasp_success_stays_a_success(monkeypatch):
    from virturoid.services.agent_design_tools import _task_outcome_for_agent
    from virturoid.services.task_executor import run_task
    _stub_pick_place(monkeypatch, grasp_model="contact", rate=0.75)
    out = _task_outcome_for_agent(run_task(_spec(), _gripper_arm()))
    assert out["success"] is True and out["score"] == 1.0
    assert out["grasp"] == {"model": "contact", "simulated": True, "success_rate": 0.75, "blocks_placed": 3,
                            "blocks_total": 4}
    assert "warning" not in out and "with_idealized_grasp" not in out


def test_submit_task_carries_the_same_disclosure(monkeypatch):
    from virturoid.services import session_state as S
    from virturoid.services.agent_design_tools import submit_task
    _stub_pick_place(monkeypatch, grasp_model="idealized_pin", rate=0.75)
    rid = S.put_robot(_gripper_arm(), prompt="arm", label="arm")
    try:
        out = submit_task({"robot_id": rid, "steps": [{"skill": "pick_place"}],
                           "goal": [{"op": PredicateOp.OBJECTS_PLACED.value}]})
    finally:
        S.forget_robot(rid)
    assert out["ok"] and out["success"] is False and out["with_idealized_grasp"]["success"] is True
    assert out["steps"][0]["metric"] == 0.75 and out["grasp_model"] == "idealized_pin"


# ------------------------------------------------------------------------------------ the real arm, real skill
@pytest.mark.skipif(not _MUJOCO, reason="the pick-place skill needs MuJoCo")
def test_the_ingested_panda_tells_the_agent_which_grasp_ran():
    p = os.path.join(_MENAGERIE, "franka_emika_panda")
    if not os.path.exists(p):
        pytest.skip(f"MuJoCo Menagerie not cached at {p}")
    from virturoid.services.agent_tools import call_tool
    env = call_tool("ingest_project", {"project_path": p, "description": "a Franka Emika Panda arm with its hand"})
    assert env["ok"], env
    out = call_tool("run_task", {"robot_id": env["result"]["robot_id"],
                                 "goal": "pick up the cube in front of you and place it 20 cm to the left"})
    r = out["result"]
    assert r["feasible"] is True
    assert r["grasp_model"] in ("contact", "idealized_pin")
    assert isinstance(r["grasp_success_rate"], float)
    step = r["steps"][0]
    assert step["skill"] == "pick_place" and isinstance(step["metric"], float) and step["detail"]
    if r["grasp_model"] == "idealized_pin":
        assert r["success"] is False and "IDEALIZED GRASP" in r["warning"]
        assert r["grasp"]["why_pinned"]
