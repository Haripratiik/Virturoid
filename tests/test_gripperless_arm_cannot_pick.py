"""A skill that has to HOLD an object is infeasible on a body with nothing to hold it with.

MEASURED DEFECT (2026-10-11, through the MCP server): an imported UR5e ships no gripper, and ``verify_robot`` said
so ("NO GRIPPER in your model"). ``run_task`` then called pick_place FEASIBLE on the same robot and ran it with
the block pinned to the bare wrist flange: the morphology gate only asked whether the body was a manipulator,
and a UR5e is one. The task verifier now asks the question the skill depends on, through the same detector
``verify_robot`` and the importer read (``services/gripper_jaws.py``), so the three cannot disagree about a body.

Composed bodies are covered both ways: a composed arm with a gripper keeps pick_place, and the schema SCARA,
whose quill carries no gripper, loses it, with the reason.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

from virturoid.schemas.task_spec import Entity, Predicate, PredicateOp, SkillCall, TaskSpec

_MUJOCO = importlib.util.find_spec("mujoco") is not None
_MENAGERIE = os.path.join(os.path.expanduser("~"), ".cache", "robot_descriptions", "mujoco_menagerie")


def _pick_spec(skill: str = "pick_place"):
    op = PredicateOp.OBJECTS_PLACED if skill == "pick_place" else PredicateOp.OBJECT_LIFTED
    return TaskSpec(name="t", prompt="pick it up", entities=[Entity("cube", "object")],
                    steps=[SkillCall("s0", skill, {})], goal=[Predicate(op, [])], source="test")


def _arm(declared: str, *, jaws: bool):
    from virturoid.schemas.gene import GeneSegment, RobotGene
    segs = [GeneSegment("base", None, "box", 0.1, 0.05, 1.0),
            GeneSegment("l1", "base", "capsule", 0.3, 0.03, 0.5, "revolute", joint_lower=-2.0, joint_upper=2.0),
            GeneSegment("flange", "l1", "box", 0.05, 0.03, 0.2, "revolute", joint_lower=-2.0, joint_upper=2.0,
                        is_end_effector=True)]
    if jaws:
        segs += [GeneSegment("jaw_a", "flange", "box", 0.05, 0.01, 0.03, "prismatic", joint_lower=0.0,
                             joint_upper=0.04),
                 GeneSegment("jaw_b", "flange", "box", 0.05, 0.01, 0.03, "prismatic", joint_lower=0.0,
                             joint_upper=0.04)]
    return RobotGene(id="arm", species="test.arm", robot_class="manipulator", segments=segs,
                     end_effector_type=declared)


@pytest.mark.parametrize("skill", ["pick_place", "grasp_lift"])
def test_a_bare_flange_cannot_run_a_grasping_skill(skill):
    from virturoid.services.task_verifier import verify_task
    v = verify_task(_pick_spec(skill), _arm("none", jaws=False))
    assert not v["ok"]
    msg = " ".join(v["issues"])
    assert f"skill {skill!r} has to hold the object" in msg and "has no gripper" in msg
    assert "'flange'" in msg, "the issue names the end effector it looked at"


def test_jaws_the_declaration_forgot_still_count():
    """A body held from before the importer could see jaws ("none" + sliding jaws) is not refused."""
    from virturoid.services.task_verifier import verify_task
    assert verify_task(_pick_spec(), _arm("none", jaws=True))["ok"]


@pytest.mark.parametrize("declared", ["gripper", "suction"])
def test_a_declared_holding_tool_keeps_the_skill(declared):
    from virturoid.services.task_verifier import verify_task
    assert verify_task(_pick_spec(), _arm(declared, jaws=False))["ok"]


def test_the_vocabulary_says_which_skills_need_a_gripper():
    from virturoid.services.agent_design_tools import list_skills
    from virturoid.services.capability_registry import skill_menu
    needs = {s["skill"]: s["needs_gripper"] for s in list_skills({})["skills"]}
    assert needs["pick_place"] and needs["grasp_lift"] and not needs["navigate"] and not needs["locomote"]
    assert "pick_place" in skill_menu() and "and a gripper" in skill_menu()


@pytest.mark.skipif(not _MUJOCO, reason="composing needs MuJoCo")
def test_composed_bodies_both_ways():
    """The composed arm with a gripper keeps pick_place; the schema SCARA (a quill, no gripper) loses it."""
    from virturoid.services.agent_design_tools import get_design_schema
    from virturoid.services.anatomy_compiler import build_from_anatomy
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.task_verifier import verify_task
    arm = compose_robot("a 6-DOF robot arm with a gripper", llm=None)
    assert verify_task(_pick_spec(), arm)["ok"]
    scara = build_from_anatomy(get_design_schema({})["examples"]["scara_arm"])
    v = verify_task(_pick_spec(), scara)
    assert not v["ok"] and any("has no gripper" in i for i in v["issues"]), v


@pytest.mark.skipif(not _MUJOCO, reason="import + verify need MuJoCo")
def test_the_imported_ur5e_is_refused_with_the_same_words_verify_uses():
    p = os.path.join(_MENAGERIE, "universal_robots_ur5e")
    if not os.path.exists(p):
        pytest.skip(f"MuJoCo Menagerie not cached at {p}")
    from virturoid.services.agent_tools import call_tool
    env = call_tool("ingest_project", {"project_path": p, "description": "a UR5e arm"})
    assert env["ok"], env
    rid = env["result"]["robot_id"]
    run = call_tool("run_task", {"robot_id": rid,
                                 "goal": "pick up the cube in front of you and place it 20 cm to the left"})["result"]
    assert run["feasible"] is False and run["success"] is False and run["steps"] == []
    assert any("has no gripper" in i and "wrist_3_link" in i for i in run["issues"]), run["issues"]
    ver = call_tool("verify_robot", {"robot_id": rid})["result"]
    assert "NO GRIPPER in your model" in ver["verdict"], ver["verdict"]
    assert ver["grasp"]["gripper"]["grips"] is False
