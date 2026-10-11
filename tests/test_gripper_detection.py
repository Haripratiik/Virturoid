"""Whether a body carries a gripper is read off its STRUCTURE, by one detector every surface shares.

MEASURED DEFECT (2026-10-11, through the MCP server): the importer declared ``end_effector_type`` from segment
NAMES ("grip" in a name), so a Franka Panda (``hand``, ``left_finger``, ``right_finger``) imported as
"none". ``capability_registry._run_pick_place`` therefore never tried the real contact grasp on it and pinned
the block instead, while ``verify_robot`` (whose detector is the grasp site, i.e. sliding jaws on the end
effector) attempted a grasp on the same robot in the same session. The Piper and ARX L5 were worse off: the
importer flagged a FINGER (``link8``) as the end effector, so even the grasp site was never emitted and verify
told those customers "NO GRIPPER in your model".

These tests pin: the detector's cases on hand-built bodies (no MuJoCo needed), the agreement between the
detector and the compiler's grasp site, the real Menagerie arms it was written for (skipped when the cache is
absent), and, just as important, the bodies it must leave alone: a ToddlerBot's gear-coupled elbows, legged
robots, humanoids and dexterous hands keep the end-effector type, the end effector and the kind they had.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

from virturoid.schemas.gene import GeneSegment, RobotGene
from virturoid.services.gripper_jaws import gripper_status, jaws_on, palm_and_jaws

_MUJOCO = importlib.util.find_spec("mujoco") is not None
_MENAGERIE = os.path.join(os.path.expanduser("~"), ".cache", "robot_descriptions", "mujoco_menagerie")


def _seg(name, parent, joint=None, **kw):
    return GeneSegment(name=name, parent=parent, shape="box", length_m=kw.pop("length_m", 0.1), radius_m=0.02,
                       mass_kg=0.2, joint_type=joint, joint_lower=-1.0 if joint else None,
                       joint_upper=1.0 if joint else None, **kw)


def _arm(*tail, ee: str, coupled=None, declared="none"):
    segs = [_seg("base", None), _seg("l1", "base", "revolute"), _seg("l2", "l1", "revolute"), *tail]
    for s in segs:
        s.is_end_effector = s.name == ee
    return RobotGene(id="t", species="test.arm", robot_class="manipulator", segments=segs,
                     coupled_joints=list(coupled or []), end_effector_type=declared)


# ------------------------------------------------------------------------------------- the detector's cases
def test_sliding_jaws_on_the_flagged_palm_are_closable():
    g = _arm(_seg("palm", "l2"), _seg("jaw_a", "palm", "prismatic"), _seg("jaw_b", "palm", "prismatic"), ee="palm")
    st = gripper_status(g)
    assert st["grips"] and st["closable"] and st["palm"] == "palm"
    assert st["jaws"] == ["jaw_a", "jaw_b"]


def test_a_flag_on_one_finger_is_walked_back_to_the_palm():
    """The Piper / ARX L5 case: the importer's last-leaf fallback lands on a jaw."""
    g = _arm(_seg("palm", "l2"), _seg("jaw_a", "palm", "prismatic"), _seg("jaw_b", "palm", "prismatic"), ee="jaw_b")
    palm, jaws = palm_and_jaws(g.segments, "jaw_b")
    assert palm == "palm" and [j.name for j in jaws] == ["jaw_a", "jaw_b"]
    st = gripper_status(g)
    assert st["grips"] and not st["closable"], "the grasp site is only emitted when the PALM carries the flag"


def test_jaws_under_a_rigid_adaptor_are_found():
    """The ALOHA case: the fingers hang under a fixed `gripper_base` bolted to the flagged link."""
    g = _arm(_seg("wrist", "l2"), _seg("adaptor", "wrist"), _seg("jaw_a", "adaptor", "prismatic"),
             _seg("jaw_b", "adaptor", "prismatic"), ee="wrist")
    assert palm_and_jaws(g.segments, "wrist")[0] == "adaptor"


def test_a_lone_prismatic_leaf_beside_the_flag_is_not_called_a_jaw():
    """Cases 2 and 3 need a PAIR: a single slide one level away is as likely a linear axis as a jaw."""
    g = _arm(_seg("palm", "l2"), _seg("slide", "palm", "prismatic"), ee="slide")
    assert palm_and_jaws(g.segments, "slide") == ("slide", [])


def test_a_robotiq_style_coupled_pair_on_a_rigid_palm_is_a_gripper_the_script_cannot_close():
    """Two revolute drivers the source couples 1:1, on a rigid palm an arm leads to (Menagerie Tidybot)."""
    pair = {"a": "drv_r", "b": "drv_l", "ratio": 1.0, "offset": 0.0}
    g = _arm(_seg("gbase", "l2"), _seg("drv_r", "gbase", "revolute"), _seg("drv_l", "gbase", "revolute"),
             _seg("pad", "drv_l"), ee="pad", coupled=[pair])
    st = gripper_status(g)
    assert st["grips"] and st["palm"] == "gbase" and not st["closable"]
    assert "coupled revolute" in st["evidence"]


def test_gear_coupled_siblings_are_not_jaws():
    """ToddlerBot couples elbow/shoulder links to sibling GEAR links 1:1, structurally a jaw pair. The palm of
    such a pair is itself revolute (an arm joint), and its neck pair sits on the root torso no arm leads to."""
    on_revolute_palm = _arm(_seg("elbow", "l2", "revolute"), _seg("yaw", "elbow", "revolute"),
                            _seg("gear", "elbow", "revolute"), _seg("hand", "yaw"), ee="hand",
                            coupled=[{"a": "yaw", "b": "gear", "ratio": -1.0, "offset": 0.0}])
    assert not gripper_status(on_revolute_palm)["grips"]
    on_root = _arm(_seg("neck", "base", "revolute"), _seg("neck_gear", "base", "revolute"), ee="l2",
                   coupled=[{"a": "neck", "b": "neck_gear", "ratio": -1.0, "offset": 0.0}])
    assert not gripper_status(on_root)["grips"]


def test_a_declared_gripper_without_jaws_still_grips_but_is_not_closable():
    """Name evidence (an SO-101's revolute jaw) is kept: declared, held, but not something our script closes."""
    g = _arm(_seg("jaw", "l2", "revolute"), ee="jaw", declared="gripper")
    st = gripper_status(g)
    assert st["grips"] and not st["closable"] and st["jaws"] == []


def test_a_bare_flange_has_no_gripper():
    st = gripper_status(_arm(ee="l2"))
    assert not st["grips"] and "no jaws were found" in st["evidence"]


# ------------------------------------------------------------------------- the agreement with the grasp site
@pytest.mark.skipif(not _MUJOCO, reason="compiling the body needs MuJoCo")
@pytest.mark.parametrize("prompt", ["a 6-DOF robot arm with a gripper", "a robot arm that sorts blocks into bins"])
def test_closable_is_exactly_the_compilers_grasp_site_rule(prompt):
    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    from virturoid.services.morphology_composer import compose_robot
    g = compose_robot(prompt, llm=None)
    has_site = 'name="grasp_site"' in compile_gene_to_mjcf(g)
    assert gripper_status(g)["closable"] == has_site
    assert bool(jaws_on(g.segments, g.end_effector().name)) == has_site


# ------------------------------------------------------------------------------------------- the real arms
def _imported(pkg: str, model: str):
    p = os.path.join(_MENAGERIE, pkg, model)
    if not os.path.exists(p):
        pytest.skip(f"MuJoCo Menagerie not cached at {p}")
    from virturoid.services.robot_import import import_robot
    out = import_robot(p)
    assert out["gene"] is not None, f"{pkg} did not import"
    return out


@pytest.mark.skipif(not _MUJOCO, reason="import needs MuJoCo")
@pytest.mark.parametrize("pkg,model,palm", [("franka_emika_panda", "panda.xml", "hand"),
                                            ("agilex_piper", "piper.xml", "link6"),
                                            ("i2rt_yam", "yam.xml", "link_6")])
def test_a_real_parallel_jaw_arm_imports_as_a_gripper_with_its_palm_as_the_end_effector(pkg, model, palm):
    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    g = _imported(pkg, model)["gene"]
    assert g.end_effector_type == "gripper"
    assert g.end_effector().name == palm
    st = gripper_status(g)
    assert st["grips"] and st["closable"]
    assert 'name="grasp_site"' in compile_gene_to_mjcf(g), "the importer and the grasp site must agree"


@pytest.mark.skipif(not _MUJOCO, reason="import needs MuJoCo")
@pytest.mark.parametrize("pkg,model", [("universal_robots_ur5e", "ur5e.xml"), ("kinova_gen3", "gen3.xml")])
def test_a_bare_arm_still_imports_without_a_gripper(pkg, model):
    g = _imported(pkg, model)["gene"]
    assert g.end_effector_type == "none"
    assert not gripper_status(g)["grips"]


@pytest.mark.skipif(not _MUJOCO, reason="import needs MuJoCo")
@pytest.mark.parametrize("pkg,model,ee,kind", [
    ("unitree_go2", "go2.xml", "RR_calf", "legged"),          # a quadruped: a calf is not a palm
    ("unitree_g1", "g1.xml", "right_wrist_yaw_link", "legged"),  # a humanoid
    ("leap_hand", "right_hand.xml", "th_ds", "manipulator"),  # a dexterous hand: revolute, uncoupled fingers
    ("wonik_allegro", "right_hand.xml", "th_tip", "manipulator"),
])
def test_legged_humanoid_and_hand_bodies_are_left_exactly_as_they_were(pkg, model, ee, kind):
    """The sweep's other half: every one of these imported as "none" with this end effector and this kind before
    the detector existed, and must still. (Full 79-model before/after table: the change's report.)"""
    from virturoid.services.task_matched_eval import robot_kind
    g = _imported(pkg, model)["gene"]
    assert g.end_effector_type == "none"
    assert g.end_effector().name == ee
    assert robot_kind(g) == kind


@pytest.mark.skipif(not _MUJOCO, reason="import needs MuJoCo")
def test_a_robotiq_on_an_arm_is_seen_by_its_coupled_drivers():
    """The Tidybot's Kinova carries a Robotiq 2F-85: no sliding jaws and no "grip" in any name, only two revolute
    drivers the source couples 1:1. The flag lands on a finger pad four links down; the palm is the gripper base."""
    g = _imported("stanford_tidybot", "tidybot.xml")["gene"]
    assert g.end_effector_type == "gripper"
    assert g.end_effector().name == "base"
    st = gripper_status(g)
    assert st["grips"] and not st["closable"] and set(st["jaws"]) == {"left_driver", "right_driver"}


@pytest.mark.skipif(not _MUJOCO, reason="import + verify need MuJoCo")
def test_a_named_revolute_jaw_is_a_gripper_verify_cannot_close_not_a_missing_one():
    """The SO-ARM100 names its jaws (Fixed_Jaw / Moving_Jaw) and has no grasp site. verify_robot used to tell this
    customer "NO GRIPPER in your model"; the importer and verify now agree it HAS one our script cannot close."""
    from virturoid.services import session_state as S
    from virturoid.services.agent_tools import call_tool
    g = _imported("trs_so_arm100", "so_arm100.xml")["gene"]
    assert g.end_effector_type == "gripper"
    rid = S.put_robot(g, prompt="so-arm100", label="so-arm100")
    try:
        res = call_tool("verify_robot", {"robot_id": rid})["result"]
    finally:
        S.forget_robot(rid)
    assert "NO GRIPPER" not in res["verdict"], res["verdict"]
    assert "NO GRASP VERDICT: our grasp script cannot close your gripper" in res["verdict"], res["verdict"]
    assert res["grasp"]["gripper"]["grips"] is True and res["grasp"]["gripper"]["closable"] is False


@pytest.mark.skipif(not _MUJOCO, reason="import needs MuJoCo")
def test_the_hand_the_importer_now_sees_is_not_billed_as_a_new_gripper():
    """Declaring the Panda's gripper puts an end-effector line on its BOM. It is the customer's own Franka Hand:
    a catalog EQUIVALENT, outside the bill and outside the mass, like the actuator lines. (Measured through
    export_held: price_usd stays $931.44 and the line's $5,500 moves to already_fitted_usd.)"""
    from virturoid.services.bom_builder import build_bom
    bom = build_bom(_imported("franka_emika_panda", "panda.xml")["gene"])
    grip = [ln for ln in bom["lines"] if ln["category"] == "gripper"]
    assert len(grip) == 1
    assert grip[0]["in_price_total"] is False and grip[0]["in_mass_total"] is False
    assert "EQUIVALENT for the gripper already on your robot" in grip[0]["detail"]
    billed = round(sum(ln["price_usd"] for ln in bom["lines"] if ln["in_price_total"]), 2)
    assert bom["totals"]["price_usd"] == billed
