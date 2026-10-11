"""An arm's reach is MEASURED by forward kinematics, never added up from its link lengths.

MEASURED DEFECT (2026-10-11, the "bring your own arm" journey through the MCP server): ``verify_robot`` read an
imported Franka Panda as "ARTICULATES (reach 1.71 m)" and printed the same 1.71 m on the spec sheet and the
verification certificate. Franka's datasheet says 855 mm. The number was the sum of every actuated segment's
length, the two finger joints and every offset link included, at full length, as if the joint ranges let the
whole chain lie straight. The same rule read a UR5e (850 mm) at 1.274 m and a Kinova Gen3 (902 mm) at 1.045 m.

``services.arm_reach`` measures the farthest the end effector's origin gets from the arm's first joint axis over
joint configurations sampled inside the declared limits. These tests pin that it lands near the datasheets on
the real arms (when the Menagerie is cached; otherwise those skip), that the verdict and every surface that
carries it report that number, and the properties that make it a measurement rather than a guess: it cannot
exceed the chain it is measured along, it does not move when the arm is mounted somewhere else, and it does not
move with the seed.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="forward kinematics needs MuJoCo")

_MENAGERIE = os.path.join(os.path.expanduser("~"), ".cache", "robot_descriptions", "mujoco_menagerie")

#: (package, model file, datasheet reach in metres)
_DATASHEET = [("franka_emika_panda", "panda.xml", 0.855),
              ("universal_robots_ur5e", "ur5e.xml", 0.850),
              ("kinova_gen3", "gen3.xml", 0.902)]


def _imported(pkg: str, model: str):
    p = os.path.join(_MENAGERIE, pkg, model)
    if not os.path.exists(p):
        pytest.skip(f"MuJoCo Menagerie not cached at {p}")
    from virturoid.services.robot_import import import_robot
    g = import_robot(p)["gene"]
    assert g is not None, f"{pkg} did not import"
    return g


def _composed_arm():
    from virturoid.services.morphology_composer import compose_robot
    return compose_robot("a 6-DOF robot arm with a gripper", llm=None)


@pytest.mark.parametrize("pkg,model,datasheet", _DATASHEET)
def test_a_real_arm_reads_within_ten_percent_of_its_datasheet(pkg, model, datasheet):
    """Measured 0.856 / 0.925 / 0.841 for 0.855 / 0.850 / 0.902 (from the first joint's axis; from its anchor point
    they read 0.858 / 0.927 / 0.968). The link sum read 1.708 / 1.274 / 1.045."""
    from virturoid.services.arm_reach import arm_reach
    gene = _imported(pkg, model)
    r = arm_reach(gene)
    assert r is not None
    assert abs(r["reach_m"] - datasheet) / datasheet <= 0.10, (pkg, r["reach_m"], datasheet)
    link_sum = sum(float(s.length_m or 0.0) for s in gene.actuated_joints())
    assert r["reach_m"] < link_sum, "the measured reach must not exceed the old link-sum overstatement"


def test_verify_reports_the_measured_reach_in_the_verdict_it_always_used():
    """The verdict wording is unchanged ("ARTICULATES (reach X m)"); only the number in it is now true."""
    from virturoid.services import session_state as S
    from virturoid.services.agent_tools import call_tool
    from virturoid.services.arm_reach import arm_reach
    gene = _imported("franka_emika_panda", "panda.xml")
    rid = S.put_robot(gene, prompt="panda reach", label="panda reach")
    try:
        res = call_tool("verify_robot", {"robot_id": rid, "mode": "quick"})["result"]
    finally:
        S.forget_robot(rid)
    expected = arm_reach(gene)["reach_m"]
    assert res["reach_m"] == expected
    assert res["verdict"].startswith(f"ARTICULATES (reach {expected:.2f} m)"), res["verdict"]
    assert "1.71" not in res["verdict"]
    assert res["reach_from"]["to_body"] == "hand"
    assert "forward kinematics" in res["note"]


def test_reach_is_bounded_by_the_chain_it_is_measured_along():
    """A sampled maximum of a distance along a chain can never exceed that chain at full length."""
    from virturoid.services.arm_reach import arm_reach
    from virturoid.services.design_critic import chain_to_ee
    gene = _composed_arm()
    r = arm_reach(gene)
    assert r is not None and r["reach_m"] > 0.2
    chain = sum(float(s.length_m or 0.0) for s in chain_to_ee(gene) if s.parent is not None)
    assert r["reach_m"] <= chain + 1e-6, (r["reach_m"], chain)


def test_reach_does_not_move_when_the_arm_is_mounted_higher():
    """It is measured from the arm's own first joint, so the stand it sits on is not part of the number."""
    from virturoid.schemas.gene import RobotGene
    from virturoid.services.arm_reach import arm_reach
    gene = _composed_arm()
    raised = RobotGene.from_dict(gene.to_dict())
    raised.base_height_m = 0.75
    assert arm_reach(raised)["reach_m"] == pytest.approx(arm_reach(gene)["reach_m"], abs=1e-3)


def test_reach_is_stable_across_seeds():
    """6000 samples: the max moves by under a percent from seed to seed. Measured over seeds 0-4 (2026-10-11, from
    the first joint's axis): this composed arm 0.696-0.701, the Panda 0.856-0.857, the UR5e 0.921-0.925, the Gen3
    0.839-0.841."""
    from virturoid.services.arm_reach import arm_reach
    gene = _composed_arm()
    vals = [arm_reach(gene, seed=s)["reach_m"] for s in range(4)]
    assert max(vals) - min(vals) <= 0.02 * max(vals), vals


def test_a_body_with_no_articulated_chain_has_no_reach():
    """Nothing to sample is not a reach of zero metres: the caller is told there is no arm to measure."""
    from virturoid.schemas.gene import GeneSegment, RobotGene
    from virturoid.services.arm_reach import arm_reach
    g = RobotGene(id="brick", species="test.brick", robot_class="manipulator", end_effector_type="none",
                  segments=[GeneSegment("base", None, "box", 0.2, 0.1, 1.0, is_end_effector=True)])
    assert arm_reach(g) is None
