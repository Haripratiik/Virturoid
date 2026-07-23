"""B2 + #214 (2026-07-24 audit): ingesting a customer robot must (1) keep the customer's OWN geometry as the
held robot, (2) report ITS honest verdict -- never a canonical template's walk reported as the customer's,
(3) offer any walkable template as an explicit opt-in with undo, and (4) classify a fixed-base multi-limb body
as legged so verify uses the right rubric (not the arm rubric).
"""
from __future__ import annotations

import importlib.util
import os
import tempfile

import pytest

_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="import + eval need MuJoCo")


def _fixed_base_quad_urdf(tmp) -> str:
    urdf = ('<?xml version="1.0"?>\n<robot name="customer_quad">\n'
            '<link name="trunk"><inertial><mass value="6.0"/><inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" '
            'ixz="0" iyz="0"/></inertial><visual><geometry><box size="0.5 0.25 0.1"/></geometry></visual></link>\n')
    for nm, x, y in [("FL", 0.2, 0.12), ("FR", 0.2, -0.12), ("HL", -0.2, 0.12), ("HR", -0.2, -0.12)]:
        urdf += (f'<link name="{nm}_leg"><inertial><mass value="0.8"/><inertia ixx="0.01" iyy="0.01" izz="0.01" '
                 f'ixy="0" ixz="0" iyz="0"/></inertial><visual><geometry><cylinder radius="0.03" length="0.3"/>'
                 f'</geometry></visual></link>\n'
                 f'<joint name="{nm}_hip" type="revolute"><parent link="trunk"/><child link="{nm}_leg"/>'
                 f'<axis xyz="0 1 0"/><limit lower="-1.2" upper="1.2" effort="20" velocity="3"/>'
                 f'<origin xyz="{x} {y} 0"/></joint>\n')
    urdf += "</robot>"
    p = os.path.join(tmp, "q.urdf")
    open(p, "w").write(urdf)
    return tmp


def test_fixed_base_quad_classifies_legged_not_manipulator():
    """#214: a fixed-base body with >=3 symmetric revolute limbs off the root is legged, not an arm."""
    from virturoid.services.robot_import import import_robot
    from virturoid.services.task_matched_eval import robot_kind
    tmp = _fixed_base_quad_urdf(tempfile.mkdtemp())
    g = import_robot(os.path.join(tmp, "q.urdf"))["gene"]
    assert robot_kind(g) == "legged", f"fixed-base quad classified {robot_kind(g)}"
    # a single-chain arm must NOT be caught by the new rule
    from virturoid.services.morphology_composer import compose_robot
    assert robot_kind(compose_robot("a 6-axis robot arm with a gripper", llm=None)) == "manipulator"


def test_ingest_keeps_customer_body_and_reports_its_own_verdict():
    from virturoid.services import session_state as S
    from virturoid.services.agent_tools import call_tool
    tmp = _fixed_base_quad_urdf(tempfile.mkdtemp())
    r = call_tool("ingest_project", {"path": tmp, "description": "my patrol quadruped"}).get("result", {})
    g = S.get_robot(r["robot_id"])
    names = [s.name for s in g.segments]
    assert any(n.endswith("_leg") for n in names), f"customer link names not preserved: {names}"
    assert "torso" not in names and "neck" not in names, "held body is a composed template, not the import"
    iv = r.get("imported_verdict")
    assert iv is not None and "own imported geometry" in iv["body"]
    # verify runs the legged rubric on the customer's body (not the arm rubric)
    v = call_tool("verify_robot", {"robot_id": r["robot_id"], "mode": "quick"})["result"]
    assert v.get("kind") == "legged"


def test_walkable_template_is_opt_in_and_undoable():
    from virturoid.services import session_state as S
    from virturoid.services.agent_tools import call_tool
    tmp = _fixed_base_quad_urdf(tempfile.mkdtemp())
    r = call_tool("ingest_project", {"path": tmp, "description": "patrol quad"}).get("result", {})
    rid = r["robot_id"]
    before = [s.name for s in S.get_robot(rid).segments]
    call_tool("edit_robot", {"robot_id": rid, "ops": [{"op": "adopt_walkable_template"}]})
    after = [s.name for s in S.get_robot(rid).segments]
    call_tool("edit_robot", {"robot_id": rid, "ops": [{"op": "undo"}]})
    restored = [s.name for s in S.get_robot(rid).segments]
    assert restored == before, "undo did not restore the customer's original body"
