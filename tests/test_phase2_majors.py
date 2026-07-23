"""Phase 2 majors (2026-07-24 MVP-readiness audit): M4 xacro guard in the gene lane, M5 advanced-tool
discoverability, M10 authored-scene materials. (M6 alive-flag lives in test_reward_loop; M1/M9 have their
own suites; M2 is a browser-verified frontend fix.)
"""
from __future__ import annotations

import importlib.util
import os
import tempfile

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


def test_m5_advanced_tools_are_discoverable_in_the_mcp_view():
    """M5: train_reward / generate_fusion / generate_control_scripts must appear in the consolidated MCP
    tools/list so a BYOK client can discover them (they were callable-by-name but invisible)."""
    from virturoid.services.agent_tools import tool_specs
    names = {s["name"] for s in tool_specs(view="mcp")}
    for t in ("train_reward", "generate_fusion", "generate_control_scripts"):
        assert t in names, f"{t} not discoverable in the MCP tools/list"


@pytest.mark.skipif(not _MUJOCO, reason="import needs MuJoCo")
def test_m4_import_robot_refuses_a_xacro_template():
    """M4: a xacro template must be REFUSED with an expand message, not held as a phantom 'mobile_base'."""
    from virturoid.services.robot_import import import_robot
    xac = ('<?xml version="1.0"?>\n<robot name="r" xmlns:xacro="http://ros.org/xacro">\n'
           '<xacro:property name="w" value="0.1"/>\n<link name="base"/>\n</robot>')
    r = import_robot(xac)
    assert r["valid"] is False and r["gene"] is None
    assert r["warnings"] and "xacro" in r["warnings"][0].lower() and "expand" in r["warnings"][0].lower()
    # a .urdf.xacro path (even if its text somehow parsed) is refused by suffix too
    d = tempfile.mkdtemp()
    p = os.path.join(d, "robot.urdf.xacro")
    open(p, "w").write('<?xml version="1.0"?>\n<robot name="r"><link name="base"/></robot>')
    assert import_robot(p)["valid"] is False


@pytest.mark.skipif(not _MUJOCO, reason="scene compose needs MuJoCo")
def test_m4_plain_urdf_still_imports():
    from virturoid.services.robot_import import import_robot
    plain = ('<?xml version="1.0"?>\n<robot name="ok"><link name="base"><inertial><mass value="1"/>'
             '<inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial></link></robot>')
    assert import_robot(plain)["gene"] is not None


@pytest.mark.skipif(not _MUJOCO, reason="scene compose needs MuJoCo")
def test_m10_authored_scene_composes_with_materials_not_bare_floor():
    """M10: the authored scene's shared materials (mat_gray/red/blue) are injected so the composed robot+scene
    model COMPILES and carries the obstacles, instead of silently dropping to a bare floor."""
    import mujoco

    from virturoid.services.ai_native_tools import _compose_scene_xml, _scene_from_dict
    from virturoid.services.agent_tools import call_tool
    from virturoid.services.morph_policy import robot_mjcf
    from virturoid.services import session_state as S

    os.environ["VIRTUROID_SESSION_DIR"] = tempfile.mkdtemp(prefix="m10_")
    rid = call_tool("create_robot", {"prompt": "a rover"})["result"]["robot_id"]
    sc = call_tool("create_scene", {"task": "sort red and blue blocks into bins",
                                    "theme": "warehouse", "seed": 1})["result"]
    gene = S.get_robot(rid)
    sg = _scene_from_dict(S.get_scene(sc["scene_id"]))
    comp = _compose_scene_xml(gene, sg)
    assert comp is not None, "authored scene fell back to bare floor (composition returned None)"
    model = mujoco.MjModel.from_xml_string(comp)              # must compile with the injected materials
    bare = mujoco.MjModel.from_xml_string(robot_mjcf(gene))
    assert model.ngeom > bare.ngeom, "scene obstacles did not make it into the composed model"
