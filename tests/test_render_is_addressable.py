"""A render you cannot put next to the previous render is half a tool.

`render_view` wrote ONE filename per (robot, view) and overwrote it. An engineer walking the customer journey
on a real Menagerie Go2 lost their "before" picture twice inside one session: once to the `add_limb` they had
rendered the body in order to inspect, and once when a second `add_limb` destroyed the first attempt's result.
Before/after is the single most common thing anyone does with a render, and the tool made it impossible without
copying files by hand -- which an agent driving us over MCP has no way to do at all.

The fix is not an argument the caller has to remember. The filename carries a stamp of the BODY and the CAMERA,
so:

  * an edit can never land on an earlier picture (the property that was violated),
  * re-rendering an unchanged robot is idempotent and reuses its own file (no clutter, and it answers "did
    anything change?" without a diff),
  * `op:'undo'` comes back to the render it left,
  * and the earlier pictures of the same robot ride back in the payload, so the comparison needs no filesystem.

Everything here goes through `agent_tools.call_tool` -- the dispatcher a customer's agent uses -- because the
subject is the TOOL's contract, not the renderer's.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("VIRTUROID_DISABLE_GAIT_HINTS", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None

pytestmark = pytest.mark.skipif(not _MUJOCO, reason="rendering needs MuJoCo")


def _call(name, args=None):
    from virturoid.services.agent_tools import call_tool
    return call_tool(name, args or {})


def _result(name, args=None):
    env = _call(name, args)
    assert env["ok"] is True, env
    return env["result"]


@pytest.fixture(scope="module")
def held_arm():
    """A held robot obtained the way a customer does. A 4-DOF arm because the SUBJECT is the filename, not the
    picture, and an arm renders in a fraction of a quadruped's time."""
    schema = _result("get_design_schema")
    return _result("submit_design", {"graph": schema["examples"]["scara_arm"]})["robot_id"]


def _taller(rid, factor: float = 1.15):
    """One localized edit, through the same door -- the customer's actual 'before/after' trigger.

    `gate_non_regression` off because the SUBJECT here is the filename, not the design: repeatedly lengthening
    the same arm eventually trips the torque-margin gate, and an edit refused for a good reason would leave the
    body unchanged and quietly turn "the render was not overwritten" into a tautology."""
    return _result("edit_robot", {"robot_id": rid, "gate_non_regression": False, "ops": [
        {"op": "scale_group", "args": {"group": "arms", "dims": "length", "factor": factor}}]})


def test_an_edit_cannot_destroy_the_render_taken_before_it(held_arm):
    """THE defect, stated as the property that was violated: the 'before' still opens after the 'after'."""
    before = _result("render_view", {"robot_id": held_arm})
    before_path, before_bytes = Path(before["path"]), before["bytes"]
    assert before_path.is_file()

    _taller(held_arm)
    after = _result("render_view", {"robot_id": held_arm})

    assert after["path"] != before["path"], "the edited body overwrote the picture of the unedited one"
    assert before_path.is_file(), f"the 'before' render was destroyed: {before_path}"
    assert before_path.stat().st_size == before_bytes, "the 'before' file survived in name only"
    assert Path(after["path"]).is_file()
    assert before_path.read_bytes() != Path(after["path"]).read_bytes(), (
        "two different bodies produced byte-identical pictures -- the stamp is not keyed to the body")


def test_a_second_edit_does_not_destroy_the_first_ones_render(held_arm):
    """The engineer's SECOND loss, which is a different case: not before-vs-after but after-vs-after."""
    seen = []
    for _ in range(3):
        seen.append(Path(_result("render_view", {"robot_id": held_arm})["path"]))
        _taller(held_arm)
    assert len({p.name for p in seen}) == 3, f"three distinct bodies wrote {len({p.name for p in seen})} files"
    for p in seen:
        assert p.is_file(), f"{p} was clobbered by a later edit"


def test_re_rendering_an_unchanged_robot_reuses_its_own_file(held_arm):
    """The other half of the contract. Uniqueness by timestamp or by counter would spray a file per call; the
    stamp is derived from the body, so an unchanged robot is idempotent -- and SAYS it was unchanged, which is
    a cheaper answer to "did my edit do anything?" than comparing pixels."""
    first = _result("render_view", {"robot_id": held_arm})
    again = _result("render_view", {"robot_id": held_arm})
    assert again["path"] == first["path"]
    assert again["state_id"] == first["state_id"]
    assert again["reused_existing_file"] is True, "an unchanged robot must land on its own file"
    assert first["reused_existing_file"] is False or first["state_id"] == again["state_id"]


def test_undo_comes_back_to_the_picture_it_left(held_arm):
    """A stamp keyed to the BODY (rather than to a revision counter, which only ever goes up) means the render
    of a reverted robot is the render it had before the edit -- the same file, not a fourth copy."""
    before = _result("render_view", {"robot_id": held_arm})
    _taller(held_arm)
    _result("edit_robot", {"robot_id": held_arm, "op": "undo"})
    back = _result("render_view", {"robot_id": held_arm})
    assert back["path"] == before["path"], "undo produced a different picture of the same body"
    assert back["state_id"] == before["state_id"]


def test_the_camera_is_part_of_the_address(held_arm):
    """Two angles of one body are two pictures. Keying only on the body would have them overwrite each other,
    which is the original defect wearing a different hat."""
    a = _result("render_view", {"robot_id": held_arm, "azimuth": 50.0})
    b = _result("render_view", {"robot_id": held_arm, "azimuth": 140.0})
    assert a["path"] != b["path"] and a["state_id"] != b["state_id"]
    assert Path(a["path"]).is_file() and Path(b["path"]).is_file()


def test_the_two_views_do_not_share_a_file(held_arm):
    """`view='collision'` draws a different model of the same body; it must not evict the visual one."""
    vis = _result("render_view", {"robot_id": held_arm, "view": "visual"})
    coll = _result("render_view", {"robot_id": held_arm, "view": "collision"})
    assert vis["path"] != coll["path"]
    assert Path(vis["path"]).is_file() and Path(coll["path"]).is_file()
    assert "collision" in Path(coll["path"]).name


def test_a_label_is_added_and_never_substituted(held_arm):
    """Ergonomics, not a requirement: nobody should have to invent a filename, and a caller who WANTS a
    readable one must not be able to reintroduce the loss by reusing it. So the label rides ALONGSIDE the
    stamp -- `before` on two different bodies is two files, both of which open."""
    first = _result("render_view", {"robot_id": held_arm, "label": "before"})
    assert "before" in Path(first["path"]).name and first["label"] == "before"
    _taller(held_arm)
    second = _result("render_view", {"robot_id": held_arm, "label": "before"})
    assert "before" in Path(second["path"]).name
    assert second["path"] != first["path"], "the same label on a different body overwrote the earlier render"
    assert Path(first["path"]).is_file() and Path(second["path"]).is_file()


def test_a_label_cannot_write_outside_the_render_directory(held_arm):
    """The label lands in a filename, so it is caller-controlled path text and is treated as such."""
    from virturoid.services.ai_native_tools import _render_dir
    r = _result("render_view", {"robot_id": held_arm, "label": "../../escaped"})
    p = Path(r["path"]).resolve()
    assert _render_dir().resolve() in p.parents, f"a label escaped the render directory: {p}"
    assert p.is_file()


def test_the_earlier_pictures_come_back_with_the_result(held_arm):
    """An agent driving us over MCP has no `ls`. If the comparison needs a file listing, the comparison does
    not happen -- so the tool hands back the renders of this robot that this call did not touch."""
    from virturoid.services.ai_native_tools import _EARLIER_RENDERS_MAX
    _taller(held_arm)
    first = _result("render_view", {"robot_id": held_arm})
    _taller(held_arm)
    second = _result("render_view", {"robot_id": held_arm})

    earlier = second["earlier_renders"]
    assert first["path"] in earlier, "the immediately-preceding render is not offered for comparison"
    assert second["path"] not in earlier, "the current render must not be listed as one of the earlier ones"
    assert len(earlier) <= _EARLIER_RENDERS_MAX, "an unbounded list would blow a client's response cap"
    for p in earlier:
        assert Path(p).is_absolute() and Path(p).is_file(), f"offered a path that does not open: {p}"


def test_the_advertised_schema_declares_the_label(held_arm):
    """A parameter a strict MCP client cannot send is a parameter that does not exist."""
    from virturoid.services.agent_tools import TOOLS
    props = TOOLS["render_view"]["parameters"]["properties"]
    assert "label" in props and props["label"]["type"] == "string"
    assert "view" in props and "azimuth" in props and "elevation" in props
