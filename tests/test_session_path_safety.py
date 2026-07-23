"""B0 (2026-07-24 audit): robot_id / scene_id are the containment boundary. A caller — including the public
`ingest_project` MCP tool passing an agent-supplied id — must never be able to write a session file outside
`build/sessions/`. Before the fix, `put_robot(robot_id="..\\..\\..\\Windows\\Temp\\evil")` escaped the repo.
"""
from __future__ import annotations

import os

import pytest


@pytest.fixture()
def sessions(tmp_path, monkeypatch):
    monkeypatch.setenv("VIRTUROID_SESSIONS_DIR", str(tmp_path / "sessions"))
    import importlib

    from virturoid.services import session_state as S
    importlib.reload(S)                                   # pick up the env var + reset in-memory caches
    return S, tmp_path


_EVIL_IDS = [
    "..\\..\\evil", "../../evil", "..\\..\\..\\..\\Windows\\Temp\\evil",
    "/etc/passwd", "C:\\Windows\\Temp\\evil", "foo/bar/baz", "....//....//evil",
]


def test_traversal_robot_ids_stay_inside_sessions_dir(sessions):
    S, tmp_path = sessions
    from virturoid.schemas.gene import RobotGene  # noqa: F401 - just to have a real-ish object
    outside_before = set(tmp_path.rglob("*.json")) | set((tmp_path.parent).glob("*.json"))
    for evil in _EVIL_IDS:
        rid = S.put_robot({"dummy": True}, robot_id=evil)          # a dict stands in for a gene (put stores as-is)
        # the returned id is sanitized to a flat token
        assert "/" not in rid and "\\" not in rid and ".." not in rid and ":" not in rid
        # the file that got written lives under sessions/robots/, nowhere else
        p = S._robot_path(rid).resolve()
        assert (tmp_path / "sessions" / "robots") in p.parents, f"{evil} -> {p} escaped"
    # nothing was created outside the sessions tree
    stray = [f for f in tmp_path.rglob("*.json") if "sessions" not in f.parts]
    assert stray == [], f"files written outside sessions/: {stray}"


def test_traversal_scene_ids_stay_inside_sessions_dir(sessions):
    S, tmp_path = sessions
    for evil in _EVIL_IDS:
        sid = S.put_scene({"objects": []}, scene_id=evil)
        p = S._scene_path(sid).resolve()
        assert (tmp_path / "sessions" / "scenes") in p.parents, f"{evil} -> {p} escaped"


def test_safe_id_is_stable_for_normal_ids(sessions):
    S, _ = sessions
    for good in ("robot_74b94017", "scene_abc123", "quad.anatomy-1"):
        assert S._safe_id(good) == good                            # legit ids pass through unchanged


def test_ingest_project_tool_cannot_escape(sessions, tmp_path):
    """The measured attack path: the public tool passing a traversal robot_id must not write outside."""
    S, _ = sessions
    # put_robot is the sink input_training_tools.py:364 calls with args["robot_id"]; confirm the sink is safe
    rid = S.put_robot({"dummy": True}, robot_id="..\\..\\evil_tool")
    assert not (tmp_path.parent / "evil_tool.json").exists()
    assert not (tmp_path / "evil_tool.json").exists()
    assert S.get_robot(rid) is not None                            # and the robot is still retrievable by its safe id
