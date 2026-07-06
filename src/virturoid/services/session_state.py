"""AI-native session state (docs/ai_native_plan.md P0) — the stateful store both frontends (the MCP server
and the in-app assistant) share so INCREMENTAL edits work: a robot/scene is created once, held under an id,
and then EDITED in place ("make it taller") rather than regenerated. Each edit lands as ONE undo step.

Design: purely additive, stdlib-only, thread-safe. Genes/scenes are snapshotted as plain dicts (``to_dict``)
so a snapshot is an immutable deep copy — undo restores it exactly. A bounded ring (``_UNDO_MAX``) per id keeps
memory flat. Mirrors the honest, in-process style of ``job_registry`` (a dict + a lock, no external deps).
"""
from __future__ import annotations

import threading
import uuid

_LOCK = threading.RLock()
_ROBOTS: dict[str, dict] = {}      # id -> {"gene": RobotGene, "undo": [dict...], "label": str, "prompt": str}
_SCENES: dict[str, dict] = {}      # id -> {"scene": dict, "undo": [dict...], "task": str, "theme": str}
_UNDO_MAX = 20


def _rid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ------------------------------------------------------------------ robots
def put_robot(gene, *, prompt: str = "", label: str = "created", robot_id: str | None = None) -> str:
    """Register a NEW robot gene under a fresh id (or replace one wholesale). Returns the id."""
    rid = robot_id or _rid("robot")
    with _LOCK:
        _ROBOTS[rid] = {"gene": gene, "undo": [], "label": label, "prompt": prompt}
    return rid


def get_robot(robot_id: str):
    """The live RobotGene for an id, or None."""
    with _LOCK:
        rec = _ROBOTS.get(robot_id)
        return rec["gene"] if rec else None


def commit_robot(robot_id: str, new_gene, *, label: str) -> bool:
    """Apply an EDIT: push the current gene onto the undo ring, then set the new one. One undo step per edit."""
    with _LOCK:
        rec = _ROBOTS.get(robot_id)
        if rec is None:
            return False
        rec["undo"].append(rec["gene"].to_dict())
        if len(rec["undo"]) > _UNDO_MAX:
            rec["undo"].pop(0)
        rec["gene"] = new_gene
        rec["label"] = label
        return True


def undo_robot(robot_id: str):
    """Restore the previous gene (pop the undo ring). Returns the restored gene, or None if nothing to undo."""
    from virturoid.schemas.gene import RobotGene
    with _LOCK:
        rec = _ROBOTS.get(robot_id)
        if not rec or not rec["undo"]:
            return None
        rec["gene"] = RobotGene.from_dict(rec["undo"].pop())
        rec["label"] = "undo"
        return rec["gene"]


def robot_meta(robot_id: str) -> dict:
    with _LOCK:
        rec = _ROBOTS.get(robot_id)
        if rec is None:
            return {}
        return {"robot_id": robot_id, "prompt": rec["prompt"], "label": rec["label"],
                "undo_depth": len(rec["undo"])}


# ------------------------------------------------------------------ scenes
def put_scene(scene: dict, *, task: str = "", theme: str = "", scene_id: str | None = None) -> str:
    sid = scene_id or _rid("scene")
    with _LOCK:
        _SCENES[sid] = {"scene": dict(scene), "undo": [], "task": task, "theme": theme}
    return sid


def get_scene(scene_id: str) -> dict | None:
    with _LOCK:
        rec = _SCENES.get(scene_id)
        return dict(rec["scene"]) if rec else None


def commit_scene(scene_id: str, new_scene: dict, *, theme: str | None = None) -> bool:
    with _LOCK:
        rec = _SCENES.get(scene_id)
        if rec is None:
            return False
        rec["undo"].append(dict(rec["scene"]))
        if len(rec["undo"]) > _UNDO_MAX:
            rec["undo"].pop(0)
        rec["scene"] = dict(new_scene)
        if theme is not None:
            rec["theme"] = theme
        return True


def undo_scene(scene_id: str) -> dict | None:
    with _LOCK:
        rec = _SCENES.get(scene_id)
        if not rec or not rec["undo"]:
            return None
        rec["scene"] = rec["undo"].pop()
        return dict(rec["scene"])


def scene_meta(scene_id: str) -> dict:
    with _LOCK:
        rec = _SCENES.get(scene_id)
        if rec is None:
            return {}
        return {"scene_id": scene_id, "task": rec["task"], "theme": rec["theme"],
                "undo_depth": len(rec["undo"])}


def reset() -> None:
    """Clear all sessions (tests)."""
    with _LOCK:
        _ROBOTS.clear(); _SCENES.clear()
