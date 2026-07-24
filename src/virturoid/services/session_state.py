"""AI-native session state (docs/ai_native_plan.md P0 + agent_first_plan.md G-B) — the stateful store both
frontends share so INCREMENTAL edits work AND the app can SEE an external agent's work.

FILE-BACKED (G-B): a stdio MCP client (Claude Code/Codex) spawns our MCP server as its OWN process, so an
in-memory store would be invisible to the running ``ui_server`` viewer (measured: cross-process get failed).
Every ``put/commit/undo`` writes ``build/sessions/{robots,scenes}/<id>.json`` (atomic replace); ``get``
reloads when the on-disk copy is newer than the cached one. So the agent's process and the app process share
one source of truth — the app becomes the live viewer of the agent designing. In-memory dict stays a cache.
Thread-safe, stdlib-only. Genes/scenes snapshot as ``to_dict`` so undo restores an exact deep copy.
"""
from __future__ import annotations

import json
import os
import re
import threading
import uuid
from pathlib import Path

_LOCK = threading.RLock()
_ROBOTS: dict[str, dict] = {}      # id -> {"gene", "undo": [dict...], "label", "prompt", "_mtime"}
_SCENES: dict[str, dict] = {}      # id -> {"scene": dict, "undo": [dict...], "task", "theme", "_mtime"}
_UNDO_MAX = 20


def _dir() -> Path:
    return Path(os.environ.get("VIRTUROID_SESSIONS_DIR") or "build/sessions")


_UNSAFE_ID = re.compile(r"[^A-Za-z0-9._-]")


def _safe_id(raw: str) -> str:
    """A filesystem-safe session id — the containment boundary for robot_id / scene_id.

    A caller (e.g. the public ``ingest_project`` MCP tool) may pass an agent-supplied id; without this a value
    like ``..\\..\\..\\Windows\\Temp\\evil`` escaped the sessions dir and wrote arbitrary JSON at the repo root
    (measured in the 2026-07-24 audit — B0). We strip every path separator, drive colon, and ``..`` to a single
    flat token, so a session file can ONLY ever land inside ``build/sessions/{robots,scenes}/``. Sanitized here
    AND at the path choke point (defense in depth), the same way ``agent_tools.safe_build_path`` confines builds.
    """
    s = _UNSAFE_ID.sub("_", str(raw or "")).strip("._")
    return (s or "session")[:128]


def _robot_path(rid: str) -> Path:
    return _dir() / "robots" / f"{_safe_id(rid)}.json"


def _scene_path(sid: str) -> Path:
    return _dir() / "scenes" / f"{_safe_id(sid)}.json"


def _rid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _atomic_write(path: Path, data: dict) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, default=str), encoding="utf-8")
    os.replace(tmp, path)                                      # atomic on the same filesystem
    return path.stat().st_mtime


# ------------------------------------------------------------------ robots
def _write_robot(rid: str, rec: dict) -> None:
    try:
        rec["_mtime"] = _atomic_write(_robot_path(rid),
                                      {"gene": rec["gene"].to_dict(), "undo": rec["undo"],
                                       "label": rec["label"], "prompt": rec["prompt"]})
    except Exception:  # noqa: BLE001 - persistence is the cross-process bonus; the in-memory store still works
        pass


def _load_robot(rid: str) -> dict | None:
    p = _robot_path(rid)
    if not p.exists():
        return None
    try:
        from virturoid.schemas.gene import RobotGene
        d = json.loads(p.read_text(encoding="utf-8"))
        return {"gene": RobotGene.from_dict(d["gene"]), "undo": d.get("undo", []),
                "label": d.get("label", ""), "prompt": d.get("prompt", ""), "_mtime": p.stat().st_mtime}
    except Exception:  # noqa: BLE001
        return None


def _fresh_robot(rid: str) -> dict | None:
    """Cached record, reloaded from disk if another process wrote a newer copy."""
    rec = _ROBOTS.get(rid)
    p = _robot_path(rid)
    disk = p.stat().st_mtime if p.exists() else None
    if rec is None or (disk is not None and disk > rec.get("_mtime", 0.0)):
        loaded = _load_robot(rid)
        if loaded is not None:
            _ROBOTS[rid] = loaded
            rec = loaded
    return rec


# The on-disk robot session store is a convenience cache (cross-process reuse), not a database of record — an
# exported package is the durable artifact. Left uncapped it grows without bound (measured 3,800+ files / 55 MB
# on a dev box), turning /api/sessions and every dir scan into O(N). Keep the most-recent N by mtime.
_SESSION_CAP = int(os.environ.get("VIRTUROID_SESSION_CAP", "500"))


def prune_sessions(cap: int | None = None) -> int:
    """Delete the oldest robot session files beyond ``cap`` (most-recent kept). Returns how many were removed.
    Best-effort and lock-free-safe: a file another process is writing is simply skipped. Never raises."""
    keep = _SESSION_CAP if cap is None else cap
    d = _dir() / "robots"
    if keep <= 0 or not d.exists():
        return 0
    try:
        files = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return 0
    removed = 0
    for p in files[keep:]:
        try:
            p.unlink(); removed += 1
        except OSError:
            pass
    return removed


def prune_scene_sessions(cap: int | None = None) -> int:
    """Cap SCENE sessions the same way prune_sessions caps robots -- delete the oldest scene files (and drop them
    from the in-memory map) beyond ``cap``, most-recent kept. Before this, robots were capped but scenes grew
    without bound over a long agent session (2026-07-24 audit). Best-effort; never raises."""
    keep = _SESSION_CAP if cap is None else cap
    d = _dir() / "scenes"
    if keep <= 0 or not d.exists():
        return 0
    try:
        files = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return 0
    removed = 0
    for p in files[keep:]:
        try:
            p.unlink(); removed += 1
            with _LOCK:
                _SCENES.pop(p.stem, None)                     # drop the in-memory record too
        except OSError:
            pass
    return removed


_gc_ticks = {"n": 0, "swept_once": False}


def put_robot(gene, *, prompt: str = "", label: str = "created", robot_id: str | None = None) -> str:
    rid = _safe_id(robot_id) if robot_id else _rid("robot")   # confine caller-supplied ids (B0)
    with _LOCK:
        rec = {"gene": gene, "undo": [], "label": label, "prompt": prompt, "_mtime": 0.0}
        _ROBOTS[rid] = rec
        _write_robot(rid, rec)
    # GC housekeeping (never fails a build): sweep ONCE on first write in a process so accumulated cruft from
    # earlier runs clears immediately, then amortize to ~once per cap/10 writes so the hot path stays free.
    try:
        _gc_ticks["n"] += 1
        if not _gc_ticks["swept_once"] or _gc_ticks["n"] % max(1, _SESSION_CAP // 10) == 0:
            _gc_ticks["swept_once"] = True
            prune_sessions()
    except Exception:  # noqa: BLE001 - GC is housekeeping; it must never fail a build
        pass
    return rid


def get_robot(robot_id: str):
    with _LOCK:
        rec = _fresh_robot(robot_id)
        return rec["gene"] if rec else None


def commit_robot(robot_id: str, new_gene, *, label: str) -> bool:
    with _LOCK:
        rec = _fresh_robot(robot_id)
        if rec is None:
            return False
        rec["undo"].append(rec["gene"].to_dict())
        if len(rec["undo"]) > _UNDO_MAX:
            rec["undo"].pop(0)
        rec["gene"] = new_gene
        rec["label"] = label
        _write_robot(robot_id, rec)
        return True


def undo_robot(robot_id: str):
    from virturoid.schemas.gene import RobotGene
    with _LOCK:
        rec = _fresh_robot(robot_id)
        if not rec or not rec["undo"]:
            return None
        rec["gene"] = RobotGene.from_dict(rec["undo"].pop())
        rec["label"] = "undo"
        _write_robot(robot_id, rec)
        return rec["gene"]


def robot_meta(robot_id: str) -> dict:
    with _LOCK:
        rec = _fresh_robot(robot_id)
        if rec is None:
            return {}
        return {"robot_id": robot_id, "prompt": rec["prompt"], "label": rec["label"],
                "undo_depth": len(rec["undo"])}


# /api/sessions cache: (directory signature) -> parsed listing. MEASURED live 2026-07-22: the sessions dir had
# accumulated 3,416 robot files (47 MB) and this function re-opened + JSON-parsed EVERY one on EVERY request,
# while the /sessions viewer polls it each 2.5 s — the first cold hit exceeded 30 s and concurrent sweeps
# stacked. The signature (file count + newest mtime) is a single cheap directory scan; content is re-parsed
# only when a session file actually changed. Entries are summary-only, so staleness risk is nil beyond the
# signature itself.
_LIST_CACHE: dict = {"sig": None, "out": []}


def list_robots() -> list[dict]:
    """All robot sessions on disk (for the app viewer's /api/sessions) — id + label + prompt."""
    d = _dir() / "robots"
    if not d.exists():
        return []
    files = sorted(d.glob("*.json"))
    try:
        sig = (len(files), max((p.stat().st_mtime_ns for p in files), default=0))
    except OSError:
        sig = None                                            # racing writer -> just re-parse this once
    with _LOCK:
        if sig is not None and _LIST_CACHE["sig"] == sig:
            return list(_LIST_CACHE["out"])
    out = []
    for p in files:
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
            out.append({"robot_id": p.stem, "label": j.get("label"), "prompt": j.get("prompt"),
                        "robot_class": (j.get("gene") or {}).get("robot_class")})
        except Exception:  # noqa: BLE001
            pass
    with _LOCK:
        _LIST_CACHE["sig"], _LIST_CACHE["out"] = sig, out
    return list(out)


# ------------------------------------------------------------------ scenes
def _write_scene(sid: str, rec: dict) -> None:
    try:
        rec["_mtime"] = _atomic_write(_scene_path(sid),
                                      {"scene": rec["scene"], "undo": rec["undo"],
                                       "task": rec["task"], "theme": rec["theme"]})
    except Exception:  # noqa: BLE001
        pass


def _fresh_scene(sid: str) -> dict | None:
    rec = _SCENES.get(sid)
    p = _scene_path(sid)
    disk = p.stat().st_mtime if p.exists() else None
    if rec is None or (disk is not None and disk > rec.get("_mtime", 0.0)):
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                rec = {"scene": d["scene"], "undo": d.get("undo", []), "task": d.get("task", ""),
                       "theme": d.get("theme", ""), "_mtime": p.stat().st_mtime}
                _SCENES[sid] = rec
            except Exception:  # noqa: BLE001
                pass
    return rec


def put_scene(scene: dict, *, task: str = "", theme: str = "", scene_id: str | None = None) -> str:
    sid = _safe_id(scene_id) if scene_id else _rid("scene")   # confine caller-supplied ids (B0)
    with _LOCK:
        rec = {"scene": dict(scene), "undo": [], "task": task, "theme": theme, "_mtime": 0.0}
        _SCENES[sid] = rec
        _write_scene(sid, rec)
    prune_scene_sessions()                                    # cap scenes too, not just robots (2026-07-24 audit)
    return sid


def get_scene(scene_id: str) -> dict | None:
    with _LOCK:
        rec = _fresh_scene(scene_id)
        return dict(rec["scene"]) if rec else None


def commit_scene(scene_id: str, new_scene: dict, *, theme: str | None = None) -> bool:
    with _LOCK:
        rec = _fresh_scene(scene_id)
        if rec is None:
            return False
        rec["undo"].append(dict(rec["scene"]))
        if len(rec["undo"]) > _UNDO_MAX:
            rec["undo"].pop(0)
        rec["scene"] = dict(new_scene)
        if theme is not None:
            rec["theme"] = theme
        _write_scene(scene_id, rec)
        return True


def undo_scene(scene_id: str) -> dict | None:
    with _LOCK:
        rec = _fresh_scene(scene_id)
        if not rec or not rec["undo"]:
            return None
        rec["scene"] = rec["undo"].pop()
        _write_scene(scene_id, rec)
        return dict(rec["scene"])


def scene_meta(scene_id: str) -> dict:
    with _LOCK:
        rec = _fresh_scene(scene_id)
        if rec is None:
            return {}
        return {"scene_id": scene_id, "task": rec["task"], "theme": rec["theme"],
                "undo_depth": len(rec["undo"])}


def reset(*, wipe_disk: bool = False) -> None:
    """Clear the in-memory cache (tests). ``wipe_disk`` also removes the session files."""
    with _LOCK:
        _ROBOTS.clear(); _SCENES.clear()
        if wipe_disk:
            import shutil
            shutil.rmtree(_dir(), ignore_errors=True)
