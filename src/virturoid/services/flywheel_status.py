"""Moat status (Theme 2): one legible, cumulative view of how the platform is compounding across ALL builds.

The flywheel only matters if it visibly accumulates. ``flywheel_runner`` measures compounding WITHIN a run;
this reads the PERSISTENT memory (the SQLite ``MemoryDB`` + the ``flywheel_manifest.jsonl`` ledger) to answer
"across every build this workspace has ever done, what has the moat banked, and is warm-start actually being
used?" — the artifact a stakeholder reads to see the self-improving loop is real (startup plan §13 memory).

Pure, offline, best-effort: a missing DB/manifest reads as an empty (honest) status rather than an error.
"""

from __future__ import annotations

import json
from pathlib import Path

from virturoid.services.memory_store import DEFAULT_MEMORY_DIR


def _read_manifest(memory_dir: Path) -> list[dict]:
    """Every banked-skill ledger row ever appended (one per trained skill/policy). Empty if none."""
    path = Path(memory_dir) / "flywheel_manifest.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def moat_status(memory_dir=DEFAULT_MEMORY_DIR) -> dict:
    """Aggregate the persistent flywheel memory into a single compounding-status dict (+ a human summary).

    Returns ``{counts, skills, warm_start, species_tree, summary}`` where ``warm_start`` reports how many of
    the trainable builds actually reused prior learning — the headline "next robot learns from the last".
    """
    memory_dir = Path(memory_dir)
    counts = {"runs": 0, "designs": 0, "skills": 0, "lessons": 0, "species_tree": 0}
    species_nodes = 0
    try:
        from virturoid.services.memory_db import MemoryDB

        db_path = memory_dir / "virturoid_memory.db"
        if db_path.exists():
            with MemoryDB(db_path) as db:
                s = db.stats()
                counts = {k: int(s.get(k, 0)) for k in counts}
                species_nodes = len(db.species_tree_nodes())
    except Exception:  # noqa: BLE001 - status must never crash
        pass

    manifest = _read_manifest(memory_dir)
    # Per banked skill: how many builds contributed, best success, how many warm-started.
    per_skill: dict[str, dict] = {}
    for r in manifest:
        sid = r.get("skill_id", "?")
        e = per_skill.setdefault(sid, {"skill_id": sid, "builds": 0, "best_success": 0.0,
                                       "warm_started": 0, "trainers": set(), "task_types": set()})
        e["builds"] += 1
        e["best_success"] = max(e["best_success"], float(r.get("success_rate") or 0.0))
        if r.get("warm_started"):
            e["warm_started"] += 1
        if r.get("trainer"):
            e["trainers"].add(r["trainer"])
        if r.get("task_type"):
            e["task_types"].add(r["task_type"])
    skills = [{**e, "trainers": sorted(e["trainers"]), "task_types": sorted(e["task_types"])}
              for e in sorted(per_skill.values(), key=lambda d: d["best_success"], reverse=True)]

    trainable = len(manifest)
    warm = sum(1 for r in manifest if r.get("warm_started"))
    warm_start = {
        "trainable_builds": trainable,
        "warm_started_builds": warm,
        "utilization": round(warm / trainable, 3) if trainable else 0.0,
    }

    summary = (
        f"Moat status: {counts['runs']} builds banked -> "
        f"{counts['designs']} designs, {len(skills)} distinct skills/policies, {counts['lessons']} lessons, "
        f"{species_nodes} species-tree nodes. "
        f"Warm-start reused prior learning in {warm}/{trainable} trainable builds "
        f"({warm_start['utilization']:.0%}). "
        + (f"Best banked skill: {skills[0]['skill_id']} ({skills[0]['best_success']:.0%}, "
           f"{skills[0]['builds']} builds)." if skills else "No skills banked yet.")
    )
    return {
        "counts": counts,
        "skills": skills,
        "warm_start": warm_start,
        "species_tree_nodes": species_nodes,
        "compounding": (counts["designs"] + len(skills)) >= 2 and (counts["runs"] >= 2),
        "summary": summary,
    }
