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


def brain_metrics(db) -> dict:
    """The robotics-AI layers this session added, read straight from their tables/ledgers so the visible status
    is TRACEABLE, not asserted: the physics-verified transfer ledger, the episode/tip corpora, the gated
    transfer-metric state, and the per-KIND provenance compounding deltas (gait warm-start vs the honest
    design-search-gain vs measured warm-vs-cold)."""
    conn = db.conn if hasattr(db, "conn") else db
    out = {"transfer_ledger": {"trials": 0, "credible": 0, "bodies": 0}, "episodes": 0, "tips": 0,
           "provenance_by_kind": {}, "embedding": {}}

    def _q(sql, default=None):
        try:
            return conn.execute(sql).fetchone()
        except Exception:  # noqa: BLE001 - a table may not exist yet
            return default

    row = _q("SELECT COUNT(*) n, COALESCE(SUM(credible),0) c, COUNT(DISTINCT dst_gene_id) b FROM transfer_trials")
    if row is not None:
        out["transfer_ledger"] = {"trials": int(row["n"]), "credible": int(row["c"]), "bodies": int(row["b"])}
    for otype, key in (("episode", "episodes"), ("tip", "tips")):
        r = _q(f"SELECT COUNT(*) n FROM vectors WHERE obj_type='{otype}'")
        out[key] = int(r["n"]) if r is not None else 0
    try:
        for r in conn.execute("SELECT kind, COUNT(*) n, AVG(delta) d FROM provenance GROUP BY kind").fetchall():
            out["provenance_by_kind"][r["kind"]] = {"edges": int(r["n"]),
                                                    "avg_delta": round(float(r["d"]), 4) if r["d"] is not None else None}
    except Exception:  # noqa: BLE001
        pass
    try:                                                          # the gated embedding metric's state (honest)
        from virturoid.services.body_metric import _load
        bundle = _load()
        out["embedding"] = {"metric_proven": bool(bundle and bundle.get("proven")),
                            "held_out_triplet_acc": (bundle or {}).get("held_out_triplet_acc"),
                            "baseline_held_out_triplet_acc": (bundle or {}).get("baseline_held_out_triplet_acc"),
                            "active": "learned_metric" if (bundle and bundle.get("proven")) else "baseline_29d"}
    except Exception:  # noqa: BLE001
        out["embedding"] = {"active": "baseline_29d", "metric_proven": False}
    try:                                                          # WS-G: concept memory lifecycle (traceable)
        from virturoid.services.concept_grounding import concept_summary
        out["concepts"] = concept_summary(db)
    except Exception:  # noqa: BLE001
        out["concepts"] = {"total": 0, "verified": 0}
    return out


def corpus_factory_status(memory_dir) -> dict:
    """WS-H dashboard: read the corpus-factory manifest so the Brain panel shows the moat GROWING — cumulative
    ANNECS, nights run, corpus size, class balance. Traceable to the manifest file; empty (honest) if none."""
    path = Path(memory_dir) / "corpus_factory.json"
    if not path.exists():
        return {"nights": 0, "cumulative_annecs": 0, "corpus_size": 0, "class_balance": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"nights": 0, "cumulative_annecs": 0, "corpus_size": 0, "class_balance": {}}
    admitted = data.get("admitted", [])
    balance: dict[str, int] = {}
    for a in admitted:
        niche = a.get("niche") or []
        kind = niche[0] if niche else "?"
        balance[kind] = balance.get(kind, 0) + 1
    return {"nights": int(data.get("nights", 0)), "cumulative_annecs": int(data.get("cumulative_annecs", 0)),
            "corpus_size": len(admitted), "class_balance": balance,
            "rejected_totals": data.get("rejected_totals", {})}


def moat_status(memory_dir=DEFAULT_MEMORY_DIR) -> dict:
    """Aggregate the persistent flywheel memory into a single compounding-status dict (+ a human summary).

    Returns ``{counts, skills, warm_start, species_tree, summary}`` where ``warm_start`` reports how many of
    the trainable builds actually reused prior learning — the headline "next robot learns from the last".
    """
    memory_dir = Path(memory_dir)
    counts = {"runs": 0, "designs": 0, "skills": 0, "lessons": 0, "species_tree": 0}
    species_nodes = 0
    reuse_edges = seeded_builds = 0
    brain = {"transfer_ledger": {"trials": 0, "credible": 0, "bodies": 0}, "episodes": 0, "tips": 0,
             "provenance_by_kind": {}, "embedding": {"active": "baseline_29d", "metric_proven": False}}
    try:
        from virturoid.services.memory_db import MemoryDB

        db_path = memory_dir / "virturoid_memory.db"
        if db_path.exists():
            with MemoryDB(db_path) as db:
                s = db.stats()
                counts = {k: int(s.get(k, 0)) for k in counts}
                species_nodes = len(db.species_tree_nodes())
                try:                                          # ACTUAL-reuse signal: recorded warm-start edges
                    from virturoid.services.robotics_vector_memory import RoboticsVectorMemory
                    _cs = RoboticsVectorMemory(db).compounding_summary()
                    reuse_edges = int(_cs.get("edges", 0) or 0)
                    seeded_builds = int(_cs.get("seeded_builds", 0) or 0)
                except Exception:  # noqa: BLE001
                    pass
                try:
                    brain = brain_metrics(db)                  # the P1-P3 brain layers (transfer ledger, metric, …)
                except Exception:  # noqa: BLE001
                    pass
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
        "provenance_reuse_edges": reuse_edges,
    }
    # HONEST compounding: the flywheel COMPOUNDS only when later builds actually REUSE prior learning — a warm-start
    # or a recorded provenance edge. Merely accumulating designs is NOT compounding (the old flag conflated them).
    reused = warm > 0 or seeded_builds > 0
    accumulating = (counts["designs"] + len(skills)) >= 2 and counts["runs"] >= 2

    summary = (
        f"Moat status: {counts['runs']} builds banked -> "
        f"{counts['designs']} designs, {len(skills)} distinct skills/policies, {counts['lessons']} lessons, "
        f"{species_nodes} species-tree nodes. "
        f"Warm-start reused prior learning in {warm}/{trainable} trainable builds "
        f"({warm_start['utilization']:.0%}); {reuse_edges} provenance reuse edges. "
        + (f"Best banked skill: {skills[0]['skill_id']} ({skills[0]['best_success']:.0%}, "
           f"{skills[0]['builds']} builds). " if skills else "No skills banked yet. ")
        + ("COMPOUNDING (prior learning is being reused)." if reused
           else "Accumulating but NOT yet compounding (0 reuse recorded)." if accumulating
           else "Empty flywheel.")
        + f" Brain: {brain['transfer_ledger']['trials']} verified transfer trials "
          f"({brain['transfer_ledger']['credible']} credible), {brain['episodes']} episodes, "
          f"embedding={brain['embedding'].get('active', 'baseline_29d')}"
        + (" (learned metric ADOPTED — beat baseline held-out)."
           if brain["embedding"].get("metric_proven") else " (gated: baseline until a metric beats it held-out).")
    )
    factory = corpus_factory_status(memory_dir)                    # WS-H: the corpus factory's growth dashboard
    return {
        "counts": counts,
        "skills": skills,
        "warm_start": warm_start,
        "species_tree_nodes": species_nodes,
        "compounding": bool(reused),
        "accumulating": bool(accumulating),
        "brain": brain,
        "corpus_factory": factory,
        "summary": summary + (f" Factory: {factory['cumulative_annecs']} verified bodies across "
                              f"{factory['nights']} nights." if factory["nights"] else ""),
    }
