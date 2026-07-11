"""Concept memory × embedding unification (master_plan_v6 §WS-G).

Two parallel knowledge systems should ground each other: **concept memory** (the LLM names a concept; physics
evidence promotes it candidate→evaluated→verified) and the **embedding** (bodies as z_body vectors). A verified
concept should own a *region* of body-space, and a brand-new concept should be grounded by its embedding
neighbours BEFORE its first build — the robotics AI grounding a word the way an LLM grounds language.

The hard rule from concept_memory is preserved exactly: **routing is exact-alias only, never fuzzy.** So this
module keeps two things strictly separate:

  * ``ground_concept`` first tries the deterministic exact-alias route (``recall_verified_route``). If it hits, the
    concept is ROUTED — no similarity involved.
  * Only when there is NO exact route (a genuinely novel word) does it produce an embedding **grounding report**:
    the nearest VERIFIED concepts by z_body + distance + their physics outcomes, clearly labelled as *advisory
    similarity* the model may reason over — never a silent substitution of one concept's route for another.

Reads the concepts table directly (never mutates it), so it composes with whatever concept_memory does upstream.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


def _json(v):
    if isinstance(v, (list, dict)):
        return v
    try:
        return json.loads(v) if v else []
    except (TypeError, json.JSONDecodeError):
        return []


def verified_concepts(db) -> list[dict]:
    """Every concept the physics evidence promoted to ``verified`` (with its execution outcome + species)."""
    conn = db.conn if hasattr(db, "conn") else db
    try:
        rows = conn.execute("SELECT * FROM concepts WHERE state='verified'").fetchall()
    except Exception:  # noqa: BLE001 - the table may not exist yet
        return []
    out = []
    for r in rows:
        d = dict(r)
        ev = _json(d.get("evidence"))
        best = max((float(e.get("success_rate", 0.0)) for e in ev if e.get("verified")), default=None)
        out.append({"concept_id": d.get("concept_id"), "label": d.get("label"),
                    "execution_family": d.get("execution_family"), "task_type": d.get("task_type"),
                    "species_pattern": d.get("species_pattern"), "best_success": best})
    return out


def _cos(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def _z_body(gene):
    from virturoid.services.robotics_vector_memory import embed_body
    return embed_body(gene)


def concept_neighbors(db, query_gene, *, k: int = 3) -> list[dict]:
    """The k VERIFIED concepts whose banked body is embedding-nearest to ``query_gene`` — each with its outcome.
    Pure similarity ranking (advisory); never returns a route the caller should silently adopt."""
    if query_gene is None:
        return []
    from virturoid.services.robotics_vector_memory import RoboticsVectorMemory
    genes = RoboticsVectorMemory(db)._species_genes()
    q = _z_body(query_gene)
    scored = []
    for c in verified_concepts(db):
        g = genes.get(c.get("species_pattern"))
        if g is None:
            continue
        sim = _cos(q, _z_body(g))
        scored.append({"concept": c["label"], "similarity": round(sim, 3),
                       "execution_family": c.get("execution_family"), "task_type": c.get("task_type"),
                       "outcome": (f"builds as a {c.get('execution_family')} doing {c.get('task_type')}"
                                   + (f" (verified {c['best_success']:.0%})" if c.get("best_success") else "")),
                       "best_success": c.get("best_success")})
    scored.sort(key=lambda s: s["similarity"], reverse=True)
    return scored[:k]


def ground_concept(memory_dir, concept: str, query_gene=None, *, aliases: list[str] | None = None,
                   k: int = 3) -> dict:
    """Ground a concept for the model. Returns either a deterministic exact-alias ROUTE, or (for a novel word) an
    advisory embedding grounding report — the robotics AI grounding a new word in what has actually been verified.

    Contract: a novel concept NEVER inherits a route; the grounding is similarity-only, and the model decides.
    """
    from virturoid.services.concept_memory import recall_verified_route
    routed = recall_verified_route(Path(memory_dir), concept, aliases=aliases)
    if routed:
        return {"concept": concept, "routed": True,
                "route": {"execution_family": routed.get("execution_family"),
                          "task_type": routed.get("task_type"), "species_pattern": routed.get("species_pattern")},
                "note": "exact verified route (canonical label or an LLM-proposed EXACT alias) — deterministic, "
                        "not a similarity guess"}
    # a genuinely novel word: ground it by its embedding neighbours (advisory)
    neighbors = []
    try:
        from virturoid.services.memory_db import MemoryDB
        db_path = Path(memory_dir) / "virturoid_memory.db"
        if db_path.exists() and query_gene is not None:
            with MemoryDB(db_path) as db:
                neighbors = concept_neighbors(db, query_gene, k=k)
    except Exception:  # noqa: BLE001 - grounding is advisory; a missing corpus just yields an empty report
        neighbors = []
    return {"concept": concept, "routed": False, "grounding": neighbors,
            "note": ("these are the embedding-NEAREST VERIFIED concepts (advisory similarity), NOT a route — a "
                     "new word never inherits another concept's execution route silently. The model reasons over "
                     "the neighbours + their physics outcomes and decides." if neighbors else
                     "no verified concepts to ground against yet — this word is genuinely new to the corpus")}


def concept_summary(db) -> dict:
    """Concept-memory counts by lifecycle state — for the Brain panel (traceable, read straight from the table)."""
    conn = db.conn if hasattr(db, "conn") else db
    out = {"candidate": 0, "evaluated": 0, "verified": 0}
    try:
        for r in conn.execute("SELECT state, COUNT(*) n FROM concepts GROUP BY state").fetchall():
            out[str(r["state"])] = int(r["n"])
    except Exception:  # noqa: BLE001
        pass
    out["total"] = sum(out.get(s, 0) for s in ("candidate", "evaluated", "verified"))
    return out
