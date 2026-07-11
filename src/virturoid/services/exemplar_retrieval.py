"""Query-specific verified-exemplar retrieval — scores OMITTED (master_plan_v6 §8.2.2 / WS-B.2).

RoboMorph reported its largest validity delta from retrieved exemplars — AND its sharpest trap: **putting the
exemplars' fitness scores in the prompt caused mode collapse** (the model just re-proposes the current best). So
the rule here is structural, not advisory: retrieval returns the *structure and provenance* of the embedding-
nearest VERIFIED bodies and **never their fitness/success/reward**. The model adapts a proven shape; it is never
told which shape "won", so it keeps exploring. ``omit_scores`` is enforced by construction and asserted in tests.

This upgrades ``get_design_schema``'s generic role exemplars to *query-specific* nearest verified designs: pass a
draft ``graph`` (or a robot_class) and the schema comes back grounded in the closest things that actually built.
"""
from __future__ import annotations

# fields a summary must NEVER contain — the mode-collapse guard (asserted by test_exemplar_retrieval)
_FORBIDDEN_SCORE_FIELDS = frozenset({"fitness", "score", "success", "success_rate", "reward", "forward_m",
                                     "verdict", "credible", "value", "metric"})


def structural_summary(gene) -> dict:
    """A SCORE-FREE structural description of a body — what to adapt, not how well it did. Composition only."""
    from virturoid.services.heldout_set import _leg_chain_count
    from virturoid.services.task_matched_eval import robot_kind
    segs = gene.segments
    shapes: dict[str, int] = {}
    for s in segs:
        shapes[getattr(s, "shape", "capsule")] = shapes.get(getattr(s, "shape", "capsule"), 0) + 1
    return {
        "robot_class": getattr(gene, "robot_class", None),
        "kind": robot_kind(gene),
        "n_segments": len(segs),
        "dof": len(gene.actuated_joints()),
        "limb_chains": _leg_chain_count(gene),
        "total_length_m": round(sum(float(getattr(s, "length_m", 0.0)) for s in segs), 3),
        "shape_counts": shapes,
        "base_mount": getattr(gene, "base_mount", None),
    }


def _assert_score_free(summary: dict) -> dict:
    """Belt-and-braces: strip any forbidden score field that ever leaks into a summary."""
    return {k: v for k, v in summary.items() if k not in _FORBIDDEN_SCORE_FIELDS}


def verified_exemplars(query_gene, *, k: int = 3, memory_dir=None, min_sim: float = 0.0) -> list[dict]:
    """The ``k`` embedding-nearest VERIFIED (buildable) bodies to ``query_gene`` — each as ``{exemplar,
    similarity, robot_class, structure}`` with **no fitness/score**. Best-effort: an empty corpus returns []."""
    try:
        from virturoid.services.agent_tools import safe_build_path
        from virturoid.services.memory_db import MemoryDB
        from virturoid.services.robotics_vector_memory import BODY, RoboticsVectorMemory
        mem = (safe_build_path(None, "memory") if memory_dir is None else memory_dir)
        db_path = (mem / "virturoid_memory.db") if hasattr(mem, "__truediv__") else mem
        if not db_path.exists():
            return []
        with MemoryDB(db_path) as db:
            vm = RoboticsVectorMemory(db)
            if vm.count(BODY) == 0:
                vm.index_species_bodies()
            hits = vm.nearest_bodies(query_gene, k=k, min_sim=min_sim)
            genes = vm._species_genes()
    except Exception:  # noqa: BLE001 - grounding is an accelerant; a missing/blank corpus never blocks design
        return []
    out: list[dict] = []
    for h in hits:
        meta = h.get("meta") or {}
        if not meta.get("buildable"):                       # VERIFIED-only: never ground on an unbuilt body
            continue
        sp = h.get("obj_id")
        row = {"exemplar": sp, "similarity": round(float(h.get("similarity", 0.0)), 3),
               "robot_class": meta.get("robot_class")}
        g = genes.get(sp)
        if g is not None:
            row["structure"] = _assert_score_free(structural_summary(g))
        out.append(row)
    return out


def query_gene_from_args(args: dict):
    """Build a query gene from design args: compile a draft ``graph`` if present (iterative-design retrieval),
    else None (the caller keeps the generic role exemplars). Never raises — a bad draft simply yields None."""
    graph = args.get("graph")
    if isinstance(graph, dict) and graph.get("parts"):
        try:
            from virturoid.services.anatomy_compiler import build_from_anatomy
            return build_from_anatomy(graph)
        except Exception:  # noqa: BLE001
            return None
    return None


def exemplar_grounding(args: dict, *, k: int = 3) -> dict:
    """The get_design_schema hook: query-specific verified exemplars for a draft design (``args['graph']``),
    scores omitted. Empty (omitted) when there's no draft to query against or the corpus is blank."""
    qg = query_gene_from_args(args)
    if qg is None:
        return {}
    ex = verified_exemplars(qg, k=k)
    if not ex:
        return {}
    return {"verified_exemplars": ex,
            "note": ("the embedding-nearest VERIFIED bodies to your draft — adapt their STRUCTURE (segment/DOF "
                     "composition, proportions). Fitness scores are intentionally omitted: adapt a proven shape, "
                     "don't copy 'the winner' (that collapses diversity). Similarity is morphology distance only.")}
