"""Harness banking — write a verified design-search win back into the flywheel (breakthrough plan H5 close /
Engineer-mode step). This is the compounding moat: a search that passes the honesty gate banks its winning
edit as a morphology-keyed tip + a provenance edge, so the NEXT search on a similar body recalls it and
warm-starts — the transfer-delta metric that Claude+MCP structurally cannot have (no cross-session memory).

Verified-only (banks only ``report.solved``), reusing the Phase-4 ``record_verified_knowledge`` (which re-gates
success >= target, so a caller can't accidentally bank a non-win) + a provenance edge stamping the measured
search fitness and the winning edit. Idempotent. Best-effort — a memory failure never breaks the search.
"""

from __future__ import annotations

from pathlib import Path

_HEADLINE = {"locomotion": "forward_m", "manipulation": "success_rate"}


def bank_search_result(report, *, gene, memory_dir, task_type: str = "locomotion",
                       gate_target: float = 0.30, species_pattern: str | None = None) -> dict:
    """Bank ``report.best`` into ``memory_dir`` IFF the search produced a verified win. Returns a summary dict.

    Records: a verified tip keyed to the body's species ("<class> walks/grasps via <edit>"), and a provenance
    edge (kind=design_search, delta=fitness, meta=the winning edit + eval count). ``species_pattern`` is looked
    up / created via ``auto_place_species`` when not given."""
    if report is None or not report.solved or report.best is None:
        return {"banked": False, "reason": "no verified win to bank"}
    node = report.best
    fam = "manipulation" if task_type in ("grasp", "grasp_lift", "pick_place", "pick_place_sort", "stack",
                                          "shelf", "push", "transport", "manipulation") else "locomotion"
    headline = float(node.artifact["metrics"].get(_HEADLINE[fam], 0.0))

    try:
        from virturoid.services.knowledge_writer import record_verified_knowledge
        from virturoid.services.memory_db import MemoryDB
        from virturoid.services.robotics_vector_memory import SKILL, RoboticsVectorMemory
        from virturoid.services.species_discovery import auto_place_species
        db_path = Path(memory_dir) / "virturoid_memory.db"
        with MemoryDB(db_path) as db:
            sp = species_pattern or auto_place_species(gene, db)["species_pattern"]
            edit = node.spec if isinstance(node.spec, dict) else {"spec": str(node.spec)}
            wb = record_verified_knowledge(
                db, gene, sp, task_type=task_type, success=headline, target=gate_target,
                outcome={"edit_kind": edit.get("edit_kind"), "params": edit.get("params"),
                         **node.artifact["metrics"]})
            vm = RoboticsVectorMemory(db)
            vm.record_provenance(SKILL, f"{sp}:{task_type}", kind="design_search", delta=float(node.fitness),
                                 meta={"edit": edit, "evals": report.n_evals, "verdict": node.verdict})
        return {"banked": bool(wb.get("wrote_tip")), "tip": wb.get("tip"), "species": sp,
                "headline": round(headline, 3), "fitness": node.fitness}
    except Exception as exc:  # noqa: BLE001 - banking is best-effort; never break the caller
        return {"banked": False, "reason": f"{type(exc).__name__}: {exc}"}
