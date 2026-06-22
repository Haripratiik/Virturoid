"""Flywheel runner (Theme 2): run many build cycles against ONE shared memory and measure the compounding.

Virturoid's moat is that every build makes the next one better — the design flywheel banks bodies, the
skill flywheel banks trained policies, and later builds warm-start from both (startup plan §13 memory,
§32 training). That claim is only worth anything if it is *measurable*, so this runner drives a sequence
of autonomous builds through a single ``memory_dir`` and snapshots, per cycle, how the banked memory
grows and whether each build had prior assets to warm-start from. The returned report is the moat made
auditable — and a regression guard against "flywheel rot" (memory that stops compounding).

The build step is injected (``build_fn``) so this is unit-testable without the slow real physics/training
loop; the default is the real :func:`virturoid.services.autonomous_build.autonomous_build`.
"""

from __future__ import annotations

from pathlib import Path

_MEMORY_TABLES = ("designs", "skills", "runs", "lessons", "species_tree")


def _memory_snapshot(memory_dir: Path) -> dict:
    """Counts of what is banked right now (best-effort; a fresh/missing DB reads as all-zero)."""
    snap = {t: 0 for t in _MEMORY_TABLES}
    try:
        from virturoid.services.memory_db import MemoryDB

        db_path = Path(memory_dir) / "virturoid_memory.db"
        if not db_path.exists():
            return snap
        with MemoryDB(db_path) as db:
            stats = db.stats()
        return {t: int(stats.get(t, 0)) for t in _MEMORY_TABLES}
    except Exception:  # noqa: BLE001 - observability must never crash the runner
        return snap


def run_flywheel(prompts, *, memory_dir, out_root, train: bool = False,
                 target_success_rate: float = 0.8, build_fn=None, progress=None) -> dict:
    """Run one autonomous build per prompt against a SHARED ``memory_dir`` and measure the compounding.

    Every cycle builds into ``out_root/cycle_<i>`` but reads/writes the same memory, so designs and
    skills banked by earlier cycles are available to later ones. For each cycle we capture the memory
    snapshot *before* and *after* the build, whether the build reused prior memory, and its success —
    so the returned report shows banked assets growing and warm-start availability rising over time.

    Returns ``{"cycles": [...], "totals": {...}, "compounding": bool}`` where ``compounding`` is the
    honest verdict: did the banked-asset count strictly grow across the run?
    """
    if build_fn is None:
        from virturoid.services.autonomous_build import autonomous_build as build_fn  # noqa: N806

    memory_dir = Path(memory_dir)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    def emit(stage: str, message: str, data: dict | None = None) -> None:
        if progress is not None:
            progress({"stage": stage, "message": message, "data": data or {}})

    cycles: list[dict] = []
    for i, prompt in enumerate(prompts):
        before = _memory_snapshot(memory_dir)
        warm_start_available = (before["designs"] + before["skills"]) > 0
        emit("cycle_start", f"Cycle {i}: building for '{prompt}' "
             f"(prior banked: {before['designs']} designs, {before['skills']} skills).",
             {"cycle": i, "before": before})

        out_dir = out_root / f"cycle_{i}"
        report = build_fn(prompt, out_dir, target_success_rate=target_success_rate,
                          train=train, memory_dir=memory_dir)
        after = _memory_snapshot(memory_dir)

        rec = {
            "cycle": i,
            "prompt": prompt,
            "robot_class": getattr(report, "robot_class", None),
            "task_type": getattr(report, "task_type", None),
            "species": getattr(report, "species", None),
            "success_rate": float(getattr(report, "final_success_rate", 0.0) or 0.0),
            "succeeded": bool(getattr(report, "succeeded", False)),
            "memory_reused": bool(getattr(report, "memory_reused", False)),
            "warm_start_available": warm_start_available,
            "banked_before": before,
            "banked_after": after,
            "designs_added": after["designs"] - before["designs"],
            "skills_added": after["skills"] - before["skills"],
        }
        cycles.append(rec)
        emit("cycle_done", f"Cycle {i}: {'succeeded' if rec['succeeded'] else 'did not reach target'} "
             f"({rec['success_rate']:.0%}); banked now {after['designs']} designs, {after['skills']} skills.",
             {"cycle": i, "record": rec})

    first, last = (cycles[0]["banked_before"], cycles[-1]["banked_after"]) if cycles else ({}, {})
    total_banked_first = sum(first.get(t, 0) for t in _MEMORY_TABLES)
    total_banked_last = sum(last.get(t, 0) for t in _MEMORY_TABLES)
    warm_started_cycles = sum(1 for c in cycles if c["warm_start_available"])
    totals = {
        "cycles": len(cycles),
        "designs_banked": last.get("designs", 0),
        "skills_banked": last.get("skills", 0),
        "warm_start_cycles": warm_started_cycles,
        "succeeded_cycles": sum(1 for c in cycles if c["succeeded"]),
        "mean_success": round(sum(c["success_rate"] for c in cycles) / len(cycles), 4) if cycles else 0.0,
    }
    return {
        "cycles": cycles,
        "totals": totals,
        # Honest verdict: across the run the shared memory strictly accumulated banked assets.
        "compounding": total_banked_last > total_banked_first and len(cycles) >= 2,
    }
