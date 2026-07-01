"""The Design Brain panel — the moat, measured in one call (the demo's data backbone).

The USP is "it gets smarter with every design." That claim is only credible if it is *measurable and
rising*, so this aggregates the two signals the live build loop now writes to a shared memory dir:

  * **MAP-Elites archive** (``design_archive.json``): coverage (how many distinct morphology niches the
    design brain has illuminated) + QD-score (their summed quality) — the *breadth* of learned design space.
  * **Provenance ledger** (the robotics vector memory): warm-start edges + the measured mean success-delta
    each new build extracted from the accumulated prior — the *compounding* itself.

`design_brain_summary(memory_dir)` returns both in one dict, so a demo/report/API can show two numbers
rising build-over-build (coverage/QD and provenance mean-delta) — the second-derivative-positive visual that
no from-scratch generator can produce. Best-effort + pure aggregation (never raises); returns zeros when a
fresh memory dir has nothing banked yet.
"""

from __future__ import annotations

from pathlib import Path


def design_brain_summary(memory_dir) -> dict:
    """Aggregate the MAP-Elites archive + provenance compounding for a shared ``memory_dir`` into the
    Design Brain metrics. Returns ``{archive_coverage, qd_score, best_score, provenance_edges,
    seeded_builds, mean_delta, positive_fraction}`` (zeros/None when nothing is banked yet)."""
    memory_dir = Path(memory_dir)
    out = {"archive_coverage": 0, "qd_score": 0.0, "best_score": None,
           "provenance_edges": 0, "seeded_builds": 0, "mean_delta": None, "positive_fraction": None}

    try:  # MAP-Elites archive (the Curator's illumination breadth)
        from virturoid.services.map_elites_archive import MapElitesArchive
        arc_path = memory_dir / "design_archive.json"
        if arc_path.exists():
            s = MapElitesArchive.load(arc_path).summary()
            out["archive_coverage"] = s["coverage"]
            out["qd_score"] = s["qd_score"]
            out["best_score"] = s["best_score"]
    except Exception:  # noqa: BLE001 - pure aggregation; never raise
        pass

    try:  # provenance compounding (the flywheel's measured reuse benefit)
        from virturoid.services.memory_db import MemoryDB
        db_path = memory_dir / "virturoid_memory.db"
        if db_path.exists():
            with MemoryDB(db_path) as db:
                vm = db.vector_memory()
                if vm is not None:
                    c = vm.compounding_summary()
                    out["provenance_edges"] = c["edges"]
                    out["seeded_builds"] = c["seeded_builds"]
                    out["mean_delta"] = c["mean_delta"]
                    out["positive_fraction"] = c["positive_fraction"]
    except Exception:  # noqa: BLE001
        pass

    # a one-line headline for the demo/report
    out["headline"] = (f"{out['archive_coverage']} design niches illuminated "
                       f"(QD {out['qd_score']}); {out['provenance_edges']} warm-start edges"
                       + (f", mean +{out['mean_delta']} per reuse" if out["mean_delta"] else ""))
    return out
