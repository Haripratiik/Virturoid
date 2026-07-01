"""Agentic design-cycle demo (Pillars 1+2): watch the robot-design brain get smarter across builds.

Runs the Designer -> Trainer(best-response) -> Curator(MAP-Elites) loop over a few legged prompts against
ONE shared memory, and prints the moat metrics that make the USP ("the self-improving robot-design brain")
concrete:
  - Designer: each build warm-starts its co-design from the best prior banked design (not from scratch).
  - Trainer: each candidate BODY is scored by the best CONTROLLER trainable on it (Stackelberg / honest
    locomotion score) -> the search optimizes for TRAINABILITY, and returns the follower it found.
  - Curator: the result is inserted into a MAP-Elites archive, illuminating DIVERSE morphology niches.
  - Memory: a provenance edge records the measured compounding (Δ vs the warm-started prior).

CPU-only (MuJoCo). Run:  PYTHONPATH=src python scripts/agentic_design_cycle_demo.py
"""

import tempfile
from pathlib import Path

from virturoid.fixtures.gene_library import quadruped_gene
from virturoid.services.design_flywheel import co_design_with_memory
from virturoid.services.map_elites_archive import MapElitesArchive
from virturoid.services.memory_db import MemoryDB

PROMPTS = [
    "a quadruped robot that walks",
    "a fast lightweight quadruped that trots",
    "a six-legged walker",          # a different body plan -> a new archive niche (illumination)
]


def main():
    with tempfile.TemporaryDirectory() as td:
        arc_path = Path(td) / "design_archive.json"
        with MemoryDB(Path(td) / "mem.db") as db:
            print("=== Agentic design cycle: does the design brain compound? ===\n")
            for i, prompt in enumerate(PROMPTS):
                r = co_design_with_memory(quadruped_gene(prompt), prompt, db, iterations=1, population=2,
                                          seed=i, best_response=True, br_samples=2, br_iters=1,
                                          br_steps=200, archive_path=arc_path)
                ctrl = r.get("best_controller") or {}
                summ = r.get("archive_summary") or {}
                print(f"build {i}: '{prompt}'")
                print(f"  warm_started={r['warm_started']}  best_value={r['best_value']:+.3f}"
                      f"  delta_vs_prior={r.get('provenance_delta')}")
                print(f"  best-response controller: freq={ctrl.get('freq')} "
                      f"thigh_amp={ctrl.get('thigh_amp')} leg_flip={ctrl.get('leg_flip')}")
                print(f"  curator: {r.get('archive_action')} -> coverage={summ.get('coverage')} "
                      f"QD={summ.get('qd_score')}\n")

            comp = db.vector_memory().compounding_summary()
            arc = MapElitesArchive.load(arc_path)
            best = arc.best() or {}
            print("=== Moat metrics (the compounding, made measurable) ===")
            print(f"  provenance: {comp['edges']} warm-start edges over {comp['seeded_builds']} builds, "
                  f"mean delta={comp['mean_delta']}")
            print(f"  MAP-Elites: {arc.coverage()} niches illuminated, QD-score={arc.qd_score()}")
            print(f"  best design: score={best.get('score')} niche={best.get('descriptor')}")
            print("\nA competitor starting cold has no archive and no provenance-verified warm-starts;\n"
                  "the gap widens with every build. That is the moat.")


if __name__ == "__main__":
    main()
