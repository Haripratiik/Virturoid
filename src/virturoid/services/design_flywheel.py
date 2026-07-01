"""Design flywheel (Theme 3 moat, CPU): co-designed bodies are banked and WARM-START future builds, so
the platform's designs improve across builds (and across customers) without any GPU.

Each build of a given morphology+task: (1) look up the best banked design, (2) warm-start the general
co-design from it (start the search at the prior best, not from scratch), (3) bank the result. The
second build of a similar robot therefore begins where the first left off — the self-improving loop the
flywheel vision calls for, here over DESIGNS (the policy/skill flywheel is the GPU track). Reuses the
existing ``MemoryDB`` (runs/designs/species tables) — one flywheel, not a parallel store.
"""

from __future__ import annotations

import json


def co_design_with_memory(gene, prompt: str, db, *, iterations: int = 3, population: int = 6,
                          seed: int = 0, best_response: bool = False, archive_path=None,
                          br_samples: int = 3, br_iters: int = 1, br_steps: int = 400) -> dict:
    """Warm-start general co-design from the best banked design for this (class, task), run it, and bank
    the result. Returns the co-design result plus ``warm_started`` + ``prior_best``.

    ``best_response=True`` scores candidate bodies by their best-response controller (the Stackelberg /
    trainability objective — Pillar 1). When this build warm-started from a prior design, a provenance
    edge (with the measured search improvement Δ = best - baseline over the prior best) is recorded in the
    robotics vector memory so cross-build compounding is *provable, not asserted* (Pillar 2). ``archive_path``
    (a JSON MAP-Elites archive) makes this the full Designer→Trainer→Curator step: the result is inserted
    into its morphology niche, so the flywheel illuminates DIVERSE bodies (guards diversity collapse)."""
    from virturoid.services.gene_codesign import co_design_general
    from virturoid.services.task_matched_eval import robot_kind

    cls = gene.robot_class
    task = robot_kind(gene)
    prior = db.best_design(cls, task)
    warm = (prior or {}).get("converged_design")
    if isinstance(warm, str):
        try:
            warm = json.loads(warm)
        except Exception:  # noqa: BLE001
            warm = None

    r = co_design_general(gene, prompt, iterations=iterations, population=population, seed=seed,
                          warm_start=warm if isinstance(warm, dict) else None, best_response=best_response,
                          br_samples=br_samples, br_iters=br_iters, br_steps=br_steps)

    db.record_run(prompt=prompt, robot_class=cls, task_type=task, converged_design=r["changed"],
                  success_rate=r["best_value"], species=getattr(gene, "species", None),
                  design_source="co_design_general")
    r["warm_started"] = isinstance(warm, dict)
    r["prior_best"] = (prior or {}).get("success_rate")
    r["provenance_delta"] = None
    if r["warm_started"]:
        # The compounding proof: record what seeded this build + the measured improvement over it.
        try:
            vm = db.vector_memory()
            if vm is not None:
                child = getattr(gene, "species", None) or f"{cls}.{task}"
                parent = (prior or {}).get("species") or f"{cls}.{task}.prior"
                delta = round(float(r["best_value"]) - float(r["baseline_value"]), 4)
                vm.record_provenance("design", str(child), parent_type="design", parent_id=str(parent),
                                     kind="warm_start", delta=delta,
                                     meta={"robot_class": cls, "task": task, "prompt": prompt})
                r["provenance_delta"] = delta
        except Exception:  # noqa: BLE001 - provenance is observability; never break the flywheel
            pass
    # Curator: illuminate the MAP-Elites archive with this build (best body + its follower), so the
    # flywheel covers DIVERSE morphology niches rather than a single global best.
    r["archive_action"] = None
    if archive_path is not None:
        try:
            from pathlib import Path

            from virturoid.services.map_elites_archive import MapElitesArchive
            arc = MapElitesArchive.load(archive_path) if Path(archive_path).exists() else MapElitesArchive()
            r["archive_action"] = arc.insert(r["gene"], r["best_value"], controller=r.get("best_controller"),
                                             meta={"robot_class": cls, "task": task, "prompt": prompt})
            arc.save(archive_path)
            r["archive_summary"] = arc.summary()
        except Exception:  # noqa: BLE001 - the archive is observability; never break the flywheel
            pass
    return r


def topology_codesign_with_memory(gene, prompt: str, db, policy, *, n_range=range(3, 8),
                                  steps: int = 600, seed: int = 0) -> dict:
    """Topology flywheel: INVENT the best leg count (zero-shot scored by the shared policy), warm-started
    by the banked best topology for this (class, task), then bank the winner. So the flywheel improves
    SHAPE across builds, not just scale. Returns the search result + ``warm_started`` + ``prior_best``."""
    from virturoid.services.task_matched_eval import robot_kind
    from virturoid.services.topology_codesign import search_leg_count

    cls = gene.robot_class
    task = robot_kind(gene)
    prior = db.best_design(cls, task)
    warm = (prior or {}).get("converged_design")
    if isinstance(warm, str):
        try:
            warm = json.loads(warm)
        except Exception:  # noqa: BLE001
            warm = None
    prior_n = warm.get("n_legs") if isinstance(warm, dict) else None
    # warm-start: focus the search in a window around the banked best topology (else the full range)
    rng = range(max(3, prior_n - 1), prior_n + 2) if isinstance(prior_n, int) else n_range

    res = search_leg_count(policy, n_range=rng, steps=steps,
                           baseline_n=(prior_n if isinstance(prior_n, int) else 4))
    db.record_run(prompt=prompt, robot_class=cls, task_type=task,
                  converged_design={"n_legs": res["best_n_legs"], "score": res["best_score"]},
                  success_rate=res["best_score"], species=getattr(gene, "species", None),
                  design_source="topology_codesign")
    res["warm_started"] = isinstance(prior_n, int)
    res["prior_best"] = (prior or {}).get("success_rate")
    return res
