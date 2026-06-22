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
                          seed: int = 0) -> dict:
    """Warm-start general co-design from the best banked design for this (class, task), run it, and bank
    the result. Returns the co-design result plus ``warm_started`` + ``prior_best``."""
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
                          warm_start=warm if isinstance(warm, dict) else None)

    db.record_run(prompt=prompt, robot_class=cls, task_type=task, converged_design=r["changed"],
                  success_rate=r["best_value"], species=getattr(gene, "species", None),
                  design_source="co_design_general")
    r["warm_started"] = isinstance(warm, dict)
    r["prior_best"] = (prior or {}).get("success_rate")
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
