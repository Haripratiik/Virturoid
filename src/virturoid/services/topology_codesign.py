"""Topology-mutating co-design V1 (the generative-design moat) — INVENT topology, don't just tune it.

Existing co-design (`gene_codesign.co_design_general`) does CEM over scalar body params on a FIXED
topology. This searches over the topology itself (leg count, DOF-per-leg) and scores each candidate
**zero-shot with the shared morphology-agnostic MorphPolicy** — no per-candidate training, because one
attention policy drives any token count (the cross-morphology transfer we demonstrated). Never-regress.
This is the BodyGen idea (arXiv:2503.00533) in its simplest evolutionary form on our stack; the general
gene-operator version (add/remove arbitrary limbs via MorphologyBuilder) is the next step
(docs/topology_codesign_design.md). Relates to [[building-blocks-not-species]], [[general-codesign-flywheel]].
"""

from __future__ import annotations


def score_topology(gene, policy, *, steps: int = 600) -> float:
    """Zero-shot task score for a candidate topology: drive it with the SHARED policy and measure forward
    travel; reject bodies that go non-finite or fall over (so the ranking reflects body quality, not
    control luck). Returns ``-inf`` for a rejected body."""
    from virturoid.services.morph_policy import rollout_morph

    r = rollout_morph(gene, policy, steps=steps)
    # Reject bodies that BLOW UP (non-finite) or COLLAPSE to the ground. We use a height_ratio FLOOR, not the
    # strict ``upright`` bool (z>0.5*z0): zero-shot transfer of one shared policy onto a novel topology often
    # crouch-walks (forward in a low stance, height_ratio ~0.35-0.45) — a viable, rankable gait, NOT a fall.
    # The strict upright gate became real after the standing-spawn fix and rejected every crouch-walker as a
    # fall, making the search vacuous (all -inf). See [[task-effectiveness-loop]].
    if not r.get("finite", False) or float(r.get("height_ratio", 0.0)) < 0.3:
        return float("-inf")
    return float(r["forward"])


def search_topology(candidates: dict, policy, *, steps: int = 600, baseline: str | None = None) -> dict:
    """Score a dict ``{name: gene}`` of candidate topologies zero-shot with ONE shared policy and pick the
    best (never-regress: if ``baseline`` is given, the winner only replaces it on a strict improvement).
    Returns ``{best, best_gene, best_score, baseline_score, history}``."""
    history = {name: round(score_topology(g, policy, steps=steps), 4) for name, g in candidates.items()}
    best = max(history, key=history.get)
    base_score = history.get(baseline) if baseline in history else None
    if baseline in history and history[best] <= history[baseline]:
        best = baseline                                   # never regress off the incumbent
    return {"best": best, "best_gene": candidates[best], "best_score": history[best],
            "baseline_score": base_score, "history": history}


def search_morphology(policy, *, leg_counts=range(3, 7), dof_options=(2, 3), steps: int = 600,
                      baseline=(4, 3)) -> dict:
    """2-D topology co-design: invent the best (leg count, DOF-per-leg) jointly, each scored zero-shot by
    the shared policy, never-regress. Demonstrates topology search over MULTIPLE structural dimensions —
    e.g. whether the extra hip-yaw DOF (3 vs 2) earns its keep for a given leg count."""
    from virturoid.services.steerable_body import steerable_quadruped

    candidates = {(n, dof): steerable_quadruped(species=f"evo_{n}l{dof}d", n_legs=n, dofs_per_leg=dof)
                  for n in leg_counts for dof in dof_options}
    res = search_topology(candidates, policy, steps=steps, baseline=baseline)
    return {"best": res["best"], "best_gene": res["best_gene"], "best_score": res["best_score"],
            "baseline": baseline, "baseline_score": res["baseline_score"], "history": res["history"]}


def search_leg_count(policy, *, n_range=range(3, 8), steps: int = 600, baseline_n: int = 4) -> dict:
    """V1 topology co-design: invent the best LEG COUNT for a free legged body, scored zero-shot by the
    shared policy. Reuses ``steerable_quadruped`` (radial mounts for any leg count). Returns the best
    leg count + its gene + the full score history."""
    from virturoid.services.steerable_body import steerable_quadruped

    candidates = {n: steerable_quadruped(species=f"evo_{n}leg", n_legs=n) for n in n_range}
    res = search_topology(candidates, policy, steps=steps, baseline=baseline_n)
    return {"best_n_legs": res["best"], "best_gene": res["best_gene"], "best_score": res["best_score"],
            "baseline_n_legs": baseline_n, "baseline_score": res["baseline_score"],
            "history": res["history"]}
