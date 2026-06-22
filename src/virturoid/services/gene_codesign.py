"""Gene co-design (plan Phase A/D): CEM over a gene's parameters + its controller.

The keystone makes any gene buildable; this makes it *good*. A cross-entropy search jointly
optimizes the body (limb-length scale, actuator torque scale, arm mount pitch) and the brain
(controller gains + phase length) against REAL MuJoCo pick-and-place success. This is the
"physics disposes" half: the LLM/seed proposes a gene, and physics-grounded search refines
it — which is exactly how a from-scratch humanoid goes from running to performing.

Gradient-free CEM is the right default here because pick-and-place is contact-rich and
discontinuous (analytic contact gradients are unreliable — Suh et al. 2022), per
docs/ai_architecture_plan.md.
"""

from __future__ import annotations

import dataclasses

from virturoid.schemas.gene import RobotGene
from virturoid.services.gene_build import default_controller_params, evaluate_gene_pick_place

# Search vector: [length_scale, torque_scale, mount_pitch, kp, kd]. Bounds keep variants
# physically sane and the controller stable.
_BOUNDS = {
    "length_scale": (0.8, 1.25),
    "torque_scale": (0.8, 2.0),
    "mount_pitch": (0.0, 1.6),     # radians; applied to the first pitched actuated segment
    "kp": (6.0, 70.0),
    "kd": (1.0, 9.0),
}
_KEYS = list(_BOUNDS)


def _variant(gene: RobotGene, vec: dict, *, ground: bool = False) -> RobotGene:
    """Apply a search vector to produce a variant gene (body params only). With ``ground=True`` the variant
    is physically GROUNDED (real masses from material+geometry, a real actuator per joint) before it is
    returned — so co-design over a grounded substrate optimizes real-physics bodies, not guessed scalars.
    The search's ``torque_scale`` then acts as a torque-DEMAND multiplier that selects a real actuator."""
    pitched_done = False
    segs = []
    for s in gene.segments:
        ns = dataclasses.replace(s)
        if s.joint_type in ("revolute", "prismatic"):
            ns.length_m = round(s.length_m * vec["length_scale"], 5)
            if s.actuator_torque_nm:
                ns.actuator_torque_nm = round(s.actuator_torque_nm * vec["torque_scale"], 3)
            # Set the mount pitch on the first pitched (axis~y) actuated segment.
            if not pitched_done and abs(s.joint_axis[1]) > 0.5:
                ns.mount_euler = (s.mount_euler[0], round(vec["mount_pitch"], 4), s.mount_euler[2])
                pitched_done = True
        elif s.parent is not None:
            ns.length_m = round(s.length_m * vec["length_scale"], 5)
        segs.append(ns)
    variant = dataclasses.replace(gene, id=f"{gene.id}_v", segments=segs)
    if ground:
        from virturoid.services.grounded_physics import ground_gene
        ground_gene(variant)
    return variant


def _seed_vec(gene: RobotGene) -> dict:
    params = default_controller_params(gene) or {"kp": 9.0, "kd": 1.4}
    # Seed mount pitch from the gene's existing first pitched segment.
    mp = 0.0
    for s in gene.segments:
        if s.joint_type in ("revolute", "prismatic") and abs(s.joint_axis[1]) > 0.5:
            mp = s.mount_euler[1]
            break
    return {"length_scale": 1.0, "torque_scale": 1.0, "mount_pitch": mp,
            "kp": float(params["kp"]), "kd": float(params["kd"])}


def co_design_general(gene: RobotGene, prompt: str = "", *, iterations: int = 3, population: int = 6,
                      seed: int = 0, elite_frac: float = 0.4, warm_start: dict | None = None,
                      ground: bool = False) -> dict:
    """Body co-design for ANY morphology — CEM over (length_scale, torque_scale) scored by the topology-
    dispatched task metric (``task_matched_eval.evaluate_robot``): a quadruped is tuned for locomotion
    distance, a rover for goal-reach rate, an arm for grasp/pick-place. One routine instead of a
    per-class optimizer (Theme 3). Never regresses (the baseline body is always a candidate). CPU MuJoCo.

    Returns ``{gene, baseline_value, best_value, history, changed}`` — ``gene`` is the best variant.
    """
    import numpy as np

    from virturoid.services.task_matched_eval import evaluate_robot

    seedv = _seed_vec(gene)
    lo = np.array([_BOUNDS["length_scale"][0], _BOUNDS["torque_scale"][0]])
    hi = np.array([_BOUNDS["length_scale"][1], _BOUNDS["torque_scale"][1]])

    def eval_vec(v) -> float:
        vec = {"length_scale": float(v[0]), "torque_scale": float(v[1]),
               "mount_pitch": seedv["mount_pitch"], "kp": seedv["kp"], "kd": seedv["kd"]}
        try:
            return float(evaluate_robot(_variant(gene, vec, ground=ground), prompt=prompt)["value"])
        except Exception:  # noqa: BLE001 - a broken variant scores worst, never crashes the search
            return -1e9

    rng = np.random.default_rng(seed)
    # Warm-start the search mean from a banked best design (the flywheel: each build starts from the
    # prior best for this morphology+task, not from scratch).
    if warm_start:
        mean = np.clip(np.array([float(warm_start.get("length_scale", 1.0)),
                                 float(warm_start.get("torque_scale", 1.0))]), lo, hi)
        std = np.array([0.08, 0.2])
    else:
        mean = np.array([1.0, 1.0]); std = np.array([0.15, 0.4])
    base = eval_vec(mean)
    best_val, best_v = base, mean.copy()
    history = [round(base, 4)]
    for _ in range(iterations):
        pop = np.clip(rng.normal(mean, std, (population, 2)), lo, hi)
        pop[0] = mean                                   # always keep the current best (never-regress)
        scores = np.array([eval_vec(v) for v in pop])
        order = np.argsort(scores)[::-1]
        n_elite = max(2, int(population * elite_frac))
        elite = pop[order[:n_elite]]
        mean, std = elite.mean(0), elite.std(0) + 1e-3
        if scores[order[0]] > best_val:
            best_val, best_v = float(scores[order[0]]), pop[order[0]].copy()
        history.append(round(best_val, 4))
    best_vec = {"length_scale": float(best_v[0]), "torque_scale": float(best_v[1]),
                "mount_pitch": seedv["mount_pitch"], "kp": seedv["kp"], "kd": seedv["kd"]}
    return {"gene": _variant(gene, best_vec, ground=ground), "baseline_value": round(base, 4),
            "best_value": round(best_val, 4), "history": history,
            "changed": {"length_scale": round(float(best_v[0]), 3),
                        "torque_scale": round(float(best_v[1]), 3)}}


def co_optimize_gene(gene: RobotGene, scenes, *, assignments: dict | None = None, iterations: int = 4,
                     population: int = 8, elite: int = 3, seed: int = 0, surrogate=None,
                     screen_keep: float = 0.5) -> dict:
    """CEM over (body + controller) maximizing real pick-place success. Returns the best.

    Returns {gene, controller_params, success_rate, baseline_success, history}. The returned
    gene is always valid; the best controller params feed run_pick_place_episode.

    If ``surrogate`` (a trained GeneGNN/fitness model with ``.predict``) is given, it is used
    as a cheap **screen** (plan Phase C): each generation's candidates are pre-ranked by the
    surrogate and only the top ``screen_keep`` fraction get expensive MuJoCo evaluation —
    spending sim budget on the genes the learned model thinks are promising.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    mean = np.array([_seed_vec(gene)[k] for k in _KEYS], dtype=float)
    lo = np.array([_BOUNDS[k][0] for k in _KEYS])
    hi = np.array([_BOUNDS[k][1] for k in _KEYS])
    std = (hi - lo) * 0.25

    def score(vec_arr) -> tuple[float, dict, dict]:
        vec = {k: float(np.clip(v, lo[i], hi[i])) for i, (k, v) in enumerate(zip(_KEYS, vec_arr))}
        cand = _variant(gene, vec)
        params = {"kp": vec["kp"], "kd": vec["kd"], "phase_steps": 300}
        r = evaluate_gene_pick_place(cand, scenes, params=params, assignments=assignments)
        return r["success_rate"], vec, params

    baseline, _, _ = score(mean)
    best = (baseline, dict(zip(_KEYS, mean)), {"kp": float(mean[3]), "kd": float(mean[4]), "phase_steps": 300})
    history = [round(baseline, 3)]

    for _ in range(iterations):
        pop = rng.normal(mean, std, size=(population, len(_KEYS)))
        pop = np.clip(pop, lo, hi)
        idxs = list(range(population))
        if surrogate is not None and population > 2:
            # Cheap screen: keep only the surrogate's top fraction for real MuJoCo eval.
            ranked = sorted(idxs, key=lambda i: surrogate.predict(_variant(
                gene, {k: float(pop[i][j]) for j, k in enumerate(_KEYS)})), reverse=True)
            idxs = ranked[: max(2, int(population * screen_keep))]
        scored = [score(pop[i]) for i in idxs]
        scored.sort(key=lambda t: t[0], reverse=True)
        if scored[0][0] > best[0]:
            best = scored[0]
        elites = np.array([[s[1][k] for k in _KEYS] for s in scored[:elite]])
        mean = elites.mean(axis=0)
        std = elites.std(axis=0) + 1e-3
        history.append(round(scored[0][0], 3))

    best_success, best_vec, best_params = best
    return {
        "gene": _variant(gene, best_vec),
        "controller_params": best_params,
        "success_rate": round(best_success, 3),
        "baseline_success": round(baseline, 3),
        "history": history,
    }
