"""Stackelberg best-response co-design (Pillar 1 keystone): score a BODY by the best CONTROLLER you
can train on it — not by one fixed CPG.

This is "make it learn, don't teach it" made structural. The consensus of the co-design literature
(Eureka -> RoboMoRe -> Debate2Create -> Stackelberg PPO) is that a design's fitness must be the
performance of its *best-response* controller (the Stackelberg follower), so the search selects bodies
that are EASY TO CONTROL — trainability as the objective, per docs/usp_and_two_pillar_plan.md Pillar 1
and the research dossier's rec #1.

It fixes a concrete, logged failure ([[locomotion-eval-generality]]): the co-design/eval loop scored a
legged body with ONE bare CPG, so an anatomy-compiled walker that would step under a controller ADAPTED
to it was scored 0.0/"fell". Here the outer CEM over morphology (the leader) scores each candidate body
by an inner best-response controller search (the follower) — a short, warm-startable CPU search over the
gait's CPG shape + PD gains — evaluated by an HONEST, anti-Goodhart metric (a real WALK: survived +
foot-contact cadence + sustained-upright, never raw forward-distance which a slide games).

CPU-only: the inner search reuses ``recipe_rollout_morph`` (the trot-CPG recipe that actually walks
bodies on CPU). The GPU phase swaps the inner follower for MJX-PPO convergence behind the same seam.
"""

from __future__ import annotations

from virturoid.schemas.gene import RobotGene

# The follower's action space: the trot-CPG gait shape + PD gains. Bounds keep the gait stable and the
# oscillation physical. (leg_flip — the diagonal/forward direction — is searched discretely.)
_CTRL_BOUNDS = {
    "freq": (0.8, 3.0),        # CPG stride frequency (Hz-ish)
    "thigh_amp": (0.2, 1.0),   # hip/thigh swing amplitude (rad)
    "calf_amp": (0.3, 1.2),    # knee/calf amplitude (rad)
    "kp": (12.0, 60.0),        # position-PD stiffness
    "kd": (1.0, 4.0),          # position-PD damping
}
_CTRL_KEYS = list(_CTRL_BOUNDS)


def _default_ctrl() -> dict:
    from virturoid.services.morph_policy import CPG_DEFAULT
    return {"freq": float(CPG_DEFAULT["freq"]), "thigh_amp": float(CPG_DEFAULT["thigh_amp"]),
            "calf_amp": float(CPG_DEFAULT["calf_amp"]), "kp": 32.0, "kd": 1.5}


def honest_locomotion_score(r: dict) -> float:
    """A single anti-Goodhart fitness for a locomotion rollout: reward forward travel ONLY when the body
    genuinely steps upright (cadence >= 1 foot-liftoff/s AND sustained-upright >= 0.5), heavily discount a
    slide (cadence 0) or a fall, plus small shaping for uprightness/cadence/survival. So a real WALK ranks
    strictly above an upright SLIDE, which ranks above a forward LUNGE-and-fall — the ordering a raw
    forward-distance KPI gets wrong (see [[honest-locomotion-audit]])."""
    forward = max(0.0, float(r.get("forward", 0.0)))
    cadence = float(r.get("cadence", 0.0))
    upright = float(r.get("upright_frac", 0.0))
    survived = bool(r.get("survived", False))
    walk_gate = cadence >= 1.0 and upright >= 0.5           # a real stepping, upright gait
    base = forward if walk_gate else 0.1 * forward          # discount non-walks by 10x
    return round(base + 0.15 * upright + 0.03 * min(cadence, 8.0) + (0.1 if survived else 0.0), 4)


def best_response_locomotion(gene: RobotGene, *, samples: int = 6, iters: int = 2, steps: int = 500,
                             warm_start: dict | None = None, seed: int = 0,
                             try_both_flips: bool = True) -> dict:
    """The follower's best-response for a legged body: a short CEM over the CPG shape + PD gains, returning
    the best HONEST score and the controller that achieved it. Warm-started from a banked controller config
    (the flywheel) when given; both gait directions (``leg_flip``) are tried so a forward gait is found
    rather than an accidental backward one. CPU MuJoCo via ``recipe_rollout_morph``.

    Returns ``{score, controller, detail, history}`` where ``controller`` (incl. ``leg_flip``) reproduces
    the best gait for this exact body.
    """
    import numpy as np

    from virturoid.services.morph_policy import CPG_DEFAULT, recipe_rollout_morph

    lo = np.array([_CTRL_BOUNDS[k][0] for k in _CTRL_KEYS])
    hi = np.array([_CTRL_BOUNDS[k][1] for k in _CTRL_KEYS])
    d = _default_ctrl()
    mean = np.clip(np.array([float((warm_start or {}).get(k, d[k])) for k in _CTRL_KEYS]), lo, hi)
    std = (hi - lo) * 0.25
    rng = np.random.default_rng(seed)
    flips = (True, False) if try_both_flips else (bool((warm_start or {}).get("leg_flip", True)),)

    def eval_ctrl(varr, flip) -> tuple[float, dict]:
        v = {k: float(np.clip(varr[i], lo[i], hi[i])) for i, k in enumerate(_CTRL_KEYS)}
        cpg = dict(CPG_DEFAULT)
        cpg.update(freq=v["freq"], thigh_amp=v["thigh_amp"], calf_amp=v["calf_amp"], leg_flip=flip)
        try:
            r = recipe_rollout_morph(gene, steps=steps, kp=v["kp"], kd=v["kd"], cpg=cpg, seed=seed)
        except Exception:  # noqa: BLE001 - a broken controller/body scores worst, never crashes the search
            return -1e9, {}
        return honest_locomotion_score(r), r

    best = (-1e9, dict(zip(_CTRL_KEYS, mean.tolist())), True, {})
    history: list[float] = []
    for _ in range(max(1, iters)):
        pop = np.clip(rng.normal(mean, std, (samples, len(_CTRL_KEYS))), lo, hi)
        pop[0] = mean                                          # always keep the incumbent (never-regress)
        scored = []
        for row in pop:
            for flip in flips:
                s, r = eval_ctrl(row, flip)
                scored.append((s, row, flip, r))
        scored.sort(key=lambda t: t[0], reverse=True)
        if scored[0][0] > best[0]:
            top = scored[0]
            best = (float(top[0]), {k: float(top[1][i]) for i, k in enumerate(_CTRL_KEYS)}, bool(top[2]), top[3])
        n_elite = max(2, (samples * len(flips)) // 2)
        elite = np.array([t[1] for t in scored[:n_elite]])
        mean, std = elite.mean(0), elite.std(0) + 1e-3
        history.append(round(best[0], 4))
    return {"score": round(best[0], 4), "controller": {**best[1], "leg_flip": best[2]},
            "detail": best[3], "history": history}


def best_response_score(gene: RobotGene, prompt: str = "", *, samples: int = 6, iters: int = 2,
                        steps: int = 500, warm_start: dict | None = None, seed: int = 0) -> dict:
    """A body's Stackelberg fitness = its best-response controller's honest score. Legged bodies get the
    locomotion best-response search; other morphologies fall back to the task-matched fixed-controller eval
    (the follower search for manipulation/nav is a later increment — the locomotion gap is the logged one).

    Returns ``{score, controller, kind, detail}``.
    """
    from virturoid.services.task_matched_eval import evaluate_robot, robot_kind

    kind = robot_kind(gene)
    if kind == "legged":
        br = best_response_locomotion(gene, samples=samples, iters=iters, steps=steps,
                                      warm_start=warm_start, seed=seed)
        return {"score": br["score"], "controller": br["controller"], "kind": kind, "detail": br["detail"]}
    try:
        value = float(evaluate_robot(gene, prompt=prompt)["value"])
    except Exception:  # noqa: BLE001 - a broken body scores worst
        value = -1e9
    return {"score": round(value, 4), "controller": None, "kind": kind, "detail": {}}


def co_design_stackelberg(gene: RobotGene, prompt: str = "", *, iterations: int = 2, population: int = 4,
                          seed: int = 0, elite_frac: float = 0.4, warm_start: dict | None = None,
                          ground: bool = False, br_samples: int = 5, br_iters: int = 1,
                          br_steps: int = 500) -> dict:
    """Leader-follower co-design: an outer CEM over the BODY (length/torque scale) where each candidate is
    scored by its inner best-response CONTROLLER (``best_response_score``), so the search optimizes for
    TRAINABILITY rather than for flattering one fixed CPG. Never regresses (the seed body is always a
    candidate). Returns ``{gene, best_controller, baseline_value, best_value, history, changed}`` — the
    winning body AND the follower that controls it (ready to deploy / warm-start the GPU trainer).
    """
    import numpy as np

    from virturoid.services.gene_codesign import _BOUNDS, _seed_vec, _variant

    seedv = _seed_vec(gene)
    lo = np.array([_BOUNDS["length_scale"][0], _BOUNDS["torque_scale"][0]])
    hi = np.array([_BOUNDS["length_scale"][1], _BOUNDS["torque_scale"][1]])

    def eval_body(v) -> tuple[float, dict | None]:
        vec = {"length_scale": float(v[0]), "torque_scale": float(v[1]),
               "mount_pitch": seedv["mount_pitch"], "kp": seedv["kp"], "kd": seedv["kd"]}
        try:
            variant = _variant(gene, vec, ground=ground)
            br = best_response_score(variant, prompt=prompt, samples=br_samples, iters=br_iters,
                                     steps=br_steps, warm_start=warm_start, seed=seed)
            return br["score"], br["controller"]
        except Exception:  # noqa: BLE001 - a broken variant scores worst, never crashes the search
            return -1e9, None

    rng = np.random.default_rng(seed)
    mean = np.clip(np.array([float((warm_start or {}).get("length_scale", 1.0)),
                             float((warm_start or {}).get("torque_scale", 1.0))]), lo, hi)
    std = np.array([0.1, 0.3])
    base_score, base_ctrl = eval_body(mean)
    best_score, best_v, best_ctrl = base_score, mean.copy(), base_ctrl
    history = [round(base_score, 4)]
    for _ in range(max(1, iterations)):
        pop = np.clip(rng.normal(mean, std, (population, 2)), lo, hi)
        pop[0] = mean                                          # never-regress: keep the incumbent
        scored = [(sc, ct, v) for (sc, ct), v in ((eval_body(v), v) for v in pop)]
        scored.sort(key=lambda t: t[0], reverse=True)
        if scored[0][0] > best_score:
            best_score, best_ctrl, best_v = scored[0][0], scored[0][1], scored[0][2].copy()
        n_elite = max(2, int(population * elite_frac))
        elite = np.array([t[2] for t in scored[:n_elite]])
        mean, std = elite.mean(0), elite.std(0) + 1e-3
        history.append(round(best_score, 4))
    best_vec = {"length_scale": float(best_v[0]), "torque_scale": float(best_v[1]),
                "mount_pitch": seedv["mount_pitch"], "kp": seedv["kp"], "kd": seedv["kd"]}
    return {"gene": _variant(gene, best_vec, ground=ground), "best_controller": best_ctrl,
            "baseline_value": round(base_score, 4), "best_value": round(best_score, 4),
            "history": history,
            "changed": {"length_scale": round(float(best_v[0]), 3),
                        "torque_scale": round(float(best_v[1]), 3)}}
