"""AI-assisted locomotion training — the plan's training flywheel (roadmap Themes 2/3, [[reasoned-redesign-loop]],
[[general-codesign-flywheel]]). Instead of cold-training every body from scratch, the trainer:

  1. RECALLS the species/class TIP from memory — the training budget that reached the target last time (the
     cross-build lesson), so it starts where prior learning left off instead of guessing.
  2. WARM-STARTS the policy from the species' banked weights (the flywheel) so it continues, not restarts.
  3. TRAINS on the efficient parallel sim — the GPU/MJX path (2048 envs) when the box is healthy, else the
     parallel-CPU ES (population fanned across cores).
  4. LEARNS FROM THE SIM: it reads the rollout (did it fall? upright-but-slow?) and ADAPTS — escalating the
     budget and continuing from the current policy — until the body clears the task target.
  5. BANKS the policy + RECORDS the budget that worked as a tip, so the NEXT build of this class trains faster.

The result: each build of a class is quicker to train than the last, because the agent reuses what it learned.
"""

from __future__ import annotations

from pathlib import Path

_DEFAULT_BUDGET = {"generations": 30, "pop": 24, "steps": 320}
_LESSON_CODE = "locomotion_budget"


def _resolve_workers(workers):
    if workers is not None:
        return workers
    import multiprocessing as mp
    import sys
    return 1 if "pytest" in sys.modules else max(1, mp.cpu_count() - 1)


def _diagnose(travel: float, upright: bool, target: float) -> str:
    """Read the sim result -> the reason it's short of target (drives the adaptation, not just a blind bump)."""
    if not upright:
        return "fell_over"                       # stability problem -> longer, gentler training
    if travel < 0.4 * target:
        return "barely_moves"                    # not learning forward -> much bigger budget
    return "upright_but_slow"                    # close -> a moderate budget bump


def _gpu_train_and_diagnose(gene, weights, *, iters, envs, models_dir, init_npz, progress):
    """One GPU burst with the given reward WEIGHTS (+ trot-CPG), then a CPU diagnosis of the trained gait
    (cadence/upright/forward from recipe_rollout_morph). Returns (npz_path, diagnosis) or (None, None)."""
    from pathlib import Path as _P

    from virturoid.services.gpu_trainer import default_training_recipe, train_gene_on_gpu
    from virturoid.services.morph_policy import MorphPolicy, recipe_rollout_morph
    out = str(_P(models_dir) / f"ai_gait_{gene.robot_class or 'legged'}.npz")
    npz = train_gene_on_gpu(gene, out_path=out, iters=iters, envs=envs,   # AUTO recipe per body (cpg/adaptive/deltas)
                            reward_weights=weights, init_npz=init_npz, progress=progress,
                            **default_training_recipe(gene))
    if not npz:
        return None, None
    r = recipe_rollout_morph(gene, MorphPolicy.from_npz(npz), steps=900)
    fwd, cad = float(r.get("forward", 0.0)), float(r.get("cadence", 0.0))
    diag = {"speed": round(fwd / 18.0, 3), "cadence": cad, "upright": r.get("upright_frac"),
            "fell": not r.get("survived"), "height_held": r.get("height_ratio"),
            "walking": bool(r.get("survived") and cad >= 1.0 and fwd > 0.15),
            "gait_quality": round(fwd, 3),
            "summary": f"fwd={fwd:.2f}m cadence={cad:.1f}/s upright={r.get('upright_frac')} survived={r.get('survived')}"}
    return npz, diag


def ai_critic_gait_loop(gene, *, llm=None, rounds: int = 3, iters: int = 35, envs: int = 1024,
                        models_dir: str = "models", progress=None, _train_diagnose=None,
                        memory_dir=None) -> dict:
    """The AI-assisted REWARD loop (Eureka / Language-to-Rewards), wired end-to-end: train a GPU burst with the
    current reward weights -> diagnose the gait -> the LLM gait_critic REDESIGNS the weights -> retrain (warm-started),
    until it walks or ``rounds`` is hit. Offline-safe: ``llm=None`` -> one default-weight run, no critic (graceful).
    Returns ``{npz, walking, rounds, weights, history}``. ``_train_diagnose`` is injectable for tests.

    Phase 2 RAG: with ``memory_dir`` set, the gait_critic gets a PRIOR-KNOWLEDGE block (trainer tips +
    nearest walking skill's recipe for this body) so it redesigns the reward from what worked before."""
    from virturoid.services.gait_critic import GAIT_REWARD_KNOBS, critique_gait

    def say(m):
        if progress:
            progress(m)
    prior = ""
    if memory_dir is not None:
        try:
            from virturoid.services.knowledge_context import assemble_prior_knowledge
            prior = assemble_prior_knowledge(memory_dir, gene=gene, task_type="locomotion")
        except Exception:  # noqa: BLE001 - RAG is best-effort; never break training
            prior = ""
    td = _train_diagnose or _gpu_train_and_diagnose
    weights = {k: float(v[2]) for k, v in GAIT_REWARD_KNOBS.items()}     # start from the trainer's default weights
    history, best_npz, walking, init_npz = [], None, False, None
    for rnd in range(max(1, rounds)):
        npz, diag = td(gene, weights, iters=iters, envs=envs, models_dir=models_dir,
                       init_npz=init_npz, progress=progress)
        if npz is None:
            break
        best_npz, init_npz = npz, "runs/app_gene.npz"                   # warm-start the next burst from this one
        history.append({"weights": dict(weights), "gait_quality": diag.get("gait_quality"),
                        "is_walking": diag.get("walking"), "summary": diag.get("summary", "")})
        if diag.get("walking"):
            walking = True
            say(f"AI-critic loop: walking at round {rnd + 1} ({diag.get('summary', '')})")
            break
        if llm is None:                                                # no LLM -> can't redesign the reward; stop
            say("AI-critic loop: no LLM backend — kept default reward weights (offline)")
            break
        prop = critique_gait(diag, weights, llm, history=history, prior_knowledge=prior)
        if not prop or not prop.get("weights"):
            break
        weights = prop["weights"]
        say(f"AI-critic round {rnd + 1}: {prop.get('key_change', 'redesigned reward weights')}")
    return {"npz": best_npz, "walking": walking, "rounds": len(history), "weights": weights, "history": history}


def _measure_travel(gene, policy, *, steps: int = 2400) -> dict:
    """Roll ``policy`` out on ``gene`` with the CORRECT controller for its TYPE and return ``{forward, upright}``.

    A recipe/CPG policy (``obs_mean`` or ``cpg`` set — banked GPU policies, recipe-ES policies) MUST be driven by
    ``recipe_rollout_morph`` (the SAME PD-to-default + CPG-prior rollout + obs-norm it was trained/banked under);
    the bare-residual ``rollout_morph`` drives it WITHOUT its gait prior so it reads as FALLEN — the exact
    misrouting ``learn_locomotion.locomotion_episode`` already fixed. Measuring every candidate through the same
    no-CPG rollout made the assisted loop under-measure banked/GPU recipe policies, so it skipped reuse and
    retrained needlessly. A plain policy (no recipe metadata) uses ``rollout_morph``."""
    from virturoid.services.morph_policy import recipe_rollout_morph, rollout_morph
    if getattr(policy, "obs_mean", None) is not None or getattr(policy, "cpg", None) is not None:
        r = recipe_rollout_morph(gene, policy, steps=steps)
        return {"forward": float(r.get("forward", 0.0)),
                "upright": bool(r.get("survived", False)) or float(r.get("height_ratio", 0.0)) > 0.6}
    r = rollout_morph(gene, policy, steps=steps)
    return {"forward": float(r.get("forward", 0.0)), "upright": bool(r.get("upright", False))}


def train_assisted(gene, *, target: float = 0.35, models_dir: str = "models", memory_dir: str = "build/memory",
                   max_rounds: int = 3, prefer_gpu: bool = True, workers: int | None = None,
                   progress=None) -> dict:
    """AI-assisted training for ``gene``. Returns provenance:
    ``{score, baseline, travel_m, target, reached, rounds, used_tip, warm_started, backend, budget}``."""
    import mujoco

    from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
    from virturoid.services.learn_locomotion import banked_policy_for, learn_locomotion
    from virturoid.services.memory_db import MemoryDB
    from virturoid.services.morph_graph import encode_robot
    from virturoid.services.morph_policy import MorphPolicy

    cls = gene.robot_class or "legged"
    workers = _resolve_workers(workers)
    db_path = Path(memory_dir) / "virturoid_memory.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    def say(m):
        if progress:
            progress(m)

    # 1) RECALL the banked training tip (the budget that reached target before) for this class.
    used_tip = False
    budget = dict(_DEFAULT_BUDGET)
    try:
        with MemoryDB(db_path) as db:
            tip = db.recall_lesson(cls, _LESSON_CODE)
        if tip and tip.get("params"):
            p = tip["params"]
            budget = {"generations": int(p.get("generations", budget["generations"])),
                      "pop": int(p.get("pop", budget["pop"])), "steps": int(p.get("steps", budget["steps"]))}
            used_tip = True
            say(f"recalled training tip for {cls}: {budget} (worked before)")
    except Exception:  # noqa: BLE001 - memory is best-effort
        pass

    # 2) WARM-START from the banked species policy (the flywheel), and FIRST measure its travel: if it already
    # clears the target, REUSE it with NO training (the cheapest, fastest path — the whole point of the flywheel).
    travel_of = lambda pol: _measure_travel(gene, pol, steps=2400)  # route recipe/cpg vs plain (deploy==measure) # noqa: E731
    warm = banked_policy_for(gene, models_dir=models_dir)
    warm_started = warm is not None
    best_policy, best_travel, best_upright, baseline = None, float("-inf"), False, None
    if warm is not None:
        r = travel_of(warm)
        best_policy, best_travel, best_upright = warm, float(r["forward"]), bool(r["upright"])
        baseline = best_travel
        if best_travel >= target:
            say(f"banked policy already clears target ({best_travel:+.3f} >= {target}) — REUSING, no training")

    # 2b) CROSS-EMBODIMENT warm-start (the transfer moat): if no SAME-species policy clears the target, sweep the
    # whole banked pool for the policy that best TRANSFERS forward to THIS body and warm-start from it if it
    # travels further. This is the measured fix for bodies that train BACKWARD from scratch (e.g. a hexapod
    # warm-started from a forward quad -> the morphology-agnostic residual carries "how to push forward"). Only
    # runs when a same-species reuse hasn't already won, so well-served bodies pay nothing.
    if best_travel < target:
        try:
            from virturoid.services.transfer_seed import transfer_policy_for
            tpol, tnpz, _ranked = transfer_policy_for(gene, models_dir=models_dir)
            if tpol is not None:
                tr = travel_of(tpol)
                if float(tr["forward"]) > best_travel:
                    best_policy, best_travel, best_upright = tpol, float(tr["forward"]), bool(tr["upright"])
                    warm_started = True
                    if baseline is None:
                        baseline = best_travel
                    say(f"cross-embodiment warm-start from {tnpz.split('/')[-1]} "
                        f"(transfers {best_travel:+.3f} m forward to this body)")
        except Exception:  # noqa: BLE001 - transfer recall is best-effort; never break training
            pass

    # 3) GPU (efficient parallel sim) once when healthy + still short of target; else fall to parallel-CPU below.
    backend = "reused" if best_travel >= target else "parallel-cpu"
    if prefer_gpu and best_travel < target:
        try:
            from virturoid.services.gpu_trainer import gpu_available, train_gene_on_gpu
            if gpu_available(timeout=15):
                say("GPU healthy — training on MJX (2048 envs)…")
                out = train_gene_on_gpu(gene, out_path=str(Path(models_dir) / f"gpu_{cls}.npz"),
                                        iters=max(120, budget["generations"] * 4), envs=2048,
                                        progress=progress, timeout=1200)
                if out:
                    gp = MorphPolicy.from_npz(out)
                    r = travel_of(gp)
                    if baseline is None:
                        baseline = float(r["forward"])
                    if float(r["forward"]) > best_travel:
                        best_policy, best_travel, best_upright, backend = gp, float(r["forward"]), bool(r["upright"]), "gpu-mjx"
        except Exception:  # noqa: BLE001 - any GPU hiccup -> parallel-CPU
            pass

    # 4) parallel-CPU ES with ADAPTATION. Each round WARM-STARTS FROM THE BEST policy so far (never off a
    # degraded one), and we KEEP THE BEST by the real task metric (2400-step travel) — so the result is never
    # worse than the banked policy or any round. The sim diagnosis drives the budget escalation.
    rounds = 0
    while best_travel < target and rounds < max_rounds:
        rounds += 1
        say(f"round {rounds}: training {budget} on {workers} cores"
            + (" (warm-started from best)" if best_policy is not None else ""))
        res = learn_locomotion(gene, generations=budget["generations"], pop=budget["pop"], steps=budget["steps"],
                               models_dir=models_dir, warm_start=best_policy, workers=workers, progress=progress)
        if baseline is None:
            baseline = res["baseline"]
        r = travel_of(res["policy"])
        travel, upright = float(r["forward"]), bool(r["upright"])
        if travel > best_travel:                               # OUTER elitism by the real task metric
            best_policy, best_travel, best_upright, backend = res["policy"], travel, upright, "parallel-cpu"
        if best_travel >= target:
            break
        diag = _diagnose(travel, upright, target)
        if diag == "barely_moves":
            budget = {"generations": int(budget["generations"] * 1.8), "pop": min(64, int(budget["pop"] * 1.5)),
                      "steps": budget["steps"]}
        elif diag == "fell_over":
            budget = {"generations": int(budget["generations"] * 1.4), "pop": budget["pop"],
                      "steps": int(budget["steps"] * 1.3)}
        else:
            budget = {"generations": int(budget["generations"] * 1.4), "pop": budget["pop"], "steps": budget["steps"]}
        say(f"  sim diagnosis: {diag} (best travel {best_travel:+.3f} < {target}) -> escalating to {budget}")

    travel, upright = best_travel, best_upright
    reached = travel >= target
    # 5) BANK the working budget as the tip for next time (idempotent; keeps the best).
    if reached:
        try:
            with MemoryDB(db_path) as db:
                db.record_lesson(cls, _LESSON_CODE, operator="es_budget", params=budget,
                                 improvement=float(travel), task_type="locomotion", source_gene=gene.id)
            say(f"banked tip for {cls}: this body reached {travel:+.3f} m with {budget}")
        except Exception:  # noqa: BLE001
            pass

    return {"score": round(float(travel), 3), "baseline": round(float(baseline if baseline is not None else 0.0), 3),
            "travel_m": round(float(travel), 3), "target": target, "reached": reached, "rounds": rounds,
            "used_tip": used_tip, "warm_started": warm_started, "backend": backend, "budget": budget}
