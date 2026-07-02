"""VIRT-Bench arms (breakthrough plan WS3, §7) — run competing DESIGN methods against the frozen tasks and score
BOTH with the independent verifier, so neither arm scores itself and the A->B delta is the *measured* value of
the search harness (the whole "beats Claude+MCP" claim in one number).

  Arm A  the fixed-pipeline baseline: compose the body, drive it with the DEFAULT controller, no search. This is
         the "Claude + MCP: one move" shape -- a single build with a stock gait, no closed loop.
  Arm B  our design-search harness: search the CPG gait direction on the cheap CPU rung with honesty-gate
         selection (services/design_search + the LLM-free cpg_grid_proposer), then submit the SELECTED best.

Both submit to virt_bench.verify_submission, which RE-RUNS the artifact in physics and applies the H1 gate --
the arm's self-claim is never consulted. This is the proof AND the development compass: a task BOTH arms fail is
the next build priority (e.g. forward hexapod locomotion, which needs the learned GPU residual, not just a CPG).
"""

from __future__ import annotations

from virturoid.services.virt_bench import get_task, list_tasks, verify_submission


def _task_body(task: dict):
    """Deterministically compose the body a locomotion task calls for (returns None for unsupported families)."""
    if task["family"] == "manipulation":                      # §3.5: manipulation tasks get a tabletop arm; the
        from virturoid.fixtures.gene_library import tabletop_arm_gene   # verifier runs the grasp/sort skill (evaluate_robot)
        if task["id"] == "M5_arm_grasp_hard":                 # WS9: the friction-hard grasp uses the PARALLEL-GRIPPER
            from virturoid.services.design_critic import add_parallel_gripper   # arm the learned residual was trained on
            return add_parallel_gripper(tabletop_arm_gene())  #   (matches the residual's obs/act dims -> it deploys)
        return tabletop_arm_gene()
    if task["family"] != "locomotion":
        return None
    from virturoid.services.steerable_body import steerable_quadruped
    if task["id"] == "L1_quad_walk":
        return steerable_quadruped(n_legs=4)
    if task["id"] == "L2_hex_walk":
        return steerable_quadruped(n_legs=6, bilateral=True)
    if task["id"] == "L3_octopod_walk":
        return steerable_quadruped(n_legs=8, bilateral=True)
    if task["id"] == "L4_decapod_walk":
        return steerable_quadruped(n_legs=10, bilateral=True)
    return None                                               # N22 hygiene: unknown legged task -> explicit
    #                                                           unsupported_task, never silently verify a WRONG body


def _zero_policy_with_cpg(gene, cpg: dict | None):
    """A zero-residual MorphPolicy carrying ``cpg`` as its gait prior -> recipe_rollout_morph drives the pure CPG
    (the verifier re-runs THIS). Isolates the searched gait from any learned residual."""
    import numpy as np

    from virturoid.services.morph_graph import encode_robot
    from virturoid.services.morph_policy import MorphPolicy, compiled_model, robot_mjcf
    graph = encode_robot(compiled_model(robot_mjcf(gene)))
    pol = MorphPolicy(graph.feature_dim)
    pol.set_params(np.zeros(pol.n_params))
    pol.cpg = dict(cpg) if cpg else None
    return pol


def run_arm_a(task_id: str, *, steps: int = 600, seed: int | None = None) -> dict:
    """Baseline arm: compose the body + the DEFAULT CPG gait, NO search -> independent verify. ``seed`` overrides
    the task's frozen verify seed (multi-seed protocol, WS4)."""
    from virturoid.services.morph_policy import CPG_DEFAULT
    task = get_task(task_id)
    gene = _task_body(task)
    if gene is None:
        return {"task": task_id, "arm": "A", "verified_pass": False, "failure_mode": "unsupported_task",
                "metrics": {}, "method": "fixed-pipeline (no search)"}
    if task["family"] == "locomotion":
        pol, method = _zero_policy_with_cpg(gene, CPG_DEFAULT), "fixed-pipeline: default CPG, no search"
    else:                                                     # manipulation: the verifier runs the built-in skill
        pol, method = None, "fixed-pipeline: default grasp skill, no search"
    res = verify_submission(task_id, gene, pol, seed=seed)    # FROZEN horizon (§3.1); seed override for WS4
    res["arm"] = "A"; res["method"] = method
    res["claimed_pass"] = bool(res.get("verified_pass"))      # A has no self-eval: it submits the default + verifies
    return res


def run_arm_b(task_id: str, *, steps: int = 600, max_evals: int = 12, on_node=None, use_memory: bool = True,
              use_gpu: bool = False, gpu_iters: int = 60, gpu_envs: int = 1024, gpu_hifi=None,
              models_dir: str = "build/models", seed: int | None = None) -> dict:
    """Full-harness arm with up to THREE candidate sources, all scored by the SAME independent verifier at the
    frozen horizon: (1) MEMORY -- recall the banked policy that best transfers to this body (the flywheel moat);
    (2) SEARCH the CPG gait direction on the cheap CPU rung; (3) GPU RESIDUAL (plan gap-closure WS1/#66) -- when
    ``use_gpu``, promote the CPU-search winner to a real GPU-trained residual (deploy-gap fixes baked in:
    decimation 10, LPF 0.2, sphere feet), WARM-started from the recalled seed iff ``use_memory`` (A+ vs B: A+
    trains the rung COLD, B warm from memory -- so ``transfer_delta`` isolates the flywheel, not the compute).
    Returns the best-verified verdict + ``searched``/``recalled``/``gpu_npz`` provenance, a per-arm ``budget``
    ledger (CPU evals, GPU iters, LLM calls -- N16 budget parity), and ``claimed_pass`` = the WINNING candidate's
    own pre-verify belief (the honesty axis; over-claim = claimed but not verified)."""
    from virturoid.services.design_search import run_design_search
    from virturoid.services.search_adapters import cpg_grid_proposer, make_locomotion_evaluate
    task = get_task(task_id)
    gene = _task_body(task)
    if gene is None:
        return {"task": task_id, "arm": "B", "verified_pass": False, "failure_mode": "unsupported_task",
                "metrics": {}, "method": "full harness", "budget": {"cpu_evals": 0, "gpu_iters": 0, "llm_calls": 0}}
    recall_steps = int(task.get("steps", 900))                 # the frozen verify horizon (used for recall + GPU verify)

    # MANIPULATION (plan gap-closure WS6): SEARCH the grasp-skill controller params on the CPU rung, then submit the
    # best to the independent verifier (the grid CONTAINS the default skill, so B can never do worse than A). This
    # closes the honest A==B on manip — the search selects a grip/timing that raises the verified success rate.
    if task["family"] != "locomotion":
        from virturoid.services.search_adapters import make_manip_evaluate, manip_grid_proposer
        mevaluate = make_manip_evaluate(gene, prompt=task["prompt"])
        mreport = run_design_search(propose=manip_grid_proposer(), evaluate=mevaluate, task_type=task["task_type"],
                                    gates=task["gates"], max_evals=max_evals, on_node=on_node)
        best_params = (mreport.best.result.get("params") if mreport.best else None)
        # keep the BEST-verified of {searched params, DEFAULT skill}: the greedy search can stop at a passing-but-
        # suboptimal grid point, so verifying the default too guarantees B is never WORSE than A (a real delta only
        # when the search genuinely beats the stock skill — honest on tasks the default already solves).
        cand = [(verify_submission(task_id, gene, best_params, seed=seed), best_params),
                (verify_submission(task_id, gene, None, seed=seed), None)]
        (res, won) = max(cand, key=lambda c: ((c[0].get("metrics") or {}).get("success_rate", 0.0) or 0.0))
        res["arm"] = "B"; res["method"] = f"manip param-search ({mreport.n_evals} evals, {mreport.stopped_reason})"
        res["searched"] = won; res["recalled"] = None; res["n_evals"] = mreport.n_evals
        res["claimed_pass"] = bool(mreport.solved)
        res["budget"] = {"cpu_evals": int(mreport.n_evals), "gpu_iters": 0, "llm_calls": 0}
        return res

    candidates = []                                            # (verified_result, method, extras, claimed_pass)

    # (1) MEMORY: the best forward transfer from the banked pool, verified zero-shot (no training, no GPU).
    recalled = None
    if use_memory:
        try:
            from virturoid.services.transfer_seed import transfer_policy_for
            # rank transfers at the FROZEN VERIFY horizon, not the short search `steps`: a banked 50Hz-decimation
            # walker needs the full episode to show its forward travel, so a short horizon mis-ranks it below a
            # policy that merely lunges early. Selecting at the deploy horizon = recall the one that actually verifies.
            pol, npz, _ranked = transfer_policy_for(gene, models_dir=models_dir, steps=recall_steps)
            if pol is not None:
                recalled = npz
                rv = verify_submission(task_id, gene, pol, seed=seed)  # FROZEN horizon (§3.1)
                candidates.append((rv, f"memory transfer-recall ({npz.split('/')[-1]})", {"recalled": npz},
                                   bool(rv.get("verified_pass"))))     # memory has no optimistic self-eval: claim==verify
        except Exception:  # noqa: BLE001 - memory is best-effort; the search path still runs
            pass

    # (2) SEARCH: the CPG gait direction on the cheap rung, honesty-gate selection.
    evaluate = make_locomotion_evaluate(gene, steps=steps)
    report = run_design_search(propose=cpg_grid_proposer(), evaluate=evaluate, task_type="locomotion",
                               gates=task["gates"], max_evals=max_evals, on_node=on_node)
    best_cpg = (report.best.result.get("cpg") if report.best else None)
    sv = verify_submission(task_id, gene, _zero_policy_with_cpg(gene, best_cpg), seed=seed)   # FROZEN (§3.1)
    # the search's OWN gate verdict on its best candidate is its CLAIM; the independent verifier confirms/denies it.
    candidates.append((sv, f"CPG-search harness ({report.n_evals} evals, {report.stopped_reason})",
                       {"searched": best_cpg}, bool(report.solved)))

    # (3) GPU RESIDUAL (WS1/#66): promote the search winner to a real learned policy. CPU search already gated the
    # spend (AlphaEvolve cascade discipline); warm-start from the recalled seed iff memory is on (A+ cold, B warm).
    gpu_iters_spent = 0
    gpu_npz = None
    if use_gpu:
        try:
            from virturoid.services.morph_policy import MorphPolicy
            if gpu_hifi is not None:
                hifi = gpu_hifi                                # test injection: a gene-bound hifi(spec) callable
            else:
                from virturoid.services.fidelity_ladder import make_gpu_locomotion_hifi
                hifi = make_gpu_locomotion_hifi(gene, iters=gpu_iters, envs=gpu_envs, verify_steps=recall_steps,
                                                decimation=10, action_lpf=0.2, sphere_feet=True,
                                                init_npz=(recalled if use_memory else None))
            gpu_iters_spent = gpu_iters
            tr = hifi({"params": dict(best_cpg or {})}) or {}
            if tr.get("trained") and tr.get("npz"):
                gpu_npz = tr["npz"]
                gp = MorphPolicy.from_npz(gpu_npz)
                gv = verify_submission(task_id, gene, gp, seed=seed)   # independent FROZEN-horizon verify (honest gate)
                warm = "warm(mem)" if (use_memory and recalled) else "cold"
                candidates.append((gv, f"GPU residual ({gpu_iters} iters, {warm})", {"gpu_npz": gpu_npz},
                                   bool(gv.get("verified_pass"))))     # selected by deploy verify: claim==verify
        except Exception:  # noqa: BLE001 - GPU is best-effort; memory+search still stand
            pass

    # keep the BEST-verified candidate: a pass beats a fail, then higher forward travel
    def _key(c):
        r = c[0]
        return (bool(r.get("verified_pass")), float((r.get("metrics") or {}).get("forward_m", 0.0)))
    best_res, method, extras, claimed = max(candidates, key=_key)
    out = dict(best_res)
    out["arm"] = "B"; out["method"] = method; out["n_evals"] = report.n_evals
    out["searched"] = best_cpg; out["recalled"] = recalled if extras.get("recalled") else None
    out["gpu_npz"] = extras.get("gpu_npz")
    out["claimed_pass"] = bool(claimed)                        # the WINNING candidate's own pre-verify belief
    out["budget"] = {"cpu_evals": int(report.n_evals), "gpu_iters": int(gpu_iters_spent), "llm_calls": 0}
    return out


def run_scoreboard(*, split: str | None = "dev", steps: int = 600, max_evals: int = 12, use_memory: bool = True,
                   models_dir: str = "build/models", families=("locomotion", "manipulation")) -> dict:
    """Run both arms over the tasks in ``split`` (``"dev"`` / ``"held_out"`` / ``None`` = all) whose family is in
    ``families`` (locomotion + manipulation by default) and return an honest, verifier-scored A-vs-B scoreboard.
    ``B_solved - A_solved`` is the measured value of the full harness -- the head-to-head number for "beats
    Claude+MCP" on this slice."""
    rows = []
    for task in list_tasks(split):
        if task["family"] not in families:
            continue
        a = run_arm_a(task["id"], steps=steps)
        b = run_arm_b(task["id"], steps=steps, max_evals=max_evals, use_memory=use_memory, models_dir=models_dir)
        rows.append({"task": task["id"], "family": task["family"],
                     "A_pass": bool(a["verified_pass"]), "B_pass": bool(b["verified_pass"]),
                     "A_metric": (a["metrics"].get("forward_m") if a["metrics"].get("forward_m") is not None
                                  else a["metrics"].get("success_rate")),
                     "B_metric": (b["metrics"].get("forward_m") if b["metrics"].get("forward_m") is not None
                                  else b["metrics"].get("success_rate")),
                     "B_searched": b.get("searched"), "B_recalled": b.get("recalled"), "B_n_evals": b.get("n_evals")})
    return {"rows": rows, "A_solved": sum(r["A_pass"] for r in rows),
            "B_solved": sum(r["B_pass"] for r in rows), "n_tasks": len(rows),
            "harness_delta": sum(r["B_pass"] for r in rows) - sum(r["A_pass"] for r in rows)}


def run_dev_scoreboard(**kw) -> dict:
    """The dev-split scoreboard (thin wrapper over ``run_scoreboard`` for back-compat)."""
    kw.setdefault("split", "dev")
    return run_scoreboard(**kw)


def _metric(res: dict):
    """The task's headline metric from a verdict (forward travel for locomotion, success rate for manipulation)."""
    m = res.get("metrics") or {}
    return m.get("forward_m") if m.get("forward_m") is not None else m.get("success_rate")


def run_head_to_head(*, split: str | None = "held_out", steps: int = 600, max_evals: int = 12,
                     models_dir: str = "build/models", families=("locomotion", "manipulation"),
                     use_gpu: bool = False, gpu_iters: int = 60, gpu_envs: int = 1024, gpu_hifi=None,
                     seed: int | None = None, prereg_path: str | None = None,
                     with_baseline: bool = False, baseline_llm=None) -> dict:
    """M5 pre-registered THREE-ARM head-to-head — the plan's decisive comparison, separating the value of the
    agentic SEARCH from the value of the flywheel MEMORY, and auditing each arm's honesty.

      A   fixed pipeline (default controller, NO search, NO memory) — the "Claude + MCP: one move" baseline.
      A+  full harness SEARCH but memory DISABLED (``run_arm_b(use_memory=False)``) — isolates the search loop.
      B   full harness + memory TRANSFER from the warm banked pool (``use_memory=True``) — adds the flywheel.

    Every arm is scored by the SAME independent verifier at the task-frozen horizon (§3.1), so no arm grades
    itself. Returns per-task rows plus the three MEASURED deltas the plan calls for:
      * ``harness_delta``  = A+solved − Asolved  (value of the agentic search over a stock build)
      * ``transfer_delta`` = Bsolved  − A+solved (value of the memory/transfer flywheel — the moat)
      * ``honesty``        = per-arm {claimed, verified, overclaim=claimed−verified} (SWE-bench-Verified check:
                             an arm's own gate verdict vs the independent verifier; overclaim>0 = the deploy gap).
    Pass ``prereg_path`` to FREEZE the protocol first (SHA256 manifest of tasks+verifier+seeds) so the run is
    pre-registered, not post-hoc — the manifest is attached under ``prereg``.

    ``with_baseline`` adds **Arm 0** — the LITERAL Claude+MCP control (plan v3 M1): an LLM agent (``baseline_llm``,
    else the env backend) drives the primitive MCP tools and submits a controller, scored by the SAME verifier.
    The headline ``baseline_delta = B_solved − A0_solved`` is the honest "beats Claude+MCP" number; until Arm 0
    is run, the defensible claim is only "beats our fixed pipeline (Arm A)"."""
    prereg = None
    arms = ("A0", "A", "A+", "B") if with_baseline else ("A", "A+", "B")
    if prereg_path is not None:                               # freeze the protocol (task+verifier+arms SHA256) FIRST
        from virturoid.services.prereg import write_prereg
        prereg = write_prereg(prereg_path, arms=arms, split=split)

    rows = []
    for task in list_tasks(split):
        if task["family"] not in families:
            continue
        a = run_arm_a(task["id"], steps=steps, seed=seed)
        a0 = None
        if with_baseline:
            from virturoid.services.virt_bench_baseline import run_arm_0
            a0 = run_arm_0(task["id"], llm=baseline_llm, seed=seed)
        # A+ = search+GPU, memory COLD (use_memory=False -> no recall AND GPU warm-start init_npz=None: N17 isolation).
        # B  = search+GPU, memory WARM. Both spend the SAME GPU iters -> harness_delta/transfer_delta compare search
        # intelligence and flywheel value, not compute (N16 budget parity, asserted via the per-arm ledger below).
        gk = dict(use_gpu=use_gpu, gpu_iters=gpu_iters, gpu_envs=gpu_envs, gpu_hifi=gpu_hifi, seed=seed)
        ap = run_arm_b(task["id"], steps=steps, max_evals=max_evals, use_memory=False, models_dir=models_dir, **gk)
        b = run_arm_b(task["id"], steps=steps, max_evals=max_evals, use_memory=True, models_dir=models_dir, **gk)
        row = {"task": task["id"], "family": task["family"],
               "A_pass": bool(a["verified_pass"]), "Aplus_pass": bool(ap["verified_pass"]),
               "B_pass": bool(b["verified_pass"]),
               "A_metric": _metric(a), "Aplus_metric": _metric(ap), "B_metric": _metric(b),
               "A_claim": bool(a.get("claimed_pass")), "Aplus_claim": bool(ap.get("claimed_pass")),
               "B_claim": bool(b.get("claimed_pass")),
               "Aplus_budget": ap.get("budget"), "B_budget": b.get("budget"),
               "B_recalled": b.get("recalled"), "B_searched": b.get("searched"), "B_gpu_npz": b.get("gpu_npz")}
        if a0 is not None:
            row.update({"A0_pass": bool(a0["verified_pass"]), "A0_metric": _metric(a0),
                        "A0_claim": bool(a0.get("claimed_pass")), "A0_budget": a0.get("budget"),
                        "A0_mode": a0.get("failure_mode")})
        rows.append(row)

    def _solved(k):
        return sum(1 for r in rows if r.get(k))

    A, Ap, B = _solved("A_pass"), _solved("Aplus_pass"), _solved("B_pass")
    honesty_arms = [("A", "A_claim", "A_pass"), ("A+", "Aplus_claim", "Aplus_pass"), ("B", "B_claim", "B_pass")]
    if with_baseline:
        honesty_arms.insert(0, ("A0", "A0_claim", "A0_pass"))
    honesty = {}
    for arm, cl, ve in honesty_arms:
        claimed, verified = _solved(cl), _solved(ve)
        honesty[arm] = {"claimed": claimed, "verified": verified, "overclaim": claimed - verified}
    # N16 budget ledger: total spend per arm (identical GPU iters between A+ and B is the parity check).
    def _budget(k):
        tot = {"cpu_evals": 0, "gpu_iters": 0, "llm_calls": 0}
        for r in rows:
            for kk, vv in (r.get(k) or {}).items():
                tot[kk] = tot.get(kk, 0) + int(vv)
        return tot
    budget = {"A+": _budget("Aplus_budget"), "B": _budget("B_budget")}
    out = {"rows": rows, "n_tasks": len(rows), "A_solved": A, "Aplus_solved": Ap, "B_solved": B,
           "harness_delta": Ap - A, "transfer_delta": B - Ap, "honesty": honesty, "budget": budget,
           "prereg": prereg}
    if with_baseline:                                          # M1: the literal Claude+MCP control
        A0 = _solved("A0_pass")
        out["A0_solved"] = A0
        out["baseline_delta"] = B - A0                         # THE headline "beats Claude+MCP" number
        budget["A0"] = _budget("A0_budget")
    return out


def _mean_sem_iqm(xs: list) -> dict:
    """Point estimate + uncertainty for a small sample of per-seed scores. Reports mean±SEM (plan v2 §3.7) AND
    the INTERQUARTILE MEAN (Agarwal et al. NeurIPS 2021, WS4): the middle-50% mean, robust to the outlier runs
    that make point estimates on few seeds misleading. ``ci95`` is the normal-approx 1.96·SEM band."""
    import math
    xs = sorted(float(x) for x in xs)
    n = len(xs)
    if n == 0:
        return {"mean": 0.0, "sem": 0.0, "ci95": 0.0, "iqm": 0.0, "n": 0}
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    sem = math.sqrt(var / n) if n > 1 else 0.0
    lo, hi = n // 4, n - n // 4                                # drop the bottom/top quartile for the IQM
    mid = xs[lo:hi] or xs
    iqm = sum(mid) / len(mid)
    return {"mean": round(mean, 4), "sem": round(sem, 4), "ci95": round(1.96 * sem, 4),
            "iqm": round(iqm, 4), "n": n}


def run_head_to_head_multiseed(*, seeds=(20260701, 20260702, 20260703), base_prereg_path: str | None = None,
                               **kw) -> dict:
    """M5 statistically-hardened head-to-head (plan gap-closure WS4): run the 3-arm comparison at ≥3 SEEDS and
    report each metric with UNCERTAINTY, not a single point estimate (few-seed point estimates are the exact trap
    §3.7 warns against). ``seeds`` overrides the per-run verify seed (N18). ``kw`` forwards to ``run_head_to_head``
    (split/steps/max_evals/use_gpu/…). Returns per-seed runs + aggregated ``A_solved``/``Aplus_solved``/``B_solved``
    and ``harness_delta``/``transfer_delta`` each as {mean, sem, ci95, iqm, n}. Writes one pre-registration
    manifest with the full seed table when ``base_prereg_path`` is given (the honest external timestamp)."""
    kw.pop("seed", None)
    kw.pop("prereg_path", None)
    prereg = None
    if base_prereg_path is not None:
        from virturoid.services.prereg import write_prereg
        prereg = write_prereg(base_prereg_path, arms=("A", "A+", "B"), seeds=list(seeds),
                              split=kw.get("split", "held_out"))
    runs = [run_head_to_head(seed=int(s), **kw) for s in seeds]
    agg = {k: _mean_sem_iqm([r[k] for r in runs])
           for k in ("A_solved", "Aplus_solved", "B_solved", "harness_delta", "transfer_delta")}
    return {"seeds": list(seeds), "n_seeds": len(seeds), "per_seed": runs, "aggregate": agg, "prereg": prereg}
