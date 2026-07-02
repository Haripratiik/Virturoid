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
        return tabletop_arm_gene()
    if task["family"] != "locomotion":
        return None
    from virturoid.services.steerable_body import steerable_quadruped
    if task["id"] == "L1_quad_walk":
        return steerable_quadruped(n_legs=4)
    if task["id"] == "L2_hex_walk":
        return steerable_quadruped(n_legs=6, bilateral=True)
    return steerable_quadruped(n_legs=4)                       # sensible default for any other legged task


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


def run_arm_a(task_id: str, *, steps: int = 600) -> dict:
    """Baseline arm: compose the body + the DEFAULT CPG gait, NO search -> independent verify."""
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
    res = verify_submission(task_id, gene, pol)               # FROZEN horizon (§3.1)
    res["arm"] = "A"; res["method"] = method
    res["claimed_pass"] = bool(res.get("verified_pass"))      # A has no self-eval: it submits the default + verifies
    return res


def run_arm_b(task_id: str, *, steps: int = 600, max_evals: int = 12, on_node=None, use_memory: bool = True,
              use_gpu: bool = False, gpu_iters: int = 60, gpu_envs: int = 1024, gpu_hifi=None,
              models_dir: str = "build/models") -> dict:
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

    # MANIPULATION: the CPG/locomotion search doesn't apply; the verifier runs the built-in skill. Arm B here =
    # the composed arm verified (memory-recall of a banked arm design/skill is the future edge; today A==B for
    # manip, reported honestly so the scoreboard covers the family without faking a delta).
    if task["family"] != "locomotion":
        res = verify_submission(task_id, gene, None)
        res["arm"] = "B"; res["method"] = "full harness: default grasp skill (no manip search yet)"
        res["searched"] = None; res["recalled"] = None; res["n_evals"] = 0
        res["claimed_pass"] = bool(res.get("verified_pass"))   # no separate self-eval here: claim == verify
        res["budget"] = {"cpu_evals": 0, "gpu_iters": 0, "llm_calls": 0}
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
                rv = verify_submission(task_id, gene, pol)             # FROZEN horizon (§3.1)
                candidates.append((rv, f"memory transfer-recall ({npz.split('/')[-1]})", {"recalled": npz},
                                   bool(rv.get("verified_pass"))))     # memory has no optimistic self-eval: claim==verify
        except Exception:  # noqa: BLE001 - memory is best-effort; the search path still runs
            pass

    # (2) SEARCH: the CPG gait direction on the cheap rung, honesty-gate selection.
    evaluate = make_locomotion_evaluate(gene, steps=steps)
    report = run_design_search(propose=cpg_grid_proposer(), evaluate=evaluate, task_type="locomotion",
                               gates=task["gates"], max_evals=max_evals, on_node=on_node)
    best_cpg = (report.best.result.get("cpg") if report.best else None)
    sv = verify_submission(task_id, gene, _zero_policy_with_cpg(gene, best_cpg))   # FROZEN horizon (§3.1)
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
                gv = verify_submission(task_id, gene, gp)      # independent FROZEN-horizon verify (honest gate)
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
                     prereg_path: str | None = None) -> dict:
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
    pre-registered, not post-hoc — the manifest is attached under ``prereg``."""
    prereg = None
    if prereg_path is not None:                               # freeze the protocol (task+verifier+arms SHA256) FIRST
        from virturoid.services.prereg import write_prereg
        prereg = write_prereg(prereg_path, arms=("A", "A+", "B"), split=split)

    rows = []
    for task in list_tasks(split):
        if task["family"] not in families:
            continue
        a = run_arm_a(task["id"], steps=steps)
        # A+ = search+GPU, memory COLD (use_memory=False -> no recall AND GPU warm-start init_npz=None: N17 isolation).
        # B  = search+GPU, memory WARM. Both spend the SAME GPU iters -> harness_delta/transfer_delta compare search
        # intelligence and flywheel value, not compute (N16 budget parity, asserted via the per-arm ledger below).
        gk = dict(use_gpu=use_gpu, gpu_iters=gpu_iters, gpu_envs=gpu_envs, gpu_hifi=gpu_hifi)
        ap = run_arm_b(task["id"], steps=steps, max_evals=max_evals, use_memory=False, models_dir=models_dir, **gk)
        b = run_arm_b(task["id"], steps=steps, max_evals=max_evals, use_memory=True, models_dir=models_dir, **gk)
        rows.append({"task": task["id"], "family": task["family"],
                     "A_pass": bool(a["verified_pass"]), "Aplus_pass": bool(ap["verified_pass"]),
                     "B_pass": bool(b["verified_pass"]),
                     "A_metric": _metric(a), "Aplus_metric": _metric(ap), "B_metric": _metric(b),
                     "A_claim": bool(a.get("claimed_pass")), "Aplus_claim": bool(ap.get("claimed_pass")),
                     "B_claim": bool(b.get("claimed_pass")),
                     "Aplus_budget": ap.get("budget"), "B_budget": b.get("budget"),
                     "B_recalled": b.get("recalled"), "B_searched": b.get("searched"), "B_gpu_npz": b.get("gpu_npz")})

    def _solved(k):
        return sum(1 for r in rows if r[k])

    A, Ap, B = _solved("A_pass"), _solved("Aplus_pass"), _solved("B_pass")
    honesty = {}
    for arm, cl, ve in (("A", "A_claim", "A_pass"), ("A+", "Aplus_claim", "Aplus_pass"), ("B", "B_claim", "B_pass")):
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
    return {"rows": rows, "n_tasks": len(rows), "A_solved": A, "Aplus_solved": Ap, "B_solved": B,
            "harness_delta": Ap - A, "transfer_delta": B - Ap, "honesty": honesty, "budget": budget,
            "prereg": prereg}
