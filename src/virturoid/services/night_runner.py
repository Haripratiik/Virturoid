"""Night-shift runner (breakthrough plan v2 §5.1/§5.2/§5.4 + M4 capstone) — the entrypoint that ASSEMBLES the
Dreamer-mode pieces into one runnable, resumable command (what scripts/ops/nightshift.service launches).

Wires together, in order:
  1. golden_suite gate      — re-run sealed regression cases; if any regressed, run in PROBE-ONLY mode (no banking).
  2. NightProposer          — 3-armed candidate generation (QD-mutation / transfer / LLM explorer), spend-guarded.
  3. filter_learnable       — keep only candidates in the AZR learnability band (cheap CPU probe).
  4. laddered_evaluate_for  — CPU-screen -> GPU-residual rung (deploy-gap fixes baked in), the production evaluate.
  5. run_night_shift        — bounded searches, verified banking to the flywheel, ANNECS-V + QD dashboard,
                              resumable JSONL journal (crash-safe: completed candidates are skipped on resume).

Everything heavy is injected (``evaluate_for``, arm callables, ``llm``) so this unit-tests with stubs — no GPU,
no MuJoCo, no LLM. Production defaults bind the real fidelity ladder + the spend-guarded LLM + the QD archive.
"""

from __future__ import annotations


def default_body_zoo():
    """A small starting body set for the mutation/transfer arms (the frontier grows as the archive fills)."""
    from virturoid.services.steerable_body import steerable_quadruped
    return [("quad", steerable_quadruped(n_legs=4)),
            ("hexapod", steerable_quadruped(n_legs=6, bilateral=True)),
            ("octopod", steerable_quadruped(n_legs=8, bilateral=True))]


_GATES = {"forward_m": 0.4, "cadence": 3.0, "upright": 0.6}


def _walk_candidate(name, gene, rng):
    """A locomotion candidate for the harness: body + a CPG-direction hint as the LLM-free heuristic proposer."""
    from virturoid.services.search_adapters import cpg_grid_proposer
    cid = f"{name}_{rng.randint(0, 1_000_000)}"
    return {"id": cid, "gene": gene, "task": f"make the {name} walk forward", "task_type": "locomotion",
            "gates": dict(_GATES), "gate_target": _GATES["forward_m"], "heuristic": cpg_grid_proposer()}


def mutate_morphology(rng):
    """A real MORPHOLOGY-mutation operator (plan gap-closure WS2-phase-2 / N20): sample a NOVEL body from the
    steerable-legged design space (n_legs x dofs_per_leg x bilateral) instead of re-picking one of 3 fixed zoo
    bodies. This is the minimal POET-style open-endedness step — the frontier can now grow new morphologies, so
    QD coverage / ANNECS-V measure genuine novelty rather than saturating on a static zoo. Returns ``(name, gene)``."""
    from virturoid.services.steerable_body import steerable_quadruped
    n_legs = rng.choice([4, 6, 8])
    dofs = rng.choice([2, 3])                                 # 3 = hip-yaw (steerable); 2 = sagittal-only
    bilateral = (n_legs != 4) and (rng.random() < 0.85)       # multi-leg bodies walk as bilateral (L,R) pairs
    gene = steerable_quadruped(n_legs=n_legs, dofs_per_leg=dofs, bilateral=bilateral)
    name = f"morph_{n_legs}L{dofs}d{'bi' if bilateral else ''}"
    return name, gene


def make_arms(zoo=None, *, seed: int = 0, mutate_prob: float = 0.5):
    """Bind the NightProposer's three arms to a body zoo. The mutate arm now, with probability ``mutate_prob``,
    emits a NOVEL morphology (``mutate_morphology``) rather than re-picking a fixed zoo body -> the frontier grows
    (WS2-phase-2). transfer picks a zoo body + relies on the laddered evaluator to warm-start from the best banked
    seed; explore is the LLM path (None here -> LLM-free night). Returns ``(mutate, transfer, explore)``."""
    import random
    zoo = zoo if zoo is not None else default_body_zoo()
    rng = random.Random(seed)

    def mutate():
        if rng.random() < mutate_prob:
            name, gene = mutate_morphology(rng)              # NOVEL body from the design space (open-endedness)
        else:
            name, gene = zoo[rng.randrange(len(zoo))]        # or refine a known body
        return _walk_candidate(name, gene, rng)

    def transfer():
        name, gene = zoo[rng.randrange(len(zoo))]
        c = _walk_candidate(name, gene, rng)
        c["arm"] = "transfer"
        return c

    return mutate, transfer, None


def _golden_policy_for(models_dir: str):
    """A ``policy_for`` for the golden suite that recalls the BANKED capability for each case's body (so golden
    protects the real flywheel capability, not the random default controller). Returns None when nothing is banked
    for that body yet -> run_golden_suite skips the case (vacuous pass) so the first night can still bank."""
    def pf(task_id):
        try:
            from virturoid.services.virt_bench import get_task
            from virturoid.services.virt_bench_arms import _task_body
            from virturoid.services.transfer_seed import transfer_policy_for
            task = get_task(task_id)
            if task.get("family") != "locomotion":
                return None
            gene = _task_body(task)
            if gene is None:
                return None
            pol, _npz, _ranked = transfer_policy_for(gene, models_dir=models_dir, steps=int(task.get("steps", 900)))
            return pol
        except Exception:  # noqa: BLE001 - golden is best-effort; no banked policy -> skip (vacuous pass)
            return None
    return pf


def run_night(*, memory_dir: str, journal_path=None, budget_evals: int = 200, n_candidates: int = 12,
              per_candidate_evals: int = 8, zoo=None, evaluate_for=None, probe=None, llm=None, archive=None,
              run_golden: bool = True, warm_start_pool: str = "build/models", seed: int = 0):
    """Run one night. Returns ``{golden, proposed, night}``. ``evaluate_for`` defaults to the real fidelity ladder
    (GPU); inject a stub for CPU tests. ``probe`` is the cheap learnability estimator (skipped if None). If the
    golden suite regresses (a BANKED capability degraded), banking is DISABLED for the night (probe-only), §5.4."""
    from virturoid.services.night_proposer import NightProposer, filter_learnable
    from virturoid.services.night_shift import laddered_evaluate_for, run_night_shift

    golden = None
    banking_ok = True
    if run_golden:
        from virturoid.services.golden_suite import load_golden_cases, run_golden_suite
        # load_golden_cases picks up floors RATCHETED by previous nights (N21); policy_for recalls the banked
        # capability each case protects (so a case is checked against the real flywheel policy, not the default).
        golden = run_golden_suite(cases=load_golden_cases(), policy_for=_golden_policy_for(warm_start_pool))
        banking_ok = golden["passed"]

    mutate, transfer, explore = make_arms(zoo, seed=seed)
    proposer = NightProposer(mutate=mutate, transfer=transfer, explore=explore, seed=seed)
    cands = proposer.propose(n_candidates)
    if probe is not None:
        cands = filter_learnable(cands, probe)

    if evaluate_for is None:
        evaluate_for = laddered_evaluate_for()               # real fidelity ladder (GPU rung)
    if archive is None:                                      # QD open-endedness dashboard (ANNECS-V/coverage/QD-score)
        from virturoid.services.qd_archive import QDArchive   # BEHAVIORAL axes (N19): (forward speed, cadence) — a
        archive = QDArchive(dims=[("forward", -0.5, 1.5), ("cadence", 0, 8)], bins=8)  # fixed zoo still fills cells

    # golden regression -> probe-only: run with a zero budget so nothing is banked (still journals what it saw)
    night = run_night_shift(cands, evaluate_for, memory_dir=memory_dir, llm=llm,
                            budget_evals=(budget_evals if banking_ok else 0),
                            per_candidate_evals=per_candidate_evals, journal_path=journal_path, archive=archive)
    sealed = None
    if banking_ok and getattr(night, "banked", 0):           # N21 ratchet: seal floors for what the night banked
        sealed = ratchet_golden_floors(warm_start_pool)
    return {"golden": golden, "banking_enabled": banking_ok, "proposed": len(cands), "night": night,
            "sealed_floors": sealed}


def ratchet_golden_floors(models_dir: str = "build/models") -> dict:
    """N21: after a banking night, re-verify each locomotion task against the best banked transfer and SEAL its
    golden floor to 0.9× the verified metric (only ever tightening). This is how new flywheel capabilities become
    drift-protected — without it the golden suite stays vacuous for everything banked after night 1. Returns the
    ``{task_id: floor}`` that were sealed/raised this call."""
    from virturoid.services.golden_suite import seal_golden_floor
    from virturoid.services.virt_bench import list_tasks, verify_submission
    from virturoid.services.virt_bench_arms import _task_body
    recall = _golden_policy_for(models_dir)
    sealed = {}
    for task in list_tasks(None):
        if task.get("family") != "locomotion":
            continue
        try:
            pol = recall(task["id"])
            if pol is None:
                continue
            gene = _task_body(task)
            if gene is None:
                continue
            r = verify_submission(task["id"], gene, pol)
            fwd = (r.get("metrics") or {}).get("forward_m")
            if fwd is not None and float(fwd) > 0.0:
                sealed[task["id"]] = seal_golden_floor(task["id"], float(fwd), metric_key="forward_m")
        except Exception:  # noqa: BLE001 - ratcheting is best-effort; never break the night's completion
            continue
    return sealed


def _main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Virturoid Dreamer-mode night shift")
    ap.add_argument("--memory-dir", default="build/memory")
    ap.add_argument("--journal", default="runs/night.jsonl")
    ap.add_argument("--budget-evals", type=int, default=200)
    ap.add_argument("--candidates", type=int, default=12)
    ap.add_argument("--resume", action="store_true", help="continue from the journal (completed ids are skipped)")
    args = ap.parse_args(argv)
    rep = run_night(memory_dir=args.memory_dir, journal_path=args.journal, budget_evals=args.budget_evals,
                    n_candidates=args.candidates)
    n = rep["night"]
    print(f"night done: proposed={rep['proposed']} banked={n.banked} novel={n.novel} "
          f"banking_enabled={rep['banking_enabled']} qd={n.qd}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
