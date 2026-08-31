"""Night-shift engine core (breakthrough plan WS2 / N1) — the autonomous self-improvement loop.

The Dreamer mode: run the design-search harness on autopilot over a stream of candidate (body, task) proposals,
bank every verified win to the flywheel, and track the open-endedness metrics — so the platform is measurably
better every morning. This is the SAME harness as Engineer mode (design_search + engineer_mode + banking); the
only difference is that a proposal policy supplies the work instead of a user.

Week-1 core is LLM-FREE and GPU-agnostic (the evaluators are injected): give it proposals + a per-candidate
evaluator and it runs bounded searches, banks solved wins (verified-only), and reports:
* ``banked`` — verified wins written to memory,
* ``novel`` (ANNECS-V) — verified wins that were NEW (a fresh banked tip; a repeat solve on a known body
  writes nothing and is not counted) — the POET-style open-endedness counter,
* a resumable ``journal`` (G8 robustness): each candidate's outcome is persisted, so a crash/restart continues
  instead of re-doing banked work.

Pure-Python orchestration; the heavy lifting is in the injected ``evaluate_for``. Unit-tests with fakes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from virturoid.services.engineer_mode import run_engineer_search
from virturoid.services.harness_banking import bank_search_result
from virturoid.services.install_paths import anchored


@dataclass
class NightReport:
    candidates_run: int = 0
    banked: int = 0
    novel: int = 0                       # ANNECS-V: verified AND newly-banked
    budget_used: int = 0
    results: list = field(default_factory=list)
    stopped_reason: str = "proposals_exhausted"
    qd: dict = None                      # QD-archive dashboard snapshot (coverage/qd_score/annecs_v) when an archive is passed


def default_night_descriptor(cand, search) -> tuple:
    """A BEHAVIORAL descriptor for the QD archive: ``(best_forward, cadence)`` (plan gap-closure N19). The old
    ``(num_actuated_joints, forward)`` keyed one axis on morphology SIZE, but the fixed zoo spans only a few n_dof
    values → most of that axis is unreachable → coverage% inflated (reads "empty" even after productive nights).
    A behavioral (speed × cadence) descriptor is the MAP-Elites-appropriate key: even ONE body fills many cells
    with diverse gaits, so coverage tracks real gait diversity. (Body-size diversity returns as an axis once WS2
    phase-2 adds a morphology-mutation operator that actually grows the frontier.)"""
    fwd = cad = 0.0
    if search.best is not None:
        m = search.best.artifact.get("metrics") or {}
        fwd = float(m.get("forward_m", m.get("forward", 0.0)) or 0.0)
        cad = float(m.get("cadence", 0.0) or 0.0)
    return (fwd, cad)


def _load_journal(path):
    if path is None or not Path(path).exists():
        return {}
    done = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rec = json.loads(line)
                done[rec["id"]] = rec
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _append_journal(path, rec):
    if path is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def laddered_evaluate_for(*, cpu_steps: int = 600, gpu_iters: int = 60, gpu_envs: int = 2048,
                          decimation: int = 10, action_lpf: float = 0.2, base_reward_weights: dict | None = None,
                          promote=None, warm_start_pool: str | None = str(anchored("build/models")), recall_steps: int = 900,
                          train_fn=None, verify_fn=None):
    """Production ``evaluate_for`` (plan v2 §5.1/N8): give each candidate a FIDELITY-LADDERED evaluate — a cheap
    CPU screen (pure-CPG rollout, seconds) that gates a GPU residual-training rung carrying the deploy-gap fixes
    (50 Hz decimation + action-LPF + keep-checkpoints, deploy==train). Returns ``evaluate_for(candidate) ->
    evaluate(spec)`` for :func:`run_night_shift`. ``train_fn``/``verify_fn`` are injectable (tests); defaults use
    the real gpu_trainer + CPU verify. This is what makes the night shift spend GPU only on CPU-survivors."""
    from virturoid.services.fidelity_ladder import make_gpu_locomotion_hifi, make_laddered_evaluate
    from virturoid.services.search_adapters import make_locomotion_evaluate

    def evaluate_for(cand):
        gene = cand["gene"]
        # TRANSFER ARM made real (§5.2, the cheapest compounding lever): warm-start the GPU rung from the banked
        # policy that best TRANSFERS forward to THIS body -- the measured fix for the from-scratch backward basin
        # (a hexapod warm-started from a forward quad). A candidate can override via cand["init_npz"].
        seed = cand.get("init_npz")
        if seed is None and warm_start_pool:
            try:
                from virturoid.services.transfer_seed import transfer_policy_for
                # rank transfers at the DEPLOY horizon (recall_steps, default 900), NOT the short screen cpu_steps:
                # a banked 50Hz-decimation walker (quaddec_fwd +0.668) needs the full episode to reveal its forward
                # travel, so a 300-step recall mis-ranks it BELOW a policy that merely lunges early -> the quad would
                # warm-start from the WRONG (hexapod) seed and fail. Same fix as run_arm_b's transfer recall.
                _pol, seed, _ranked = transfer_policy_for(gene, models_dir=warm_start_pool, steps=recall_steps)
            except Exception:  # noqa: BLE001 - transfer recall is best-effort; train from scratch if it fails
                seed = None
        screen = make_locomotion_evaluate(gene, steps=cpu_steps)
        hifi = make_gpu_locomotion_hifi(gene, iters=gpu_iters, envs=gpu_envs, decimation=decimation,
                                        action_lpf=action_lpf, base_reward_weights=base_reward_weights,
                                        init_npz=seed, train_fn=train_fn, verify_fn=verify_fn)
        # PROMOTE FIX: a body with a banked forward TRANSFER seed always earns the GPU rung -- the warm-start makes
        # it walk, so gating GPU on PURE-CPG screen survival would wrongly reject a body that only walks with the
        # learned residual (e.g. the quad, which the default CPG can't drive -> falls the screen -> never banks
        # despite a +0.668 banked seed). The cheap screen-survival gate still applies to NOVEL bodies (no seed).
        cand_promote = promote
        if cand_promote is None and seed is not None:
            cand_promote = lambda _r: True                    # transfer available -> spend GPU (it will warm-start)
        return make_laddered_evaluate(screen, hifi, promote=cand_promote)

    return evaluate_for


def run_night_shift(proposals, evaluate_for, *, memory_dir, llm=None, budget_evals: int = 200,
                    per_candidate_evals: int = 8, gate_target: float = 0.30, journal_path=None,
                    on_result=None, archive=None, descriptor_for=None) -> NightReport:
    """Run the night shift over ``proposals`` (an iterable of candidate dicts, each ``{id, gene, task,
    task_type?, gates?}``). ``evaluate_for(candidate) -> evaluate`` builds the per-candidate physics evaluator
    (fidelity-ladder adapter in production, a test double in tests). Bounded by ``budget_evals`` total search
    evaluations. ``journal_path`` (G8) makes it resumable — already-journaled candidate ids are skipped.

    ``archive`` (a ``qd_archive.QDArchive``) is optional: when given, each verified BANK is inserted into the
    MAP-Elites archive keyed by ``descriptor_for(cand, search)`` (default ``default_night_descriptor``), and the
    report carries the QD dashboard snapshot (coverage / qd_score / ANNECS-V) — the open-endedness metrics."""
    from virturoid.services.robotics_vector_memory import cosine, embed_body
    done = _load_journal(journal_path)
    rep = NightReport()
    banked_embeds: list = []                                  # ANNECS-V: distinct banked bodies (embedding-space)
    _descriptor = descriptor_for or default_night_descriptor

    for cand in proposals:
        cid = cand["id"]
        if cid in done:                                       # G8: resume — don't re-run a journaled candidate
            continue
        if rep.budget_used >= budget_evals:
            rep.stopped_reason = "budget_exhausted"
            break
        gene, task = cand["gene"], cand.get("task", "walk")
        task_type = cand.get("task_type", "locomotion")
        remaining = budget_evals - rep.budget_used
        search = run_engineer_search(
            task=task, gene=gene, evaluate=evaluate_for(cand), llm=llm, memory_dir=memory_dir,
            task_type=task_type, gates=cand.get("gates"), max_evals=min(per_candidate_evals, remaining),
            heuristic=cand.get("heuristic"))
        rep.budget_used += search.n_evals
        rep.candidates_run += 1

        banked = {"banked": False}
        if search.solved:
            banked = bank_search_result(search, gene=gene, memory_dir=memory_dir, task_type=task_type,
                                        gate_target=cand.get("gate_target", gate_target))
            if banked.get("banked"):
                rep.banked += 1
                # ANNECS-V novelty = a distinct body in MORPHOLOGY-EMBEDDING space (not tip-text): a near-identical
                # body already banked this run is coverage, not novelty (POET's minimal-criterion discipline).
                z = embed_body(gene)
                if all(cosine(z, e) < 0.98 for e in banked_embeds):
                    rep.novel += 1
                    banked_embeds.append(z)
                if archive is not None:                       # QD: place the verified win in its behavior niche
                    fit = float(search.best.fitness) if search.best else 0.0
                    try:
                        archive.add(cid, _descriptor(cand, search), fit)
                    except Exception:  # noqa: BLE001 - a bad descriptor must not break the night run
                        pass

        rec = {"id": cid, "solved": search.solved, "banked": bool(banked.get("banked")),
               "evals": search.n_evals, "stopped": search.stopped_reason,
               "best_mode": (search.best.artifact["failure_mode"] if search.best else None)}
        rep.results.append(rec)
        _append_journal(journal_path, rec)
        if on_result is not None:
            on_result(rec)

    if archive is not None:
        rep.qd = archive.snapshot()
    return rep
