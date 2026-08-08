"""The corpus factory — autonomous verified-outcome growth (master_plan_v6 §10.2 / WS-C.2).

The gated metric, the hints, the transfer ledger, the compounding claim all starve without volume + diversity.
The machinery exists; this is the missing DRIVER: a budget-capped, checkpointed, failure-isolated night that runs

    PROPOSE → FEASIBILITY GATE → VERIFY-BUILD → ADMIT (ordered gate stack) → TRANSFER QUEUE → CHECKPOINT + ratchet

The proposer is injected (an offline batch LLM agent in production — explicit dev tokens, NEVER the customer hot
path; a deterministic generator in tests): the factory is proposer-agnostic and never designs a body itself. The
**ordered admission gate stack** (cheapest first) is the load-bearing part, and every gate is un-gameable:

  0. held-out guard        — never admit a reserved held-out / whole-niche body (leak protection)
  1. feasibility           — valid tree + compiles to a finite-mass model (pre-sim reject)
  2. real-data anchor      — the label must be a PHYSICS verdict, never a metric-PREDICTED outcome
  3. verdict               — the render-level un-gameable credible verdict (never scalar-only)
  4. success floor         — fitness ≥ τ
  5. novelty / dedup       — reject a near-duplicate (raises mean pairwise similarity / fills a saturated niche)
  6. diversity retention   — per-niche keep-best (keep-hard-when-abundant)

Dashboards (a flat one with continued spend IS collapse): ANNECS (cumulative novel-AND-solved bodies), mean
pairwise embedding similarity, niche coverage. A failed body is a diagnostic record, never a campaign kill.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

GATES = ("held_out", "feasibility", "real_data_anchor", "verdict", "success_floor", "novelty", "retention")


@dataclass
class FactoryConfig:
    max_bodies: int = 20               # budget cap (bodies proposed per night)
    min_novelty: float = 0.05          # dedup floor (blended niche+embedding novelty)
    per_niche_cap: int = 8             # diversity-aware retention: keep-best beyond this per fine niche
    success_forward_floor: float = 0.0  # 'credible' already encodes the floor; hook for a stricter τ
    grow_ledger: bool = False          # cross-evaluate admitted gaits on nearest bodies (grows transfer ledger)
    grow_ledger_k: int = 3


@dataclass
class NightResult:
    admitted: list = field(default_factory=list)
    rejected: dict = field(default_factory=dict)         # gate -> count
    diagnostics: list = field(default_factory=list)      # per-candidate record (never lost)
    coverage: dict = field(default_factory=dict)
    annecs: int = 0
    mean_pairwise_similarity: float | None = None
    wall_s: float = 0.0
    # Drafts drawn vs slots filled. A night reporting 20 proposals hides how many BODIES were built to get
    # them; when a proposer resamples past reserved niches, drafts/slots is the ratio that says whether the
    # guard is steering (→1.0) or the proposer is re-rolling blind into the same wall (→ max_resample+1).
    proposer: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"admitted": self.admitted, "rejected": self.rejected, "coverage": self.coverage,
                "annecs": self.annecs, "mean_pairwise_similarity": self.mean_pairwise_similarity,
                "n_admitted": len(self.admitted), "n_diagnostics": len(self.diagnostics), "wall_s": self.wall_s,
                "proposer": self.proposer}


def _compiles_and_credible(gene) -> dict:
    """Default VERIFY-BUILD: the un-gameable physics verdict via Design-Bench (schema→compile→verdict), tagged
    ``source='physics'`` so the real-data-anchor gate can prove the label wasn't a metric prediction."""
    from virturoid.services.design_bench import evaluate_design
    row = evaluate_design(gene, verify=True)
    return {"schema_valid": bool(row["schema_valid"]), "compiles": bool(row["compiles"]),
            "credible": bool(row["credible"]), "fitness": float(row.get("fitness_raw", 0.0)),
            "verdict": row.get("verdict"), "kind": row.get("kind"), "source": "physics"}


def gait_search_verify(gene, *, max_evals: int = 6, steps: int = 500) -> dict:
    """The §10.2 VERIFY-BUILD: a bounded CPU CEM/grid gait search that tries to FIND a credible controller for a
    legged body (minutes/body), not just check the default scripted gait — so a body that CAN be solved is
    admitted, and one that can't is honestly rejected. Non-legged bodies fall back to the fast scripted verdict.
    Still ``source='physics'`` (every eval is a real rollout)."""
    from virturoid.services.task_matched_eval import robot_kind
    if robot_kind(gene) != "legged":
        return _compiles_and_credible(gene)
    # cheap structural pre-check first (schema/compile), so a broken body fails fast without a search
    base = _compiles_and_credible(gene)
    if not base["schema_valid"] or not base["compiles"]:
        return base
    try:
        from virturoid.services.design_search import run_design_search
        from virturoid.services.search_adapters import cpg_grid_proposer, make_cpg_evaluate
        rep = run_design_search(propose=cpg_grid_proposer(), evaluate=make_cpg_evaluate(gene, steps=steps),
                                task_type="locomotion", max_evals=max_evals,
                                gates={"forward_m": 0.12, "cadence": 3.0, "upright": 0.5})
        b = rep.best
        fwd = float(b.result.get("forward", 0.0)) if b else 0.0
        return {"schema_valid": True, "compiles": True, "credible": bool(rep.solved), "fitness": fwd,
                "verdict": f"gait_search solved={rep.solved} best_forward={fwd:.2f}m", "kind": "legged",
                "source": "physics", "gait_params": (b.spec.get("params") if b else None)}
    except Exception as exc:  # noqa: BLE001 - a search failure falls back to the scripted verdict, never a crash
        base["verdict_search_error"] = f"{type(exc).__name__}: {exc}"[:120]
        return base


def gait_fit_verify_fn(memory_dir):
    """VERIFY-BUILD that asks the question THE GAIT BANK actually cares about, and the one the two verifiers above
    do not: *does this body have an operating point of its own that is still walking at the settling horizon, and
    does a perturbed copy of that point walk too?*

    ``_compiles_and_credible`` judges every legged body at the shipped ``freq 1.5 / kp 32``, which is one body's
    hand-tuned numbers — so it admits bodies that need no gait learned and rejects bodies that walk perfectly well
    under a controller of their own. ``gait_search_verify`` does search, but with a 6-evaluation CPG grid at 500
    steps: 12x shorter than the horizon a walk is judged at today, and task #267 measured exactly what a short
    horizon buys — the grounded authored cat reads CREDIBLE WALK at 1500 steps and is on the floor at 2014.

    This runs the PRODUCT's fitter (``fit_gait_for_body``) against the night's OWN memory, and reports the
    operating point the body would actually ship with plus the robustness margin that decides whether it may be
    banked. Two properties are load-bearing:

    * **The night's DB, never the product's.** ``fit_gait_for_body(db=None)`` falls back to opening
      ``build/memory``, so a "clean rebuild" run that way would warm-start every fit from the very corpus it is
      supposed to be replacing. The closure holds the night's memory directory so a fresh bank stays fresh.
    * **A body whose SHIPPED DEFAULT already walks is still measured.** The fitter returns early in that case
      (correctly — it must never churn a working controller), so the default's own margin is measured here
      instead. A body+params pair that walks robustly is a legitimate corpus row whether or not the numbers were
      searched for; what is NOT legitimate is banking it with the fragility question unasked.

    COST: this is the honest, expensive verifier — measured 24-145 s per legged body for the fit, plus 4-12
    settling-horizon rollouts for the margin. A non-legged candidate falls through to the fast scripted verdict.
    """
    def verify(gene) -> dict:
        from virturoid.services.task_matched_eval import robot_kind
        if robot_kind(gene) != "legged":
            return _compiles_and_credible(gene)
        base = _compiles_and_credible(gene)
        if not base["schema_valid"] or not base["compiles"]:
            return base
        from virturoid.services.gait_flywheel import (_DEFAULT_GAIT, _SETTLE_STEPS, fit_gait_for_body,
                                                      robustness_margin)
        from virturoid.services.memory_db import MemoryDB
        with MemoryDB(Path(memory_dir) / "virturoid_memory.db") as db:
            fit = fit_gait_for_body(gene, db=db, bank=False)
        if not fit.get("ok", True):
            return {**base, "credible": False, "verdict": f"gait fit ERRORED: {fit.get('error')}",
                    "fit_error": fit.get("error")}
        if fit.get("adopted"):
            params, rob = dict(fit["params"]), dict(fit.get("robustness") or {})
            if "robustness_rel" not in rob:
                rob = {"robustness_rel": fit.get("robustness_rel"), "probes": fit.get("robustness_probes") or {},
                       "steps": _SETTLE_STEPS}
            fwd, src = float(fit.get("confirm_forward_m") or fit.get("forward_m") or 0.0), "fitted"
        elif fit.get("searched"):
            return {**base, "credible": False, "kind": "legged", "source": "physics",
                    "verdict": f"no operating point still walking at {_SETTLE_STEPS} steps ({fit.get('reason')})",
                    "n_evals": fit.get("n_evals"), "gait_source": "none"}
        else:
            # The shipped default already walks it. Measure THAT point's margin — the row is only bankable if the
            # walk survives a perturbation of itself, and nothing upstream has asked that question.
            params, src = dict(_DEFAULT_GAIT), "default"
            fwd = float(fit.get("default_forward_m") or 0.0)
            rob = robustness_margin(gene, params, steps=_SETTLE_STEPS, per_param=False)
        return {"schema_valid": True, "compiles": True, "credible": True, "fitness": abs(fwd), "kind": "legged",
                "source": "physics", "gait_params": params, "gait_source": src,
                "robustness": rob, "robustness_rel": rob.get("robustness_rel"),
                "verdict": (f"CREDIBLE WALK at {_SETTLE_STEPS} steps ({fwd:+.3f} m, {src} op-point); "
                            f"robustness_rel={rob.get('robustness_rel')}")}
    return verify


def gait_bank_fn(gene, result: dict, memory_dir) -> None:
    """BANK THE ADMITTED BODY'S GAIT — the write the corpus factory never made.

    MEASURED on this checkout: the one real factory night on record (``build/memory_factory_v7``, 2026-07-12,
    30 proposals, 1 admitted) left ``skills`` EMPTY and ``provenance`` empty; its DB holds exactly one body
    vector. ``default_bank_fn`` upserts a BODY into the vector sub-space and nothing else, so a night can admit
    a body that walks and contribute ZERO rows to the gait bank the flywheel retrieves from. That is task #207's
    "dead write path", and it means no amount of running the factory as shipped could ever grow the gait corpus.

    Banking happens HERE, after the whole ordered gate stack, not inside the verifier — so a body rejected for
    novelty or retention cannot leave a gait row behind. The params and the margin come from the verify result
    (``gait_fit_verify_fn``), so nothing is re-searched and nothing is banked that was not measured: a candidate
    with no ``robustness_rel`` is refused, exactly as ``learn_gait_flywheel`` refuses one.
    """
    if memory_dir is None:
        return
    params = result.get("gait_params")
    rob = result.get("robustness") or {}
    if not isinstance(params, dict) or rob.get("robustness_rel") is None:
        return                                        # fragile, or never measured — a bank of controllers says no
    from virturoid.services.gait_flywheel import _DeployResult, bank_gait
    from virturoid.services.memory_db import MemoryDB
    holder = _DeployResult(params, {"forward": float(result.get("fitness", 0.0)), "height_ratio": 1.0,
                                    "survived": True, "credible": True})
    with MemoryDB(Path(memory_dir) / "virturoid_memory.db") as db:
        bank_gait(db, gene, holder, robustness=rob)
    # ...and the body sub-space write too (novelty / nearest-bodies), on its OWN connection after the first has
    # closed — two live connections to one SQLite file is how a night acquires a writer-lock stall.
    default_bank_fn(gene, result, memory_dir)


def _reject(gate: str, niche, detail=None) -> dict:
    return {"admitted": False, "gate": gate, "niche": niche, "detail": detail}


def admit(gene, *, grid, corpus_genes, corpus_keys, niche_best: dict, config: FactoryConfig, verify_fn,
          prompt=None) -> dict:
    """Run one candidate through the ordered gate stack. Returns an admit/reject record with the FIRST failing
    gate (auditable). Verification runs only after the cheap structural gates pass.

    CHEAPEST-FIRST IS LITERAL. The novelty/dedup gate is pure structure — a body_key comparison and an embedding
    distance, microseconds — but it used to sit BEHIND ``verify_fn``, which for a gait-corpus night is a 39 s
    fit-plus-robustness measurement per body (measured 2026-08-08 on this checkout). A night that rejected 9
    proposals for novelty therefore spent ~6 minutes fitting controllers for bodies it had already decided were
    duplicates. Novelty depends on nothing the verifier produces, so it moves ahead of it. Admissions are
    unchanged by the move (a body failing novelty was never admitted); what changes is the bill, and the reported
    gate for a body that would have failed both — now the cheaper, more actionable one."""
    from virturoid.services.heldout_set import body_key, is_held_out
    from virturoid.services.novelty_search import embedding_novelty, niche_key
    key = niche_key(gene)
    # 0) held-out guard (leak protection) — the factory must never learn on a compounding-eval body
    if is_held_out(gene, prompt=prompt):
        return _reject("held_out", key)
    # 1) feasibility (cheap, pre-sim): a valid kinematic tree
    if gene.validate():
        return _reject("feasibility", key, "schema")
    # 2) novelty / dedup — an EXACT structural duplicate (same body_key) is rejected outright (the un-gameable
    #    duplicate signal; the embedding is coarse by design, so structure is the finer discriminator). The soft
    #    blended novelty catches the near-duplicate-but-not-identical case (the mode-collapse guard, §10.5).
    if body_key(gene) in (corpus_keys or set()):
        return _reject("novelty", key, "exact structural duplicate")
    nov = 0.5 * grid.novelty(gene) + 0.5 * embedding_novelty(gene, corpus_genes)
    if nov < config.min_novelty:
        return _reject("novelty", key, round(nov, 4))
    res = verify_fn(gene)
    if not res.get("schema_valid", True) or not res.get("compiles", True):
        return _reject("feasibility", key, "compile")
    # 3) real-data anchor: the label must be a physics verdict, NEVER a metric-predicted outcome (§10.2/§10.5)
    if res.get("source") != "physics":
        return _reject("real_data_anchor", key, res.get("source"))
    # 4) verdict: the render-level un-gameable credible verdict
    if not res.get("credible"):
        return _reject("verdict", key, res.get("verdict"))
    # 5) success floor
    if float(res.get("fitness", 0.0)) < config.success_forward_floor:
        return _reject("success_floor", key, res.get("fitness"))
    # 6) diversity-aware retention: per-niche keep-best (keep-hard-when-abundant)
    incumbent = niche_best.get(key)
    if grid.count(gene) >= config.per_niche_cap and (incumbent is not None and res["fitness"] <= incumbent):
        return _reject("retention", key, "niche full + not better than incumbent")
    return {"admitted": True, "gate": None, "niche": key, "result": res, "novelty": round(nov, 4)}


class FactoryManifest:
    """The persistent corpus manifest — enables checkpointing (resume a crashed night) + cross-night ANNECS and
    novelty continuity. Stores each admitted body's gene (so novelty/grid rebuild across nights)."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.data = {"artifact": "virturoid_corpus_factory", "version": "v1", "nights": 0,
                     "cumulative_annecs": 0, "admitted": [], "rejected_totals": {}}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                pass

    def corpus_genes(self) -> list:
        from virturoid.schemas.gene import RobotGene
        out = []
        for a in self.data.get("admitted", []):
            try:
                out.append(RobotGene.from_dict(a["gene"]))
            except Exception:  # noqa: BLE001
                continue
        return out

    def niche_best(self) -> dict:
        best: dict = {}
        for a in self.data.get("admitted", []):
            k = tuple(a.get("niche")) if a.get("niche") else None
            if k is None:
                continue
            best[k] = max(best.get(k, 0.0), float(a.get("fitness", 0.0)))
        return best

    def record_admit(self, gene, rec: dict, night: int) -> None:
        self.data.setdefault("admitted", []).append(
            {"body_key": _body_key(gene), "niche": list(rec["niche"]), "gene": gene.to_dict(),
             "fitness": rec["result"]["fitness"], "verdict": rec["result"].get("verdict"),
             "novelty": rec["novelty"], "night": night})

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, default=str), encoding="utf-8")
        return self.path


def _body_key(gene) -> str:
    from virturoid.services.heldout_set import body_key
    return body_key(gene)


def _mean_pairwise_similarity(genes) -> float | None:
    import math
    from virturoid.services.morphology_embedding import embed_gene
    genes = [g for g in genes if g is not None]
    if len(genes) < 2:
        return None
    vecs = [embed_gene(g) for g in genes]

    def _cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na > 0 and nb > 0 else 0.0
    sims = [_cos(vecs[i], vecs[j]) for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
    return round(sum(sims) / len(sims), 4) if sims else None


def run_factory_night(propose_fn, *, config: FactoryConfig | None = None, manifest_path: Path | str | None = None,
                      verify_fn=None, memory_dir: Path | str | None = None, bank_fn=None) -> NightResult:
    """Run one budget-capped factory night. ``propose_fn(context) -> (gene, prompt) | gene | None`` yields the next
    candidate (context carries thinnest classes + coverage so a smart proposer targets thin niches). Checkpoints
    to the manifest after every admit. Returns a NightResult with the admission funnel + diversity dashboards."""
    from virturoid.services.novelty_search import MapElitesGrid, niche_key
    cfg = config or FactoryConfig()
    verify = verify_fn or _compiles_and_credible
    manifest = FactoryManifest(manifest_path or (Path(memory_dir or "build/corpus") / "corpus_factory.json"))
    corpus = manifest.corpus_genes()
    grid = MapElitesGrid(corpus)
    corpus_keys = {_body_key(g) for g in corpus}
    niche_best = manifest.niche_best()
    night = int(manifest.data.get("nights", 0)) + 1
    res = NightResult()
    start = time.perf_counter()

    brief = guard_brief()          # the held-out guard as a design constraint, handed to the proposer up front
    for _ in range(cfg.max_bodies):
        context = {"thinnest_classes": grid.thinnest_classes(), "coverage": grid.coverage(), "night": night,
                   "guard": brief, "filled_niches": sorted(grid.cells), "corpus_keys": set(corpus_keys)}
        try:
            proposed = propose_fn(context)
        except Exception as exc:  # noqa: BLE001 - a proposer failure is isolated, never kills the night
            res.diagnostics.append({"stage": "propose", "error": f"{type(exc).__name__}: {exc}"[:160]})
            continue
        if proposed is None:
            break
        gene, prompt = (proposed if isinstance(proposed, tuple) else (proposed, None))
        try:
            rec = admit(gene, grid=grid, corpus_genes=corpus, corpus_keys=corpus_keys, niche_best=niche_best,
                        config=cfg, verify_fn=verify, prompt=prompt)
        except Exception as exc:  # noqa: BLE001 - one odd body is a diagnostic, not a crash
            res.diagnostics.append({"stage": "admit", "error": f"{type(exc).__name__}: {exc}"[:160]})
            continue
        if not rec["admitted"]:
            res.rejected[rec["gate"]] = res.rejected.get(rec["gate"], 0) + 1
            res.diagnostics.append({"stage": "reject", "gate": rec["gate"], "detail": rec.get("detail"),
                                    "niche": list(rec["niche"]) if rec.get("niche") else None})
            continue
        # ADMIT: update grid + corpus + niche-best, checkpoint, and (best-effort) bank
        grid.add(gene)
        corpus.append(gene)
        corpus_keys.add(_body_key(gene))
        k = niche_key(gene)
        niche_best[k] = max(niche_best.get(k, 0.0), rec["result"]["fitness"])
        manifest.record_admit(gene, rec, night)
        res.admitted.append({"body_key": _body_key(gene), "niche": list(k),
                             "fitness": rec["result"]["fitness"], "verdict": rec["result"].get("verdict")})
        manifest.save()                                   # checkpoint after every admit (crash-resumable)
        if bank_fn is not None:
            try:
                bank_fn(gene, rec["result"], memory_dir)
            except Exception as exc:  # noqa: BLE001 - banking is an accelerant, never blocks admission
                res.diagnostics.append({"stage": "bank", "error": f"{type(exc).__name__}: {exc}"[:160]})

    # cross-night ANNECS: admitted bodies are BOTH novel (passed gate 5) AND solved (credible) — POET's ANNECS
    manifest.data["cumulative_annecs"] = int(manifest.data.get("cumulative_annecs", 0)) + len(res.admitted)
    manifest.data["nights"] = night
    for g, c in res.rejected.items():
        manifest.data.setdefault("rejected_totals", {})[g] = \
            manifest.data.get("rejected_totals", {}).get(g, 0) + c
    manifest.save()
    res.annecs = manifest.data["cumulative_annecs"]
    res.coverage = grid.coverage()
    # DISTINCT BODIES is the flywheel's binding constraint (#274) and the dashboards did not report it: a corpus
    # of 20 bodies that are 3 shapes reads as 20 everywhere. total_bodies vs distinct_body_keys says which.
    res.coverage["distinct_body_keys"] = len(corpus_keys)
    res.mean_pairwise_similarity = _mean_pairwise_similarity(corpus)
    res.wall_s = round(time.perf_counter() - start, 2)
    stats = getattr(propose_fn, "stats", None)
    if isinstance(stats, dict):
        res.proposer = {**stats, "slots": cfg.max_bodies,
                        "drafts_per_slot": round(stats.get("drafts", 0) / max(1, cfg.max_bodies), 2)}
    # RATCHET the gated metric against the (possibly grown) ledger — adopts only if it beats baseline held-out
    if memory_dir is not None:
        try:
            from virturoid.services.memory_db import MemoryDB
            from virturoid.services.transfer_ledger import ratchet_metric
            with MemoryDB(Path(memory_dir) / "virturoid_memory.db") as db:
                res.diagnostics.append({"stage": "ratchet", **ratchet_metric(db)})
        except Exception as exc:  # noqa: BLE001
            res.diagnostics.append({"stage": "ratchet", "error": f"{type(exc).__name__}: {exc}"[:160]})
    return res


def guard_brief() -> dict:
    """The held-out guard stated as a DESIGN BRIEF for the proposer — what a body must avoid to be admissible.

    Passed into every proposal context by ``run_factory_night``. A code proposer reads ``max_limbs`` and picks
    targets inside it; an LLM proposer gets the same dict in its prompt. This is the half of "steer, don't
    filter" that happens BEFORE a body exists."""
    from virturoid.services.heldout_set import design_constraints
    return design_constraints()


def held_out_aware(propose_fn, *, max_resample: int = 4, on_reserved=None):
    """Make the held-out guard STEER the proposer instead of filtering it after the fact.

    WHY THIS EXISTS, AND WHY THE FIRST VERSION DID NOT WORK. MEASURED 2026-07-12 (v7-C1): a live night admitted
    1/30 with rejected={held_out: 25}. This wrapper was written for exactly that and the waste did not move —
    a 40-body gated night still logged {held_out: 24}. Re-measured 2026-08-08, the reason was not the wrapper's
    shape but the guard underneath it: ``_niche_many_limb`` counted the composer's NECK AND TAIL as limbs, so
    100% of the factory's own legged prompt bank (18 of 18) composed to a "many-limbed" body. Resampling cannot
    escape a partition that contains the entire family being proposed; the wrapper faithfully re-rolled four
    times into the same wall and handed the fifth body to the gate. The guard is fixed at the source
    (``heldout_set``, HELD_OUT_VERSION v2); this wrapper is fixed here.

    Two changes, both about spending a DRAFT instead of a SLOT:

    * ``on_reserved(gene, prompt, why)`` — the reject is reported back to the proposer with the guard's
      ``explain()`` payload, so a proposer that can retarget (ours blacklists the target it just drew) does not
      re-roll blind. A resample loop with no feedback is a random search over the same distribution.
    * A proposer returning ``None`` mid-resample means THAT DRAFT failed, not that the bank is exhausted. The
      old loop returned None on the first such hiccup, and ``run_factory_night`` reads None as "stop" — so one
      transient composition error ended the night and forfeited the rest of the budget silently.

    NEVER weakens the leak guard: the gate stack's held-out check stays authoritative on the FINAL body, and a
    held-out body that survives exhausted resamples is still rejected there."""
    from virturoid.services.heldout_set import explain
    stats = {"drafts": 0, "reserved_drafts": 0, "failed_drafts": 0}

    def wrapped(context):
        last = None
        for _ in range(max(1, max_resample + 1)):
            proposed = propose_fn(context)
            stats["drafts"] += 1
            if proposed is None:
                stats["failed_drafts"] += 1
                if last is None:
                    return None                               # nothing drawn at all -> the bank really is done
                continue                                      # a failed DRAFT is not an exhausted bank
            last = proposed
            gene, prompt = (proposed if isinstance(proposed, tuple) else (proposed, None))
            try:
                why = explain(gene, prompt=prompt)
            except Exception:  # noqa: BLE001 - an unclassifiable body is not our reason to resample
                return proposed
            if not why["held_out"]:
                return proposed
            stats["reserved_drafts"] += 1
            if on_reserved is not None:
                try:
                    on_reserved(gene, prompt, why)             # steer: tell the proposer WHICH wall it hit
                except Exception:  # noqa: BLE001 - feedback is an accelerant, never a failure mode
                    pass
        return last                                            # exhausted -> hand it over; the gate still rules
    wrapped.stats = stats    # drafts spent vs slots filled — the number that says whether steering is working
    return wrapped


def default_bank_fn(gene, result: dict, memory_dir) -> None:
    """The minimal banking that makes an admitted body part of the retrievable corpus: upsert it into the BODY
    vector sub-space (buildable=True) so exemplar_retrieval / novelty / nearest_bodies see it, + provenance."""
    if memory_dir is None:
        return
    from virturoid.services.memory_db import MemoryDB
    from virturoid.services.robotics_vector_memory import BODY, RoboticsVectorMemory, embed_body
    with MemoryDB(Path(memory_dir) / "virturoid_memory.db") as db:
        vm = RoboticsVectorMemory(db)
        vm.upsert(BODY, f"factory_{_body_key(gene)}", embed_body(gene),
                  {"robot_class": gene.robot_class, "buildable": True, "source": "corpus_factory"})
