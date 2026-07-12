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

    def to_dict(self) -> dict:
        return {"admitted": self.admitted, "rejected": self.rejected, "coverage": self.coverage,
                "annecs": self.annecs, "mean_pairwise_similarity": self.mean_pairwise_similarity,
                "n_admitted": len(self.admitted), "n_diagnostics": len(self.diagnostics), "wall_s": self.wall_s}


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


def _reject(gate: str, niche, detail=None) -> dict:
    return {"admitted": False, "gate": gate, "niche": niche, "detail": detail}


def admit(gene, *, grid, corpus_genes, corpus_keys, niche_best: dict, config: FactoryConfig, verify_fn,
          prompt=None) -> dict:
    """Run one candidate through the ordered gate stack. Returns an admit/reject record with the FIRST failing
    gate (auditable). Verification runs only after the cheap structural gates pass."""
    from virturoid.services.heldout_set import body_key, is_held_out
    from virturoid.services.novelty_search import embedding_novelty, niche_key
    key = niche_key(gene)
    # 0) held-out guard (leak protection) — the factory must never learn on a compounding-eval body
    if is_held_out(gene, prompt=prompt):
        return _reject("held_out", key)
    # 1) feasibility (cheap, pre-sim): a valid kinematic tree
    if gene.validate():
        return _reject("feasibility", key, "schema")
    res = verify_fn(gene)
    if not res.get("schema_valid", True) or not res.get("compiles", True):
        return _reject("feasibility", key, "compile")
    # 2) real-data anchor: the label must be a physics verdict, NEVER a metric-predicted outcome (§10.2/§10.5)
    if res.get("source") != "physics":
        return _reject("real_data_anchor", key, res.get("source"))
    # 3) verdict: the render-level un-gameable credible verdict
    if not res.get("credible"):
        return _reject("verdict", key, res.get("verdict"))
    # 4) success floor
    if float(res.get("fitness", 0.0)) < config.success_forward_floor:
        return _reject("success_floor", key, res.get("fitness"))
    # 5) novelty / dedup — an EXACT structural duplicate (same body_key) is rejected outright (the un-gameable
    #    duplicate signal; the embedding is coarse by design, so structure is the finer discriminator). The soft
    #    blended novelty catches the near-duplicate-but-not-identical case (the mode-collapse guard, §10.5).
    if body_key(gene) in (corpus_keys or set()):
        return _reject("novelty", key, "exact structural duplicate")
    nov = 0.5 * grid.novelty(gene) + 0.5 * embedding_novelty(gene, corpus_genes)
    if nov < config.min_novelty:
        return _reject("novelty", key, round(nov, 4))
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

    for _ in range(cfg.max_bodies):
        context = {"thinnest_classes": grid.thinnest_classes(), "coverage": grid.coverage(), "night": night}
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
    res.mean_pairwise_similarity = _mean_pairwise_similarity(corpus)
    res.wall_s = round(time.perf_counter() - start, 2)
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


def held_out_aware(propose_fn, *, max_resample: int = 4):
    """Wrap a proposer so a night's proposal budget is not WASTED on bodies the held-out guard will trivially
    reject. MEASURED 2026-07-12 (v7-C1): a live-gpt-5.5 night admitted 1/30 with rejected={held_out: 25} — the
    LLM designs bodies that land in reserved niches (extra limbs → many_limb, aquatic/aerial, wheel-leg hybrid),
    burning 25 of 30 slots before any real gate ran. On a held-out proposal this resamples up to ``max_resample``
    times and yields the first non-held-out body.

    This NEVER weakens the leak guard: the gate stack's held-out check stays authoritative (a held-out body that
    slips through after exhausted resamples is still rejected there). The wrapper only stops the budget bleeding —
    exactly the 'smart proposer targets thin niches' the factory was designed for (avoiding reserved niches is
    part of being smart). Pure w.r.t. the guard; needs no corpus access."""
    from virturoid.services.heldout_set import is_held_out

    def wrapped(context):
        proposed = propose_fn(context)
        tries = 0
        while proposed is not None and tries < max_resample:
            gene, prompt = (proposed if isinstance(proposed, tuple) else (proposed, None))
            try:
                held = is_held_out(gene, prompt=prompt)
            except Exception:  # noqa: BLE001 - an unclassifiable body is not our reason to resample
                return proposed
            if not held:
                return proposed
            proposed = propose_fn(context)                    # this slot fell in a reserved niche -> try again
            tries += 1
        return proposed                                       # first non-held-out, or None (bank done), or last try
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
