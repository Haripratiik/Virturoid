"""Gait flywheel — make learned locomotion COMPOUND via specific, tokenized retrieval.

Audit finding: learned gaits banked as *designs*, never as reusable *policies*; provenance edges were never
recorded; so the flywheel didn't actually compound (moat_status: 578 builds, ~0 skills reused). And a measured
truth: warm-start only compounds when the prior FITS the new body — so retrieval must be SPECIFIC (robot-type +
task + morphology), and the transfer must be SCREENED (a bad prior must never hurt).

This module closes that loop for gait_search:
  * bank_gait   — store the learned gait PARAMS as a retrievable skill keyed by (robot_class, task, species) in the
                  MemoryDB skill bank (base_config), so a future body can recall the actual controller, not a design.
  * recall_gait — retrieve the best banked gait for the CLOSEST prior body (recall_skill keys class+task, prefers
                  nearest species; falls back to the vector-memory nearest-body sub-space).
  * learn_gait_flywheel — recall -> SCREENED warm-start search (gait_search injects+screens the prior) -> bank the
                  result -> record a PROVENANCE edge with the measured delta, so compounding is provable, not asserted.

Deterministic; CPU MuJoCo (via gait_search); standard-library persistence (MemoryDB + vector memory).
"""

from __future__ import annotations

import json

LOCOMOTION = "locomotion"
_FWD_NORM = 1.5   # metres that maps to success_rate 1.0 (a strong learned quad walk ~1.85 m)


def _class_of(gene) -> str:
    cls = getattr(gene, "robot_class", None)
    if cls:
        return str(cls)
    try:
        from virturoid.services.task_matched_eval import robot_kind
        return robot_kind(gene)
    except Exception:  # noqa: BLE001
        return "legged"


def bank_gait(db, gene, result, *, task: str = LOCOMOTION) -> str | None:
    """Bank a learned gait's PARAMS as a retrievable skill (keyed by class+task+species). Returns the skill_id.

    Only banks a DEPLOYABLE result (survived + real forward) — never a fall (honesty: the bank must stay a bank of
    working controllers). ``record_skill`` is idempotent-keep-best, so a weaker later gait can't clobber a stronger one.
    """
    if not getattr(result, "best_survived", False) or abs(getattr(result, "best_forward", 0.0)) < 0.15:
        return None
    cls = _class_of(gene)
    species = getattr(gene, "species", None) or cls
    gene_id = getattr(gene, "id", None) or cls
    skill_id = f"gait::{cls}::{gene_id}"[:96]
    success = min(1.0, abs(result.best_forward) / _FWD_NORM)
    db.record_skill(
        skill_id, cls, task, success_rate=success, species=species, gene_id=gene_id,
        base_config={"gait_params": result.best_params, "forward_m": round(result.best_forward, 4),
                     "height_ratio": round(result.best_height_ratio, 3), "controller": "crawl_gait"},
        notes="learned deployable gait (gait_search); base_config.gait_params IS the deploy controller")
    return skill_id


def recall_gait(db, gene, *, task: str = LOCOMOTION) -> dict | None:
    """Recall the best banked gait PARAMS for the closest prior body of this class+task. None if nothing banked."""
    cls = _class_of(gene)
    species = getattr(gene, "species", None) or cls
    sk = db.recall_skill(cls, task, species=species)
    if not sk:
        return None
    bc = sk.get("base_config")
    if isinstance(bc, str):
        try:
            bc = json.loads(bc)
        except json.JSONDecodeError:
            return None
    if not isinstance(bc, dict):
        return None
    return bc.get("gait_params")


def learn_gait_flywheel(gene, db, *, generations: int = 8, pop: int = 20, steps: int = 900,
                        seed: int = 0, workers: int = 1, bank: bool = True, vm=None) -> dict:
    """Recall a specific prior -> SCREENED warm-start search -> bank -> record provenance. The compounding loop.

    Returns ``{forward_m, height_ratio, survived, reused_prior, prior_transfer_forward, banked_skill,
    compounding_delta}`` where ``compounding_delta`` = how much the search beat the prior's zero-shot transfer
    (the measured value the reuse added; None when nothing was reused).
    """
    from virturoid.services.gait_search import search_gait

    prior = recall_gait(db, gene)
    res = search_gait(gene, generations=generations, pop=pop, steps=steps, seed=seed,
                      workers=workers, warm_start=prior)
    skill_id = bank_gait(db, gene, res) if bank else None

    compounding_delta = None
    if prior is not None and res.prior_transfer_forward is not None:
        compounding_delta = round(abs(res.best_forward) - abs(res.prior_transfer_forward), 4)
        if skill_id is not None:                              # record the warm-start edge with the measured delta
            try:
                if vm is None:
                    from virturoid.services.robotics_vector_memory import RoboticsVectorMemory
                    vm = RoboticsVectorMemory(db)
                vm.record_provenance("skill", skill_id, parent_type="skill", parent_id="gait_prior",
                                     kind="gait_warm_start", delta=compounding_delta,
                                     meta={"prior_transfer_forward": res.prior_transfer_forward,
                                           "final_forward": res.best_forward})
            except Exception:  # noqa: BLE001 - provenance is best-effort; never fail the learn
                pass

    return {
        "forward_m": round(res.best_forward, 4), "height_ratio": round(res.best_height_ratio, 3),
        "survived": bool(res.best_survived), "reused_prior": prior is not None,
        "prior_transfer_forward": (round(res.prior_transfer_forward, 4)
                                   if res.prior_transfer_forward is not None else None),
        "banked_skill": skill_id, "compounding_delta": compounding_delta,
        "params": res.best_params,
    }
