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
# A gait prior changes a controller's phase/amplitude targets.  Weak embedding
# matches are not evidence that those targets fit the new kinematic tree.
_MIN_GAIT_TRANSFER_SIMILARITY = 0.55


def _class_of(gene) -> str:
    cls = getattr(gene, "robot_class", None)
    if cls:
        return str(cls)
    try:
        from virturoid.services.task_matched_eval import robot_kind
        return robot_kind(gene)
    except Exception:  # noqa: BLE001
        return "legged"


def _deploy_sim_config(gene) -> dict:
    """The sim config a gait verdict was measured/deployed under, PINNED with the bank (Thesis A / dossier risk
    #10: a physics verdict is sim-config-RELATIVE). A recalled 'walks 0.65 m' then carries the timestep, gravity
    and engine it's valid under, so a consumer on a different sim sees the mismatch instead of trusting a stale
    number. Best-effort: derived from the SAME compiler the deploy uses; falls back to the deploy descriptor."""
    cfg = {"controller": "crawl_gait", "engine": "mujoco"}
    try:
        import mujoco

        from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
        m = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene, spawn_z=standing_spawn_z(gene)))
        cfg["timestep"] = round(float(m.opt.timestep), 6)
        cfg["gravity_z"] = round(float(m.opt.gravity[2]), 4)
        cfg["mujoco"] = getattr(mujoco, "__version__", "?")
    except Exception:  # noqa: BLE001 - the descriptor alone still identifies the deploy controller/engine
        cfg["mujoco"] = "unknown"
    return cfg


def bank_gait(db, gene, result, *, task: str = LOCOMOTION) -> str | None:
    """Bank a learned gait's PARAMS as a retrievable skill (keyed by class+task+species). Returns the skill_id.

    Only banks a DEPLOYABLE result (survived + real forward) — never a fall (honesty: the bank must stay a bank of
    working controllers). ``record_skill`` is idempotent-keep-best, so a weaker later gait can't clobber a stronger one.

    UN-GAMEABLE: also rejects a non-CREDIBLE result — a body that SLIDES forward (feet never leave the ground) or
    rears/lurches survives and covers distance but is NOT a walk; banking it would let a slide masquerade as a gait
    and be recalled/reused. (A result double without ``best_credible`` is assumed credible for back-compat.)
    """
    if not getattr(result, "best_survived", False) or abs(getattr(result, "best_forward", 0.0)) < 0.15:
        return None
    if not getattr(result, "best_credible", True):
        return None
    cls = _class_of(gene)
    species = getattr(gene, "species", None) or cls
    gene_id = getattr(gene, "id", None) or cls
    skill_id = f"gait::{cls}::{gene_id}"[:96]
    success = min(1.0, abs(result.best_forward) / _FWD_NORM)
    db.record_skill(
        skill_id, cls, task, success_rate=success, species=species, gene_id=gene_id,
        base_config={"gait_params": result.best_params, "forward_m": round(result.best_forward, 4),
                     "height_ratio": round(result.best_height_ratio, 3), "controller": "crawl_gait",
                     "sim_config": _deploy_sim_config(gene)},
        notes="learned deployable gait (gait_search); base_config.gait_params IS the deploy controller")
    # Index the skill into the vector memory by THIS BODY'S morphology embedding (the robotics tokenization), so a
    # future body recalls it by structural similarity — cross-body, not just an exact class-string match.
    try:
        from virturoid.services.robotics_vector_memory import (SKILL, RoboticsVectorMemory, _body_latent,
                                                                embed_skill)
        vm = RoboticsVectorMemory(db)
        vec = embed_skill(f"{task} {cls} {species}", gene, success_rate=success, latent=_body_latent(gene))
        vm.upsert(SKILL, skill_id, vec, {"robot_class": cls, "task_type": task, "species": species})
    except Exception:  # noqa: BLE001 - vector indexing is a retrieval accelerant; the DB bank is the source of truth
        pass
    return skill_id


def _gait_params_for_skill(db, skill_id: str) -> dict | None:
    row = db.conn.execute("SELECT base_config FROM skills WHERE skill_id=?", (skill_id,)).fetchone()
    if not row or not row["base_config"]:
        return None
    try:
        bc = json.loads(row["base_config"]) if isinstance(row["base_config"], str) else row["base_config"]
    except json.JSONDecodeError:
        return None
    return bc.get("gait_params") if isinstance(bc, dict) and bc.get("controller") == "crawl_gait" else None


def recall_gait(db, gene, *, task: str = LOCOMOTION) -> dict | None:
    """Recall the best banked gait PARAMS for the closest prior body. Embedding (morphology) retrieval FIRST — the
    robotics tokenization — so a new body finds the gait of the most structurally-similar banked body even across
    class labels; falls back to the exact (class, task, species) skill match. None if nothing is banked."""
    # 1) TOKENIZED retrieval: nearest banked gait skill by body-morphology embedding (z_body ⊕ z_task).
    try:
        from virturoid.services.robotics_vector_memory import RoboticsVectorMemory
        for hit in RoboticsVectorMemory(db).nearest_skills(
            gene, task, k=6, min_sim=_MIN_GAIT_TRANSFER_SIMILARITY,
        ):
            params = _gait_params_for_skill(db, hit.get("obj_id", ""))
            if params:
                return params
    except Exception:  # noqa: BLE001 - fall back to the exact string match
        pass
    # 2) exact (class, task, species) — the string key (prefers nearest species).
    cls = _class_of(gene)
    species = getattr(gene, "species", None) or cls
    sk = db.recall_skill(cls, task, species=species)
    # The SQL fallback has only class/task + species-token ranking, not a
    # morphology distance.  Without the vector screen, only reuse an exact
    # species match; a different body can enter through the scored vector path.
    if sk and (sk.get("species") or cls) == species:
        params = _gait_params_for_skill(db, sk.get("skill_id", ""))
        if params:
            return params
    return None


def _recall_gait_source(db, gene, *, task: str = LOCOMOTION) -> tuple[dict, str] | None:
    """Recall gait params with the exact skill identifier that supplied them.

    ``recall_gait`` predates the provenance ledger and intentionally exposes only controller parameters to its
    deploy callers.  The learning path needs the concrete parent skill as well, otherwise it cannot make an
    auditable warm-start edge.
    """
    try:
        from virturoid.services.robotics_vector_memory import RoboticsVectorMemory
        for hit in RoboticsVectorMemory(db).nearest_skills(
            gene, task, k=6, min_sim=_MIN_GAIT_TRANSFER_SIMILARITY,
        ):
            skill_id = str(hit.get("obj_id", ""))
            params = _gait_params_for_skill(db, skill_id)
            if params:
                return params, skill_id
    except Exception:  # noqa: BLE001 - exact lookup remains a valid fallback
        pass
    cls = _class_of(gene)
    species = getattr(gene, "species", None) or cls
    skill = db.recall_skill(cls, task, species=species)
    if skill and (skill.get("species") or cls) == species:
        skill_id = str(skill.get("skill_id", ""))
        params = _gait_params_for_skill(db, skill_id)
        if params:
            return params, skill_id
    return None


_DEFAULT_GAIT = {"freq": 1.5, "hip_amp": 0.9, "knee_amp": 1.0, "duty": 0.25, "kp": 32.0, "kd": 1.5}


class _DeployResult:
    """A result-shaped holder carrying DEPLOY-horizon metrics for banking."""
    def __init__(self, params, r):
        self.best_params = params
        self.best_forward = float(r["forward"])
        self.best_height_ratio = float(r["height_ratio"])
        self.best_survived = bool(r["survived"])
        self.best_credible = bool(r.get("credible", False))


def learn_gait_flywheel(gene, db, *, generations: int = 10, pop: int = 20, steps: int = 900,
                        deploy_steps: int = 1500, seed: int = 0, workers: int = 1, bank: bool = True,
                        vm=None) -> dict:
    """Recall a specific prior -> SCREENED warm-start search -> DEPLOY-SELECT vs the default -> bank -> provenance.

    Deploy-select (honesty): the search optimizes at ``steps``, but the winner is re-measured at the longer
    ``deploy_steps`` horizon AND compared to the SHIPPED default gait there — the learned gait is banked ONLY if
    it beats the default at deploy (never trust the search horizon; the bank must always deploy BETTER than default).
    """
    from virturoid.services.gait_search import evaluate_gait, search_gait

    recalled = _recall_gait_source(db, gene)
    prior, prior_skill_id = recalled if recalled is not None else (None, None)
    res = search_gait(gene, generations=generations, pop=pop, steps=steps, seed=seed,
                      workers=workers, warm_start=prior)
    # DEPLOY-SELECT at the deploy horizon: learned winner vs the shipped default. Bank ONLY a CREDIBLE walk that
    # beats the default — a slide (fast but no real stepping) must never enter the bank.
    learned = evaluate_gait(gene, res.best_params, steps=deploy_steps)
    default = evaluate_gait(gene, _DEFAULT_GAIT, steps=deploy_steps)
    # CREDIBILITY-FIRST (un-gameable): the learned gait must itself be a credible, surviving walk. It beats the
    # default when the default is NOT a credible walk (a slide that merely covers distance must never block a real
    # walk from banking) or — when the default IS a credible walk — when it travels meaningfully further. This makes
    # the decision robust to the deploy horizon: a non-credible slide can never win by accumulating raw distance.
    learned_ok = bool(learned["survived"]) and bool(learned.get("credible", False))
    default_credible = bool(default["survived"]) and bool(default.get("credible", False))
    beats_default = learned_ok and (not default_credible
                                    or abs(learned["forward"]) > abs(default["forward"]) + 0.02)

    skill_id = None
    if bank and beats_default:
        skill_id = bank_gait(db, gene, _DeployResult(res.best_params, learned))   # bank the DEPLOY metrics

    compounding_delta = None
    prior_deploy_forward = None
    if prior is not None and res.prior_transfer_forward is not None:
        # Record a like-for-like deploy metric. The search-horizon warm-start score is useful diagnostics, but it
        # cannot be subtracted from a deploy-horizon controller result and called a compounding improvement.
        prior_deploy = evaluate_gait(gene, prior, steps=deploy_steps)
        prior_deploy_forward = float(prior_deploy["forward"])
        compounding_delta = round(float(learned["forward"]) - prior_deploy_forward, 4)
        if skill_id is not None:
            try:
                if vm is None:
                    from virturoid.services.robotics_vector_memory import RoboticsVectorMemory
                    vm = RoboticsVectorMemory(db)
                vm.record_provenance("skill", skill_id, parent_type="skill", parent_id=prior_skill_id,
                                     kind="gait_warm_start", delta=compounding_delta,
                                     meta={"horizon_steps": deploy_steps,
                                           "prior_search_transfer_forward": res.prior_transfer_forward,
                                           "prior_deploy_forward": prior_deploy_forward,
                                           "deploy_forward": learned["forward"]})
            except Exception:  # noqa: BLE001 - provenance is best-effort; never fail the learn
                pass

    return {
        "forward_m": round(learned["forward"], 4),           # the DEPLOY-horizon distance (honest)
        "search_forward_m": round(res.best_forward, 4),
        "default_forward_m": round(default["forward"], 4),
        "beats_default": bool(beats_default),
        "height_ratio": round(learned["height_ratio"], 3), "survived": bool(learned["survived"]),
        "reused_prior": prior is not None,
        "prior_transfer_forward": (round(res.prior_transfer_forward, 4)
                                   if res.prior_transfer_forward is not None else None),
        "prior_deploy_forward_m": (round(prior_deploy_forward, 4)
                                    if prior_deploy_forward is not None else None),
        "banked_skill": skill_id, "compounding_delta": compounding_delta, "params": res.best_params,
    }
