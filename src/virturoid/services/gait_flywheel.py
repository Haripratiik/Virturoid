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

import hashlib
import json
import logging

_LOG = logging.getLogger(__name__)

LOCOMOTION = "locomotion"
_FWD_NORM = 1.5   # metres that maps to success_rate 1.0 (a strong learned quad walk ~1.85 m)

#: THE IDENTITY OF THE GATE A BANKED ROW WAS ADMITTED UNDER, stamped into ``base_config['bank_gate']``.
#:
#: The fragility gate (``learn_gait_flywheel``: bank only an operating point whose perturbed copies also walk)
#: landed 2026-08-04. EVERY row banked before it, and every row banked through a call site that does not measure
#: a margin, is a row about which the fragility question was never asked. Those are not interchangeable with
#: gated rows and must never be pooled with them silently — that is precisely how ``duty``, a coordinate the
#: controller never read, came to be reported to operators as the bank's tightest-clustered parameter (#265/#266).
#: A row with no stamp is PRE-GATE by construction; ``gate_of`` says so in one word, and ``mine_gait_hints``
#: reports the split on every call so a thin gated corpus can never hide inside a fat ungated one.
BANK_GATE = "fragility_v1"

#: MEASURED AND FAILED. The fragility question WAS asked of this operating point and the answer was "no perturbed
#: copy of it walks". That is strictly more information than never asking, and it is not the same information, so
#: it gets its own word: pooling it with ``BANK_GATE`` would put a known-fragile point into the mined evidence,
#: and pooling it with ``ungated`` would throw away a measurement we paid rollouts for. Only a caller that has a
#: reason to keep a fragile precedent (``r2prime.seed_corpus`` measures the corpus mechanism; ``reward_loop``
#: carries the reward expression that certified it) ever writes one.
BANK_GATE_FRAGILE = "fragility_v1_fragile"

#: NOT MEASURED, AND THE CALLER SAYS SO. Distinct from a missing stamp, which means the row predates the gate
#: entirely. This one means: code that knows about the gate ran, had no margin in hand, and declined to buy one —
#: with the reason recorded in ``base_config['bank_gate_reason']``. The ordinary verify path is the only door that
#: writes it, because measuring there would put 4-12 settling-horizon rollouts on every single build.
BANK_UNGATED = "ungated_declared"

#: How much is KNOWN about a row's operating point, most-known first. Used only to refuse a DOWNGRADE: see the
#: keep-best note in ``bank_gait``. A missing stamp ranks lowest because it carries no measurement at all.
_GATE_RANK = {BANK_GATE: 3, BANK_GATE_FRAGILE: 2, BANK_UNGATED: 1, "ungated": 0}
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


def structural_gait_key(gene) -> str:
    """Name-independent hash of the physical kinematic tree a gait controls.

    IDs, species labels and prompts are excluded: equivalent bodies share a cache, while different dimensions,
    masses, joint limits, mounts, actuators or topology cannot collide in keep-best. Symmetric child branches are
    sorted by recursive content, so LLM-chosen segment names/order do not create false structural differences.
    """
    def q(value):
        return None if value is None else round(float(value), 6)

    def local(seg) -> dict:
        return {
            "shape": seg.shape, "length_m": q(seg.length_m), "radius_m": q(seg.radius_m),
            "mass_kg": q(seg.mass_kg), "joint_type": seg.joint_type or "fixed",
            "joint_axis": [q(v) for v in seg.joint_axis], "joint_lower": q(seg.joint_lower),
            "joint_upper": q(seg.joint_upper), "mount_euler": [q(v) for v in seg.mount_euler],
            "mount_offset": [q(v) for v in seg.mount_offset],
            "actuator_torque_nm": q(seg.actuator_torque_nm), "torque_req_nm": q(seg.torque_req_nm),
            "cross_section": ([q(v) for v in seg.cross_section] if seg.cross_section else None),
            "is_end_effector": bool(seg.is_end_effector),
        }

    def node(seg) -> dict:
        kids = [node(child) for child in gene.children_of(seg.name)]
        kids.sort(key=lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")))
        return {"link": local(seg), "children": kids}

    root = gene.root() if hasattr(gene, "root") else None
    payload = {"schema": "gait-structure-v1", "robot_class": _class_of(gene),
               "base_mount": getattr(gene, "base_mount", None),
               "end_effector_type": getattr(gene, "end_effector_type", None),
               "tree": node(root) if root is not None else []}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


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


def gate_of(base_config: dict | None) -> str:
    """WHAT IS KNOWN about this banked row's operating point, in one word.

    ``'fragility_v1'``          measured, and every perturbed copy still walked.
    ``'fragility_v1_fragile'``  measured, and it failed — a known-fragile point, kept deliberately.
    ``'ungated_declared'``      not measured; the caller had no margin and said so (reason on the row).
    ``'ungated'``               no stamp at all — the row predates the gate.

    An absent stamp is ungated, never "assumed fine": a missing error bar defaulting to "fine" is the failure
    mode the whole robustness measurement exists to remove. Consumers that pool must do so explicitly —
    ``mine_gait_hints(gated_only=True)`` keeps ONLY ``'fragility_v1'``, which is the point of the other words.
    """
    return str((base_config or {}).get("bank_gate") or "ungated")


#: Keys that describe WHERE A ROW CAME FROM rather than what it measured. They are established by a separate
#: audit pass (``scripts/bank_provenance.py``), not by the write that banks the gait, so a bank must carry them
#: forward instead of overwriting them.
_PROVENANCE_KEYS = ("row_source", "row_source_stamp", "row_source_evidence", "row_source_stamped_at")


def _banked_base_config(db, skill_id: str) -> dict | None:
    """The ``base_config`` already on ``skill_id``, or ``None`` when nothing is banked under it yet."""
    try:
        row = db.conn.execute("SELECT base_config FROM skills WHERE skill_id=?", (skill_id,)).fetchone()
    except Exception:  # noqa: BLE001 - a store without a queryable handle just has no prior row to protect
        return None
    if not row or not row["base_config"]:
        return None
    try:
        bc = json.loads(row["base_config"]) if isinstance(row["base_config"], str) else row["base_config"]
    except json.JSONDecodeError:
        return None
    return bc if isinstance(bc, dict) else None


def _banked_gate(db, skill_id: str) -> str | None:
    """The gate stamp already on ``skill_id``, or ``None`` when nothing is banked under it yet."""
    return gate_of(_banked_base_config(db, skill_id))


def _carry_provenance(prior_bc: dict | None, fresh: dict) -> dict:
    """Carry a prior row's PROVENANCE stamp onto a fresh write.

    ``MemoryDB.record_skill`` REPLACES ``base_config``, so without this a perfectly legitimate re-bank silently
    erases the quarantine stamp -- and an absent ``row_source`` reads as ``unattributed``, which is exactly the
    bucket a suite-authored row would launder itself into. OBSERVED live 2026-08-07: a row stamped by the audit
    pass was re-banked minutes later and came back unstamped.

    The gate guard above protects the row from a write that knows LESS about robustness. It cannot protect this,
    because a re-bank that knows strictly MORE is allowed through and still carries no provenance -- provenance
    is not something a banking call can know. So it is copied rather than defended.
    """
    if not prior_bc:
        return fresh
    for key in _PROVENANCE_KEYS:
        if fresh.get(key) is None and prior_bc.get(key) is not None:
            fresh[key] = prior_bc[key]
    return fresh


def bank_gait(db, gene, result, *, task: str = LOCOMOTION, cross_eval: bool = False,
              reward_expr: str = "", robustness: dict | None = None, ungated_reason: str | None = None,
              door: str | None = None) -> str | None:
    """Bank a learned gait's PARAMS as a retrievable skill (keyed by class+task+species). Returns the skill_id.

    Only banks a DEPLOYABLE result (survived + real forward) — never a fall (honesty: the bank must stay a bank of
    working controllers). ``record_skill`` is idempotent-keep-best, so a weaker later gait can't clobber a stronger one.

    UN-GAMEABLE: also rejects a non-CREDIBLE result — a body that SLIDES forward (feet never leave the ground) or
    rears/lurches survives and covers distance but is NOT a walk; banking it would let a slide masquerade as a gait
    and be recalled/reused. (A result double without ``best_credible`` is assumed credible for back-compat.)

    ``reward_expr`` (R6): the LLM/template reward EXPRESSION that certified this gait, banked alongside the params
    so a future morphology-nearby body can seed its reward search with a reward already proven on a body like it.

    ``robustness``: the ``robustness_margin`` reading the admitting caller measured for these exact params. IT IS
    THE ROW'S PROVENANCE, not decoration — a margin that PASSED stamps ``BANK_GATE``, one that FAILED stamps
    ``BANK_GATE_FRAGILE``, and a caller with no margin at all must say so via ``ungated_reason`` (``BANK_UNGATED``).
    This function deliberately does NOT measure the margin itself: at the settling horizon that is 4-12 extra
    rollouts, and ``bank_gait`` is called from the ordinary verify path (``ai_native_tools._auto_bank_gait``)
    where that cost would land on every build. So the gate is ENFORCED by the caller and RECORDED here.

    ``door`` names the call site, stamped as ``bank_door``. The census "what fraction of rows carry a real
    measured margin" is only answerable per-door if the row remembers which door it came through; with four
    doors and different budgets behind each, the pooled fraction alone tells you nothing about where to look.

    A WRITE MAY NEVER DOWNGRADE WHAT IS KNOWN. ``record_skill`` is keep-best on ``success_rate`` with ``>=``, so
    an unmeasured re-bank of a body whose row was already gated used to overwrite the stamp and the margin with
    nulls — and since an absent stamp reads as "predates the gate", the erasure was indistinguishable from
    history. Two rows both travelling >= _FWD_NORM both score 1.0, so ``>=`` makes that collision ordinary rather
    than exotic. A write whose gate knows strictly LESS than the banked row's is therefore declined (the skill_id
    is still returned: the bank does hold this body's gait, it just did not take this weaker provenance).
    """
    # SIGNED, like ``gait_quality.classify``'s own ``fwd >= 0.3`` and unlike what stood here. ``forward`` is
    # world-frame delta-x (+x forward), so ``abs`` admitted a body that walked 0.5 m BACKWARD and then scored it
    # ``success_rate 1.0`` below — into the corpus, where the next morphologically-similar body recalls it as a
    # warm start. The ``best_credible`` line under this one covers every caller that HAS a verdict, but it
    # defaults to True for back-compat, so a result double built without one (``corpus_factory`` and ``r2prime``
    # both build theirs by hand) was gated by this line alone. Sign-blind here is the same defect the deploy
    # comparison in ``learn_gait_flywheel`` carried, one write further down the pipe and harder to see, because
    # a corpus does not tell you which of its rows walked the wrong way.
    if not getattr(result, "best_survived", False) or float(getattr(result, "best_forward", 0.0)) < 0.15:
        return None
    if not getattr(result, "best_credible", True):
        return None
    cls = _class_of(gene)
    species = getattr(gene, "species", None) or cls
    gene_id = getattr(gene, "id", None) or cls
    structure_key = structural_gait_key(gene)
    skill_id = f"gait::{cls}::{structure_key}"[:96]
    success = min(1.0, max(0.0, float(result.best_forward) / _FWD_NORM))
    # WHICH GATE THIS ROW GOT IN UNDER. Three distinguishable answers, because "measured and fragile" and "never
    # measured" are different facts and only one of them cost rollouts to learn.
    if robustness:
        gate = BANK_GATE if (robustness or {}).get("robustness_rel") is not None else BANK_GATE_FRAGILE
    else:
        gate = BANK_UNGATED if ungated_reason else None
    prior_bc = _banked_base_config(db, skill_id)
    prior = gate_of(prior_bc)
    if prior is not None and _GATE_RANK.get(gate or "ungated", 0) < _GATE_RANK.get(prior, 0):
        _LOG.debug("bank_gait: kept the %s row for %s; this write only knows %s", prior, skill_id, gate)
        return skill_id                       # never trade a measured row for an unmeasured one
    db.record_skill(
        skill_id, cls, task, success_rate=success, species=species, gene_id=gene_id,
        base_config=_carry_provenance(prior_bc, {"gait_params": result.best_params,
                     "forward_m": round(result.best_forward, 4),
                     "height_ratio": round(result.best_height_ratio, 3), "controller": "crawl_gait",
                     "reward_expr": reward_expr or None, "structure_key": structure_key,
                     "sim_config": _deploy_sim_config(gene),
                     # ...and the error bar that let it in, plus WHICH DOOR it came through. A reader (or a
                     # miner) can then tell a measured row from an unmeasured one without guessing from a
                     # timestamp, and can tell which call site is producing the unmeasured ones.
                     "bank_gate": gate,
                     "bank_gate_reason": (ungated_reason if gate == BANK_UNGATED else None),
                     "bank_door": door,
                     "robustness_rel": (robustness or {}).get("robustness_rel"),
                     "robustness_probes": (robustness or {}).get("probes") or None,
                     "robustness_steps": (robustness or {}).get("steps")}),
        notes="learned deployable gait (gait_search); base_config.gait_params IS the deploy controller")
    # Index the skill into the vector memory by THIS BODY'S morphology embedding (the robotics tokenization), so a
    # future body recalls it by structural similarity — cross-body, not just an exact class-string match.
    try:
        from virturoid.services.robotics_vector_memory import (SKILL, RoboticsVectorMemory, _body_latent,
                                                                embed_skill)
        vm = RoboticsVectorMemory(db)
        vec = embed_skill(f"{task} {cls} {species}", gene, success_rate=success, latent=_body_latent(gene))
        # the gene rides INLINE in the vector meta so transfer cross-eval can recover the neighbour's BODY
        # (not just its params) without depending on the species vault being populated
        vm.upsert(SKILL, skill_id, vec, {"robot_class": cls, "task_type": task, "species": species,
                                         "structure_key": structure_key, "gene": gene.to_dict()})
    except Exception:  # noqa: BLE001 - vector indexing is a retrieval accelerant; the DB bank is the source of truth
        pass
    # TRANSFER LEDGER (opt-in; batch contexts like night-shift): replay this gait on the K nearest banked bodies
    # and theirs on this one — every verified outcome densifies the ground truth the gated metric learns from,
    # so the embedding upgrades itself with usage. Bounded (2·K short rollouts); never blocks the hot build path.
    if cross_eval:
        from virturoid.services.transfer_ledger import cross_evaluate_on_bank
        cross_evaluate_on_bank(db, gene, result.best_params, skill_id=skill_id)
    return skill_id


def gate_census(db, *, task: str = LOCOMOTION) -> dict:
    """The bank's provenance split — rows per gate, rows per door, and the gated fraction of each door.

    THE QUESTION THIS ANSWERS is "what fraction of what we hold has a real measured error bar, and which call
    site is producing the rest". A single pooled fraction cannot answer the second half, and the second half is
    the actionable one: three of the four doors used to pass no margin, and until the row remembered its door
    there was no way to see that from the bank itself. ``door=None`` counts rows banked before doors were named.
    """
    import collections
    rows = db.conn.execute("SELECT base_config FROM skills WHERE task_type=?", (task,)).fetchall()
    gates: collections.Counter = collections.Counter()
    per_door: dict[str, collections.Counter] = {}
    for r in rows:
        try:
            bc = json.loads(r["base_config"]) if isinstance(r["base_config"], str) else (r["base_config"] or {})
        except json.JSONDecodeError:
            bc = {}
        bc = bc if isinstance(bc, dict) else {}
        g = gate_of(bc)
        gates[g] += 1
        per_door.setdefault(str(bc.get("bank_door") or "unnamed"), collections.Counter())[g] += 1
    n = sum(gates.values())
    return {"rows": n, "by_gate": dict(gates),
            "gated_fraction": round(gates.get(BANK_GATE, 0) / n, 3) if n else 0.0,
            "by_door": {d: {"rows": sum(c.values()), "by_gate": dict(c),
                            "gated_fraction": round(c.get(BANK_GATE, 0) / sum(c.values()), 3)}
                        for d, c in sorted(per_door.items())}}


def _gait_params_for_skill(db, skill_id: str) -> dict | None:
    row = db.conn.execute("SELECT base_config FROM skills WHERE skill_id=?", (skill_id,)).fetchone()
    if not row or not row["base_config"]:
        return None
    try:
        bc = json.loads(row["base_config"]) if isinstance(row["base_config"], str) else row["base_config"]
    except json.JSONDecodeError:
        return None
    return bc.get("gait_params") if isinstance(bc, dict) and bc.get("controller") == "crawl_gait" else None


def _leg_count(gene) -> int:
    """Count WEIGHT-BEARING legs (not all limb chains): a root-child chain with >=2 revolute joints terminating
    in a welded foot pad. This distinguishes legs from tails/necks/antennae — measured necessary: a 'dog' has 6
    chains (4 legs + tail + neck) but 4 LEGS, so counting all chains falsely rejected a quad↔quad recall while
    the leg count (4 vs a hexapod's 6) is exactly the axis the 29-D vector blurs (§3.I2)."""
    root = gene.root() if hasattr(gene, "root") else None
    if root is None:
        return 0
    legs = 0
    for child in gene.children_of(root.name):
        chain = [child]
        node = child
        while gene.children_of(node.name):
            node = gene.children_of(node.name)[0]
            chain.append(node)
        n_rev = sum(1 for s in chain if s.joint_type == "revolute")
        if n_rev >= 2 and chain[-1].joint_type in (None, "fixed"):
            legs += 1
    return legs


def _structural_recall(db, gene, task: str) -> tuple[dict, str, str] | None:
    """Embedding retrieval + a HARD STRUCTURAL PRE-FILTER (flywheel_breakthrough_plan §3.I2, measured): at cosine
    0.98 the 29-D vector blurs leg-count and stance, so a hexapod recalled a quadruped's gait (0.9939) over the
    real hexapod (0.9291). Two corrections, highest priority first:
      1. EXACT structural cache — a banked gait for the SAME structural signature (body_key) is a deterministic
         repeat-build reuse (the floor baseline the plan must beat); return it verbatim.
      2. LEG-COUNT equality — a candidate whose banked body has a different limb-chain count than the query is
         rejected (a 4-leg gait must never seed a 6-leg body), before any cosine ranking.
    Returns (params, skill_id, match_kind ∈ {exact_cache, leg_count, fallback}) or None. Candidates carry their
    body INLINE in the skill vector meta; a candidate with no inline gene is only a last-resort fallback.
    """
    from virturoid.schemas.gene import RobotGene
    q_legs, q_key = _leg_count(gene), structural_gait_key(gene)
    try:
        from virturoid.services.robotics_vector_memory import RoboticsVectorMemory
        hits = RoboticsVectorMemory(db).nearest_skills(gene, task, k=8, min_sim=_MIN_GAIT_TRANSFER_SIMILARITY)
    except Exception:  # noqa: BLE001
        hits = []
    leg_match: tuple | None = None
    no_gene: tuple | None = None
    for hit in hits:
        sid = str(hit.get("obj_id", ""))
        params = _gait_params_for_skill(db, sid)
        if not params:
            continue
        meta = hit.get("meta") or {}
        bg = None
        if isinstance(meta.get("gene"), dict):
            try:
                bg = RobotGene.from_dict(meta["gene"])
            except Exception:  # noqa: BLE001
                bg = None
        if bg is not None and structural_gait_key(bg) == q_key:
            return params, sid, "exact_cache"                # deterministic same-body reuse wins outright
        if bg is None:
            no_gene = no_gene or (params, sid, "fallback")   # can't verify structure -> last resort
            continue
        if _leg_count(bg) == q_legs and leg_match is None:
            leg_match = (params, sid, "leg_count")           # first (highest-sim) leg-count-consistent match
    return leg_match or no_gene


def recall_gait(db, gene, *, task: str = LOCOMOTION) -> dict | None:
    """Recall the best banked gait PARAMS for the closest prior body — embedding retrieval with a hard structural
    pre-filter (exact-structure cache, then leg-count equality — §3.I2), falling back to the exact (class, task,
    species) skill match. None if nothing structurally-appropriate is banked."""
    hit = _structural_recall(db, gene, task)
    if hit is not None:
        return hit[0]
    # exact (class, task, species) string key (prefers nearest species) — unchanged fallback.
    cls = _class_of(gene)
    species = getattr(gene, "species", None) or cls
    sk = db.recall_skill(cls, task, species=species)
    if sk and (sk.get("species") or cls) == species:
        params = _gait_params_for_skill(db, sk.get("skill_id", ""))
        if params:
            return params
    return None


def recall_reward_hints(db, gene, *, task: str = LOCOMOTION, k: int = 6, min_sim: float = 0.2) -> list[dict]:
    """R6 — the moat, through the reward loop. Return the reward EXPRESSIONS that certified credible gaits on the
    morphology-nearest banked bodies, each with the neighbour's similarity + forward distance. A new robot's
    reward search seeds from these (a reward already proven on a body SHAPED like it) instead of cold-starting
    from generic templates. Empty when nothing similar is banked. Deduplicated, best (nearest × fastest) first."""
    try:
        from virturoid.services.robotics_vector_memory import RoboticsVectorMemory
        seen, out = set(), []
        for hit in RoboticsVectorMemory(db).nearest_skills(gene, task, k=k, min_sim=min_sim):
            row = db.conn.execute("SELECT base_config FROM skills WHERE skill_id=?",
                                  (str(hit.get("obj_id", "")),)).fetchone()
            if not row or not row["base_config"]:
                continue
            try:
                bc = json.loads(row["base_config"]) if isinstance(row["base_config"], str) else row["base_config"]
            except (json.JSONDecodeError, TypeError):
                continue
            expr = (bc or {}).get("reward_expr")
            if isinstance(expr, str) and expr.strip() and expr not in seen:
                seen.add(expr)
                out.append({"reward_expr": expr, "similarity": round(float(hit.get("similarity", 0.5)), 3),
                            "forward_m": float((bc or {}).get("forward_m", 0.0))})
        out.sort(key=lambda h: (h["similarity"], h["forward_m"]), reverse=True)
        return out
    except Exception:  # noqa: BLE001 - no vector index yet -> caller cold-starts from templates
        return []


def _recall_gait_source(db, gene, *, task: str = LOCOMOTION) -> tuple[dict, str] | None:
    """Recall gait params with the exact skill identifier that supplied them.

    ``recall_gait`` predates the provenance ledger and intentionally exposes only controller parameters to its
    deploy callers.  The learning path needs the concrete parent skill as well, otherwise it cannot make an
    auditable warm-start edge.

    ``db is None`` is a COLD START, not an error: the build path fits an op-point even when no corpus is
    reachable (or hints are switched off for a hermetic bench run), and a cold start simply has no prior.
    """
    if db is None:
        return None
    hit = _structural_recall(db, gene, task)                 # same hard structural pre-filter as recall_gait
    if hit is not None:
        return hit[0], hit[1]
    cls = _class_of(gene)
    species = getattr(gene, "species", None) or cls
    skill = db.recall_skill(cls, task, species=species)
    if skill and (skill.get("species") or cls) == species:
        skill_id = str(skill.get("skill_id", ""))
        params = _gait_params_for_skill(db, skill_id)
        if params:
            return params, skill_id
    return None


_DEFAULT_GAIT = {"freq": 1.5, "hip_amp": 0.9, "knee_amp": 1.0, "kp": 32.0, "kd": 1.5}

#: THE PERTURBATION THE SEARCH ITSELF OPTIMISES AGAINST, and how many probes each surviving candidate pays for.
#:
#: 1e-2 because that is the SIM-TO-REAL number: hardware tolerance on a PD gain and on a commanded step
#: frequency is ~1%, so an operating point that only walks inside a tighter band than that is not a controller
#: anybody can deploy — it is a property of one float in one simulator. Measured (task #267) at the point the
#: fitter used to adopt for the grounded authored cat: 0 of 8 perturbed copies survive at 1e-2.
#:
#: 2 probes, not 4, because this runs per CANDIDATE rather than once per fit, and it is SCREENED — only a
#: candidate that is already credible at its own parameters pays for probes at all (measured on the cat, 2 of
#: 97). A knife edge is a knife edge: it does not need four probes to be caught, and the adopted point is
#: re-measured afterwards on the full ``_ROBUST_LADDER`` at ``_ROBUST_N`` with HELD-OUT draws anyway.
#: *** DEFAULTED OFF: THE ROBUSTNESS-AWARE SEARCH WAS MEASURED AND IT DOES NOT WORK. *** Controlled A/B, 3 base
#: seeds x seed_restarts=3 per arm per body, max_evals=96, search_steps=deploy_steps=6000, arm OFF = this set to
#: None (documented at ``search_gait`` as restoring the pre-337f44a single-rollout search exactly):
#:     adoptions          6/15 -> 8/15   (it adopts MORE, which reads like a win and is not one)
#:     STURDY adoptions   4/6 = 67%  ->  5/8 = 63%   -- the sturdy FRACTION got WORSE
#:     wall clock         +24% overall, +62% on the dog
#:     hexapod            winners BYTE-IDENTICAL across both arms on all 3 seeds -- zero effect
#: Both extra adoptions are fragile, and the dog's seed-202 winner was adopted with its own in-search screen
#: reading ``search_robust_frac: "0/2"``. A search that reports zero-of-two robust and adopts anyway is not
#: selecting for robustness. The reported margin does not predict either: on cat/202 it moved 0.01 -> 0.001,
#: i.e. the disclosed number got WORSE while the point was still adopted.
#: The MACHINERY is kept (pass ``robust_rel=`` explicitly) because the measurement is worth reproducing and the
#: neighbourhood scoring is correct in itself -- what is refuted is that scoring a 2-probe neighbourhood at the
#: END of a CEM whose population already collapsed can find a robust basin the search never visited.
#: [[knife-edge-optima]]: the fitter's point is strictly dominated by one that is 30% FASTER and 12x more
#: robust, and REACHABILITY is the binding constraint -- 96 evals yield 2 credible candidates of 97 at one seed
#: and zero at two others. A different SAMPLING strategy is the open lead, not a different ranking rule.
_SEARCH_ROBUST_REL = None
_SEARCH_ROBUST_N = 2

#: The perturbation draws the REPORTED margin is measured on. Deliberately unrelated to the ones ``search_gait``
#: scores candidates against (it derives those from the search seed), because a margin measured on the same
#: draws the winner was selected under is an error bar quoted on its own training set — it would read high
#: exactly when the search had overfitted the probe, which is the one case it exists to catch.
_MARGIN_SEED = 20260802


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
                        vm=None, max_evals: int | None = None, stop_on_credible: bool = False,
                        recall: bool = True, robust_rel: float | None = _SEARCH_ROBUST_REL,
                        robust_n: int = _SEARCH_ROBUST_N, deadline: float | None = None) -> dict:
    """Recall a specific prior -> SCREENED warm-start search -> DEPLOY-SELECT vs the default -> bank -> provenance.

    Deploy-select (honesty): the search optimizes at ``steps``, but the winner is re-measured at the longer
    ``deploy_steps`` horizon AND compared to the SHIPPED default gait there — the learned gait is banked ONLY if
    it beats the default at deploy (never trust the search horizon; the bank must always deploy BETTER than default)
    AND ONLY IF IT SURVIVES A PERTURBATION OF ITSELF (``robustness_margin``); see the note at the gate below for the
    measurement that made the second condition necessary. ``robustness_rel`` is reported either way, so a caller
    that adopts a fragile point does so knowingly rather than by omission.

    ``recall=False`` runs the same search with NO prior (a cold run) while still banking to ``db``. Callers use it
    to RE-RUN a body whose warm-started search failed — see ``fit_gait_for_body`` for the measurement that made
    that necessary.

    THE HORIZON IS PART OF THE ANSWER, so it is reported: ``horizon_steps`` is ``deploy_steps`` and ``settled``
    says whether that was long enough for ``gait_quality.settling`` to check the body was STILL WALKING at the
    end. A caller that passes a small ``deploy_steps`` for budget reasons (``adapt_gait`` uses 600) gets
    ``settled=False`` and must not present the result as a walk claim — measured, the grounded authored cat is
    "credible" at 1500 and falls at step 2014 (task #267). ``fit_gait_for_body``, which DOES make the claim,
    passes the settling horizon for both ``steps`` and ``deploy_steps``.
    """
    from virturoid.services.gait_search import evaluate_gait, search_gait

    recalled = _recall_gait_source(db, gene) if recall else None
    prior, prior_skill_id = recalled if recalled is not None else (None, None)
    # ROBUSTNESS-AWARE BY DEFAULT (task #267). The search is asked for a controller, not for the single highest
    # rollout it can find: every candidate that walks at its own parameters is re-scored across a neighbourhood
    # of size ``robust_rel``, ranked robust-first, and the credible-early-stop only fires for a candidate whose
    # perturbed copies walk too. ``robust_rel=None`` restores the old single-rollout search exactly, which is
    # what the before/after ablation for this task was measured with.
    res = search_gait(gene, generations=generations, pop=pop, steps=steps, seed=seed,
                      workers=workers, warm_start=prior, max_evals=max_evals,
                      stop_on_credible=stop_on_credible, robust_rel=robust_rel, robust_n=robust_n,
                      deadline=deadline)
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
    # SIGNED, never ``abs``. ``forward`` is world-frame delta-x and it is signed at every producer (+x forward,
    # -x backward, ``gait_search.reward_features_from_rollout``), so an unsigned comparison ranks a gait that
    # walks BACKWARD faster ABOVE one that walks forward. It was latent HERE — this branch is only reached when
    # both arms are credible, and ``gait_quality.classify`` requires ``forward >= 0.3``, so both were positive by
    # construction — but latent is not the same as harmless: the identical unsigned rule at
    # ``ai_native_tools._honest_gait`` (the verify deploy-select) is NOT gated on credibility and does compare
    # two backward numbers. Writing it signed here removes the pattern that gets copied.
    beats_default = learned_ok and (not default_credible
                                    or float(learned["forward"]) > float(default["forward"]) + 0.02)

    # ...AND THE HORIZON THE CLAIM IS MADE AT HAS TO BE ONE A WALK CAN BE JUDGED AT.
    #
    # ``credible`` above is ``gait_quality.classify`` at ``deploy_steps``, and below ``_SETTLE_MIN_STEPS`` the
    # settling gate is OFF — so at a short horizon "credible" means "had gone somewhere by the time we stopped
    # looking", not "walks". Callers judge at wildly different horizons (``fit_gait_for_body`` 6000,
    # ``input_training_tools._learn_gait`` 1500, ``train_held`` 600) and every one of them prints the same word.
    #
    # MEASURED 2026-08-10 on the real Menagerie Go2, through ``train_held``'s own 600-step deploy horizon:
    #
    #     searched winner   600 steps   forward +0.233 m   CROUCH (low/unstable stance)
    #     searched winner  6000 steps   forward -0.279 m   FELL by ROLL-OVER (roll 139 / pitch 82 deg)
    #
    # The sign of the headline number FLIPS between the two horizons. And it has already cost a customer's robot
    # once: ``trained_controller``'s module docstring records an authored horse that trained "solved,
    # beats_default (0.868 vs 0.471 at the search's 600-step deploy horizon)", landed, and then verified 3.305 m
    # CREDIBLE WALK -> 2.107 m FELL by YAW-DRIFT. That is this exact gap, and undo was the only thing that saved it.
    #
    # So a winner that reads credible at a horizon too short to say gets ONE confirming rollout at the settling
    # horizon. It is paid for only on the success path (a non-credible winner is already refused and skips it),
    # it is free for callers that already judge at or above ``_SETTLE_MIN_STEPS``, and on a body that falls the
    # rollout terminates at the fall — measured on the Go2, the 6000-step confirmation costs 0.77 s, LESS than
    # the 600-step rollout it is checking. What it is NOT is a longer search: ``fit_gait_for_body``'s docstring
    # records the ablation that judging longer than you search "just adopts nothing", and that ruling stands for
    # the SEARCH. This is the last step before a claim reaches a customer's robot, where adopting nothing is the
    # correct answer if nothing walks.
    from virturoid.services.gait_quality import _SETTLE_MIN_STEPS
    settling_check = None
    if learned_ok and int(deploy_steps) < _SETTLE_MIN_STEPS:
        _s = evaluate_gait(gene, res.best_params, steps=_SETTLE_STEPS)
        settling_check = {"steps": int(_SETTLE_STEPS), "credible": bool(_s.get("credible", False)),
                          "survived": bool(_s["survived"]), "forward_m": round(float(_s["forward"]), 4),
                          "verdict": str(_s.get("verdict", "")),
                          "judged_at": int(deploy_steps)}
    # THE ONE BOOLEAN A DOOR SHOULD PRINT AS "credible", and the sentence that justifies it. ``credible`` is a
    # measurement at one horizon, ``survived`` is a much weaker fact, and ``beats_default`` is a BANK criterion
    # about other bodies — three different things that had all been printed under the same word by one door or
    # another. This is the claim: it walked, at a horizon a walk can be judged at.
    #
    # IT LEADS WITH THE ``classify()`` VERDICT, every branch, because of where this sentence ENDS UP. Callers
    # quote it inline — ``trained_controller`` renders "did not produce a credible walk (<this>)" and, for
    # ``apply='always'``, "landed an operating point whose own verdict was <this>". Written explanation-first it
    # produced, measured on the Go2: "whose own verdict was the winning operating point survived but is not a
    # walk at 600 steps: CROUCH". The verdict is in there, and it is the last thing the sentence says. Leading
    # with it costs nothing and makes the quote read as a verdict in both frames.
    credible_walk = learned_ok and (settling_check is None or settling_check["credible"])
    if credible_walk:
        _cw_reason = (f"CREDIBLE WALK at {(settling_check or {}).get('steps', deploy_steps)} steps "
                      f"({(settling_check or {}).get('forward_m', round(float(learned['forward']), 4))} m)")
    elif not bool(learned["survived"]):
        _cw_reason = (f"{learned.get('verdict')} — the winning operating point did not survive its own "
                      f"{deploy_steps}-step rollout")
    elif not learned_ok:
        _cw_reason = (f"{learned.get('verdict')} — the winning operating point survived its {deploy_steps}-step "
                      f"rollout but is not a walk ({round(float(learned['forward']), 4)} m)")
    else:
        _cw_reason = (f"{settling_check['verdict']} at the {_SETTLE_STEPS}-step horizon a walk is judged at "
                      f"({settling_check['forward_m']} m) — it read credible only at the {deploy_steps}-step "
                      f"horizon it was searched over ({round(float(learned['forward']), 4)} m)")

    # THE ERROR BAR IS PART OF THE DECISION, not a footnote printed beside it. A credible walk that no perturbed
    # copy of itself can repeat is one lucky float, and BANKING one is how the flywheel comes to serve a fall.
    # MEASURED 2026-08-02 (task #267) on the grounded authored cat: the op-point the fitter adopts
    # (freq 1.700 / hip 1.080 / knee 0.939 / kp 94.27 / kd 14.0) travels 3.817 m and holds its rate to step 6000
    # — and is 0/8 at BOTH rel 1e-2 and 1e-3, and falls at step 8754. It was ALREADY IN THE BANK, recalled as
    # this body's own prior, winning at evaluation 1 in 3 s. That is the settling defect one horizon out, with
    # the flywheel now doing the propagating: the corpus had memorised a fall and served it as a warm start.
    #
    # A robustness probe is the horizon-FREE version of "is it still walking later", and measured here it
    # predicts exactly that: the cat's 8/8-robust point (freq 2.709 / kp 70.93) is still walking at 12000 steps
    # with a flat rate, while the 0/8 point falls at 8754. Chasing the horizon upward cannot be the answer —
    # every horizon ends somewhere — so the bank is gated on the MARGIN, not only on the verdict.
    rob: dict = {"robustness_rel": None, "probes": {}}
    if beats_default and credible_walk:
        # HELD-OUT DRAWS. ``robustness_margin`` defaults to ``_MARGIN_SEED``, which is not the seed the search
        # scored candidates under — so a robustness-aware search that overfitted its own probe draws is caught
        # here rather than confirmed here.
        #
        # ``and credible_walk`` is a COST guard, not a second gate: a point already refuted at the settling
        # horizon cannot bank, and the ladder is up to 12 rollouts. Its consequence is that ``robustness_rel``
        # reads ``None`` for such a point, which means "not measured — we had already declined", and
        # ``not_banked_reason`` says which of the two it was.
        rob = robustness_margin(gene, res.best_params, steps=deploy_steps)
    sturdy = rob["robustness_rel"] is not None

    skill_id = None
    if bank and beats_default and sturdy and credible_walk:
        # ...and the margin RIDES WITH IT. A gate that leaves no trace on the row it admitted cannot be audited
        # afterwards, which is how a bank came to hold 97 rows nobody could sort into measured and unmeasured.
        skill_id = bank_gait(db, gene, _DeployResult(res.best_params, learned),   # bank the DEPLOY metrics
                             robustness=rob, door="learn_gait_flywheel")

    compounding_delta = None
    prior_deploy_forward = None
    if prior is not None and res.prior_transfer_forward is not None:
        # Record a like-for-like deploy metric. The search-horizon warm-start score is useful diagnostics, but it
        # cannot be subtracted from a deploy-horizon controller result and called a compounding improvement.
        prior_deploy = evaluate_gait(gene, prior, steps=deploy_steps)
        prior_deploy_forward = float(prior_deploy["forward"])
        compounding_delta = round(float(learned["forward"]) - prior_deploy_forward, 4)
        # Record the edge for EVERY measured reuse, not only the ones that won.
        #
        # This used to be gated on `skill_id is not None` — i.e. only when the warm-started run beat the
        # default well enough to bank a NEW skill. A reuse that measured worse recorded nothing, so the
        # ledger could only ever contain wins and `compounding_summary()['positive_fraction']` was
        # vacuously 1.0: compounding was UNFALSIFIABLE by construction. Measured 2026-07-21 on a quad whose
        # second (warm-started) run came in at delta -0.3 — a real, like-for-like measurement that the ledger
        # silently discarded.
        #
        # When a new skill banked, the edge is skill->skill as before. When nothing banked, the thing that was
        # seeded is the BUILD, so the child is the gene: "this body's gait run warm-started from skill X and
        # measured delta d". Either way the delta is real and the sign is preserved.
        try:
            if vm is None:
                from virturoid.services.robotics_vector_memory import RoboticsVectorMemory
                vm = RoboticsVectorMemory(db)
            _meta = {"horizon_steps": deploy_steps,
                     "prior_search_transfer_forward": res.prior_transfer_forward,
                     "prior_deploy_forward": prior_deploy_forward,
                     "deploy_forward": learned["forward"]}
            # NEITHER OF THESE IS A CONTROL ARM, and the ``_no_bank`` suffix has been read as one (2026-08-07:
            # "the CONTROL arm, bank withheld ... the control BEATS the treatment"). It means no NEW skill was
            # banked FROM this run. Both branches are the SAME code path with the SAME prior read from the bank,
            # split on the OUTCOME -- so ``gait_warm_start`` is the winners' bucket (+0.3753 m on the live bank)
            # and ``gait_warm_start_no_bank`` the losers' (+0.0020 m). Comparing them measures the split, not the
            # bank; the honest pooled figure is +0.0786 m over 594 edges. They are also not comparable with
            # ``gait_hint_deploy``, whose delta is (recall - default) on one body; this delta is (search result -
            # its own starting point), which has no default in it at all.
            _arm = {"is_control_arm": False,
                    "delta_means": "deploy_forward - prior_deploy_forward (the SEARCH's gain over its own seed)",
                    "bank_was_read": True}
            if skill_id is not None:
                vm.record_provenance("skill", skill_id, parent_type="skill", parent_id=prior_skill_id,
                                     kind="gait_warm_start", delta=compounding_delta, meta={**_meta, **_arm})
            else:
                vm.record_provenance("gene", getattr(gene, "id", "") or "gene",
                                     parent_type="skill", parent_id=prior_skill_id,
                                     kind="gait_warm_start_no_bank", delta=compounding_delta,
                                     meta={**_meta, **_arm, "banked": False,
                                           "why": "this run banked no new skill (it did not beat the shipped "
                                                  "default); the bank was still READ for the prior"})
        except Exception:  # noqa: BLE001 - provenance is best-effort; never fail the learn
            pass

    return {
        "forward_m": round(learned["forward"], 4),           # the DEPLOY-horizon distance (honest)
        "search_forward_m": round(res.best_forward, 4),
        "default_forward_m": round(default["forward"], 4),
        "beats_default": bool(beats_default),
        # FOUR DIFFERENT FACTS, FOUR DIFFERENT KEYS. They were being collapsed into one word by the doors that
        # print this dict, in both directions: ``train_held`` printed ``survived`` under the name ``credible``,
        # and passed ``survived and beats_default`` into a parameter whose contract is "the run's own un-gameable
        # verdict". Whoever reports one of these must be able to name which one.
        #   survived        the body was still up when the rollout stopped
        #   credible        classify() said CREDIBLE WALK -- AT ``horizon_steps``, which may be too short to say
        #   credible_walk   THE CLAIM: credible, and not refuted at the horizon a walk is judged at
        #   beats_default   a BANK criterion: credible AND better than the free shipped default (about OTHER bodies)
        "credible": bool(learned.get("credible", False)),
        "verdict": str(learned.get("verdict", "")),          # the classify() STRING, so a decline can diagnose
        "default_credible": bool(default_credible),
        "default_verdict": str(default.get("verdict", "")),
        "credible_walk": bool(credible_walk),
        "credible_walk_reason": _cw_reason,
        "settling_check": settling_check,                    # None = not needed (already judged long enough)
        "horizon_steps": int(deploy_steps),                  # how long anyone actually looked...
        "settled": learned.get("holds_rate") is not None,    # ...and whether that was long enough to judge
        # ...and how much of that horizon the winner's rollout ACTUALLY INTEGRATED. ``horizon_steps`` is the
        # number a decline sentence quotes; this is the number that says whether the quote was earned. ``None``
        # when the evaluator did not report it (a stubbed one in a unit test), which is "unknown", never "zero".
        "deploy_steps_integrated": learned.get("steps_integrated"),
        "rates": learned.get("rates"),
        # ...and how much of a perturbation the answer survives, which is the part a horizon can never tell you.
        "robustness_rel": rob["robustness_rel"],
        "robustness_probes": rob["probes"],
        "robustness_per_param": rob.get("per_param_rel"),
        "robustness_note": margin_sentence(rob) if (beats_default and credible_walk) else None,
        # The WHOLE margin measurement, carried intact. Rebuilding it field-by-field downstream is how
        # ``margin_sentence`` came to print "per parameter (n=None each)": the sample sizes were dropped on the
        # way out and the sentence lost the one number that makes its two halves comparable.
        "robustness": dict(rob),
        # What the search itself saw, at the perturbation it was scored under — reported separately from the
        # held-out margin above so the two can DISAGREE visibly instead of silently agreeing.
        "search_robust_frac": getattr(res, "best_robust_frac", None),
        "search_robust_rel": getattr(res, "robust_rel", None),
        # THE HONEST COST: every physics rollout this call ran — the CEM's, the two deploy-horizon comparisons,
        # the prior's deploy check, and the margin ladder. ``n_evals`` counts CANDIDATES and is now smaller than
        # the work done, so reporting it alone would understate robustness-aware search precisely where it costs.
        "n_rollouts": (int(getattr(res, "n_rollouts", 0)) + 2 + (1 if prior_deploy_forward is not None else 0)
                       + int(rob.get("n_rollouts") or 0) + (1 if settling_check is not None else 0)),
        "n_rollouts_margin": int(rob.get("n_rollouts") or 0),
        # WHICH gate refused, named. There are now two ways past ``beats_default`` to end with no row, and a
        # sentence that names the wrong one is the defect this whole change is about.
        "not_banked_reason": (
            None if skill_id or not (bank and beats_default) else
            (f"the winning operating point is credible at the {deploy_steps}-step horizon it was judged at and "
             f"NOT at the {_SETTLE_STEPS}-step horizon a walk is judged at ({settling_check['verdict']}, "
             f"{settling_check['forward_m']} m) — banking it is how a corpus comes to serve a fall as a warm start"
             if not credible_walk else
             f"the winning operating point does not survive a {min(_ROBUST_LADDER):g} relative "
             f"perturbation of its own parameters ({rob['probes']}) — a bank of controllers "
             f"must not hold one lucky float, which a later body would recall as a warm start")),
        "n_evals": int(getattr(res, "n_evals", 0)),
        "stopped_reason": getattr(res, "stopped_reason", "generation_limit"),
        "height_ratio": round(learned["height_ratio"], 3), "survived": bool(learned["survived"]),
        "reused_prior": prior is not None,
        "prior_transfer_forward": (round(res.prior_transfer_forward, 4)
                                   if res.prior_transfer_forward is not None else None),
        "prior_deploy_forward_m": (round(prior_deploy_forward, 4)
                                    if prior_deploy_forward is not None else None),
        "banked_skill": skill_id, "compounding_delta": compounding_delta, "params": res.best_params,
    }


# The op-point every legged body is judged at TODAY. It is one body's hand-tuned numbers (the canonical fanned
# template's), which is exactly why only that body walks under it — see docs/breaking_the_cotuning_wall.md §1.5.
#
# ``duty`` WAS a sixth entry here and was removed with the search dimension (2026-08-01, task #265):
# ``crawl_gait_rollout`` never read it, so every fitted/banked ``duty`` was an unconstrained coordinate whose
# CEM variance-collapse was then mined and shown to operators as the bank's tightest-clustered parameter. See
# the note at ``gait_search.PARAM_NAMES`` for the measurement and for what reviving it would require. Banked
# rows keep the key; nothing reads it, and ``recall_gait`` still returns those dicts verbatim.
_FIT_PARAMS = ("freq", "hip_amp", "knee_amp", "kp", "kd")

#: The horizon a walk is JUDGED at. 4x the old 1500-step deploy horizon, which was itself the number the whole
#: pipeline optimised against — and the grounded authored cat's adopted op-point falls at step 2014, i.e. 514
#: steps after the old verdict stopped looking (task #267). 4x is not a tuned constant; it is "materially
#: longer than the thing being claimed", and it is cross-checked at 12000 (8x) in the robustness ladder tests.
_SETTLE_STEPS = 6000

#: Relative perturbation sizes the adopted operating point is probed at, largest first, and how many joint
#: perturbations per rung. The reported margin is the LARGEST rung at which every probe still walks.
_ROBUST_LADDER = (1e-1, 1e-2, 1e-3)
_ROBUST_N = 4

#: (``_SEARCH_ROBUST_REL`` / ``_SEARCH_ROBUST_N`` / ``_MARGIN_SEED`` are defined ABOVE ``learn_gait_flywheel``
#: because they are default ARGUMENT values there, and Python binds those at definition time.)


# ------------------------------------------------------------------ making a TEST SUITE able to afford this
#
# THE COST IS REAL AND IT IS THE POINT. ``fit_gait_for_body`` is a bounded CEM search over 6000-step rollouts,
# and MEASURED on this checkout (2026-08-05, one CPU, grounded bodies straight out of ``create_robot``):
#
#     grounded authored hexapod    24.2 s     3 evals   ADOPTS
#     grounded authored cat        56.0 s    87 evals   ADOPTS
#     grounded authored dog       124.7 s   360 evals   ADOPTS NOTHING  (the whole budget spent on an honest no)
#
# That is the price of judging a body with a controller of its own instead of throwing the body away, and NONE
# of it should be negotiated down in the product. But a test suite that calls ``create_robot`` ~50 times pays it
# ~50 times, and ``tests/test_ai_native.py`` alone builds the SAME "a quadruped robot dog" NINE times — 19
# minutes of re-deriving one answer. Two switches, BOTH DEFAULT-OFF so a product run is byte-identical:
#
#   VIRTUROID_GAIT_FIT_CACHE=1  memoize the fit on the body's STRUCTURAL key for the life of the process.
#                               Loses no coverage whatsoever: every structurally distinct body still runs the
#                               real search, including the expensive negative. Only the Nth identical rebuild is
#                               free. This is the one the suite leans on.
#   VIRTUROID_SKIP_GAIT_FIT=1   the fit is a DISCLOSED no-op. For a test whose subject is something else
#                               entirely (the tool dispatcher, a scene, a URDF round-trip) and which would
#                               otherwise pay two minutes to re-establish a locomotion fact it never reads.
#                               It is deliberately visible in the result (``skipped: True`` and a reason that
#                               names the variable) rather than silent, because a fit that returns
#                               "adopted: False" quietly is the exact measurement artefact this module exists
#                               to remove -- see the ``ok=False`` handling at the bottom of ``fit_gait_for_body``.
#
# THE CACHE IS OPT-IN AT THE CALL SITE (``cache=True``), and only the BUILD PATH opts in. That is deliberate
# and it was found the hard way: keying on ``structural_gait_key`` alone is NOT safe, because that function
# answers for a gene with no segments at all -- every stub gene in tests/test_gait_fit_robustness.py hashes to
# the same 'b4b292ac85f6e215' -- and pinning the monkeypatched ``evaluate_gait``/``_one_search`` by ``id()``
# does not save it either, since CPython reuses the address once monkeypatch drops the previous stub. The first
# version of this cache duly handed ``test_restarts_are_bounded`` another test's adopted walk. A test that
# drives the fitter DIRECTLY must always get a real run, so the fitter never caches unless asked.
_FIT_CACHE: dict[tuple, tuple[dict, dict]] = {}

#: Prefixed onto a replayed fit's ``reason`` on BOTH the returned dict and the gene's own metadata. One
#: constant so a reader can strip it to compare two fits' substance, and so the two surfaces cannot drift.
_REPLAY_TAG = "[REPLAYED from an identical body's fit --"


def _fit_cache_key(gene, kw: dict):
    """The identity of THIS fit, or ``None`` when it must not be cached."""
    import os

    if os.environ.get("VIRTUROID_GAIT_FIT_CACHE") != "1":
        return None
    segs = getattr(gene, "segments", None)
    if not segs:                                     # no body -> no structural identity worth keying on
        return None
    try:
        body = structural_gait_key(gene)
    except Exception:  # noqa: BLE001 - a partial gene has no structural identity -> never cached
        return None
    return (body, tuple(sorted(kw.items()))) if body else None


def clear_fit_cache() -> None:
    """Drop the memoized fits. For a test that deliberately wants the search to run again."""
    _FIT_CACHE.clear()


def _remember(key, gene, out: dict) -> dict:
    """Snapshot a COMPLETED fit under ``key`` and return ``out`` unchanged.

    Both halves of the result are kept: the disclosure dict AND the metadata the fit wrote onto the gene, since
    an adopted operating point lives on ``metadata['gait_params']`` and a caller handed back only the dict would
    be holding a body that never got its controller.

    A CRASH (``ok=False``) is never cached. It is not an answer about the body, so the next caller must get a
    real attempt rather than a memoized failure.
    """
    if key is None or not out.get("ok", True):
        return out
    md = getattr(gene, "metadata", None) or {}
    _FIT_CACHE[key] = (dict(out), {k: dict(md[k]) for k in ("gait_params", "gait_fit")
                                   if isinstance(md.get(k), dict)})
    return out


def robustness_margin(gene, params: dict, *, steps: int = _SETTLE_STEPS, ladder=_ROBUST_LADDER,
                      n: int = _ROBUST_N, seed: int = _MARGIN_SEED, per_param: bool = True,
                      per_param_n: int | None = None) -> dict:
    """How much relative perturbation does this operating point survive? Returns the largest rung of ``ladder``
    at which ALL ``n`` jointly-perturbed copies are still a credible walk, or ``None`` below the finest rung —
    plus, when ``per_param``, the SAME margin measured one parameter at a time.

    A verdict without this is missing its error bar. MEASURED 2026-08-01 (task #267) at the fitted op-points the
    product actually ships: the canonical template tolerates >=1e-1 on all five parameters, the grounded
    authored cat's adopted point tolerates less than 1e-5 on four of the five (a 2.4e-5 relative change in step
    frequency flips it from +0.958 m CREDIBLE WALK to +0.500 m FELL by ROLL-OVER) — four orders of magnitude
    apart, and both printed the same two words. A number that fragile is not a property of the controller you
    could deploy; it is a property of one lucky float.

    THE PER-PARAMETER BREAKDOWN IS WHERE THAT COMPARISON LIVES, which is why it is on by default. The joint
    margin answers "how far can everything move at once" and collapses to a single scalar; the per-parameter
    margins answer "which knob is the customer not allowed to touch", and that is the actionable half — a body
    whose margin is 1e-1 on four axes and 1e-3 on ``freq`` needs its step frequency pinned, not a redesign.
    Each parameter is probed ALONE (the others held exactly), so the number is attributable.

    ``seed`` defaults to ``_MARGIN_SEED``, which is HELD OUT from the draws ``search_gait`` optimises against.
    Measuring the margin on the search's own probe draws would report the fit, not the point.

    ``per_param_n`` defaults to ``n``, i.e. the two estimates use the SAME number of draws. It was briefly 2,
    for cost, and the grounded authored cat printed the result: "survives a 0.1 relative perturbation of all 5
    parameters at once; per parameter: kd 0.001, freq 0.1, ...". Both halves were honest small samples and the
    sentence still read as a contradiction, which is the failure mode this whole task is about. Equal-sized
    samples plus the disclosed ``n`` in ``margin_sentence`` make disagreement legible as sampling noise instead.

    COST. The joint ladder costs ``n`` rollouts per rung and STOPS at the first rung that passes, so a robust
    point costs ``n`` and a fragile one ``n * len(ladder)``. The per-parameter pass adds at best
    ``len(params) * per_param_n`` (every axis sturdy at the top rung) and at worst that times ``len(ladder)``:
    for five parameters, a 3-rung ladder and ``n=4``, 20 to 60 rollouts. Measured on the grounded bodies at the
    6000-step horizon that is roughly 10-60 s — real, and it is the price of an error bar rather than an
    assertion. It is paid ONCE per adopted operating point, not per candidate.
    """
    import random

    from virturoid.services.gait_search import evaluate_gait, perturbed_params

    spent = [0]

    def _walks(q) -> bool:
        spent[0] += 1
        res = evaluate_gait(gene, q, steps=steps)
        return bool(res.get("credible")) and bool(res.get("survived"))

    def _ladder(keys, probes_n, rng) -> tuple[float | None, dict]:
        probes: dict[str, str] = {}
        for rel in ladder:
            ok = sum(int(_walks(perturbed_params(params, rel, rng, keys=keys))) for _ in range(probes_n))
            probes[f"{rel:g}"] = f"{ok}/{probes_n}"
            if ok == probes_n:
                return float(rel), probes          # ladder is largest-first; the first pass IS the margin
        return None, probes

    keys = tuple(k for k in _FIT_PARAMS if k in params)
    best, probes = _ladder(keys, int(n), random.Random(seed))
    out = {"robustness_rel": best, "probes": probes, "steps": int(steps), "n": int(n),
           "note": (f"every one of {n} jointly-perturbed copies still walks at rel {best:g}" if best is not None
                    else f"fails below rel {min(ladder):g} — this operating point is one lucky float, "
                         f"not a controller")}
    pp_n = int(n if per_param_n is None else per_param_n)
    if per_param and keys:
        out["per_param_rel"] = {k: _ladder((k,), pp_n, random.Random(seed + i + 1))[0]
                                for i, k in enumerate(keys)}
        out["per_param_n"] = pp_n
    out["n_rollouts"] = spent[0]                  # what the error bar COST, disclosed with the error bar
    return out


def margin_sentence(rob: dict, *, head: bool = True) -> str:
    """The error bar as one line of English, for printing NEXT TO the two-word verdict.

    THE HONESTY DEFECT task #267 names is not that the margin was unmeasured — it is that a point tolerating
    1e-1 on every parameter and a point tolerating less than 1e-5 on four of five both print "CREDIBLE WALK",
    four orders of magnitude apart, with nothing beside them to tell a reader which one they are holding.

    Parameters are listed TIGHTEST FIRST, because the tightest one is the operating constraint: it is the knob
    a deployment has to pin, and it is what the joint scalar averages away. ``head=False`` gives only that list,
    for a caller that has already said the joint part in its own words.

    THE SAMPLE SIZES ARE PRINTED. Both halves are small-sample estimates over random perturbation draws, and
    they can disagree — the grounded authored cat has produced "0.1 jointly" beside "kd 0.001" from the same
    measurement. Stating ``n`` makes that read as the sampling noise it is, instead of as a contradiction the
    reader has to resolve by guessing which number to believe.
    """
    pp = rob.get("per_param_rel") or {}
    tight = ", ".join(f"{k} <{min(_ROBUST_LADDER):g}" if v is None else f"{k} {v:g}"
                      for k, v in sorted(pp.items(), key=lambda kv: (kv[1] is not None, kv[1] or 0.0)))
    pn = rob.get("per_param_n")
    tail = f"; per parameter (n={pn} each): {tight}" if tight else ""
    if not head:
        return tail.lstrip("; ") if tail else "per-parameter margin not measured"
    rel, n = rob.get("robustness_rel"), rob.get("n") or _ROBUST_N
    lead = (f"survives a {rel:g} relative perturbation of all {len(pp) or len(_FIT_PARAMS)} parameters at once "
            f"({n}/{n} probes)" if rel is not None else
            f"survives NO perturbation as large as {min(_ROBUST_LADDER):g} in any of {n} probes — one lucky "
            f"float, not a controller")
    return lead + tail


#: The rollout function every rung of this module searches — ``gait_search.evaluate_gait`` calls it
#: unconditionally, and the five parameters in ``_FIT_PARAMS`` are ITS parameters.
_SEARCHED_CONTROLLER = "morph_policy.crawl_gait_rollout"

#: Below this fraction of the requested horizon, a search has not MEASURED the body it is reporting on. Not a
#: physics constant — a DISCLOSURE threshold. 2% of the 6000-step horizon is 120 steps (0.24 s of simulated
#: time), and a decline that quotes "still walking at 6000 steps" after 120 steps of physics is quoting a
#: horizon nothing ever reached. The live inchworm books ~30 steps (0.5%) per rollout, 379 times over.
_DEGENERATE_INTEGRATION_FRAC = 0.02

#: Wall-clock ceiling for ONE per-body gait fit, in seconds. A CUSTOMER-PATIENCE bound, not a search parameter —
#: see ``_budget_from_env`` for the override chain and ``search_gait``'s ``deadline`` for where it is enforced.
#:
#: WHY 180, AND WHY NOT SMALLER. The fit is the thing that makes an authored body walk at all (docs/
#: breaking_the_cotuning_wall.md; 7/7 proportion variants), so the cost of cutting it too early is the whole
#: correctness win. The number is therefore set from the measured ADOPTION times, NOT from the failure times.
#: Every adoption on this checkout landed well inside a minute of searching — grounded authored hexapod 24.2 s
#: / 3 evals, cat 56.0 s / 87 evals, and through ``call_tool`` a biped that adopted a +3.732 m robust walk at
#: EVALUATION 1 off a flywheel recall. Every body that searched longer than two minutes adopted nothing (dog
#: 124.7 s; large quadruped, 8-legged spider). 180 s is ~3x the slowest measured adoption, which is the margin
#: for a slower machine or a harder body, and it cuts the stage that dominates a slow build by ~3x: MEASURED
#: end to end through ``create_robot`` on the large quadruped, the gait-fit stage goes 615.4 s -> 188.2 s.
#:
#: THE CALL IS NOT THE SEARCH. Compose is LLM-driven and unbounded — measured across five runs of the same
#: three prompts it ranged 16.8 s to 253.7 s — so end-to-end totals move for reasons this budget does not
#: control, and the number to compare before/after is the gait-fit stage.
#:
#: IT IS NOT A SILENT TRUNCATION. A fit the clock stops reports ``stopped_by_budget`` and a reason that says the
#: search did not finish, precisely so the budget can never be mistaken for a finding about the body.
_DEFAULT_FIT_BUDGET_S = 180.0


def crawl_deployment_match(gene) -> dict:
    """Will the controller THIS FITTER SEARCHES ever be the controller this body is DRIVEN by?

    A SEARCH THAT CANNOT SUCCEED MUST NOT BE REPORTED AS A SEARCH THAT FAILED. Measured on the live inchworm
    (docs/body_vs_controller_ruling.md, defect D1): ``fit_gait_for_body`` spent "360 gaits (379 physics
    rollouts)" in **1.3 seconds** and told the customer "none was still walking at the 6000-step horizon". Every
    one of those rollouts was the same unactuated collapse, because the body is a limbless serial spine and
    ``crawl_gait_rollout``'s wave gait drives ``build_appendage_map(...).legs`` — of which it has none. Meanwhile
    ``ai_native_tools._honest_gait`` routes exactly that body to ``_honest_serpentine`` BEFORE it ever reads
    ``metadata['gait_params']``, so nothing this fitter could have adopted would have reached the robot anyway.
    That sentence reads to a customer as a measured negative about their design; it is a measurement artefact,
    the same class as a NaN twin issuing a certificate.

    THIS FUNCTION IS THE MATRIX, in the order the product actually dispatches (``verify_robot`` first, then
    ``_honest_gait``), so the two can be compared rather than assumed equal:

        structure                       deployed by the product        this fitter searches   applicable?
        aerial (rotors / class)         aerial._honest_fly             crawl                  NO
        aquatic (class / metadata)      _honest_swim                   crawl                  NO
        LIMBLESS SERIAL SPINE           _honest_serpentine             crawl                  NO   <- D1
        mobile (wheels)                 _honest_drive                  crawl                  NO
        manipulator / spray / other     _honest_reach / no verdict     crawl                  NO
        legged, 0 measured legs         crawl, driving nothing         crawl                  NO
        legged, 2 legs (biped)          crawl, then _honest_biped      crawl                  YES (the hard case)
        legged, >=3 legs                crawl                          crawl                  YES

    Only the last two rows are searchable. The rest get an honest refusal, never a decline.

    THE BIPED ROW IS SEARCHABLE, and a proposal to refuse it is RECORDED HERE BECAUSE IT WAS WRONG. On
    2026-08-10 "the crawl engine cannot balance two legs, so that search could never have succeeded" was
    proposed as a cheap a-priori refusal on this row; the FIRST measurement taken to justify it refuted it — a
    composed 2-legged body adopted at evaluation 1, +3.732 m at the 6000-step horizon against the default's
    0.498, rate rising, 4/4 robustness probes — and the bank already held two ``gait::humanoid::*`` locomotion
    rows, one at success 1.0. The distinction this function exists to police is exactly the one that proposal
    blurred: "expensive, and often unsuccessful" is not "structurally impossible", and only the second may be
    refused before measuring. Bounding the cost of the hard case is the BUDGET's job, not this matrix's.

    FAIL-OPEN, DELIBERATELY. Anything this cannot determine (no MuJoCo, a partial gene, a stub in a unit test)
    returns ``applicable=True`` with ``measured=False``: a diagnostic must never be the thing that stops a real
    body from being fitted. The cost of a false "applicable" is the search we run today; the cost of a false
    "not applicable" is a body silently denied its own controller.
    """
    out = {"searched_controller": _SEARCHED_CONTROLLER, "deployed_controller": _SEARCHED_CONTROLLER,
           "applicable": True, "match": True, "measured": False, "kind": None, "n_legs": None, "why": ""}
    try:
        from virturoid.services.aquatic import _is_serial_spine
        from virturoid.services.body_kind import body_kind, measured_appendages
        # A REFUSAL IS ONLY ALLOWED ON A BODY THAT COMPILED. ``body_kind`` NEVER RAISES — a gene with no
        # segments, or one segment, or a partial stub falls through every structural branch to its terminal
        # "manipulator (fixed base + articulation)". Refusing on that answer would deny a controller to a body
        # nothing has actually looked at, which is the same failure this function exists to stop, pointed the
        # other way. ``measured_appendages`` compiles the body (and caches it), so ``measured`` is the honest
        # "we have seen this robot" flag, and every branch below is gated on it.
        amap = measured_appendages(gene)
        if not amap.get("measured"):
            return {**out, "why": "this body would not compile for the structural measurement, so which "
                                  "controller it deploys is unknown — searching rather than refusing on a "
                                  "classification nothing could check"}
        bk = body_kind(gene)
        out["kind"] = bk.kind
        out["measured"] = True
        if bk.kind == "aerial":
            return {**out, "deployed_controller": "aerial._honest_fly", "applicable": False, "match": False,
                    "why": "this body is FLOWN (rotor thrust + a geometric flight controller), never walked — "
                           "verify_robot routes it to _honest_fly and would never read a fitted leg gait"}
        if bk.kind == "aquatic":
            return {**out, "deployed_controller": "ai_native_tools._honest_swim", "applicable": False,
                    "match": False,
                    "why": "this body is SWUM in real MuJoCo fluid — verify_robot routes it to _honest_swim and "
                           "would never read a fitted leg gait"}
        if bk.kind == "mobile":
            return {**out, "deployed_controller": "ai_native_tools._honest_drive", "applicable": False,
                    "match": False, "why": "this body DRIVES on wheels — verify_robot routes it to _honest_drive"}
        if bk.kind != "legged":
            return {**out, "deployed_controller": f"ai_native_tools verdict for kind={bk.kind!r}",
                    "applicable": False, "match": False,
                    "why": f"verify_robot gives a {bk.kind} body no leg-gait verdict at all, so a fitted crawl "
                           f"operating point could never reach it"}
        # THE SPINE TEST IS SCOPED TO LEGGED BODIES, and the scope is load-bearing rather than tidy: a 6-DOF ARM
        # is also "a single non-branching revolute chain", so ``_is_serial_spine`` says True for it. Only
        # ``_honest_gait`` consults that predicate, and verify_robot has already sent a manipulator to
        # ``_honest_reach`` before ``_honest_gait`` is ever reached — so testing it earlier here would refuse an
        # arm for the wrong reason and print the wrong controller name in the disclosure. Measured: it did.
        if _is_serial_spine(gene):
            # The D1 row.
            return {**out, "deployed_controller": "morph_policy.serpentine_rollout", "applicable": False,
                    "match": False,
                    "why": "this body is a LIMBLESS SERIAL SPINE, driven by the serpentine travelling-wave "
                           "controller (ai_native_tools._honest_serpentine), not by the leg crawl gait this "
                           "fitter searches — it has no legs for that wave to drive, so every candidate would "
                           "be the same unactuated collapse"}
        legs = int(amap.get("legs") or 0)
        out["n_legs"] = legs
        if legs == 0:
            return {**out, "applicable": False, "match": False,
                    "why": "the crawl wave gait has NO ACTUATED LEG to drive on this body (the appendage map "
                           "measures zero), so every operating point produces the identical unactuated rollout "
                           "and the search cannot distinguish them"}
        if legs == 2:
            # A BIPED IS SEARCHABLE. "The crawl engine cannot balance two legs, so that search could never have
            # succeeded" was proposed as a cheap refusal here on 2026-08-10 and REFUTED the same day by the
            # first measurement taken to justify it: a composed 2-legged body (8 DOF, legs=2) adopted an
            # operating point at EVALUATION 1 — +3.732 m at the 6000-step horizon against the default's 0.498,
            # rate rising +0.41 -> +0.62 m/1000 (accelerating, not decaying), surviving a 1e-2 perturbation of
            # all five parameters at once, 4/4 probes. The bank independently agrees: it already holds two
            # ``gait::humanoid::*`` locomotion rows, one at success 1.0, and a row only enters after beating the
            # default at the deploy horizon.
            #
            # Task #206 is still true and is NOT this claim. It measured ONE humanoid on which the scripted
            # crawl fell at step 487 while learned control held 6000/6000. "Hard, and often better served by
            # learned control" is a different statement from "impossible", and only the second would license a
            # refusal. What bounds the cost of the hard case is the BUDGET, which cannot deny a body a walk it
            # can actually do.
            return {**out, "why": "biped: _honest_gait runs the crawl FIRST (so a fitted point does ship if it "
                                  "is credible) and falls back to _honest_biped's static-balance verdict only "
                                  "when it is not — the search is applicable, and MEASURED to succeed on a "
                                  "composed biped (+3.732 m at 6000 steps, 4/4 robustness probes). It is the "
                                  "hard case, not the impossible one: where the crawl does fail, task #206 "
                                  "measured learned control balancing the same body 6000/6000"}
        return {**out, "why": f"legged, {legs} legs measured — the crawl gait this fitter searches is the one "
                              f"_honest_gait deploys"}
    except Exception:  # noqa: BLE001 - a diagnostic must never block a real fit; see the docstring
        return {**out, "measured": False,
                "why": "could not determine which controller this body deploys; proceeding with the search "
                       "rather than refusing one"}


def _integrated(res) -> int | None:
    """Physics steps a rollout result actually integrated, or ``None`` when the evaluator did not say.

    ``None`` is NOT zero. A stubbed evaluator in a unit test reports nothing here, and treating silence as "no
    physics happened" would turn every contract test's honest decline into a degeneracy alarm.
    """
    v = (res or {}).get("steps_integrated")
    return int(v) if v is not None else None


def fit_gait_for_body(gene, *, deploy_steps: int = _SETTLE_STEPS, search_steps: int = _SETTLE_STEPS,
                      max_evals: int = 96,
                      warm_evals: int = 24, generations: int = 8, pop: int = 24, seed: int = 0,
                      db=None, bank: bool = True, seed_restarts: int = 3, cache: bool = False,
                      budget_s: float | None = None) -> dict:
    """Give ONE GROUNDED body its OWN operating point, and cache it on ``gene.metadata['gait_params']``.

    Call this AFTER ``ground_and_repair``. The whole point is that a body must be judged with a controller fitted
    to IT before anything decides it cannot walk: measured on this checkout, the authored dog / cat / horse /
    cheetah all score ``0.000`` through ``evaluate_robot`` at the shipped ``freq 1.5 / kp 32`` op-point and all
    reach a CREDIBLE WALK with an op-point of their own. ``anatomy_compiler.ensure_walkable_quad`` reads that
    ``0.000`` and substitutes a template for the customer's design, so the substitution is really a measurement
    artefact. See docs/breaking_the_cotuning_wall.md §0/§1.

    Two properties keep this BOUNDED and safe to put on the interactive build path — bounded, not cheap; the
    "~0.5 s" this docstring used to promise was written against a 1500-step horizon and one seed, and the same
    commit that wrote it raised the horizon to 6000 and the restarts to 3. RE-MEASURED 2026-08-05: a grounded
    authored hexapod 24.2 s / 3 evals, a cat 56.0 s / 87, a dog 124.7 s / 360 and no adoption:

    * **Nothing is searched for a body that already walks.** The shipped default is measured first; if it is
      already a CREDIBLE WALK at the deploy horizon the body keeps it and this returns after that ONE rollout
      having changed nothing. So no robot that ships today can move.
    * **Nothing is adopted that does not beat the default at the DEPLOY horizon.** That gate lives in
      ``learn_gait_flywheel`` (which also recalls a structural prior and banks the result), never in the search's
      own optimistic score — measured, a cat credible at an 800-step search horizon FELL at 1500.

    THE SEARCH ITSELF IS ROBUSTNESS-AWARE (task #267, 2026-08-06). Every candidate that walks at its own
    parameters is re-scored across a neighbourhood of relative size ``_SEARCH_ROBUST_REL`` — the sim-to-real
    tolerance, ~1e-2 — and ranked robust-first, and the credible-early-stop fires only for a candidate whose
    perturbed copies walk too. That closes the loop the two paragraphs below only annotated: ranking winners by
    sturdiness AFTER the fact cannot help if the only winner the optimiser ever produces is a spike, and a
    single-rollout objective produces spikes by construction, because a spike is what the maximum of a rough
    function looks like. MEASURED before -> after on the grounded authored cat, whose adopted point was 0/8 at
    both 1e-2 and 1e-3: see the task #267 report for the per-body table and the cost.

    STURDINESS LEADS THE RANKING, and a FRAGILE WIN DOES NOT END THE SEARCH. Until 2026-08-02 this stopped at
    the first draw the horizon called credible, measured its robustness, and shipped it either way — so the
    error bar was an annotation on a decision already taken. That is how the settling defect survived its own
    fix: on the grounded authored cat the adopted point holds its rate to 6000 and is ON THE FLOOR BY STEP 8754,
    0/8 perturbed copies surviving at both rel 1e-2 and 1e-3. A horizon can always be outlived; a perturbation
    probe is the horizon-free form of the same question, and here it separates the two cases cleanly (the cat's
    8/8-robust point is still walking at 12000). A fragile winner is therefore kept only as a FALLBACK while the
    remaining seed attempts look for a controller — and if none is found it is still adopted (the body does
    travel and does hold its rate; declining it would be the same dishonesty pointed the other way) but is
    flagged ``fragile`` and refused entry to the bank, so no later body recalls it as a warm start.

    ``search_steps == deploy_steps == _SETTLE_STEPS``, AND THEY MUST STAY EQUAL. Both were 1500, and 1500 was
    exactly where the claim broke: the grounded authored cat's adopted point reads CREDIBLE WALK at 1500 and
    falls at step 2014 (task #267). Raising only the JUDGING horizon does not fix it, it just adopts nothing —
    MEASURED on this checkout, three variants at the same 96-eval x 3-seed budget:

        search 1500 / judge 1500          cat ADOPTS (+0.64 m/1000 -> falls at 2014)   dog ADOPTS (rate -> 0.23)
        search 1500 / judge 6000 + rate   cat ADOPTS NOTHING (141 evals)               dog ADOPTS NOTHING
        search 6000 / judge 6000 + rate   cat ADOPTS a real walk, rate 0.769 -> 0.636  dog ADOPTS NOTHING

    because ``stop_on_credible`` stops at the first candidate the SEARCH horizon calls credible, so screening
    cheaply and judging honestly just spends the whole budget rediscovering drift. An optimiser may never be
    scored at a shorter horizon than the one its answer is judged at — that IS the defect, stated generally.

    ``warm_evals``/``max_evals`` + credible-early-stop bound the cost. THIS IS NOT FREE, and the numbers are
    MEASURED before -> after, same grounded bodies, same machine:

        body whose default already walks   0.7 s /   0 evals ->   1.0 s /   0 evals   (one rollout, 4x longer)
        grounded authored cat              2.7 s /   8 evals ->  42.0 s /  66 evals   adopts a REAL walk
        grounded authored dog             11.9 s /  30 evals -> 145.3 s / 360 evals   adopts NOTHING

    So a fit costs ~1.4x when the body already walks, ~15x when it finds a walk, and ~12x when it proves it
    cannot — the dog's 145 s is the whole budget spent establishing a negative, which is both the expensive
    direction and the honest one. Against the 8-170 s range an end-to-end build used to occupy, the ceiling for
    a legged body moves into the low tens of minutes for the pathological case. A caller that needs it cheaper
    should cut ``seed_restarts`` or ``max_evals``, NEVER the horizon: a shorter horizon does not make the fit
    cheaper, it makes the answer wrong. Search runs SERIAL in this mode by construction (``gait_search``
    disables the process pool so the early stop actually saves rollouts), which also keeps the build path free
    of multiprocessing.

    ``seed_restarts`` — THE SEARCH IS SEED-FRAGILE ON AUTHORED QUADRUPEDS, so one draw is not a verdict.
    MEASURED 2026-08-01 (task #265), same grounded body, same code, same 96-eval budget: the authored dog reaches
    a CREDIBLE WALK on seeds 0/3/4 and FELLS by ROLL-OVER on 1/2 (winning travel +0.796 to +2.069 m, a 2.6x
    spread); the authored cat wins on 1 of 3 seeds. At a fixed ``seed=0`` that makes a customer's robot walk or
    fall on a draw nobody chose. This is NOT a regression introduced by fitting per body — it was always there,
    hidden because ``ensure_walkable_quad`` threw the body away before anything searched it. Stage 1 exposed it by
    finally trusting the fitter, so the fix is a ROBUST fitter, not going back to substituting the body.
    A failed search therefore RESTARTS from a decorrelated seed, up to ``seed_restarts`` attempts, keeping the
    best; a body that succeeds on the first attempt pays nothing extra.

    Returns a disclosure dict (also stored on ``metadata['gait_fit']``). Never raises — but a genuine CRASH is
    reported as ``ok=False`` with ``error``, distinguishable from a considered decline (``ok=True``,
    ``adopted=False``), because a swallowed exception that reads as "this body has no better gait" is the exact
    measurement artefact this function exists to eliminate.

    ``budget_s`` IS THE CUSTOMER'S HALF OF THAT COST, and it is why the paragraphs above are no longer the whole
    story. Everything they say about honesty is about the ANSWER; this is about the WAIT. Measured 2026-08-10
    across 20 real ``create_robot`` calls through ``agent_tools.call_tool``, the range is 0.5 s to 634 s, and the
    three worst — a large quadruped 634 s, an 8-legged spider 552 s, a biped 270 s — adopted NOTHING.

    TWO OF THOSE THREE NO LONGER REPRODUCE, which is worth saying because it is the flywheel working rather
    than this change: re-measured the same day after the corpus had grown, the spider's fit is 111 s (it ADOPTS,
    at evaluation 14) and the biped's is 86 s for the full 360 evaluations. Only the large quadruped still
    saturates, and it is the case the budget binds on — 615.4 s -> 188.2 s of gait fitting. A ceiling set from
    the failures would have been tuned to a moving target; set from the ADOPTIONS, it is not. So the
    default is a ceiling (``_DEFAULT_FIT_BUDGET_S``, overridable per call or by ``VIRTUROID_GAIT_FIT_BUDGET_S``,
    ``0`` for unbounded), set from the measured ADOPTION times rather than the failure times so that no body
    known to walk is cut. Three properties keep it from becoming a quiet way to stop measuring:

    * the clock is checked at EVALUATION boundaries only, never inside a rollout — the horizon is never shortened,
      which is the one economy this function's own docstring forbids;
    * a fit the clock stopped sets ``stopped_by_budget`` and says in its ``reason`` that the search DID NOT
      FINISH. It must never read as "none was still walking";
    * the test suite pins it OFF (``tests/conftest.py``), so no assertion about whether a body walks becomes a
      function of how fast the machine running it is.

    AN EVALUATION BUDGET SITS BESIDE THE CLOCK, and that third bullet is why it had to. A wall clock cannot say
    WHICH candidate it stopped at — MEASURED 2026-08-12, two runs of the same 8-legged spider build on the same
    box, both stopped by the same 180 s ceiling, stopped at 144 of 360 gaits and at 70 of 360 — so "this build
    was budgeted" is not a fact any test can pin without pinning the hardware, and the stopping point is not a
    property of the robot. An evaluation is a unit the physics owns, so the same cap cuts the
    same search at the same candidate anywhere. It is carried on the build (``build_progress.evals_budget``),
    shared by every fit in the build for the same reason the clock is, and it is ``None`` unless a caller asked
    for one: the default is still the full search, which is the thing that makes an authored body walk.
    MEASURED on the grounded authored dog, ``evals_budget=8``: 8 evaluations, 14 rollouts, 6.4 s against the
    same body's 124 s unbudgeted fit, and ``stopped_by_eval_budget`` with a reason that says the search did not
    finish rather than that the body did not walk.

    ``cache=True`` lets an identical body reuse an identical fit, but ONLY when ``VIRTUROID_GAIT_FIT_CACHE=1``;
    the build path asks for it and a test suite turns it on. ``VIRTUROID_SKIP_GAIT_FIT=1`` makes the whole fit a
    disclosed no-op. Both default OFF — see the block above ``_FIT_CACHE`` for what they cost and what they buy.
    """
    import os
    import time

    from virturoid.services.gait_search import _HI, _LO, evaluate_gait

    out: dict = {"ok": True, "searched": False, "adopted": False, "reason": "", "n_evals": 0,
                 "horizon_steps": int(deploy_steps), "stopped_by_budget": False,
                 "stopped_by_eval_budget": False}
    if os.environ.get("VIRTUROID_SKIP_GAIT_FIT") == "1":
        # SAY SO. A caller reading `searched: False, adopted: False` off a real fit is being told "this body was
        # measured and kept its default"; here it means "nothing was measured", and those must not look alike.
        out["skipped"] = True
        out["reason"] = ("gait fitting was DISABLED for this process by VIRTUROID_SKIP_GAIT_FIT=1 — this body "
                         "was never searched, so nothing here is a finding about whether it can walk")
        _stash(gene, out)
        return out
    # DID THE CALLER TURN FITTING OFF FOR THIS WHOLE BUILD? Checked here rather than at the one call site that
    # knows about it, because a build fits up to three bodies through two modules — measured, ``tune_gait:
    # false`` left ``ensure_walkable_quad`` fitting anyway and overwriting the disclosure with a real result,
    # so the control said "off" and the clock said otherwise.
    from virturoid.services import build_progress as _P
    _b = _P.current()
    if _b is not None and not getattr(_b, "fit_gait", True):
        out.update({"skipped": True, "searched": False, "adopted": False, "n_evals": 0, "n_rollouts": 0,
                    "reason": ("gait fitting was TURNED OFF for this build (tune_gait=false), so this body was "
                               "never measured and nothing here says whether it can walk. It ships on the "
                               "shipped default operating point, which is another robot's hand-tuned numbers "
                               "— every authored quadruped measures 0.000 m there — and the walkability gate "
                               "judges it at those numbers too, so an untuned body is also likelier to be "
                               "replaced by a template. Rebuild with tune_gait=true, or call learn_gait, "
                               "before reading any walk verdict as a fact about this design")})
        _stash(gene, out)
        return out
    # DOES THE CONTROLLER THIS FITTER SEARCHES EVER REACH THIS BODY? Asked BEFORE a single rollout is paid for.
    # See ``crawl_deployment_match`` for the matrix and for the measurement that made this necessary: a search
    # over a controller the body is never driven by is not a search, and its empty result is not a finding.
    deploy = crawl_deployment_match(gene)
    out["deployed_controller"] = deploy["deployed_controller"]
    if not deploy["applicable"]:
        out.update({"skipped": True, "not_applicable": True, "searched": False, "adopted": False,
                    "n_evals": 0, "n_rollouts": 0, "deployment": deploy,
                    "reason": (f"NO SEARCH WAS RUN, and nothing here is a finding about this body. This fitter "
                               f"searches {_SEARCHED_CONTROLLER} and its five parameters "
                               f"{', '.join(_FIT_PARAMS)}; this body is driven by "
                               f"{deploy['deployed_controller']} instead — {deploy['why']}. Running the search "
                               f"anyway would spend hundreds of rollouts that measure nothing and then report a "
                               f"decline, which a reader takes for a measured negative about the design")})
        if deploy["deployed_controller"] == "morph_policy.serpentine_rollout":
            # Say what it would take, because "not applicable" must not be read as "not fittable". The serpentine
            # controller exposes exactly five parameters of its own (amp, wavenum, freq, kp, kd — the same arity
            # as _FIT_PARAMS) and _honest_serpentine calls it with the defaults for all five. What is missing is
            # not a search; it is a CREDIBILITY BAR for undulatory gaits. `gait_quality.classify` is written for
            # legged travel (cadence, support fraction, roll/pitch of an upright trunk) and `_honest_serpentine`
            # gates on a bare `planar_m > 0.15` with no settling and no yaw. Optimising against that threshold
            # would manufacture exactly the gate-scrape the ruling documents for the starfish (0.302 m over a
            # 0.30 m bar), so the honest order is bar first, search second.
            out["fittable_in_principle"] = True
            out["what_would_make_it_fittable"] = (
                "morph_policy.serpentine_rollout exposes (amp, wavenum, freq, kp, kd) and _honest_serpentine "
                "calls it with the defaults for all five, so this body class has never had ANY operating point "
                "fitted to it. Wiring a search needs a credibility bar for undulatory travel first — "
                "gait_quality.classify measures legged cadence/support/upright, and _honest_serpentine's bare "
                "'planar_m > 0.15' is a threshold an optimiser would scrape rather than a walk it would find")
        _stash(gene, out)
        return out
    # ``budget_s`` IS PART OF THE KEY. Two fits of the same body under different budgets are two different
    # searches — one may have finished and one may have been cut — and replaying the cut one as the answer to
    # the unbounded question is the same class of error as replaying a crash. The BUILD's remaining clock is
    # deliberately NOT in the key: within one build, replaying an identical body's answer to a later, shorter
    # window is exactly what the cache is for.
    _ckey = _fit_cache_key(gene, {"deploy_steps": deploy_steps, "search_steps": search_steps,
                                  "max_evals": max_evals, "warm_evals": warm_evals,
                                  "generations": generations, "pop": pop, "seed": seed,
                                  "bank": bank, "seed_restarts": seed_restarts,
                                  "budget_s": budget_s if budget_s is not None else _budget_from_env(None),
                                  # THE EVAL CAP *IS* IN THE KEY, unlike the clock's remaining window one line
                                  # up, and the asymmetry is deliberate. A cached answer to a LONGER clock is
                                  # strictly better than the shorter one it replaces. A cached answer to a
                                  # SMALLER eval cap is a TRUNCATED search, and replaying it for an uncapped
                                  # caller would hand them a partial result carrying "stopped by the evaluation
                                  # budget" for a budget they never set — a partial search presented as a
                                  # completed one, which is the exact defect this budget exists to prevent.
                                  "evals_budget": getattr(_P.current(), "evals_budget", None),
                                  "db": db is not None}) if cache else None
    if _ckey is not None and _ckey in _FIT_CACHE:
        # A structurally identical body already paid for this exact search. Replay BOTH mutations the real call
        # makes -- the adopted operating point and the disclosure -- so the returned gene is indistinguishable
        # from a freshly fitted one; returning the dict alone would hand back a body with no gait_params.
        #
        # AND IT SAYS IT IS A REPLAY, because the counts describe a search THIS CALL DID NOT RUN. Measured
        # 2026-08-13: a second structurally identical fit inside a build whose evaluation ledger was already
        # exhausted returned ``{searched: true, n_evals: 6}`` -- six evaluations reported by a call that spent
        # zero, and no way for a reader to tell. The ledger itself was never overspent (the budget holds); what
        # was wrong is a claim about work done, which is the same defect as every other number in this codebase
        # that was true where it was taken and quoted somewhere else.
        #
        # The counts are KEPT rather than zeroed: they are the honest provenance of the answer being handed
        # back, and blanking them would trade one misreading for another ("this answer came from nothing").
        # ``replayed_from_cache`` is what makes them readable as history instead of as this call's invoice.
        cached_out, cached_md = _FIT_CACHE[_ckey]
        md = dict(getattr(gene, "metadata", None) or {})
        md.update({k: dict(v) for k, v in cached_md.items()})
        gene.metadata = md
        replay = dict(cached_out)
        _md_fit = md.get("gait_fit")
        # BOTH SURFACES OR NEITHER. The first version of this prefixed only the RETURNED dict, leaving
        # ``gene.metadata['gait_fit']['reason']`` on the original string -- so the same fit read two different
        # ways depending on where you looked at it, which is the defect this codebase has spent the day
        # removing (four surfaces disagreeing about one robot, #215/#218). ``test_a_cached_fit_returns_the_same
        # _answer_AND_the_same_body_state`` caught it by asserting the two agree, and it was right to.
        for _tgt in (replay, _md_fit if isinstance(_md_fit, dict) else None):
            if _tgt is None:
                continue
            _tgt["replayed_from_cache"] = True
            _tgt["evals_spent_by_this_call"] = 0
            if _tgt.get("n_evals") and not str(_tgt.get("reason") or "").startswith(_REPLAY_TAG):
                _tgt["reason"] = (f"{_REPLAY_TAG} this call ran no rollouts; the {_tgt['n_evals']} "
                                  f"evaluation(s) described below were spent by the first one] "
                                  + str(_tgt.get("reason") or ""))
        return replay
    try:
        _t_roll0 = time.monotonic()
        default = evaluate_gait(gene, _DEFAULT_GAIT, steps=deploy_steps)
        # WHAT ONE ROLLOUT COSTS ON *THIS* BODY, measured rather than guessed, and free because the default-gait
        # probe had to run anyway. It is the only honest basis for telling a customer up front what the fit may
        # cost: rollout cost varies by an order of magnitude across bodies (an 8-legged spider integrates far
        # more contacts than a hexapod), so a fixed advertised number would be wrong for almost every robot.
        _t_roll = max(1e-6, time.monotonic() - _t_roll0)
        out["rollout_s"] = round(_t_roll, 3)
        out["default_forward_m"] = round(float(default["forward"]), 4)
        out["default_verdict"] = default.get("verdict")
        out["default_rates"] = default.get("rates")
        # THE DEEPEST ANY ROLLOUT ACTUALLY GOT. A decline quotes a horizon ("none was still walking at the
        # 6000-step horizon"); this is the number that says whether the quote was earned. Kept as a running
        # maximum over every deploy-horizon rollout this call observes, and left ``None`` when nothing reported
        # it — see ``_integrated``.
        deepest = _integrated(default)
        if bool(default["survived"]) and bool(default.get("credible", False)):
            # Already walks under the shipped controller -> never churn it. This is what keeps the change
            # additive: only bodies that would otherwise be discarded pay for (or receive) a new op-point.
            out["reason"] = "the shipped default gait is already a credible walk for this body"
            _stash(gene, out)
            return _remember(_ckey, gene, out)
        n_evals = 0
        n_rollouts = 1                                   # the default-gait probe above is already one rollout
        # REFUSE IN ADVANCE WHERE THREE ROLLOUTS SAY THE SEARCH HAS NOTHING TO RANK.
        #
        # THIS IS A SAMPLE, NOT A PROOF, and the word "provably" stood here until 2026-08-13. ``_FIT_PARAMS``
        # is 5-dimensional, so the two corners below are 2 of its 32; the step from "both extremes collapse" to
        # "so does everything between them" is a monotonicity assumption over that box, and nothing forbids a
        # body that dies at all-low and all-high gains but survives at, say, high ``freq`` with low ``kp``.
        # Twice today a claim of this exact shape — a bound asserted over an optimiser's reachable set without
        # sweeping it — was measured false in ``sysid/fit.py`` and retracted. Calling this one a proof would be
        # the third.
        #
        # WHAT IS MEASURED: swept 20 composed+grounded bodies (inchworm, snake, starfish, biped, 8-legged
        # spider, humanoid, giraffe, one-legged hopper, ostrich, kangaroo, flamingo, centipede, crab, turtle,
        # gecko, mantis, millipede, long-legged, top-heavy biped, tiny insect) and the refusal fires on NONE of
        # them — every default rollout cleared the floor, lowest ostrich 159 and hopper 181 of 6000. So its
        # blast radius today is zero measured bodies, which is an absence of evidence rather than a proof of
        # safety, and the reason to keep it narrow.
        #
        # WHY IT EXISTS AT ALL: ``degenerate_search`` below catches the same case only AFTER the whole budget
        # is spent — the live inchworm booked "360 gaits (379 physics rollouts)" before anything noticed that
        # none integrated more than ~30 of 6000 steps. The finding was right and the price was absurd.
        # Degeneracy is the property that the OUTPUT DOES NOT DEPEND ON THE INPUT, and three rollouts are
        # strong EVIDENCE of it even though they are not a demonstration.
        #
        # Deliberately narrow, and it FAILS OPEN in both directions: the probes only run when the default
        # ITSELF collapsed (so a body that gets anywhere never pays for them), any probe that survives past the
        # floor cancels the refusal, and an evaluator that does not report ``steps_integrated`` (every stubbed
        # one) leaves ``deepest`` at ``None`` and is searched normally. This is the a-priori refusal the biped
        # was wrongly proposed for: the criterion is a measurement of THIS body, not a class it belongs to.
        floor = max(1, int(_DEGENERATE_INTEGRATION_FRAC * deploy_steps))
        if deepest is not None and deepest < floor:
            corners = [{k: _LO[k] for k in _FIT_PARAMS}, {k: _HI[k] for k in _FIT_PARAMS}]
            probes = [_integrated(evaluate_gait(gene, c, steps=deploy_steps)) for c in corners]
            n_rollouts += len(corners)
            reached = [p for p in probes if p is not None]
            deepest = max([deepest, *reached]) if reached else deepest
            if reached and max(reached) < floor:
                out.update({"searched": False, "adopted": False, "n_evals": 0, "n_rollouts": n_rollouts,
                            "degenerate_search": True, "deepest_rollout_steps": deepest,
                            "probe_rollouts": n_rollouts,
                            "reason": (
                                f"NO SEARCH WAS RUN, and nothing here is a finding about this body. Three probe "
                                f"rollouts — the shipped default and both extreme corners of the "
                                f"{len(_FIT_PARAMS)}-parameter box — ALL collapsed by step {deepest} of "
                                f"{deploy_steps} ({deepest / max(1, deploy_steps):.1%} of the horizon). Those are "
                                f"3 samples of a {len(_FIT_PARAMS)}-parameter box and not a proof that nothing "
                                f"between them survives — but when the most and least aggressive operating "
                                f"points available collapse identically, a search over them would spend "
                                f"hundreds of rollouts and then report a decline that reads as a measured "
                                f"negative about the design. Check that this is a body the leg crawl "
                                f"gait can drive at all (crawl_deployment_match says it is driven by "
                                f"{out.get('deployed_controller')}), or give it a learned controller "
                                f"(train_held)")})
                _stash(gene, out)
                return _remember(_ckey, gene, out)
        learned: dict = {}
        best: dict = {}
        attempts = max(1, int(seed_restarts))
        # THE BUDGET IS SET *AFTER* THE DEFAULT PROBE, deliberately: a body whose default already walks never
        # reaches here, so the clock only ever runs on a body that is actually being searched.
        # WHOSE CLOCK. ``create_robot`` can run this fitter up to three times (the authored body, the
        # walkability gate's candidate, then a substituted body), so three independent ceilings would honour
        # "spend up to N seconds" while taking 3N. The rule, in precedence order:
        #
        #   1. an explicit ``budget_s`` argument is THIS fit's own ceiling;
        #   2. a build in progress contributes ITS deadline, and the earlier of the two always wins;
        #   3. with no explicit argument inside a build, the build's clock is taken verbatim — INCLUDING when
        #      it is unbounded. Falling back to the per-fit default there is the bug that made
        #      ``gait_budget_s: 0`` silently cap at 180 s;
        #   4. outside any build, the environment/default chain applies.
        _now = time.monotonic()
        _build = _P.current()
        # ``claim_deadline`` STARTS the build's shared clock here, at the first search that actually needs it,
        # and returns the same instant to every later fit in the build. Reading a deadline fixed at build
        # construction instead let a 253.65 s composer hand this function 0.0 s of budget — see build_progress.
        _bd = _build.claim_deadline() if hasattr(_build, "claim_deadline") else None
        if budget_s is None:
            deadline = _bd if _build is not None else (
                None if _budget_from_env(None) is None else _now + _budget_from_env(None))
        else:
            _own = _budget_from_env(budget_s)
            deadline = None if _own is None else _now + _own
            if _bd is not None:
                deadline = _bd if deadline is None else min(deadline, _bd)
        budget = None if deadline is None else round(max(0.0, deadline - _now), 1)
        out["budget_s"] = budget
        _worst = _t_roll * (warm_evals + max_evals) * attempts
        # THE EVALUATION LEDGER lives on the BUILD, like the clock, so the up-to-three fits of one build share
        # one budget instead of getting one each. ``None`` on every build that did not ask for a cap, and then
        # every line that reads it is a no-op: the fit still runs, in full, by default. That is the
        # non-negotiable half of task #291 and it is why the default here is None rather than a number.
        _evb = getattr(_build, "evals_budget", None) if _build is not None else None
        out["evals_budget"] = _evb
        _P.say(f"fitting an operating point to this body: one {deploy_steps}-step rollout costs {_t_roll:.1f}s "
               f"here, and the search may run up to {(warm_evals + max_evals) * attempts} of them "
               f"(~{_worst / 60:.0f} min)"
               + (f", capped at {budget:.0f}s by the build budget" if budget is not None
                  else " — UNBOUNDED (no build budget set)")
               + (f"; and at {_evb} evaluations for the whole build"
                  if getattr(_build, "evals_budget", None) else ""))
        out["estimated_worst_case_s"] = round(_worst, 1)
        attempts_run = 0
        _try = 0
        for _try in range(attempts):
            # NO FIRST-ATTEMPT EXEMPTION HERE, unlike the clock immediately below. The clock forgives its first
            # attempt because wall time is spent by stages this fit does not control (a 253 s composer would
            # otherwise cancel a search that had not started). An evaluation is only ever spent BY a search, so
            # the same exemption would be the 3N bug in another currency: three fits in one build, each granted
            # "one honest look" past an exhausted ledger, run three searches on a budget of zero.
            _rem = _build.evals_remaining() if _build is not None else None
            if _rem is not None and _rem < 1:
                out["stopped_by_eval_budget"] = True
                break
            # BETWEEN attempts as well as inside the search: a restart is a whole fresh budget, and starting one
            # with the clock already spent is the single largest overshoot available. Never on the FIRST
            # attempt: a body deserves one honest look even under a budget already spent by an earlier stage of
            # the same build, and refusing there would make the fit depend on what ran before it.
            if deadline is not None and time.monotonic() >= deadline and _try:
                out["stopped_by_budget"] = True
                break
            attempts_run += 1
            # Decorrelated restart seeds. A failed search is a failed DRAW, not a verdict on the body (see
            # ``seed_restarts`` above), so retry before concluding this body cannot walk. Retry ONLY on failure
            # OR ON A FRAGILE WIN: a body that finds a STURDY op-point first time costs exactly what it did before.
            kw = dict(generations=generations, pop=pop, steps=search_steps, deploy_steps=deploy_steps,
                      seed=seed + _try * 1009)
            learned, n_try = _one_search(gene, db=db, bank=bank, warm_evals=warm_evals, max_evals=max_evals,
                                         out=out, kw=kw, deadline=deadline, eval_cap=_rem)
            n_evals += n_try
            if _build is not None:
                # CHARGE WHAT WAS ACTUALLY DRAWN, not what was allowed. An attempt that stops early on a credible
                # walk must not bill the budget for candidates it never evaluated, or the SECOND fit of the build
                # would be refused a search this one had left it room for.
                _build.spend_evals(n_try)
            n_rollouts += int(learned.get("n_rollouts_attempt") or learned.get("n_rollouts") or 0)
            _att = learned.get("deploy_steps_integrated")
            if _att is not None:
                deepest = int(_att) if deepest is None else max(deepest, int(_att))
            if not learned.get("beats_default"):
                continue
            _ensure_margin(gene, learned, deploy_steps)
            if not best or _rank(learned) > _rank(best):
                best = learned
            if learned.get("robustness_rel") is not None:
                break                    # a controller, not one lucky float -> stop paying for further draws
        # ATTEMPTS THAT RAN, not the index the loop reached. Breaking out on the clock at ``_try == 2``
        # left this reporting "3 seed attempt(s)" for two searches -- a count of work that was never done,
        # inside the very sentence that has to say how much WAS.
        out["seed_attempts"] = max(1, attempts_run)
        learned = best or learned
        out["searched"] = True
        out.update({k: learned.get(k) for k in ("forward_m", "n_evals", "stopped_reason", "beats_default",
                                                "credible", "survived", "reused_prior", "banked_skill",
                                                "search_robust_frac", "search_robust_rel")})
        out["n_evals"] = n_evals
        out["n_rollouts"] = n_rollouts
        if learned.get("beats_default"):
            # STORE WHAT WAS VERIFIED. This used to round to 4 decimals on the way into the cache, so the body
            # deployed a controller the fitter had never measured. MEASURED 2026-08-01 on the grounded authored
            # cat (seed 0): validated freq 2.21322363 -> +0.958 m CREDIBLE WALK; stored freq 2.2132 -> +0.500 m
            # FELL by ROLL-OVER. A 2.4e-5 relative change in step frequency flipped the verdict — these walks sit
            # on a knife edge, so the exact float that was verified is the only one safe to ship.
            params = {k: float(learned["params"][k]) for k in _FIT_PARAMS if k in learned["params"]}
            # RE-VERIFY THE STORED CONTROLLER (belt and braces). Adoption must never accept anything
            # ``gait_quality.classify`` rejects; the adoption gate in ``learn_gait_flywheel`` already requires
            # `credible`, and the disagreement seen on the cat came from shipping different numbers than were
            # gated, not from the gate. Confirming the stored dict closes that class of drift for good.
            confirm = evaluate_gait(gene, params, steps=deploy_steps)
            _ci = _integrated(confirm)
            if _ci is not None:
                deepest = _ci if deepest is None else max(deepest, _ci)
            out["confirm_forward_m"] = round(float(confirm["forward"]), 4)
            out["confirm_verdict"] = confirm.get("verdict")
            out["confirm_rates"] = confirm.get("rates")          # the rate profile behind the verdict, not just a total
            out["holds_rate"] = confirm.get("holds_rate")
            if not (bool(confirm["survived"]) and bool(confirm.get("credible", False))):
                out["adopted"] = False
                out["reason"] = (f"searched {n_evals} gaits; the winner did not reproduce as a credible walk when "
                                 f"re-measured from the stored parameters ({confirm.get('verdict')}) — not adopted")
                _stash(gene, out)
                return _remember(_ckey, gene, out)
            # THE ERROR BAR, measured during the search (``_ensure_margin``) so it could STEER it rather than
            # merely annotate the result. A verdict with no robustness number cannot distinguish a controller
            # from one lucky float, and this fitter has shipped both under the same two words.
            _ensure_margin(gene, learned, deploy_steps)
            rob = dict(learned.get("robustness") or
                       {"robustness_rel": learned.get("robustness_rel"),
                        "probes": learned.get("robustness_probes") or {},
                        "per_param_rel": learned.get("robustness_per_param")})
            out["robustness_rel"] = rob["robustness_rel"]
            out["robustness_probes"] = rob["probes"]
            # THE PER-PARAMETER ERROR BAR, which is the half of this that a reader can act on: it is what
            # separates "this body walks, keep its step frequency within 10%" from "this body walks, and one
            # part in 10^5 of step frequency rolls it over". Both used to print CREDIBLE WALK and nothing else.
            out["robustness_per_param"] = rob["per_param_rel"]
            out["robustness_note"] = margin_sentence(rob)
            out["fragile"] = rob["robustness_rel"] is None
            md = dict(getattr(gene, "metadata", None) or {})
            md["gait_params"] = params
            gene.metadata = md
            out["adopted"] = True
            out["params"] = params
            out["deepest_rollout_steps"] = deepest      # reported on BOTH outcomes; an adoption earns its horizon
            out["degenerate_search"] = False
            _rob = (margin_sentence(rob) if rob["robustness_rel"] is not None else
                    f"is FRAGILE — no perturbed copy of it survives even a {min(_ROBUST_LADDER):g} relative "
                    f"change ({rob['probes']}), a sturdier point was looked for over "
                    f"{out.get('seed_attempts')} seed attempt(s) and not found, and it is NOT banked for reuse "
                    f"({margin_sentence(rob, head=False)})")
            out["reason"] = (f"searched {n_evals} gaits ({n_rollouts} physics rollouts) for this body over "
                             f"{out.get('seed_attempts')} seed attempt(s); adopted one that is still walking at "
                             f"step {deploy_steps} ({out['confirm_forward_m']} m, rate {_rate_str(confirm)}) vs "
                             f"the default's {out['default_forward_m']} m; it {_rob}"
                             + (" — and the search was cut short by the build budget, so a longer one may find "
                                "a better point (this one is verified either way)"
                                if out.get("stopped_by_budget") or out.get("stopped_by_eval_budget") else ""))
        else:
            # THE HORIZON IN THE SENTENCE HAS TO HAVE BEEN REACHED BY SOMETHING. A decline is a claim about the
            # body, so it may only quote a horizon some rollout actually integrated. When the deepest one
            # collapsed in the first couple of percent, the honest report is that the search measured NOTHING —
            # 379 rollouts in 1.3 s is arithmetically impossible for real physics, and until now nothing noticed
            # (docs/body_vs_controller_ruling.md D1). This is the GENERAL net: it does not know or care why the
            # body dies instantly, only that a horizon nobody reached must not be quoted as one that was.
            degenerate = deepest is not None and deepest < floor
            out["deepest_rollout_steps"] = deepest
            out["degenerate_search"] = bool(degenerate)
            if degenerate:
                out["reason"] = (
                    f"searched {n_evals} gaits ({n_rollouts} physics rollouts) for this body over "
                    f"{out.get('seed_attempts')} seed attempt(s) AND MEASURED NOTHING: the deepest rollout "
                    f"collapsed at step {deepest} of {deploy_steps} "
                    f"({deepest / max(1, deploy_steps):.1%} of the horizon), so no candidate was ever judged at "
                    f"the horizon this result is about. NOTHING HERE IS A FINDING about whether this body can "
                    f"walk — a search whose rollouts all die on contact cannot tell two operating points apart. "
                    f"The body is driven by {out.get('deployed_controller')}; check that it is a body the leg "
                    f"crawl gait can drive at all (crawl_deployment_match) before reading this as a negative")
            elif out.get("stopped_by_budget"):
                # A SEARCH THE CLOCK STOPPED IS A SEARCH THAT DID NOT FINISH — never a finding about the body.
                # This is the same rule as the degeneracy net one branch up and as the applicability refusal
                # above it, applied to the third way a search can fail to be a measurement. Saying "none was
                # still walking at the horizon" here would be exactly the sentence this whole module exists to
                # stop: a budget the customer set, reported back to them as a fact about their robot.
                out["reason"] = (
                    f"THE SEARCH WAS STOPPED BY THE BUILD BUDGET, not by an answer. It ran {n_evals} of a "
                    f"possible {(warm_evals + max_evals) * attempts} gaits ({n_rollouts} physics rollouts) over "
                    f"{out.get('seed_attempts')} of {attempts} seed attempt(s) in {out.get('budget_s'):.0f}s and "
                    f"had not found an operating point that beats the shipped default. NOTHING HERE SAYS THIS "
                    f"BODY CANNOT WALK — the remaining budget was never spent. One rollout costs "
                    f"{out.get('rollout_s')}s on this body, so finishing needs roughly "
                    f"{out.get('estimated_worst_case_s', 0) / 60:.0f} min: re-run with a larger "
                    f"gait_budget_s (or 0 for unbounded) to complete it. The body ships on the shipped default "
                    f"gait meanwhile")
            elif out.get("stopped_by_eval_budget"):
                # THE SAME RULE IN THE OTHER CURRENCY. A caller who capped the search at N evaluations gets back
                # "your cap stopped this", never "none was still walking" — the second sentence is a measurement
                # this build did not make. Unlike the clock, this one reproduces: the same cap on any machine
                # cuts the search at the same candidate, which is why it is the budget a test can pin.
                out["reason"] = (
                    f"THE SEARCH WAS STOPPED BY THE EVALUATION BUDGET, not by an answer. It ran {n_evals} of a "
                    f"possible {(warm_evals + max_evals) * attempts} gaits ({n_rollouts} physics rollouts) over "
                    f"{out.get('seed_attempts')} of {attempts} seed attempt(s), capped at "
                    f"{out.get('evals_budget')} evaluations for the whole build, and had not found an operating "
                    f"point that beats the shipped default. NOTHING HERE SAYS THIS BODY CANNOT WALK — the "
                    f"candidates past the cap were never drawn. Re-run with a larger gait_max_evals (or omit it "
                    f"for the full search) to complete it. The body ships on the shipped default gait meanwhile")
            else:
                _depth = ("" if deepest is None else
                          f" (the deepest rollout reached step {deepest} of {deploy_steps})")
                out["reason"] = (f"searched {n_evals} gaits ({n_rollouts} physics rollouts) for this body over "
                                 f"{out.get('seed_attempts')} seed attempt(s); none was still walking at the "
                                 f"{deploy_steps}-step horizon{_depth}")
    except Exception as exc:  # noqa: BLE001 - fitting an op-point must never BLOCK a build...
        # ...but it must never masquerade as one either. A swallowed KeyError used to return in 0 s looking
        # exactly like "searched, adopted nothing", so a crash read as a considered decline — the same artefact
        # this function exists to remove. Callers and the customer-facing disclosure now see ok=False + error.
        _LOG.warning("fit_gait_for_body failed for gene %s", getattr(gene, "id", "?"), exc_info=True)
        out["ok"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["reason"] = (f"gait fit FAILED for this body ({type(exc).__name__}) — this is an error, not a finding: "
                         f"the body was never measured, so nothing here says it cannot walk")
    _stash(gene, out)
    return _remember(_ckey, gene, out)


def _rate_str(res: dict) -> str:
    """"1.09 -> 0.87 m/1000 steps" — the travel RATE at the start and the end of the judged run, which is the
    thing a net displacement hides. Empty when the horizon was too short to have a profile."""
    rates = res.get("rates") or {}
    if len(rates) < 2:
        return "rate not measurable at this horizon"
    ks = sorted(int(k) for k in rates)
    return f"{rates[ks[0]]:+.2f} -> {rates[ks[-1]]:+.2f} m/1000 steps"


def _ensure_margin(gene, learned: dict, steps: int) -> float | None:
    """The winner's robustness margin, measured ONCE and carried on the result dict.

    ``learn_gait_flywheel`` measures it at the bank gate, so on the normal path this is a lookup. It is
    re-measured only for a caller (or a test) whose search result predates the field — never guessed, because a
    missing error bar defaulting to "fine" is precisely the failure mode this whole task exists to remove.
    """
    if "robustness_rel" not in learned:
        rob = robustness_margin(gene, learned.get("params") or {}, steps=steps)
        learned["robustness_rel"] = rob["robustness_rel"]
        learned["robustness_probes"] = rob["probes"]
        learned["robustness_per_param"] = rob.get("per_param_rel")
        learned["robustness_note"] = margin_sentence(rob)
        learned["robustness"] = dict(rob)
        learned["n_rollouts_margin"] = int(rob.get("n_rollouts") or 0)
    return learned.get("robustness_rel")


def _rank(learned: dict) -> tuple:
    """Order two search winners: STURDINESS FIRST, distance only as a tie-break.

    Ranking by distance is what picked the fall. MEASURED 2026-08-02 (task #267) on the grounded authored cat,
    the full 96-evaluation budget at seed 0 produced exactly two credible candidates out of 97, and the one that
    travelled 0.16% further (3.817 m vs 3.811 m) was the fragile one — 0/4 perturbed copies surviving against
    3/4 for its near-twin. A selection rule that reads the last 6 mm of displacement will take the knife edge
    every time, so displacement is only consulted once sturdiness has tied.
    """
    if not learned.get("beats_default"):
        return (-2.0, 0.0)
    rel = learned.get("robustness_rel")
    # Signed, for the same reason as everywhere else in this file: a tie-break on |distance| ranks a backward
    # walk above a forward one. Latent — this line is only reached once ``beats_default`` holds, and that now
    # requires a signed win over the default by a credible winner — and written signed so it stays latent.
    return (float(rel) if rel is not None else -1.0, float(learned.get("forward_m") or 0.0))


def _one_search(gene, *, db, bank: bool, warm_evals: int, max_evals: int, out: dict, kw: dict,
                deadline: float | None = None, eval_cap: int | None = None):
    """One SHORT first pass plus the full-budget re-run. Returns ``(learned, n_evals_used)``.

    ``eval_cap`` is the build's REMAINING evaluation budget (``None`` = unbounded, which is the default and the
    only thing a build that never asked for a cap can see). It bounds both passes together rather than each, and
    it is reported through ``out['stopped_by_eval_budget']`` whenever it actually bit — a truncated search is a
    search that did not finish, and the rule this module is built on is that such a search may never come back to
    the customer as a finding about their body.

    ``warm_evals`` is small (24) because a GOOD PRIOR should win almost immediately; ``max_evals`` (96) is the
    real budget. The re-run used to be gated on ``reused_prior``, which meant a body with NO prior — every cold
    start, and every hermetic run with hints disabled — was searched with the warm budget and never got the
    other 96. That is a quarter of the intended budget spent on the hardest case, and it is a DECLINE-SHAPED
    bug: the fitter reports "none was still walking", which reads as a fact about the body.

    It only became load-bearing when the settling gate made `credible` mean "walks" (task #267): on the grounded
    authored cat at the 6000-step horizon, 24 evals x 3 seeds finds nothing and declines, while the full budget
    finds a walk. Declining a body that demonstrably walks is the same dishonesty pointed the other way.

    A FRAGILE WIN IS ALSO A REASON TO KEEP LOOKING, which is why the re-run is no longer conditional on failure.
    An earlier revision of this docstring claimed the full-budget settling-aware search finds the cat
    ``freq 2.709 / kp 70.93``, "8/8 robust at BOTH rel 1e-3 and 1e-2". RE-MEASURED 2026-08-02, that is HALF true
    and the false half mattered: the POINT is real and verifies exactly as described (8/8 at both rungs, flat
    rate 0.547 -> 0.546, still walking at 12000 steps, 6.75 m) — but THIS SEARCH DOES NOT FIND IT. Cold at seeds
    0 / 1009 / 2018 it returns ``freq 1.700 / kp 94.27``: 0/8 at both rungs and on the floor by step 8754. Two
    of those three seeds find no credible candidate at all, and the full 96 evals at seed 0 turn up exactly two
    credible candidates out of 97. So the budget is not the binding constraint — reachability is, and quoting a
    point the code cannot reach as though it were the code's output is the same class of false claim as quoting
    a verdict from a horizon that ends before the fall.
    """
    _warm = warm_evals if eval_cap is None else max(1, min(int(warm_evals), int(eval_cap)))
    if eval_cap is not None and _warm < warm_evals:
        out["stopped_by_eval_budget"] = True
    learned = _open_db_and_learn(gene, db=db, bank=bank, max_evals=_warm, deadline=deadline, **kw)
    n_evals = int(learned.get("n_evals") or 0)
    # THE CLOCK'S VERDICT IS TRACKED SEPARATELY FROM THE WINNER'S. ``learned`` ends up being whichever pass
    # RANKED higher, so reading the stop reason off it alone loses the case where the COLD re-run is the one the
    # budget cut -- and the caller would then print a decline for a search that never finished, the exact
    # sentence this budget exists not to produce.
    out_of_time = str(learned.get("stopped_reason")) == "budget"
    rollouts = int(learned.get("n_rollouts") or 0)
    # A FRAGILE WIN COUNTS AS AN EMPTY ONE HERE. The warm pass can end in 1 evaluation because the recalled
    # prior IS the answer — and MEASURED 2026-08-02 on the grounded authored cat, the prior the bank held was
    # 0/8 robust and falls at step 8754, so "won at evaluation 1 in 3 s" was the corpus handing back a fall it
    # had memorised. Spending the real budget to look for something sturdier is the point of having a budget.
    if not learned.get("beats_default") or _ensure_margin(gene, learned, int(kw["deploy_steps"])) is None:
        # SCREEN THE PRIOR AT THE SEARCH LEVEL. This module's contract is that "a bad prior must never hurt",
        # and gait_search states that an injected prior "is simply one of `pop` candidates and does no harm".
        # MEASURED 2026-08-01 on the grounded authored dog, that is false: the prior is re-injected as
        # candidate 0 of EVERY generation and is selected into the ELITE set, which drags the CEM mean with
        # it. Warm (prior kp 57.0, freq 1.31): 96 evaluations, best -0.075 m, nothing adopted. Cold, same
        # body, same seed: credible at evaluation 25, +1.715 m, adopted. The prior did not fail to help — it
        # actively prevented the walk this body can do. So when a warm run comes back empty, re-run it cold
        # and keep whichever actually beat the default. (The deeper fix belongs in gait_search's elite
        # selection and is not this change.) With no prior to screen, the same re-run is simply the rest of
        # the budget, which is why it is no longer conditional on one having been recalled.
        _cold_cap = max_evals if eval_cap is None else min(int(max_evals), int(eval_cap) - n_evals)
        if eval_cap is not None and _cold_cap < max_evals:
            out["stopped_by_eval_budget"] = True
        if _cold_cap < 1:
            # THE LEDGER IS EMPTY MID-ATTEMPT. Returning here rather than running a 1-candidate re-run keeps the
            # cap honest; the warm pass's answer stands and the disclosure above says the search was cut.
            learned["n_rollouts_attempt"] = rollouts
            if out_of_time:
                out["stopped_by_budget"] = True
            return learned, n_evals
        cold = _open_db_and_learn(gene, db=db, bank=bank, max_evals=_cold_cap, recall=False,
                                  deadline=deadline, **kw)
        n_evals += int(cold.get("n_evals") or 0)
        rollouts += int(cold.get("n_rollouts") or 0)
        out_of_time = out_of_time or str(cold.get("stopped_reason")) == "budget"
        if learned.get("reused_prior"):
            out["prior_screened_out"] = True
        if cold.get("beats_default"):
            _ensure_margin(gene, cold, int(kw["deploy_steps"]))
        if _rank(cold) > _rank(learned):     # sturdiness first; see ``_rank`` for why distance cannot lead
            learned = cold
    # BOTH passes' physics, not just the winner's: this is what the attempt COST, which is the number a caller
    # deciding whether it can afford robustness-aware search needs. Each pass's own count already includes the
    # margin ladder it paid for, so this is not double-counted.
    learned["n_rollouts_attempt"] = rollouts
    if out_of_time:
        out["stopped_by_budget"] = True
    return learned, n_evals


def _budget_from_env(explicit: float | None) -> float | None:
    """Resolve the wall-clock ceiling for one fit: explicit argument > ``VIRTUROID_GAIT_FIT_BUDGET_S`` > default.

    ``0`` (or anything <= 0) means UNBOUNDED — search until the eval budget is spent, which is exactly today's
    behaviour and what the test suite pins, so no assertion about whether a body walks becomes a function of how
    fast the machine running it happens to be.
    """
    import os

    if explicit is None:
        raw = os.environ.get("VIRTUROID_GAIT_FIT_BUDGET_S")
        try:
            explicit = _DEFAULT_FIT_BUDGET_S if raw is None else float(raw)
        except ValueError:
            explicit = _DEFAULT_FIT_BUDGET_S
    return None if float(explicit) <= 0 else float(explicit)


def _stash(gene, out: dict) -> None:
    try:
        md = dict(getattr(gene, "metadata", None) or {})
        md["gait_fit"] = dict(out)
        gene.metadata = md
    except Exception:  # noqa: BLE001 - disclosure is best-effort
        pass


def _open_db_and_learn(gene, *, db, bank: bool, recall: bool = True, deadline: float | None = None,
                       **kw) -> dict:
    """Run ``learn_gait_flywheel`` against the product memory DB, falling back to a DB-less search.

    The flywheel contract is consult-before / bank-after, so the build path uses the real corpus. But a build must
    not fail because the memory directory is unavailable (or hints are disabled for a hermetic bench run), hence
    the in-memory fallback.
    """
    import os

    if db is not None:
        return learn_gait_flywheel(gene, db, workers=1, bank=bank, stop_on_credible=True, recall=recall,
                                   deadline=deadline, **kw)
    if os.environ.get("VIRTUROID_DISABLE_GAIT_HINTS") == "1":
        # The bench measures the composer+compiler deterministically; a corpus that grew between runs would make
        # the gate number float, which is the exact flicker VIRTUROID_DISABLE_GAIT_HINTS was introduced to stop.
        return learn_gait_flywheel(gene, None, workers=1, bank=False, stop_on_credible=True, recall=False,
                                   deadline=deadline, **kw)
    try:
        from virturoid.services.agent_tools import safe_build_path
        from virturoid.services.memory_db import MemoryDB
        mem = safe_build_path(None, "memory")
        mem.mkdir(parents=True, exist_ok=True)
        with MemoryDB(mem / "virturoid_memory.db") as _db:
            return learn_gait_flywheel(gene, _db, workers=1, bank=bank, stop_on_credible=True, recall=recall,
                                       deadline=deadline, **kw)
    except Exception:  # noqa: BLE001 - no corpus is a cold start, not a failure
        return learn_gait_flywheel(gene, None, workers=1, bank=False, stop_on_credible=True, recall=False,
                                   deadline=deadline, **kw)
