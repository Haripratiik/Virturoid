"""Design-Bench — measure LLM design quality as a staged validity funnel (master_plan_v6 §8.1, WS-A).

Strict LLM-first made the model's design quality the product; T9 measured *coverage* (did we build something?),
never *quality* (is what we built valid, credible, faithful, diverse?). This module is that measurement — the
converged harness shape from the research: **a staged funnel with deterministic physics as the ONLY selection
authority**, reported against a single absolute denominator so no stage can hide behind survivorship.

The funnel (each stage strictly downstream of the last, all rates over ``n_attempts``):

    schema_valid@1  →  compile@1  →  verdict@1 (HEADLINE)  →  fitness (ref/efficiency-normalised)

plus the guards the research made non-negotiable: **diversity** (embedding spread + duplicate rate — the
RoboMorph mode-collapse alarm), **spec-faithfulness** (deterministic checks of every parseable constraint the
prompt literally asserted), **verdict fragility** (does a credible verdict survive a mild mass perturbation, or
was it a knife-edge sim exploit?), **quality-per-physics-eval** (the real cost axis), and a **per-model matrix**
(aggregate numbers across LLM tiers are meaningless — the customer brings the brain).

Physics is judge; nothing here is scored on assertion. It designs no robot — it only grades designs it is handed.
"""
from __future__ import annotations

import math

# Pinned so the number is reproducible across runs/machines (harness rule §8.1).
PINNED_SIM_CONFIG = {"legged_steps": 800, "mobile_steps": 500, "manip_steps": 400,
                     "aquatic_steps": 1500, "aerial_steps": 1600, "fragility_mass_scale": 1.10}

# The POSITIVE verdict strings the honesty engines actually emit (grepped from ai_native_tools, not guessed):
# legged→CREDIBLE, mobile→DRIVES, manipulator→ARTICULATES, grasp→PICKS UP, aquatic→SWIMS, aerial→FLIES,
# serpentine→CRAWLS. Anything else (SLIDE/LURCHES/TIPPED/STUCK/STANDS…) is a non-credible verdict.
_CREDIBLE_PREFIXES = ("CREDIBLE", "DRIVES", "ARTICULATES", "PICKS UP", "SWIMS", "FLIES", "CRAWLS")

# ---------------------------------------------------------------- the per-case OUTCOME vocabulary
# A boolean per case cannot say the one thing the bench most needs to say. MEASURED on the live battery
# (2026-08-08 + 2026-08-07 arms): the product REFUSED elephant__appearance ("center of mass falls outside the
# foot support polygon"), palletizer__construction ("5 joint(s) near/over their static hold torque"),
# mobile_manip__appearance and starfish__appearance. Under a boolean those land as ``False`` -- indistinguishable
# from a body that was built and fell over -- so the floor could only record them as failures, and the offline
# floor recorded three of the same prompt ids as ``true``. A gate written on that boolean protects bodies the
# product declines to build, and would read a HONESTY IMPROVEMENT (a new refusal) as a regression.
#
# So the recorded expectation is tri-state. ``refused`` is a first-class CORRECT outcome; only ``failed`` is a
# defect. The headline ``verdict@1`` is deliberately left alone -- a refused prompt is still an UNSERVED prompt,
# so it must not count as credible -- and the new ``correct@1`` sits beside it for the honesty question.
OUTCOME_CREDIBLE = "credible"        #: built, and the physics verdict is credible
OUTCOME_REFUSED = "refused"          #: the product declined to produce a body (strict-mode grounding refusal)
OUTCOME_FAILED = "failed"            #: a design exists (or was attempted) and does not earn a credible verdict
OUTCOME_UNVERIFIED = "unverified"    #: structural run only (verify=False); no physics was asked for
OUTCOMES = (OUTCOME_CREDIBLE, OUTCOME_REFUSED, OUTCOME_FAILED, OUTCOME_UNVERIFIED)


# ---------------------------------------------------------------- physics verdict (gene-direct)
def _verdict_for_gene(gene, *, quick: bool = True) -> dict:
    """The un-gameable motion verdict for a gene, dispatched by STRUCTURE (robot_kind) — the same honesty engine
    ``verify_robot`` uses, but gene-direct so the bench needs no session plumbing. Returns the raw verdict dict."""
    from virturoid.services.ai_native_tools import (_honest_drive, _honest_fly, _honest_gait, _honest_reach,
                                                    _honest_swim)
    from virturoid.services.task_matched_eval import robot_kind
    kind = robot_kind(gene)
    md = getattr(gene, "metadata", None) or {}
    C = PINNED_SIM_CONFIG
    try:
        if kind == "aerial" or md.get("rotor_offsets"):
            res = _honest_fly(gene, steps=C["aerial_steps"])
        elif kind == "aquatic" or md.get("aquatic"):
            res = _honest_swim(gene, steps=C["aquatic_steps"])
        elif kind == "legged":
            res = _honest_gait(gene, steps=C["legged_steps"], render=False, tag="bench")
        elif kind == "mobile":
            res = _honest_drive(gene, steps=C["mobile_steps"])
        elif kind == "manipulator":
            res = _honest_reach(gene, steps=C["manip_steps"])
        else:
            res = {"kind": kind, "verdict": f"{kind.upper()}: no locomotion verdict", "credible_walk": False}
    except Exception as exc:  # noqa: BLE001 - an odd body yields an honest failed verdict, never a crash
        res = {"kind": kind, "verdict": f"could not simulate ({type(exc).__name__})", "error": str(exc)[:160]}
    res.setdefault("kind", kind)
    return res


def _is_credible(res: dict) -> bool:
    return bool(str(res.get("verdict", "")).upper().startswith(_CREDIBLE_PREFIXES)
                or res.get("credible") or res.get("credible_walk"))


def _fitness_raw(res: dict, kind: str) -> float:
    """The kind-appropriate performance scalar from a verdict (metres of forward travel / reach)."""
    for key in (("forward_m", "distance_m") if kind in ("legged", "aquatic", "aerial") else
                ("distance_m", "forward_m") if kind == "mobile" else
                ("reach_m", "planar_m")):
        v = res.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return 0.0


# reference performance per kind (metres) — a credible real-robot-class scale, so fitness is normalised to a
# human/robot reference (Eureka's human-normalised score), never an absolute that rewards a big body.
_REFERENCE_M = {"legged": 1.5, "mobile": 0.8, "manipulator": 0.5, "aquatic": 0.8, "aerial": 1.0}


def _volume(gene) -> float:
    """A monotone body-volume proxy (sum of per-segment solid volumes, m^3) — the denominator for the
    efficiency-normalised fitness guard against big-body gaming (RoboMoRe)."""
    vol = 0.0
    for s in gene.segments:
        L = float(getattr(s, "length_m", 0.0)); r = float(getattr(s, "radius_m", 0.0))
        vol += math.pi * r * r * L + (4.0 / 3.0) * math.pi * r ** 3   # capsule ~ cylinder + end caps
    return max(vol, 1e-6)


# ---------------------------------------------------------------- spec-faithfulness (§8.1.9, deterministic)
def _wheel_count(gene) -> int:
    return sum(1 for s in gene.segments
               if getattr(s, "shape", None) == "cylinder" and s.joint_type == "revolute")


def _spec_faithfulness(gene, constraints: dict) -> dict:
    """Deterministic rule checks for every parseable constraint the prompt asserted. Returns per-constraint
    booleans + an overall ``faithful`` (all satisfied). Only constraints actually present are scored."""
    from virturoid.services.heldout_set import _leg_chain_count
    from virturoid.services.task_matched_eval import robot_kind
    checks: dict[str, bool] = {}
    dof = len(gene.actuated_joints())
    if "kind" in constraints:
        checks["kind"] = robot_kind(gene) == constraints["kind"]
    if "wheels" in constraints:
        checks["wheels"] = _wheel_count(gene) == int(constraints["wheels"])
    if "legs" in constraints:
        # count non-wheel limb chains (legs/arms) — the structural "how many limbs" signal
        checks["legs"] = _leg_chain_count(gene) == int(constraints["legs"])
    if "dof" in constraints:
        checks["dof"] = dof == int(constraints["dof"])
    if "dof_min" in constraints:
        checks["dof_min"] = dof >= int(constraints["dof_min"])
    if "dof_max" in constraints:
        checks["dof_max"] = dof <= int(constraints["dof_max"])
    if "arms" in constraints:
        checks["arms"] = (gene.end_effector_type in ("gripper", "hand")) or dof >= 3
    if "size_max_m" in constraints:
        checks["size_max_m"] = max((float(getattr(s, "length_m", 0.0)) for s in gene.segments), default=0.0) \
            <= float(constraints["size_max_m"])
    return {"checks": checks, "faithful": all(checks.values()) if checks else None, "n_checks": len(checks)}


# ---------------------------------------------------------------- per-design evaluation
def evaluate_design(gene, *, constraints: dict | None = None, verify: bool = True,
                    fragility: bool = False, refused: bool = False,
                    refusal_reason: str | None = None) -> dict:
    """Grade ONE design through the funnel. ``gene=None`` records a design failure honestly (schema stage).

    ``refused=True`` says the absent design is a REFUSAL — strict mode declining to produce an unsound body —
    rather than a malformed one. It is still not credible (the prompt went unserved, so ``verdict@1`` must not
    reward it) but it is routed to ``error_class="refused"`` and ``outcome="refused"``, because "we correctly
    declined" and "we emitted a broken kinematic tree" are different facts about the product and a gate written
    on the boolean alone cannot tell them apart.

    Returns a per-design row with the boolean stage passes, the physics fitness, the spec-faithfulness checks,
    ``error_class`` (the first failing stage — the routing signal for grounded repair, §8.2) and ``outcome``
    (the tri-state the per-case floors are recorded in)."""
    from virturoid.services.task_matched_eval import robot_kind
    row: dict = {"schema_valid": False, "compiles": False, "credible": False,
                 "fitness_raw": 0.0, "fitness_ref_norm": 0.0, "fitness_eff": 0.0,
                 "error_class": "schema", "kind": None, "n_physics_evals": 0,
                 "outcome": OUTCOME_FAILED, "refused": bool(refused)}
    if gene is None:
        if refused:
            row["error_class"] = "refused"
            row["outcome"] = OUTCOME_REFUSED
            row["refusal_reason"] = (str(refusal_reason)[:300] if refusal_reason else None)
        return row
    # 1) schema_valid@1 — a valid, buildable kinematic tree
    issues = gene.validate()
    if issues:
        row["error_class"] = "schema"; row["schema_issue"] = issues[0]
        return row
    row["schema_valid"] = True
    row["kind"] = robot_kind(gene)
    row["error_class"] = "compile"
    # 2) compile@1 — MJCF compiles + MuJoCo loads with finite mass (structural, no stepping)
    try:
        from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
        xml = compile_gene_to_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene))
        import mujoco
        m = mujoco.MjModel.from_xml_string(xml)
        if not math.isfinite(float(m.body_mass.sum())) or float(m.body_mass.sum()) <= 0:
            raise ValueError("non-finite/zero total mass")
        row["compiles"] = True
        row["error_class"] = "verdict"
    except Exception as exc:  # noqa: BLE001
        row["error_class"] = "compile"; row["compile_error"] = f"{type(exc).__name__}: {exc}"[:160]
        return row
    if constraints:
        row["spec"] = _spec_faithfulness(gene, constraints)
    if not verify:
        row["error_class"] = None
        row["outcome"] = OUTCOME_UNVERIFIED     # no physics was asked for; "failed" would be a claim, not a fact
        return row
    # 3) verdict@1 — the un-gameable physics verdict
    res = _verdict_for_gene(gene)
    row["n_physics_evals"] = 1
    row["verdict"] = res.get("verdict")
    # keep the honesty-engine diagnostics (roll/pitch/support/cadence…) so grounded_repair can give TYPED feedback
    row["verdict_detail"] = {k: res[k] for k in ("roll_max_deg", "pitch_max_deg", "support_frac", "cadence",
                             "height_ratio", "wheel_ground_contact_frac", "wheel_spin_radps", "reach_span_m")
                             if k in res and res[k] is not None}
    row["credible"] = _is_credible(res)
    row["fitness_raw"] = _fitness_raw(res, row["kind"])
    ref = _REFERENCE_M.get(row["kind"], 1.0)
    row["fitness_ref_norm"] = round(min(row["fitness_raw"] / ref, 2.0), 4) if ref else 0.0
    row["fitness_eff"] = round(row["fitness_raw"] / _volume(gene), 4)
    row["error_class"] = None if row["credible"] else "verdict"
    row["outcome"] = OUTCOME_CREDIBLE if row["credible"] else OUTCOME_FAILED
    # 4) verdict fragility — did a credible verdict survive a mild mass perturbation? (knife-edge exploit alarm)
    if fragility and row["credible"]:
        try:
            pert = _perturb_mass(gene, PINNED_SIM_CONFIG["fragility_mass_scale"])
            res2 = _verdict_for_gene(pert)
            row["n_physics_evals"] += 1
            row["fragile"] = bool(not _is_credible(res2))   # credible → not-credible under perturbation = fragile
        except Exception:  # noqa: BLE001
            row["fragile"] = None
    return row


def _perturb_mass(gene, scale: float):
    """A deterministic mild-DR draw: a copy of the gene with every segment mass scaled. (WS-E's certificate DR
    sweep is the full version; this is the bench's cheap fragility probe.)"""
    from dataclasses import replace
    segs = [replace(s, mass_kg=float(getattr(s, "mass_kg", 0.2)) * scale) for s in gene.segments]
    return replace(gene, segments=segs)


# ---------------------------------------------------------------- diversity (the mode-collapse guard §8.1.7)
def diversity_report(genes: list) -> dict:
    """Embedding spread over a set of designs: unique-structure ratio, mean pairwise cosine similarity, and the
    CV of a size feature. A high mean-similarity / low unique-ratio is the mode-collapse alarm."""
    from virturoid.services.heldout_set import body_key
    from virturoid.services.morphology_embedding import embed_gene
    genes = [g for g in genes if g is not None]
    n = len(genes)
    if n == 0:
        return {"n": 0, "unique_ratio": None, "mean_pairwise_similarity": None, "size_cv": None}
    keys = [body_key(g) for g in genes]
    unique_ratio = len(set(keys)) / n
    vecs = [embed_gene(g) for g in genes]

    def _cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na > 0 and nb > 0 else 0.0
    sims = [_cos(vecs[i], vecs[j]) for i in range(n) for j in range(i + 1, n)]
    mean_sim = sum(sims) / len(sims) if sims else None
    sizes = [sum(float(getattr(s, "length_m", 0.0)) for s in g.segments) for g in genes]
    mean_sz = sum(sizes) / n
    var = sum((x - mean_sz) ** 2 for x in sizes) / n
    size_cv = (math.sqrt(var) / mean_sz) if mean_sz > 0 else 0.0
    return {"n": n, "unique_ratio": round(unique_ratio, 4),
            "mean_pairwise_similarity": round(mean_sim, 4) if mean_sim is not None else None,
            "size_cv": round(size_cv, 4)}


# ---------------------------------------------------------------- the aggregated funnel
def run_bench(designs: list[dict], *, model: str = "cassette", verify: bool = True,
              fragility: bool = False) -> dict:
    """Grade a list of design records and aggregate the funnel. Each record: ``{prompt_id, family, phrasing,
    concept?, constraints?, gene}``. Rates are over ``n_attempts`` (the single absolute denominator, §8.1).

    Also reports conditional pass-rates (clearly labelled) for diagnosis, per-family and per-phrasing breakdowns,
    the diversity guard, spec-faithfulness, and quality-per-physics-eval.

    ``model`` labels the row for the matrix and is taken on trust HERE, because this function grades a bare
    list of genes and has no provenance to check it against. The provenance gate lives one level up, in
    :func:`bench_from_cassette`, which reads the label off the cassette's own rows (#208) — so prefer that
    entry point for anything whose number will be published or gated.
    """
    rows = []
    for d in designs:
        r = evaluate_design(d.get("gene"), constraints=d.get("constraints"), verify=verify, fragility=fragility,
                            refused=bool(d.get("refused")), refusal_reason=d.get("refusal_reason"))
        r.update({k: d.get(k) for k in ("prompt_id", "family", "phrasing", "concept")})
        rows.append(r)
    n = len(rows) or 1
    n_schema = sum(1 for r in rows if r["schema_valid"])
    n_compile = sum(1 for r in rows if r["compiles"])
    n_credible = sum(1 for r in rows if r["credible"])
    n_refused = sum(1 for r in rows if r["outcome"] == OUTCOME_REFUSED)
    credible_rows = [r for r in rows if r["credible"]]
    # error-class histogram over ALL attempts (the routing signal for WS-B)
    err_hist: dict[str, int] = {}
    for r in rows:
        err_hist[r["error_class"] or "none"] = err_hist.get(r["error_class"] or "none", 0) + 1
    # spec-faithfulness over designs that carried checkable constraints
    spec_rows = [r for r in rows if r.get("spec") and r["spec"].get("faithful") is not None]
    spec_faithful = sum(1 for r in spec_rows if r["spec"]["faithful"])
    # fragility over credible designs (only when computed)
    frag_rows = [r for r in credible_rows if r.get("fragile") is not None]
    n_fragile = sum(1 for r in frag_rows if r["fragile"])
    n_phys = sum(r["n_physics_evals"] for r in rows)
    genes = [d.get("gene") for d in designs]

    def _rate(x, d):
        return round(x / d, 4) if d else None

    funnel = {
        "model": model, "n_attempts": len(rows),
        # absolute funnel (every rate over n_attempts — the harness-rule denominator)
        "schema_valid@1": _rate(n_schema, len(rows)),
        "compile@1": _rate(n_compile, len(rows)),
        "verdict@1": _rate(n_credible, len(rows)),                       # THE headline
        # PER-CASE outcomes, because the headline rate cannot resolve what it is asked to resolve. The battery is
        # 20 prompts, so verdict@1 moves in steps of exactly 0.05 -- the same size as the gate's own tolerance.
        # A change that fixes one body and breaks another therefore reads as NO CHANGE AT ALL, and a single flip
        # reads as a full-tolerance regression. That ambiguity is not statistical (the cassette is deterministic
        # and hermetic, so this is an exact function of the code) -- it is purely lost information, recovered here.
        # Gate on this map, not the rate, and a regression can name the body it broke.
        "per_case": {str(r.get("prompt_id")): bool(r["credible"]) for r in rows},
        # ...and the boolean above still cannot say the one thing the LIVE battery most needed it to say. A
        # refusal reads as False, exactly like a body that fell over, so a recorded floor could not tell "the
        # product correctly declined" from "the product broke" -- and the offline floor duly marked three
        # REFUSED prompt ids `true`. This is that map, tri-state. ``outcome_regressions`` gates on it.
        "per_case_outcome": {str(r.get("prompt_id")): r["outcome"] for r in rows},
        # The honesty rate beside the capability rate. verdict@1 answers "how often did the customer get a
        # working robot"; correct@1 answers "how often did the product do the right thing" -- which counts an
        # honest refusal, because shipping a topple-prone body instead would have been the WORSE outcome.
        # Two separate numbers on purpose: neither may be quoted for the other, and verdict@1 is unchanged by
        # this addition, so no floor recorded against it is weakened.
        "refused@1": _rate(n_refused, len(rows)),
        "correct@1": _rate(n_credible + n_refused, len(rows)),
        "refusals": {str(r.get("prompt_id")): r.get("refusal_reason")
                     for r in rows if r["outcome"] == OUTCOME_REFUSED},
        # conditional (labelled, for diagnosis only — never the headline)
        "conditional": {"compile|schema": _rate(n_compile, n_schema),
                        "verdict|compile": _rate(n_credible, n_compile)},
        "error_classes": err_hist,
        "fitness": {"ref_norm_mean": round(sum(r["fitness_ref_norm"] for r in credible_rows) / len(credible_rows), 4)
                    if credible_rows else None,
                    "efficiency_mean": round(sum(r["fitness_eff"] for r in credible_rows) / len(credible_rows), 4)
                    if credible_rows else None},
        "diversity": diversity_report(genes),
        "spec_faithfulness": {"rate": _rate(spec_faithful, len(spec_rows)), "n_scored": len(spec_rows)},
        "verdict_fragility": {"rate": _rate(n_fragile, len(frag_rows)), "n_scored": len(frag_rows)}
        if frag_rows else {"rate": None, "n_scored": 0},
        "quality_per_physics_eval": round(n_credible / n_phys, 4) if n_phys else None,
    }
    # per-family + per-phrasing verdict@1 (the style-confound split, §8.1 harness rule)
    def _breakdown(key):
        out: dict[str, dict] = {}
        for r in rows:
            g = r.get(key) or "?"
            out.setdefault(g, {"n": 0, "credible": 0})
            out[g]["n"] += 1
            out[g]["credible"] += int(r["credible"])
        return {g: {"n": v["n"], "verdict@1": _rate(v["credible"], v["n"])} for g, v in out.items()}
    funnel["by_family"] = _breakdown("family")
    funnel["by_phrasing"] = _breakdown("phrasing")
    return funnel


class ProvenanceMismatch(ValueError):
    """A run label claims a generator the scored rows do not support (#208). Raised instead of printed."""


def label_for_mode(mode: str) -> str:
    """The canonical run label for a measured cassette mode. The label is a FUNCTION OF THE DATA, never of a
    CLI flag — ``scripts/design_bench.py`` used to derive it from ``--strict-llm`` on the *current* run, so
    ``--strict-llm`` without ``--record`` printed ``live_llm_v1`` over a replay of the offline fixture."""
    return {"live": "live_llm_v1", "offline": "offline_heuristic_v1",
            "mixed": "mixed_provenance_v1", "empty": "empty_cassette"}.get(mode, f"unknown_{mode}")


def check_label(label: str, mode: str) -> None:
    """Reject a label whose claim outruns the evidence. ``live`` in the label requires every scored design to
    have been authored by a model; a label without it must not sit on a fully-live cassette either — a live
    measurement quietly filed as the offline baseline is the same defect facing the other way."""
    claims_live = "live" in label.lower()
    if claims_live and mode != "live":
        raise ProvenanceMismatch(
            f"label {label!r} claims a live model but the cassette measured mode={mode!r}. "
            f"Only {label_for_mode(mode)!r} (or an explicit non-live label) is honest for these rows.")
    if mode == "live" and not claims_live:
        raise ProvenanceMismatch(
            f"label {label!r} does not say it is live, but every scored design was authored by a model "
            f"(mode={mode!r}). Use {label_for_mode(mode)!r} so the number cannot be read as the offline floor.")


def bench_from_cassette(cassette=None, *, verify: bool = True, fragility: bool = False,
                        model: str | None = None, only_recorded: bool = False):
    """Run Design-Bench over the committed cassette + battery — the deterministic CI entry point.

    HERMETIC by construction: gait-hint recall is disabled for the duration (VIRTUROID_DISABLE_GAIT_HINTS), so
    the gate measures the composer+compiler alone. Before this, verdict@1 floated with whatever gaits the
    session DB happened to contain (measured 0.50 empty vs 0.55 banked, 2026-07-22) and the regression gate
    flickered at its floor depending on test order. The product path keeps hints on; only the bench opts out.

    PROVENANCE (#208): ``model`` now defaults to the label the cassette's own rows justify, and an explicit
    label that contradicts them raises :class:`ProvenanceMismatch`. The funnel carries a ``provenance`` block
    so a reader of the JSON can tell a replay of the deterministic builders from a live-model measurement
    without having to know which flags the run was invoked with.
    """
    import os
    from virturoid.services import design_battery as B
    from virturoid.services.design_cassette import DesignCassette
    cas = cassette or DesignCassette()
    prov = cas.provenance()
    if model is None:
        model = label_for_mode(prov["mode"])
    else:
        check_label(model, prov["mode"])
    designs = []
    excluded_infra: list[str] = []
    for rec in B.battery():
        pid = B.prompt_id(rec)
        # A SUBSET cassette (the small live arm, #208) holds only some prompts. Scoring the absent ones would
        # book 15 phantom schema failures and make the live verdict@1 uncomparable with the full replay, so
        # ``only_recorded`` narrows the denominator instead -- and the funnel says so, loudly, below.
        if only_recorded and not cas.has(pid):
            continue
        # #280: an HTTP 429 is NOT a design failure. MEASURED on the 2026-08-08 live battery -- 14 of 16
        # "refusals" were the org's 50-requests-per-day cap on the fast model, and the funnel dutifully
        # reported verdict@1 = 0.0 for them. That is a number about an API quota wearing the product's label.
        # Transport failures leave the denominator (named, never dropped silently); design refusals stay in it,
        # because a model declining to produce a sound body IS the product's measured behaviour.
        ep = cas.entry_provenance(pid) if cas.has(pid) else None
        if ep and ep["infrastructure_failure"]:
            excluded_infra.append(pid)
            continue
        # A recorded DESIGN REFUSAL is the product declining to build an unsound body. It stays in the
        # denominator (an unserved prompt is unserved) but it is carried through as a refusal, not as a
        # nameless absent gene, so the funnel's tri-state can distinguish it from a malformed design.
        refused = bool(ep and ep["failure_kind"] == "design_refusal")
        designs.append({"prompt_id": pid, "family": rec["family"], "phrasing": rec["phrasing"],
                        "concept": rec["concept"], "constraints": rec.get("constraints"),
                        "gene": cas.get_gene(pid),
                        "refused": refused, "refusal_reason": ep["error"] if refused else None})
    prev = os.environ.get("VIRTUROID_DISABLE_GAIT_HINTS")
    os.environ["VIRTUROID_DISABLE_GAIT_HINTS"] = "1"
    try:
        out = run_bench(designs, model=model, verify=verify, fragility=fragility)
    finally:
        if prev is None:
            os.environ.pop("VIRTUROID_DISABLE_GAIT_HINTS", None)
        else:
            os.environ["VIRTUROID_DISABLE_GAIT_HINTS"] = prev
    out["battery_version"] = B.BATTERY_VERSION
    out["cassette"] = cas.summary()
    # The provenance of the NUMBER, not of the run: which generator authored the designs that were scored.
    n_battery = len(B.battery())
    out["provenance"] = dict(prov, scored_prompt_ids=sorted(d["prompt_id"] for d in designs),
                             n_battery=n_battery, is_subset=len(designs) < n_battery,
                             coverage=round(len(designs) / n_battery, 4) if n_battery else None)
    out["mode"] = prov["mode"]
    out["excluded_infrastructure_failures"] = sorted(excluded_infra)
    if excluded_infra:
        out["infrastructure_warning"] = (
            f"{len(excluded_infra)}/{n_battery} prompts were dropped from the denominator because the recording "
            f"hit TRANSPORT failures (rate limit / timeout / outage), not design failures: "
            f"{', '.join(sorted(excluded_infra))}. Those prompts are UNMEASURED here -- not failed. Re-record "
            f"them before quoting this funnel as a capability statement.")
    if out["provenance"]["is_subset"]:
        out["subset_warning"] = (f"SUBSET: {len(designs)}/{n_battery} battery prompts scored. verdict@1 here is "
                                 f"NOT comparable with a full-battery number -- use diff_funnels, which "
                                 f"restricts to the prompts both runs scored.")
    return out


# ---------------------------------------------------------------- the per-case OUTCOME gate (tri-state)
#: What each recorded outcome PROTECTS, and what counts as losing it. The whole point of the tri-state is that
#: a refusal is a correct outcome with its own regression, distinct from a credible body's:
#:
#:   floor ``credible`` → the body must still walk/drive/reach. Anything else is a loss.
#:   floor ``refused``  → the product must still DECLINE. Falling to ``failed`` means the honesty gate stopped
#:                        firing and we now ship a body that does not work — a real regression, and one the
#:                        boolean floor could not express at all (both sides read False).
#:   floor ``failed``   → tracked debt. Protected against nothing; it cannot get worse.
#:
#: Upward moves are never failures, but they are REPORTED (``improved``) so the floor gets raised deliberately
#: instead of drifting — the same discipline ``known_regressions`` enforces facing the other way.
_OUTCOME_MUST_HOLD = {OUTCOME_CREDIBLE: (OUTCOME_CREDIBLE,),
                      OUTCOME_REFUSED: (OUTCOME_REFUSED, OUTCOME_CREDIBLE),
                      OUTCOME_FAILED: OUTCOMES,
                      OUTCOME_UNVERIFIED: OUTCOMES}
_OUTCOME_RANK = {OUTCOME_FAILED: 0, OUTCOME_REFUSED: 1, OUTCOME_CREDIBLE: 2}


def outcome_regressions(floor: dict, got: dict) -> dict:
    """Compare a recorded tri-state per-case floor against a fresh run's ``per_case_outcome``.

    Returns ``{"broke": {pid: "was -> now"}, "improved": {...}, "missing": [...]}``. ``missing`` is not a
    nicety: a refusal row carries no gene, so its replayed outcome is a pure function of the failure-kind
    classifier — widen ``INFRASTRUCTURE_ERROR_MARKERS`` by one word (say "capacity", which appears in the
    palletizer refusal) and that prompt would be reclassified as transport, dropped from the denominator, and
    vanish from the map instead of failing anything. Absence is therefore a gate condition, not a skip.
    """
    broke: dict[str, str] = {}
    improved: dict[str, str] = {}
    missing = sorted(k for k in floor if k not in got)
    for pid, was in floor.items():
        now = got.get(pid)
        if now is None:
            continue
        if now not in _OUTCOME_MUST_HOLD.get(was, OUTCOMES):
            broke[pid] = f"{was} -> {now}"
        elif _OUTCOME_RANK.get(now, -1) > _OUTCOME_RANK.get(was, -1):
            improved[pid] = f"{was} -> {now}"
    return {"broke": dict(sorted(broke.items())), "improved": dict(sorted(improved.items())), "missing": missing}


#: The production outcome for a prompt the live path has never actually answered (every live attempt died on
#: transport). NOT an outcome — the absence of one. Kept out of ``OUTCOMES`` on purpose so it can never be
#: mistaken for a measurement, and so a floor entry resting on it reads as "no evidence", not "it works".
PRODUCTION_UNMEASURED = "unmeasured"


def capability_reconciliation(offline: dict, production: dict, *, prompt_ids=None) -> dict:
    """Line up an OFFLINE per-case floor against what the LIVE product actually did, prompt by prompt.

    THE DEFECT THIS EXISTS FOR. The offline floor is a floor on the deterministic builders replayed through the
    compiler and the physics — a genuinely useful gate, and NOT a statement about the product. But it is a map
    of prompt ids to ``true``, published next to ``verdict@1``, and it was read as one: three of its ``true``
    entries (``elephant__appearance``, ``palletizer__construction``, ``mobile_manip__appearance``) are prompts
    the live product **refuses**, correctly, because the anatomy graph it proposed put the centre of mass
    outside the support polygon or drove joints past their static hold torque. Nothing in the artifact said so.

    So the claim and the evidence are now stored together and checked. Every offline ``credible`` is classified:

      * ``corroborated`` — the live product built it and it earned a credible verdict. Only these are capability.
      * ``contradicted`` — the live product refused it, or built something that does not work.
      * ``unmeasured``   — the live path has never returned a real answer for this prompt (transport only).

    ``product_only`` is the same question facing the other way, and it is not decoration: MEASURED here, the one
    prompt the live product DID serve credibly (``welder__appearance``) is ``failed`` in the offline floor. A
    floor that protects nothing the product does, and fails to protect the one thing it does, is worth naming.
    """
    ids = sorted(prompt_ids) if prompt_ids is not None else sorted(set(offline) | set(production))
    rows = {pid: {"offline": offline.get(pid), "production": production.get(pid, PRODUCTION_UNMEASURED)}
            for pid in ids}
    off_credible = [p for p in ids if offline.get(p) == OUTCOME_CREDIBLE]
    corroborated = [p for p in off_credible if rows[p]["production"] == OUTCOME_CREDIBLE]
    contradicted = {p: rows[p]["production"] for p in off_credible
                    if rows[p]["production"] in (OUTCOME_REFUSED, OUTCOME_FAILED)}
    unmeasured = [p for p in off_credible if rows[p]["production"] == PRODUCTION_UNMEASURED]
    product_only = [p for p in ids
                    if rows[p]["production"] == OUTCOME_CREDIBLE and offline.get(p) != OUTCOME_CREDIBLE]
    return {"n_prompts": len(ids), "rows": rows,
            "offline_credible": off_credible, "n_offline_credible": len(off_credible),
            "corroborated": corroborated, "contradicted": dict(sorted(contradicted.items())),
            "unmeasured": unmeasured, "product_only": product_only,
            # The blunt number: of the bodies the offline gate PROTECTS, how many is the product known to build?
            "capability_claim_rate": (round(len(corroborated) / len(off_credible), 4) if off_credible else None),
            # Everything the offline floor protects without live corroboration. A reader of the offline artifact
            # must see this list; it is what stops `true` being read as "the product can do this".
            "offline_only_floor": sorted(set(contradicted) | set(unmeasured))}


# ---------------------------------------------------------------- cassette-vs-live diff (#208)
def diff_funnels(baseline: dict, candidate: dict) -> dict:
    """Compare two funnels case by case. Built for the cassette-vs-live question specifically: the aggregate
    cannot answer it (20 prompts, 0.05 per case, see ``per_case``'s note), and the two runs may not even cover
    the same prompt set when the live arm is a small sample. So the comparison is restricted to the prompts
    BOTH scored, and the restricted rates are reported next to the full ones."""
    a, b = baseline.get("per_case") or {}, candidate.get("per_case") or {}
    oa, ob = baseline.get("per_case_outcome") or {}, candidate.get("per_case_outcome") or {}
    shared = sorted(set(a) & set(b))
    gained = sorted(k for k in shared if b[k] and not a[k])
    lost = sorted(k for k in shared if a[k] and not b[k])

    def _rate(m, keys):
        return round(sum(1 for k in keys if m.get(k)) / len(keys), 4) if keys else None
    return {
        "baseline_mode": baseline.get("mode"), "candidate_mode": candidate.get("mode"),
        "baseline_model": baseline.get("model"), "candidate_model": candidate.get("model"),
        "n_shared": len(shared), "shared_prompt_ids": shared,
        "only_in_baseline": sorted(set(a) - set(b)), "only_in_candidate": sorted(set(b) - set(a)),
        # on the SHARED subset only -- the sole apples-to-apples comparison available
        "verdict@1_baseline_shared": _rate(a, shared),
        "verdict@1_candidate_shared": _rate(b, shared),
        "delta_verdict@1_shared": (None if not shared else
                                   round(_rate(b, shared) - _rate(a, shared), 4)),
        "gained": gained, "lost": lost, "n_changed": len(gained) + len(lost),
        # The same comparison in the TRI-STATE, where "lost" splits into the two very different things it was
        # hiding. MEASURED offline->live: of the 5 prompts the boolean calls lost, THREE are refusals -- the
        # candidate declined to build a body whose rest pose topples or whose joints sit over their static
        # hold torque. That is not a defect, and a comparison that cannot say so is the reason the offline
        # floor was read as a capability claim in the first place.
        #
        # NOTE these are NOT gate verdicts. Two different generators are never graded against each other
        # (``check_label`` refuses it in both directions); this is a description of how they differ.
        "outcome_changes": {k: f"{oa[k]} -> {ob[k]}" for k in shared
                            if k in oa and k in ob and oa[k] != ob[k]},
        "declined_by_candidate": sorted(k for k in shared if ob.get(k) == OUTCOME_REFUSED
                                        and oa.get(k) != OUTCOME_REFUSED),
        "built_but_broken_in_candidate": sorted(k for k in shared if oa.get(k) == OUTCOME_CREDIBLE
                                                and ob.get(k) == OUTCOME_FAILED),
        # full-battery rates, kept ONLY as context; they are not comparable when the sets differ
        "verdict@1_baseline_full": baseline.get("verdict@1"), "verdict@1_candidate_full": candidate.get("verdict@1"),
    }
