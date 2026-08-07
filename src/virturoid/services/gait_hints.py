"""Gait HINTS — the flywheel's moat done right: transferable, auto-discovered HINTS that WARM-START a fresh
adaptation per body, never a copy-pasted policy.

Copy-pasting a banked gait onto a new body is a trap (RoboMorph's mode-collapse): every user asks a slightly
different body/scene/task, and a verbatim policy that fit body A is a HEAD-START-INTO-A-WALL on body B. What
transfers is not the exact numbers but the PRINCIPLES the successful walks share — "credible quads cluster the
step frequency here", "the knee lifts more than the hip swings", "widen the stance when a body rolls over". This
module MINES those hints from whatever has actually walked (no hardcoded values — the regions ARE the data), and
hands them to a SHORT search that adapts to the new body. The banked policy is a starting hint; the deployed gait
is freshly fitted.

A hint is shown to an operator as EVIDENCE, so it must be able to be WRONG: every ``param_region`` has to clear
the three gates documented at ``_region_evidence`` before it is claimed, and one that does not is returned under
``suppressed`` with the measurement that sank it. Zero honest hints beat six invented ones (#266).

  * mine_gait_hints  — scan the banked CREDIBLE walks + the lesson store, derive transferable hints + a search
                       prior. Sharpens automatically as more walks are banked (thin at first, honest about it).
  * hint_prior       — the param dict + per-param search bounds the hints imply (the warm-start seed).
  * adapt_gait_from_hints — warm-start a SHORT gait search from that prior, scored on THIS body → ADAPTED params.

Deterministic, CPU, no LLM, no hardcoded gait numbers.
"""
from __future__ import annotations

import json
import random
import statistics as st

# The parameters worth mining a hint region FROM. ``duty`` was removed here with the search dimension
# (2026-08-01, task #265): it never reached the controller, so its banked spread was CEM variance-collapse on a
# null coordinate — and this module then reported it as "nearby walkers cluster duty near 0.25 (65 robots)", the
# TIGHTEST-looking cluster of all six, because a coordinate with no signal shrinks fastest. A mined hint must
# describe something that can move a robot. See the note at ``gait_search.PARAM_NAMES``.
_PARAM_KEYS = ("freq", "hip_amp", "knee_amp", "kp", "kd")

# ── What a ``param_region`` has to EARN before an operator is shown it as evidence (#266) ────────────────────
# Removing ``duty`` removed the worst INSTANCE and left the MECHANISM: every remaining key was still emitted
# unconditionally, in identical confident wording, on nothing but ">=2 observations exist". Measured on the real
# 90-walk bank, that announced "nearby walkers cluster kp near 56.61 (67 robots)" over an advertised cluster of
# [20.0, 240.0] — a band that SPANS AND EXCEEDS the whole [24, 240] search range — and the same for kd
# ([0.9375, 14.0] of [1, 14]). This product's differentiator is VERIFIED memory; memory that invents structure
# its own data contradicts, and shows it as evidence, is worse than no memory. Three gates, all required:
#
#   1. DISPERSION — the band's spread as a RATIO against the SAMPLING PRIOR the values were drawn from (a
#      uniform draw over the search range). A band as wide as the range is the range, restated as advice.
#   2. ASSOCIATION — distance from the advised center must actually predict banked ``forward_m``. A parameter
#      that does not move the robot has no business being advised on.
#   3. CONTROL FOR WINNERS — banked values are CEM WINNERS, and elite selection shrinks a coordinate's variance
#      whether or not it has signal (fastest where it has NONE, because the elites are then an unbiased random
#      subset). So gate 1 alone is NOT enough, and this is not a theoretical worry — SIMULATING this repo's own
#      CEM update (``gait_search.search_gait``: mean=elite.mean, std=elite.std+1e-3, prior injected as one
#      candidate) on a coordinate the fitness IGNORES produced spread ratios of 0.75 with independent seeds and
#      0.63 -> 0.19 as the warm-started prior wins more often — i.e. the whole 0.19-0.75 band is reachable with
#      ZERO signal, and the real bank's five parameters sit at 0.26-0.83 inside it. The removed ``duty``, proven
#      to have no effect, measured 0.13 — TIGHTER than every real parameter. The null model therefore has to be
#      the elite-set variance selection alone produces, and the cleanest estimate of it needs no guessed
#      hyper-parameters: resample random same-size groups from the banked winners themselves. Both arms then
#      carry the identical selection shrinkage and it cancels.
_MIN_REGION_SUPPORT = 8      # observations carrying BOTH a value and a banked distance; 3 walks is not a region
_MAX_SPREAD_VS_PRIOR = 0.5   # gate 1: at least twice as tight as a uniform draw over the search range
_ALPHA = 0.05                # gates 2 and 3
_PERMUTATIONS = 400          # p resolution 1/401; gates short-circuit, so this stays off the deploy hot path


def _search_range(key: str) -> tuple[float, float]:
    """The interval ``search_gait`` samples this parameter from — the PRIOR a mined band must beat."""
    from virturoid.services.gait_search import _HI, _LO
    return float(_LO[key]), float(_HI[key])


def _ranks(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    out = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for t in range(i, j + 1):
            out[order[t]] = (i + j) / 2.0 + 1.0                   # average rank over ties
        i = j + 1
    return out


def _corr(a: list[float], b: list[float]) -> float:
    ma, mb = st.fmean(a), st.fmean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db) if da > 0 and db > 0 else 0.0


def _region_evidence(vals: list[float], fwd: list[float], lo: float, hi: float, *, seed: int = 0) -> dict:
    """Run the three gates for ONE parameter. Returns the readings + ``ok`` (all three passed).

    ``vals``/``fwd`` are aligned: one banked walk each, its parameter value and the distance it actually
    travelled. Deterministic (fixed-seed permutations) — this is a product surface, not a notebook.

    HONESTY about what the association can and cannot see: ``forward_m`` is compared ACROSS BODIES, so a big
    body's longer stride is a confound this test does not remove. That cuts the right way — it can only make a
    spurious hint EASIER to pass, never a real one harder — and it is the correct test for the claim actually
    being made, because a ``param_region`` hint is advice offered to every body regardless of size. A parameter
    whose good band is body-specific should not be advertised as a universal region in the first place.
    """
    n = len(vals)
    n_top = max(3, n // 2)
    blank = {"support": n, "n_top": n_top, "center": None, "band": None, "spread_vs_prior": None,
             "spread_vs_banked": None, "rho_center_distance": None, "p_association": None,
             "p_vs_winner_null": None, "ok": False}
    if n < _MIN_REGION_SUPPORT:
        # Includes n == 0: rows banked before ``forward_m`` was recorded, which can seed a prior but can never
        # support a claim. Nothing below is even computed — there is no sample to describe.
        return {**blank, "why": f"only {n} banked walks carry both a value and a travelled distance "
                                f"(need {_MIN_REGION_SUPPORT}) — too few to establish a region"}
    prior_sd = (hi - lo) / 12 ** 0.5                              # stdev of a uniform draw over the search range
    full_sd = st.pstdev(vals)
    top = [vals[i] for i in sorted(range(n), key=lambda i: fwd[i], reverse=True)[:n_top]]
    spread, center = st.pstdev(top), st.median(top)
    d = {**blank, "center": round(center, 4), "band": [round(min(top), 4), round(max(top), 4)],
         "spread_vs_prior": round(spread / prior_sd, 3) if prior_sd > 0 else None,
         "spread_vs_banked": round(spread / full_sd, 3) if full_sd > 0 else None}
    if full_sd <= 0:
        return {**d, "why": "every banked walk carries the SAME value — that is one un-searched default "
                            "repeated, not a cluster discovered across robots"}
    if d["spread_vs_prior"] > _MAX_SPREAD_VS_PRIOR:
        return {**d, "why": f"the band is {d['spread_vs_prior']:.0%} as wide as a uniform draw over the "
                            f"[{lo:g}, {hi:g}] search range — that is the search range, not a cluster"}

    # GATE 2 — does distance from the advised center predict travel? The center is REFIT inside every
    # permutation, so the null accounts for the fact that the center was chosen from this same data.
    def assoc(outcome_ranks: list[float], order: list[int]) -> float:
        c = st.median([vals[i] for i in order[:n_top]])
        return _corr(_ranks([abs(v - c) for v in vals]), outcome_ranks)

    fwd_ranks = _ranks(fwd)
    rho = assoc(fwd_ranks, sorted(range(n), key=lambda i: fwd[i], reverse=True))
    d["rho_center_distance"] = round(rho, 3)
    if rho >= 0:
        return {**d, "why": "walkers FURTHER from the advised center travelled no less than walkers inside it "
                            "— the parameter does not move the outcome it is being advised for"}
    rng = random.Random(seed)
    idx, hit2, hit3 = list(range(n)), 0, 0
    for _ in range(_PERMUTATIONS):
        rng.shuffle(idx)                                          # relabel WHICH walker travelled how far
        perm = [fwd_ranks[i] for i in idx]
        if assoc(perm, sorted(range(n), key=lambda j: perm[j], reverse=True)) <= rho + 1e-12:
            hit2 += 1
        # GATE 3 — the elite-selection null: a random same-size group of the banked WINNERS, whose variance is
        # already shrunk by selection exactly as much as the real group's is.
        if st.pstdev(rng.sample(vals, n_top)) / full_sd <= d["spread_vs_banked"] + 1e-12:
            hit3 += 1
    d["p_association"] = round((hit2 + 1) / (_PERMUTATIONS + 1), 4)
    d["p_vs_winner_null"] = round((hit3 + 1) / (_PERMUTATIONS + 1), 4)
    if d["p_association"] >= _ALPHA:
        return {**d, "why": f"reshuffling which walker travelled how far reproduces this association "
                            f"{d['p_association']:.0%} of the time — it is not distinguishable from chance"}
    if d["p_vs_winner_null"] >= _ALPHA:
        return {**d, "why": f"random same-size groups of the banked CEM winners are this tight "
                            f"{d['p_vs_winner_null']:.0%} of the time — the band is what elite selection "
                            f"produces on a parameter with no effect, not evidence of one"}
    return {**d, "ok": True, "why": ""}


def _class_of(gene) -> str:
    from virturoid.services.gait_flywheel import _class_of as gc
    return gc(gene)


def _base_config(row_or_blob) -> dict:
    try:
        bc = json.loads(row_or_blob) if isinstance(row_or_blob, str) else row_or_blob
    except (json.JSONDecodeError, TypeError):
        return {}
    return bc if isinstance(bc, dict) else {}


def _forward_of(bc: dict):
    """The distance this banked walk actually travelled — the OUTCOME a mined region is tested against.
    ``None`` when the row predates ``bank_gait`` recording it: such a walk can seed a prior but can never
    support a claim, because there is nothing to check the claim against."""
    f = bc.get("forward_m")
    return abs(float(f)) if isinstance(f, (int, float)) else None


def _gate_of(bc: dict) -> str:
    from virturoid.services.gait_flywheel import gate_of
    return gate_of(bc)


def _banked_gait_params(db, robot_class: str | None, *,
                        min_success: float) -> list[tuple[dict, float | None, str]]:
    """Every banked CREDIBLE gait (optionally for a class) as (params, forward_m, bank_gate) — the evidence hints
    are mined from. NOTE the ``min_success`` filter selects on the very outcome the association gate tests, which
    truncates its range and can only make that gate HARDER to pass — conservative, in the direction that matters."""
    q = "SELECT robot_class, base_config, success_rate FROM skills WHERE task_type='locomotion'"
    out = []
    for row in db.conn.execute(q).fetchall():
        if float(row["success_rate"] or 0.0) < min_success:
            continue
        if robot_class and (row["robot_class"] or "").lower() not in (robot_class.lower(), "legged"):
            continue
        bc = _base_config(row["base_config"])
        gp = bc.get("gait_params")
        if isinstance(gp, dict):
            out.append((gp, _forward_of(bc), _gate_of(bc)))
    return out


def _vector_nearest_gaits(db, gene, *, k: int, min_sim: float) -> list[tuple[dict, float, float | None, str]]:
    """The banked gaits of the EMBEDDING-NEAREST robots to ``gene`` (cross-body, cross-class), each with its
    cosine similarity and the distance it travelled. THIS is the moat: a brand-new morphology finds robots
    SHAPED like it in the robotics vector space (``embed_body``/GeneGNN latent) and borrows THEIR gait
    principles — real transfer through a learned/structural embedding, not a class-string match. Empty when
    nothing is indexed near it."""
    try:
        from virturoid.services.gait_flywheel import LOCOMOTION
        from virturoid.services.robotics_vector_memory import RoboticsVectorMemory
        out = []
        for hit in RoboticsVectorMemory(db).nearest_skills(gene, LOCOMOTION, k=k, min_sim=min_sim):
            row = db.conn.execute("SELECT base_config FROM skills WHERE skill_id=?",
                                  (str(hit.get("obj_id", "")),)).fetchone()
            bc = _base_config(row["base_config"]) if row else {}
            gp = bc.get("gait_params")
            if isinstance(gp, dict):
                out.append((gp, float(hit.get("similarity", 0.5)), _forward_of(bc), _gate_of(bc)))
        return out
    except Exception:  # noqa: BLE001 - no vector index yet -> caller falls back to the class-string source
        return []


def mine_gait_hints(db, gene=None, robot_class: str | None = None, *, min_success: float = 0.4,
                    k: int = 8, min_sim: float = 0.2, gated_only: bool = False) -> dict:
    """AUTO-DISCOVER transferable gait hints, borrowed from the EMBEDDING-NEAREST robots. When a ``gene`` is
    given the source gaits are the vector-nearest banked robots (weighted by morphology similarity — a
    completely new body borrows from whatever it is SHAPED like, across class labels); otherwise it falls back
    to class-string filtering. Every number is derived from data — nothing hardcoded; <2 sources -> honest
    'not enough data' + the shipped default as an un-tuned prior. The hints sharpen as more sims are banked.

    A ``param_region`` is only CLAIMED when it survives the three gates in ``_region_evidence`` (#266); the ones
    that do not are returned under ``suppressed`` WITH the reason, so "no hint" reads as a measurement rather
    than as silence. ``prior``/``bounds`` are unaffected by the gates in kind: the prior is an explicitly
    UN-TUNED warm-start seed that ``search_gait`` injects as one candidate among ``pop``, not a claim about
    evidence — ``bounds``, which IS such a claim, is now populated only for regions that earned it.

    ``gated_only`` restricts the source rows to those banked UNDER the fragility gate (``bank_gate`` stamped by
    ``gait_flywheel.bank_gait``). The default is False so the live product keeps mining everything it holds — but
    ``n_gated``/``n_ungated`` are ALWAYS returned and always named in the note, because pooling rows whose
    operating point was measured for fragility with rows where the question was never asked is exactly how a
    parameter the controller never read (``duty``) came to be shown to operators as the bank's tightest cluster.
    An ungated row is not a bad row; it is a row with an unknown error bar, and that has to be visible."""
    from virturoid.services.gait_flywheel import BANK_GATE, _DEFAULT_GAIT, _class_of
    src = "vector_nearest"
    pw = _vector_nearest_gaits(db, gene, k=k, min_sim=min_sim) if gene is not None else []
    if not pw:                                                    # no vector neighbors -> class-string source
        cls = robot_class or (_class_of(gene) if gene is not None else None)
        pw = [(p, 1.0, f, g) for p, f, g in _banked_gait_params(db, cls, min_success=min_success)]
        src = "class_match" if cls else "all_banked"
    n_gated = sum(1 for _, _, _, g in pw if g == BANK_GATE)
    n_ungated = len(pw) - n_gated
    if gated_only:
        pw = [t for t in pw if t[3] == BANK_GATE]
    pw = [(p, w, f) for p, w, f, _ in pw]
    params = [p for p, _, _ in pw]
    hints: list[dict] = []
    suppressed: list[dict] = []
    prior = dict(_DEFAULT_GAIT)
    bounds: dict[str, tuple[float, float]] = {}
    if len(pw) >= 2:
        # per-param region as a SIMILARITY-WEIGHTED center (nearer robots pull the prior harder) + observed range
        for key in _PARAM_KEYS:
            vw = [(float(p[key]), w, f) for p, w, f in pw if key in p and isinstance(p[key], (int, float))]
            if len(vw) < 2:
                continue
            tot = sum(max(0.05, w) for _, w, _ in vw) or 1.0
            center = sum(v * max(0.05, w) for v, w, _ in vw) / tot
            prior[key] = round(center, 4)                         # the warm-start seed: always available
            lo, hi = _search_range(key)
            judged = [(v, f) for v, _, f in vw if f is not None]
            ev = _region_evidence([v for v, _ in judged], [f for _, f in judged], lo, hi)
            if not ev["ok"]:
                suppressed.append({"param": key, "reason": ev["why"], "support": ev["support"],
                                   "spread_vs_prior": ev["spread_vs_prior"],
                                   "p_association": ev["p_association"],
                                   "p_vs_winner_null": ev["p_vs_winner_null"]})
                continue
            band, n_j = ev["band"], len(judged)
            bounds[key] = (band[0], band[1])
            hints.append({"kind": "param_region", "param": key, "center": ev["center"], "range": band,
                          "support": ev["support"], "spread_vs_prior": ev["spread_vs_prior"],
                          "p_association": ev["p_association"], "p_vs_winner_null": ev["p_vs_winner_null"],
                          "note": f"the {ev['n_top']} furthest-travelling of {n_j} nearby "
                                  f"walkers put {key} in [{band[0]:g}, {band[1]:g}] — "
                                  f"{ev['spread_vs_prior']:.0%} as wide as a uniform search over "
                                  f"[{lo:g}, {hi:g}]; the further a walker sat from {ev['center']:g} the less it "
                                  f"travelled (rank-corr {ev['rho_center_distance']:+.2f}, p="
                                  f"{ev['p_association']:.3f}), and random same-size groups of banked winners "
                                  f"are this tight only p={ev['p_vs_winner_null']:.3f} of the time"})
        rel = [1 for p in params if float(p.get("knee_amp", 0)) > float(p.get("hip_amp", 0))]
        if params and len(rel) / len(params) >= 0.7:
            hints.append({"kind": "relation", "rule": "knee_amp > hip_amp", "support": len(params),
                          "note": f"{len(rel) / len(params):.0%} of nearby walkers lift the knee MORE than the "
                                  f"hip swings — a stepping (not sliding) gait; keep knee_amp above hip_amp"})
    # STRUCTURAL lessons for the class (the existing failure->fix hint store, auto-written on repair)
    try:
        _cls = robot_class or (_class_of(gene) if gene is not None else "")
        for L in db.lessons_for_class(_cls or "", limit=5):
            hints.append({"kind": "structural", "failure": L.get("failure_code"), "fix": L.get("operator"),
                          "note": (L.get("rationale") or f"on {L.get('failure_code')}, apply {L.get('operator')}")})
    except Exception:  # noqa: BLE001 - lessons are value-add
        pass
    src_txt = ("VECTOR-similar robots (shaped like this one, across class)" if src == "vector_nearest"
               else src.replace("_", "-") + " walks")
    # THE PROVENANCE SPLIT, said on every call. Two banks of the same size are not the same evidence when one
    # was screened for fragility and the other was not, and a reader cannot tell from the hint text alone.
    gate_txt = (f" [{n_gated} banked under the fragility gate, {n_ungated} with no measured margin"
                + ("; only the gated rows were used]" if gated_only else "; both pooled]"))
    if len(pw) < 2:
        note = ("not enough banked walks near this body yet — using the shipped default as an un-tuned prior; "
                "hints appear automatically once >=2 credible walks are banked near it in the robotics embedding")
    elif suppressed and not bounds:
        # The honest reading, and the one worth acting on: the bank holds walks but no MINABLE STRUCTURE. Saying
        # so beats six confident regions that the same data refutes.
        note = (f"{len(pw)} {src_txt} banked, but NO parameter region survived the evidence gates — none of them "
                f"clusters more tightly than the search prior, predicts how far a robot got, AND beats what CEM "
                f"winner-selection produces on a parameter with no effect. See `suppressed` for the per-parameter "
                f"reason. The prior below is an UN-TUNED warm-start seed, not evidence.")
    else:
        note = (f"hints borrowed from {len(pw)} {src_txt} — they sharpen as more sims run"
                + (f"; {len(suppressed)} parameter(s) failed the evidence gates (see `suppressed`)"
                   if suppressed else ""))
    return {"n": len(pw), "source": src, "robot_class": robot_class, "hints": hints, "prior": prior,
            "bounds": bounds, "suppressed": suppressed, "note": note + gate_txt,
            "n_gated": n_gated, "n_ungated": n_ungated, "gated_only": bool(gated_only)}


def hint_prior(hints: dict) -> dict:
    """The warm-start param dict the mined hints imply (the seed a fresh search adapts FROM, not the answer)."""
    return dict(hints.get("prior") or {})


def adapt_gait_from_hints(gene, db, *, generations: int = 4, pop: int = 10, steps: int = 600,
                          deploy_steps: int = 1200, seed: int = 0) -> dict:
    """The moat, done right: mine the class hints, WARM-START a SHORT gait search from their prior, and DEPLOY the
    result FITTED TO THIS BODY. Two bodies get two different gaits from the same hints — adaptation, not a copy.
    Returns ``{params, walked, forward_m, source, hints, adapted_from_prior}``."""
    from virturoid.services.gait_search import evaluate_gait, search_gait
    h = mine_gait_hints(db, gene=gene)                            # VECTOR-nearest robots seed the search prior
    prior = hint_prior(h)
    res = search_gait(gene, generations=generations, pop=pop, steps=steps, seed=seed, warm_start=prior)
    r = evaluate_gait(gene, res.best_params, steps=deploy_steps)
    # how far the fresh fit MOVED from the hint prior (proves it adapted, isn't a copy)
    drift = round(sum(abs(float(res.best_params.get(k, 0)) - float(prior.get(k, 0))) for k in _PARAM_KEYS), 3)
    return {"params": res.best_params, "walked": bool(r.get("credible") and r.get("survived")),
            "forward_m": round(float(r.get("forward", 0)), 3), "verdict": r.get("verdict"),
            "source": "hint_guided_adaptation", "n_hints_from": h["n"], "adapted_from_prior_by": drift,
            "hints": h["hints"]}


# ── Agent-facing surface: inspect the auto-discovered hints, and adapt a body's gait FROM them (not a copy) ──

def _flywheel_hints(args: dict) -> dict:
    from virturoid.services.memory_db import DEFAULT_DB_PATH, MemoryDB
    if not DEFAULT_DB_PATH.exists():
        return {"ok": True, "n": 0, "hints": [], "note": "no flywheel memory yet — hints appear as sims run"}
    gene = None                                                   # a held robot -> VECTOR-nearest hints for its body
    if args.get("robot_id"):
        try:
            from virturoid.services import session_state as S
            held = S.get_robot(args["robot_id"]) if hasattr(S, "get_robot") else None
            gene = held.get("gene") if isinstance(held, dict) else (getattr(held, "gene", None) or held)
        except Exception:  # noqa: BLE001
            gene = None
    with MemoryDB(DEFAULT_DB_PATH) as db:
        h = mine_gait_hints(db, gene=gene, robot_class=args.get("robot_class"))
    return {"ok": True, **h}


def _adapt_gait(args: dict) -> dict:
    from virturoid.services import session_state as S
    from virturoid.services.memory_db import DEFAULT_DB_PATH, MemoryDB
    held = S.get_robot(args.get("robot_id", "")) if hasattr(S, "get_robot") else None
    gene = held.get("gene") if isinstance(held, dict) else (getattr(held, "gene", None) or held)
    if gene is None:
        return {"ok": False, "error": f"no robot '{args.get('robot_id')}'; create_robot/submit_design first"}
    with MemoryDB(DEFAULT_DB_PATH) as db:
        out = adapt_gait_from_hints(gene, db, generations=int(args.get("generations", 4)),
                                    pop=int(args.get("pop", 10)), steps=int(args.get("steps", 600)))
    return {"ok": True, **out,
            "note": "gait FITTED to this body, warm-started from the flywheel's mined hints — not a copied policy"}


GAIT_HINT_TOOLS = {
    "flywheel_hints": {
        "description": "Inspect the flywheel's AUTO-DISCOVERED transferable gait hints for a robot class (the "
                       "param regions credible walks cluster in + relational/structural rules), mined from banked "
                       "sims. These sharpen as more robots are used — nothing is hardcoded.",
        "parameters": {"type": "object", "properties": {
            "robot_id": {"type": "string", "description": "a held robot -> hints from its VECTOR-nearest robots"},
            "robot_class": {"type": "string"}}},
        "handler": _flywheel_hints,
    },
    "adapt_gait": {
        "description": "Fit a NEW gait to a held robot by WARM-STARTING a short search from the flywheel's mined "
                       "hints (never a copy-pasted policy). Two different bodies get two different fitted gaits. "
                       "Returns the adapted params + whether it walked + how far it drifted from the hint prior.",
        "parameters": {"type": "object", "required": ["robot_id"], "properties": {
            "robot_id": {"type": "string"}, "generations": {"type": "integer"}, "pop": {"type": "integer"}}},
        "handler": _adapt_gait, "heavy": True,
    },
}
