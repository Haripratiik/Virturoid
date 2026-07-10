"""Gait HINTS — the flywheel's moat done right: transferable, auto-discovered HINTS that WARM-START a fresh
adaptation per body, never a copy-pasted policy.

Copy-pasting a banked gait onto a new body is a trap (RoboMorph's mode-collapse): every user asks a slightly
different body/scene/task, and a verbatim policy that fit body A is a HEAD-START-INTO-A-WALL on body B. What
transfers is not the exact numbers but the PRINCIPLES the successful walks share — "credible quads cluster the
step frequency here", "the knee lifts more than the hip swings", "widen the stance when a body rolls over". This
module MINES those hints from whatever has actually walked (no hardcoded values — the regions ARE the data), and
hands them to a SHORT search that adapts to the new body. The banked policy is a starting hint; the deployed gait
is freshly fitted.

  * mine_gait_hints  — scan the banked CREDIBLE walks + the lesson store, derive transferable hints + a search
                       prior. Sharpens automatically as more walks are banked (thin at first, honest about it).
  * hint_prior       — the param dict + per-param search bounds the hints imply (the warm-start seed).
  * adapt_gait_from_hints — warm-start a SHORT gait search from that prior, scored on THIS body → ADAPTED params.

Deterministic, CPU, no LLM, no hardcoded gait numbers.
"""
from __future__ import annotations

import json
import statistics as st

_PARAM_KEYS = ("freq", "hip_amp", "knee_amp", "duty", "kp", "kd")


def _class_of(gene) -> str:
    from virturoid.services.gait_flywheel import _class_of as gc
    return gc(gene)


def _banked_gait_params(db, robot_class: str | None, *, min_success: float) -> list[dict]:
    """Every banked CREDIBLE gait's params (optionally for a class) — the evidence the hints are mined from."""
    q = "SELECT robot_class, base_config, success_rate FROM skills WHERE task_type='locomotion'"
    out = []
    for row in db.conn.execute(q).fetchall():
        if float(row["success_rate"] or 0.0) < min_success:
            continue
        if robot_class and (row["robot_class"] or "").lower() not in (robot_class.lower(), "legged"):
            continue
        try:
            bc = json.loads(row["base_config"]) if isinstance(row["base_config"], str) else row["base_config"]
        except (json.JSONDecodeError, TypeError):
            continue
        gp = (bc or {}).get("gait_params")
        if isinstance(gp, dict) and gp.get("controller", "crawl_gait") in ("crawl_gait", None) or isinstance(gp, dict):
            gp = bc.get("gait_params")
        if isinstance(gp, dict):
            out.append(gp)
    return out


def mine_gait_hints(db, robot_class: str | None = None, *, min_success: float = 0.4) -> dict:
    """AUTO-DISCOVER transferable gait hints from the banked credible walks + the lesson store. Returns
    ``{n, hints, prior, note}``. Every number is derived from data — nothing hardcoded; with <2 walks it says so
    and falls back to the shipped default as the (honest, un-tuned) prior."""
    from virturoid.services.gait_flywheel import _DEFAULT_GAIT
    params = _banked_gait_params(db, robot_class, min_success=min_success)
    hints: list[dict] = []
    prior = dict(_DEFAULT_GAIT)
    bounds: dict[str, tuple[float, float]] = {}
    if len(params) >= 2:
        # per-param CREDIBLE REGION (median + IQR): where working gaits actually cluster, for THIS class
        for k in _PARAM_KEYS:
            vals = sorted(float(p[k]) for p in params if k in p and isinstance(p[k], (int, float)))
            if len(vals) >= 2:
                med = st.median(vals)
                lo, hi = vals[0], vals[-1]
                prior[k] = round(med, 4)
                bounds[k] = (round(lo, 4), round(hi, 4))
                hints.append({"kind": "param_region", "param": k, "center": round(med, 4),
                              "range": [round(lo, 4), round(hi, 4)], "support": len(vals),
                              "note": f"credible walkers cluster {k} near {med:.2f} (from {len(vals)} banked walks)"})
        # RELATIONAL hint (auto): do credible walks lift the knee more than they swing the hip?
        rel = [1 for p in params if float(p.get("knee_amp", 0)) > float(p.get("hip_amp", 0))]
        if params:
            frac = len(rel) / len(params)
            if frac >= 0.7:
                hints.append({"kind": "relation", "rule": "knee_amp > hip_amp", "support": len(params),
                              "note": f"{frac:.0%} of credible walks lift the knee MORE than the hip swings — "
                                      f"a stepping (not sliding) gait; keep knee_amp above hip_amp"})
    # STRUCTURAL lessons for this class (the existing failure->fix hint store, auto-written on repair)
    try:
        for L in db.lessons_for_class(robot_class or "", limit=5):
            hints.append({"kind": "structural", "failure": L.get("failure_code"), "fix": L.get("operator"),
                          "note": (L.get("rationale") or f"on {L.get('failure_code')}, apply {L.get('operator')}")})
    except Exception:  # noqa: BLE001 - lessons are value-add
        pass
    note = ("hints mined from banked credible walks — they SHARPEN as more sims run"
            if len(params) >= 2 else
            "not enough banked walks yet to mine a region — using the shipped default as an un-tuned prior; "
            "hints appear automatically once ≥2 credible walks are banked for this class")
    return {"n": len(params), "robot_class": robot_class, "hints": hints, "prior": prior,
            "bounds": bounds, "note": note}


def hint_prior(hints: dict) -> dict:
    """The warm-start param dict the mined hints imply (the seed a fresh search adapts FROM, not the answer)."""
    return dict(hints.get("prior") or {})


def adapt_gait_from_hints(gene, db, *, generations: int = 4, pop: int = 10, steps: int = 600,
                          deploy_steps: int = 1200, seed: int = 0) -> dict:
    """The moat, done right: mine the class hints, WARM-START a SHORT gait search from their prior, and DEPLOY the
    result FITTED TO THIS BODY. Two bodies get two different gaits from the same hints — adaptation, not a copy.
    Returns ``{params, walked, forward_m, source, hints, adapted_from_prior}``."""
    from virturoid.services.gait_search import evaluate_gait, search_gait
    h = mine_gait_hints(db, _class_of(gene))
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
    with MemoryDB(DEFAULT_DB_PATH) as db:
        h = mine_gait_hints(db, args.get("robot_class"))
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
        "parameters": {"type": "object", "properties": {"robot_class": {"type": "string"}}},
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
