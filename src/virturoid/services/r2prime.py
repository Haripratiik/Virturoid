"""R2′ — the decisive Thesis A measurement (docs/robotics_language_gen3d_research.md §9, Stage 0.1).

The claim under test: **corpus grounding raises the verified-solve rate**. Retrieving a physics-VERIFIED,
morphology-matched precedent from the flywheel corpus should make a body succeed at its task MORE OFTEN than
(a) no grounding at all and (b) a single static "expert example" reused for everything (RoboMorph's mode-collapse
baseline). We measure it on the PRIMARY corpus (gaits) with an UN-GAMEABLE verdict — the crawl-gait physics:
a body "solves" its task only when it produces a CREDIBLE walk (upright + real forward + survives the horizon),
never a slide or a lurch. No scalar to game, no LLM to pay: the physics is the judge.

Three knowledge arms, identical everywhere except the gait they deploy:
  • ``off``        — cold shipped default gait (NO grounding).
  • ``cheatsheet`` — the single BEST banked gait, reused for EVERY body (one static exemplar in the prompt).
  • ``corpus``     — the morphology-NEAREST banked gait for THIS body (retrieval), default only on a miss.

Decision rule (from the dossier, so a null result is a real, reportable outcome):
  corpus ≤ off        → retrieval adds nothing; FIX retrieval before building on it.
  corpus ≈ cheatsheet → retrieval ≈ one static example; re-scope Thesis A to "prompt-engineering++".
  corpus > both       → the retrieval lift is real (the mechanism Thesis A rests on).

This is the CPU mechanism harness (deterministic: the "designer" is the retrieval itself). The SAME arm
structure — off / one-exemplar / retrieved-precedent as in-context grounding — is what a live LLM designer
consumes, so pointing this battery at an LLM backend measures the full end-to-end claim; that run needs a live
model and is out of the zero-token CPU path.
"""
from __future__ import annotations

import json

from virturoid.services.gait_flywheel import LOCOMOTION, _DEFAULT_GAIT

KNOWLEDGE_ARMS = ("off", "cheatsheet", "corpus")

# Structurally DIVERSE legged bodies. References seed the corpus; the battery is measured on it. The battery is
# deliberately NOT identical to the references (different sizes / leg counts) so ``corpus`` must actually
# generalize by morphology, not fetch an exact copy. References are chosen where a body-FITTED gait is worth
# banking (a standard dog that the shipped default already walks contributes no distinct precedent).
_REFERENCE_PROMPTS = ("a large quadruped robot", "a small four-legged robot", "a six-legged robot")
_BATTERY_PROMPTS = ("a quadruped robot dog that walks", "a big four-legged robot",
                    "a hexapod robot", "a compact quadruped robot")


def _compose(prompt: str):
    """Compose a legged body and make sure it can STAND (so the verdict measures the GAIT, not a broken stance)."""
    from virturoid.services.morphology_composer import compose_robot
    g = compose_robot(prompt)
    try:
        from virturoid.services.anatomy_compiler import ensure_walkable_quad
        g = ensure_walkable_quad(g, prompt)
    except Exception:  # noqa: BLE001 - a non-quad (hexapod) is a no-op here
        pass
    return g


def verified_solve(gene, params: dict, *, steps: int = 1200) -> tuple[bool, dict]:
    """UN-GAMEABLE task verdict: deploy the crawl gait with ``params`` and count it solved ONLY on a CREDIBLE
    walk that survived the horizon (a slide/lurch/fall is never a solve). Returns (solved, rollout metrics)."""
    from virturoid.services.gait_search import evaluate_gait
    r = evaluate_gait(gene, params, steps=steps)
    return bool(r.get("credible") and r.get("survived")), r


def _best_banked_gait(db) -> dict | None:
    """The single strongest banked gait — the ``cheatsheet`` arm's one static exemplar for every body."""
    row = db.conn.execute(
        "SELECT base_config FROM skills WHERE task_type=? ORDER BY success_rate DESC LIMIT 1",
        (LOCOMOTION,)).fetchone()
    if not row or not row["base_config"]:
        return None
    try:
        bc = json.loads(row["base_config"]) if isinstance(row["base_config"], str) else row["base_config"]
    except json.JSONDecodeError:
        return None
    return bc.get("gait_params") if isinstance(bc, dict) else None


def arm_params(arm: str, gene, db) -> dict:
    """The gait each knowledge arm deploys for ``gene`` — the ONLY thing that differs between arms."""
    if arm == "corpus":
        from virturoid.services.gait_flywheel import recall_gait
        return dict(recall_gait(db, gene) or _DEFAULT_GAIT)        # morphology-nearest precedent (default on miss)
    if arm == "cheatsheet":
        return dict(_best_banked_gait(db) or _DEFAULT_GAIT)        # ONE static exemplar reused everywhere
    return dict(_DEFAULT_GAIT)                                     # off: cold default, no grounding


def seed_corpus(db, prompts: tuple[str, ...] = _REFERENCE_PROMPTS, *, generations: int = 8, pop: int = 16,
                steps: int = 800, deploy_steps: int = 1200, seeds: int = 2) -> list[dict]:
    """Grow the corpus the flywheel way: search a body-FITTED gait for each reference and bank the best CREDIBLE
    one, keyed by morphology. Unlike the product path (which banks only a gait that BEATS the shipped default),
    R2' banks a working precedent for EVERY reference the search can walk — so the corpus has real per-morphology
    coverage to retrieve, not a single global winner. A reference whose search finds no credible walk is skipped.

    EVERY ROW IS MEASURED FOR FRAGILITY BEFORE IT IS BANKED, and the reading rides with it. This door can afford
    the error bar where the ordinary verify path cannot: seeding one reference already costs ``seeds`` searches
    (~250 rollouts at ``generations``x``pop``), so the 4-12 rollout ladder is a few percent on top. It is measured
    at ``deploy_steps`` — the horizon the precedent was selected at — and ``bank_gait`` records that horizon.

    A FRAGILE REFERENCE IS STILL BANKED, and stamped ``fragility_v1_fragile``. Refusing it would silently change
    what R2' measures: the arms compare retrieval against no-grounding over whatever the corpus actually holds,
    and dropping references would confound "retrieval helps" with "we curated the corpus harder". The stamp keeps
    that choice visible and keeps the fragile rows out of ``mine_gait_hints(gated_only=True)``.
    """
    from virturoid.services.gait_flywheel import _DEFAULT_GAIT, _DeployResult, bank_gait, robustness_margin
    from virturoid.services.gait_search import evaluate_gait, search_gait
    banked = []
    for i, p in enumerate(prompts):
        g = _compose(p)
        try:
            # Candidate gaits: the shipped default (a credible baseline on many bodies) + a couple of search seeds.
            # CRITICAL (un-gameable): the search's fitness-best is often a fast SLIDE, not a walk — so we select the
            # best CREDIBLE + surviving candidate at the deploy horizon, never the longest slide. A reference with
            # no credible gait at this budget banks nothing (honest: the corpus only holds real walks).
            cands = [dict(_DEFAULT_GAIT)]
            for s in range(max(1, seeds)):
                try:
                    cands.append(search_gait(g, generations=generations, pop=pop, steps=steps, seed=i * 10 + s).best_params)
                except Exception:  # noqa: BLE001
                    pass
            best = None
            for params in cands:
                r = evaluate_gait(g, params, steps=deploy_steps)
                if r.get("credible") and r.get("survived") and (best is None or r["forward"] > best[1]["forward"]):
                    best = (params, r)
            # ``per_param=False``: the per-parameter breakdown is 3-5x the cost and does not enter this decision.
            rob = robustness_margin(g, best[0], steps=deploy_steps, per_param=False) if best else None
            sid = bank_gait(db, g, _DeployResult(best[0], best[1]), robustness=rob,
                            door="r2prime.seed_corpus") if best else None
            banked.append({"prompt": p, "skill_id": sid,
                           "forward_m": round(float(best[1]["forward"]), 3) if best else None,
                           "credible": best is not None,
                           "robustness_rel": (rob or {}).get("robustness_rel"),
                           "robustness_probes": (rob or {}).get("probes")})
        except Exception as exc:  # noqa: BLE001 - a reference that won't learn is skipped, not fatal
            banked.append({"prompt": p, "error": f"{type(exc).__name__}: {exc}"})
    return banked


def run_r2prime_warmstart(db, *, battery: tuple[str, ...] = _BATTERY_PROMPTS, generations: int = 5, pop: int = 10,
                          steps: int = 600, deploy_steps: int = 1200, seed: int = 0) -> dict:
    """The FAIREST, mechanism-true arm — the flywheel's actual use of the corpus. Does a SMALL search reach a
    credible walk MORE OFTEN / FURTHER when WARM-STARTED from the morphology-nearest banked gait than from cold?
    Zero-shot recall (``run_r2prime``) is the harshest test; this measures the compounding the corpus is FOR:
    grounding as a starting point a short search refines, not a drop-in final controller."""
    from virturoid.services.gait_flywheel import recall_gait
    from virturoid.services.gait_search import evaluate_gait, search_gait
    arms = {"cold": {"solved": 0, "n": 0, "rows": [], "fwd": []},
            "corpus_warmstart": {"solved": 0, "n": 0, "rows": [], "fwd": []}}
    for p in battery:
        g = _compose(p)
        prior = recall_gait(db, g)
        for arm, warm in (("cold", None), ("corpus_warmstart", prior)):
            res = search_gait(g, generations=generations, pop=pop, steps=steps, seed=seed, warm_start=warm)
            r = evaluate_gait(g, res.best_params, steps=deploy_steps)
            solved = bool(r.get("credible") and r.get("survived"))
            arms[arm]["n"] += 1
            arms[arm]["solved"] += int(solved)
            arms[arm]["fwd"].append(round(float(r.get("forward", 0)), 3))
            arms[arm]["rows"].append({"prompt": p, "solved": solved, "warm_started": warm is not None,
                                      "forward_m": round(float(r.get("forward", 0)), 3), "verdict": r.get("verdict")})
    for a in arms:
        n = arms[a]["n"]
        arms[a]["verified_solve_rate"] = round(arms[a]["solved"] / n, 3) if n else 0.0
        arms[a]["mean_forward_m"] = round(sum(arms[a]["fwd"]) / n, 3) if n else 0.0
    return arms


def run_r2prime(db, *, battery: tuple[str, ...] = _BATTERY_PROMPTS, arms: tuple[str, ...] = KNOWLEDGE_ARMS,
                steps: int = 1200) -> dict:
    """Measure verified-solve per knowledge arm over the battery. Corpus must already be seeded."""
    res = {a: {"solved": 0, "n": 0, "rows": []} for a in arms}
    for p in battery:
        g = _compose(p)
        for a in arms:
            solved, r = verified_solve(g, arm_params(a, g, db), steps=steps)
            res[a]["n"] += 1
            res[a]["solved"] += int(solved)
            res[a]["rows"].append({"prompt": p, "solved": solved, "forward_m": round(float(r.get("forward", 0)), 3),
                                   "verdict": r.get("verdict")})
    for a in arms:
        n = res[a]["n"]
        res[a]["verified_solve_rate"] = round(res[a]["solved"] / n, 3) if n else 0.0
    return res


def decision(res: dict, *, eps: float = 1e-6) -> str:
    """Translate the per-arm rates into the dossier's decision rule (a null result is a real outcome)."""
    off = res.get("off", {}).get("verified_solve_rate", 0.0)
    cs = res.get("cheatsheet", {}).get("verified_solve_rate", 0.0)
    co = res.get("corpus", {}).get("verified_solve_rate", 0.0)
    if co <= off + eps:
        return (f"corpus ({co:.0%}) <= off ({off:.0%}): retrieval adds no lift -- FIX retrieval before building on it")
    if co <= cs + eps:
        return (f"corpus ({co:.0%}) ~= cheatsheet ({cs:.0%}): retrieval ~ one static exemplar -- re-scope Thesis A "
                f"to prompt-engineering++")
    return (f"corpus ({co:.0%}) > cheatsheet ({cs:.0%}) and off ({off:.0%}): the retrieval lift is REAL "
            f"(the mechanism Thesis A rests on)")
