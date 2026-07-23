"""The autonomous, LLM-authored-reward training loop (agentic platform plan WS-R / R1).

This is the piece that makes "ask in NLP, get a trained policy with an intelligent reward" REAL instead of a
mock. It composes three existing parts into one closed loop:

  ``reward_dsl.propose_rewards``  — an LLM (the customer's own, via BYOK) writes candidate reward EXPRESSIONS
                                     over a whitelisted, AST-sandboxed feature vocabulary (no raw ``exec``).
  ``gait_search.search_gait(reward_fn=...)`` — each candidate reward STEERS a real CEM gait search on the body.
  ``reward_dsl.select_reward``     — candidates rank by a code-owned TRUSTED success (``gait_quality.classify``),
                                     and a reward that games (high return, low credibility) is flagged + dropped.

The reward is optimized; success is never the reward's to define — the Eureka/DrEureka failure modes, closed.
Pure CPU, LLM-optional (heuristic templates backfill), deterministically testable. The MJX/GPU port reuses the
same feature extractor + reward compilation; only the inner ``search_gait`` swaps for the MJX trainer.
"""

from __future__ import annotations

from virturoid.services.gait_search import evaluate_gait, search_gait
from virturoid.services.reward_dsl import propose_rewards, select_reward


def _steered_rollout_fn(gene, *, generations: int, pop: int, steps: int, seed: int):
    """Build the ``rollout_fn(compiled_reward) -> (trusted_success, reward_return)`` that ``select_reward`` needs.

    It runs a SHORT reward-steered CEM search and reports the winning gait's TRUSTED success (classify-based:
    forward distance only when the gait is credible, else 0) alongside the reward's own return. The two are
    computed independently, which is what lets ``select_reward`` catch a reward that racked up return without
    producing a credible walk."""
    def rollout_fn(compiled_reward):
        res = search_gait(gene, generations=generations, pop=pop, steps=steps, seed=seed,
                          reward_fn=compiled_reward)
        trusted = float(res.best_forward) if res.best_credible else 0.0
        return trusted, float(res.best_fitness)     # best_fitness == the reward's return under a reward_fn
    return rollout_fn


def run_intelligent_reward_loop(gene, task: str = "walk forward", *, llm=None, n_rewards: int = 4,
                                screen_generations: int = 3, screen_pop: int = 10,
                                final_generations: int = 8, final_pop: int = 24, steps: int = 800,
                                seed: int = 0, bank: bool = True, db=None) -> dict:
    """Author → screen → select → train → verify, with zero hand-written reward code.

    1. ``propose_rewards`` writes ``n_rewards`` candidate reward expressions (LLM if given, else templates).
    2. each candidate cheaply STEERS a short search; ``select_reward`` ranks by trusted success + flags gaming.
    3. the winning (non-gamed) reward drives a FINAL, longer search.
    4. the result is judged by the SAME un-gameable ``classify`` verdict ``verify_robot`` uses, then optionally
       banked to the flywheel keyed by this body's morphology.

    Returns a compact, honest report: the chosen reward, the trained gait + its verdict, and how many candidates
    were rejected for gaming. Never raises — a total failure returns ``ok: False`` with a reason."""
    try:
        cands = propose_rewards(task, n=n_rewards, llm=llm)
        rollout_fn = _steered_rollout_fn(gene, generations=screen_generations, pop=screen_pop,
                                         steps=steps, seed=seed)
        sel = select_reward(cands, rollout_fn)
        best = sel.get("best")
        ranked = [{"name": c.name, "expr": c.expr, "trusted_success": round(c.trusted_success, 4),
                   "reward_return": round(c.reward_return, 4), "gamed": c.gamed}
                  for c in sel.get("ranked", [])]

        if best is None:                                     # every candidate gamed or failed -> honest default
            res = search_gait(gene, generations=final_generations, pop=final_pop, steps=steps, seed=seed)
            v = evaluate_gait(gene, res.best_params, steps=steps)
            return {"ok": True, "reward_source": "default_fitness_fallback", "reward_expr": None,
                    "n_candidates": len(cands), "n_gamed": sel.get("n_gamed", 0), "ranked": ranked,
                    "verdict": v["verdict"], "credible": v["credible"], "forward_m": round(v["forward"], 3),
                    "gait_params": res.best_params,
                    "note": "no reward beat the honesty gate; trained on the default un-gameable fitness"}

        final = search_gait(gene, generations=final_generations, pop=final_pop, steps=steps, seed=seed,
                            reward_fn=best.compiled)
        v = evaluate_gait(gene, final.best_params, steps=steps)   # classify verdict, reward-independent
        out = {"ok": True, "reward_source": ("llm" if best.name.startswith("llm") else "template"),
               "reward_name": best.name, "reward_expr": best.expr,
               "n_candidates": len(cands), "n_gamed": sel.get("n_gamed", 0), "ranked": ranked,
               "verdict": v["verdict"], "credible": v["credible"], "forward_m": round(v["forward"], 3),
               "height_ratio": round(v["height_ratio"], 3), "gait_params": final.best_params}

        if bank and v["credible"]:
            try:
                from virturoid.services.gait_flywheel import bank_gait
                from virturoid.services.memory_db import MemoryDB
                _db = db or MemoryDB()
                bank_gait(_db, gene, final)                  # bank the verified gait keyed by morphology
                out["banked"] = True
                if db is None:
                    _db.close()
            except Exception:  # noqa: BLE001 - banking is value-add; a miss never fails the loop
                out["banked"] = False
        return out
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
