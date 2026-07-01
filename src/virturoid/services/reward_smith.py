"""Reward-smith (Pillar 1): a sample-efficient LLM reward-design loop for a slow, expensive trainer.

Reward search is a high-variance best-of-N game where each candidate costs a full (slow) training run — so
Eureka's raw ~400-runs-per-task recipe is a non-starter here. This is the budget-aware scheduler that makes
it affordable (docs/next_moves_plan.md, Move 5):

  * **ASHA / successive-halving** — screen K candidates at LOW fidelity (few envs+iters), promote the top
    ``1/eta``, bump fidelity, repeat; only the finalist(s) pay a full run. Turns ~15 full-run-equivalents into
    a gate-certified reward, ~25x cheaper than Eureka-raw.
  * **Rank by the HONESTY GATE, never the training reward** — the ranking key at every rung is the independent
    verifier (cadence + upright + survived), held structurally out of the reward, so a candidate that games
    ``fwd_vel`` earns high training reward but a low gate score and is culled automatically.
  * **Weight-first** — the inner search is over the vetted reward-term weights (low-variance, safe); a new
    reward term is proposed (outer loop) only when the weight search plateaus below the gate.

Everything heavy — training, the LLM proposer, the gate, the reflection — is INJECTED as a callable, so the
LOOP is unit-testable on CPU with a stub trainer, and the real MJX-PPO trainer + ``gait_critic`` (proposer/
reflector) + ``reward_agent`` + the honesty gate plug in unchanged. Pure orchestration, no GPU, no MuJoCo.
"""

from __future__ import annotations

from typing import Callable


def successive_halving(candidates: list, eval_fn: Callable[[object, float], float], *,
                       rungs=(0.12, 0.30, 1.0), eta: int = 3) -> tuple[list, list]:
    """ASHA successive-halving over an EXPENSIVE evaluator.

    ``eval_fn(candidate, fidelity) -> score`` returns the HONESTY-GATE score (bigger = better) of training the
    candidate at ``fidelity`` fraction of the full budget. Evaluates all candidates at ``rungs[0]``, keeps the
    top ``1/eta`` (>=1), bumps to the next fidelity, and repeats — so only the strongest survivor(s) reach the
    full (``1.0``) rung. Returns ``(final_ranking, history)`` where ``final_ranking`` is ``[(candidate, score)]``
    at the top rung (best first) and ``history[i]`` is the full ranking at rung ``i``.
    """
    if not candidates:
        return [], []
    pool = list(candidates)
    history: list = []
    scored: list = []
    for i, fidelity in enumerate(rungs):
        scored = [(c, float(eval_fn(c, float(fidelity)))) for c in pool]
        scored.sort(key=lambda t: t[1], reverse=True)
        history.append(list(scored))
        if i < len(rungs) - 1:
            keep = max(1, len(scored) // eta)
            pool = [c for c, _ in scored[:keep]]
    return scored, history


def reward_smith_cycle(propose_fn: Callable[[object, object, int], list],
                       eval_fn: Callable[[object, float], float],
                       gate_fn: Callable[[object, float], bool],
                       reflect_fn: Callable[[object, list], object], *,
                       iterations: int = 3, k: int = 8, rungs=(0.12, 0.30, 1.0), eta: int = 3,
                       seed: object = None) -> dict:
    """Run one reward-smith cycle: propose -> screen (ASHA, gate-ranked) -> full-eval survivor -> reflect.

    Injected roles (real bindings in parentheses):
      * ``propose_fn(champion, reflection, k) -> [candidate, ...]`` — K reward-weight vectors, warm-started from
        the champion + the last reflection (``gait_critic`` / ``reward_agent`` + ``gpt-4.1-mini``).
      * ``eval_fn(candidate, fidelity) -> gate_score`` — train at ``fidelity`` and return the HONEST gate score
        (MJX-PPO + the honesty gate; NOT the training reward).
      * ``gate_fn(champion, gate_score) -> bool`` — the final certification (the honesty gate's go/no-go).
      * ``reflect_fn(champion, ranking) -> reflection`` — per-term stats + the decomposed gate verdict +
        ``gait_diagnostics`` narrative + a reward<->gate negative-correlation flag (``gait_critic``).

    Stops early once the champion is gate-certified. Returns ``{champion, champion_score, certified, iterations}``
    where ``iterations`` logs each round (best score, certified, candidate count, full-fidelity eval count).
    """
    champion, champion_score, certified, reflection = seed, float("-inf"), False, None
    log: list = []
    for it in range(max(1, iterations)):
        candidates = list(propose_fn(champion, reflection, k))
        if not candidates:
            break
        ranking, _hist = successive_halving(candidates, eval_fn, rungs=rungs, eta=eta)
        best_c, best_s = ranking[0]
        if best_s > champion_score:
            champion, champion_score = best_c, best_s
        certified = bool(gate_fn(champion, champion_score))
        reflection = reflect_fn(champion, ranking)
        log.append({"iteration": it, "best_score": round(best_s, 4), "certified": certified,
                    "n_candidates": len(candidates)})
        if certified:
            break
    return {"champion": champion, "champion_score": round(champion_score, 4),
            "certified": certified, "iterations": log}


def full_fidelity_budget(k: int, iterations: int, *, rungs=(0.12, 0.30, 1.0), eta: int = 3,
                         confirm_seeds: int = 2) -> float:
    """The cost of a cycle in FULL-RUN-EQUIVALENTS (each unit = one full training run), for budgeting on a
    slow GPU. Sums each rung's ``candidates * fidelity`` (survivors shrink by ``eta`` per rung) across
    ``iterations``, + a ``confirm_seeds`` full run for the finalist. Lets a caller size K/iterations to a
    training budget (e.g. ~8 on the slowest box, ~15 at K=8/N=3, ~35 for a hard humanoid walk)."""
    per_iter = 0.0
    n = k
    for i, fidelity in enumerate(rungs):
        per_iter += n * float(fidelity)
        n = max(1, n // eta) if i < len(rungs) - 1 else n
    return round(per_iter * max(1, iterations) + confirm_seeds * 1.0, 3)
