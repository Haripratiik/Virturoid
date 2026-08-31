"""Reward reflection (agentic platform plan WS-R / R2): the Eureka-grade payload that makes the reward loop
ITERATE intelligently instead of firing once.

Eureka's core trick is not "an LLM writes a reward" -- it's the feedback: after a run, tell the model WHICH reward
component dominated, which saturated, and whether the policy actually succeeded, so its next proposal is a
correction rather than a fresh guess. This module produces that payload for our DSL rewards:

  * it decomposes the reward EXPRESSION into its top-level additive terms (via the same AST the sandbox uses) and
    scores each term's numeric contribution on the trained gait's features -- so the payload can NAME the term
    that saturated ("the +0.3*upright term contributed 0.29 while max(0,forward_vel) contributed 0.00");
  * it flags features that are pinned at their bound (upright~1, standing still) or flat (forward_vel~0);
  * it reports the honest outcome -- the ``classify`` verdict and whether the gait was credible, kept SEPARATE
    from the reward's own number (a high reward with a low verdict is the hacking signature, and the diagnosis
    says so);
  * it writes a human diagnosis the proposer (LLM via BYOK, or the template pool) consumes next round.

Pure CPU, deterministic, LLM-free. ``run_intelligent_reward_loop`` uses it to close the propose -> train ->
reflect -> re-propose loop.
"""

from __future__ import annotations

import ast

from virturoid.services.reward_dsl import REWARD_FEATURES, compile_reward


def _split_additive_terms(expr: str) -> list[tuple[float, str]]:
    """Split a reward expression into its top-level additive terms as ``(sign, source)`` pairs. ``a + 0.2*b - c``
    -> ``[(+1,'a'), (+1,'0.2*b'), (-1,'c')]``. Anything not a top-level +/- is one term. Returns ``[]`` on a
    parse failure (the caller falls back to whole-expression reflection)."""
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return []

    def walk(node, sign):
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            s = 1.0 if isinstance(node.op, ast.Add) else -1.0
            return walk(node.left, sign) + walk(node.right, sign * s)
        return [(sign, node)]

    out = []
    for sign, node in walk(tree.body, 1.0):
        try:
            src = ast.unparse(node)
        except Exception:  # noqa: BLE001
            return []
        out.append((sign, src))
    return out


def reward_term_contributions(expr: str, features: dict) -> list[dict]:
    """Each top-level additive term's signed numeric contribution on ``features``, largest magnitude first.
    Lets the reflection name the term that dominates or saturates. Falls back to a single whole-expression
    entry if the split fails."""
    terms = _split_additive_terms(expr)
    if not terms:
        try:
            return [{"term": expr, "value": round(float(compile_reward(expr)(features)), 4)}]
        except Exception:  # noqa: BLE001
            return []
    out = []
    for sign, src in terms:
        try:
            val = sign * float(compile_reward(src)(features))
        except Exception:  # noqa: BLE001
            val = 0.0
        out.append({"term": (("-" if sign < 0 else "") + src), "value": round(val, 4)})
    out.sort(key=lambda t: abs(t["value"]), reverse=True)
    return out


# Natural bounds used only to flag saturation (a feature pinned at its limit is a farming risk).
_SATURATION = {"upright": 0.9, "height_ratio": 0.95, "contact_frac": 0.95, "alive": 0.99}


def build_reflection_payload(gene, params: dict, expr: str, *, steps: int = 800) -> dict:
    """Roll the trained gait once and emit the reflection payload: per-feature values, per-term contributions,
    saturated/flat flags, the honest verdict, and a diagnosis naming what to fix. ``expr`` is the reward that
    produced ``params`` (None/"" -> feature+verdict reflection with no term split)."""
    from virturoid.services.gait_quality import classify
    from virturoid.services.gait_search import _clip, reward_features_from_rollout
    from virturoid.services.morph_policy import crawl_gait_rollout

    p = _clip(params)
    r = crawl_gait_rollout(gene, steps=steps, freq=p["freq"], hip_amp=p["hip_amp"], knee_amp=p["knee_amp"],
                           kp=p["kp"], kd=p["kd"], record_qpos=True)
    feats = reward_features_from_rollout(r)
    fwd = float(r.get("forward", 0.0))
    verdict = classify(r) if callable(classify) else "?"
    vlabel = verdict if isinstance(verdict, str) else getattr(verdict, "label", str(verdict))
    credible = vlabel.upper().startswith(("WALK", "CREDIBLE"))

    saturated = [f for f, thr in _SATURATION.items() if abs(feats.get(f, 0.0)) >= thr]
    flat = [f for f in ("forward_vel",) if abs(feats.get(f, 0.0)) < 0.05]
    contribs = reward_term_contributions(expr, feats) if expr else []
    dominant = contribs[0] if contribs else None

    # the diagnosis: a correction hint for the next proposal, phrased around the failure mode we actually see.
    diag = []
    if credible:
        diag.append(f"Credible ({vlabel}); forward {fwd:.2f} m. Reward and verdict agree.")
    else:
        diag.append(f"NOT credible ({vlabel}); forward {fwd:.2f} m.")
        if "forward_vel" in flat and dominant and dominant["value"] > 0:
            diag.append(f"forward_vel is flat ({feats['forward_vel']:.3f}) while the dominant term "
                        f"'{dominant['term']}' contributes {dominant['value']:.3f} -- the reward is being farmed "
                        f"WITHOUT moving. Raise the forward-progress weight or penalize standing still.")
        if saturated:
            diag.append(f"Saturated (pinned at bound): {', '.join(saturated)} -- these terms give no more "
                        f"gradient; the policy sits in them. Down-weight or gate them behind forward progress.")
        if not saturated and "forward_vel" not in flat:
            diag.append("The body moves but the gait is not classified as a walk (slide/lurch). Add a foot "
                        "clearance / cadence shaping term or increase the upright requirement.")

    return {
        "reward_expr": expr or None,
        "verdict": vlabel, "credible": credible, "forward_m": round(fwd, 3),
        "features": {k: round(float(v), 4) for k, v in feats.items()},
        "term_contributions": contribs,
        "dominant_term": dominant,
        "saturated_features": saturated,
        "flat_features": flat,
        "diagnosis": " ".join(diag),
    }


def format_reflection_for_proposer(payload: dict) -> str:
    """A compact text block for the LLM proposer's next round (BYOK). Names the outcome, the dominant/saturated
    terms, and the correction -- the Eureka feedback, as prose the model can act on."""
    lines = [f"Previous reward: {payload.get('reward_expr')}",
             f"Outcome: {payload.get('verdict')} (credible={payload.get('credible')}), "
             f"forward={payload.get('forward_m')} m."]
    if payload.get("term_contributions"):
        top = ", ".join(f"{t['term']}={t['value']}" for t in payload["term_contributions"][:4])
        lines.append(f"Term contributions: {top}.")
    if payload.get("saturated_features"):
        lines.append(f"Saturated: {', '.join(payload['saturated_features'])}.")
    if payload.get("flat_features"):
        lines.append(f"Flat/unmoved: {', '.join(payload['flat_features'])}.")
    lines.append(f"Diagnosis: {payload.get('diagnosis')}")
    lines.append("Write improved rewards that fix this specifically.")
    return "\n".join(lines)
