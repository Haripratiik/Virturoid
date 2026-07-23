"""Reward-as-code DSL + anti-gaming selection (breakthrough v5, B3 hardening). The existing ``reward_smith`` is the
BUDGET scheduler (ASHA over candidates, ranked by the honesty gate). This module is the missing SAFETY + HONESTY
substrate the plan calls for: a reward is an arithmetic EXPRESSION over a fixed vocabulary of per-step scalars,
parsed with ``ast`` behind a tiny allowlist so LLM-authored reward text can never reach ``eval`` of the raw
environment — the exact Eureka/DrEureka failure modes, closed:

1. **Whitelisted feature API, AST-sandboxed** — no attribute access, no calls to arbitrary names, no imports, no
   subscripting; only the fixed feature names + a handful of safe math functions.
2. **Success is SEPARATE and TRUSTED** — the reward is optimized; SUCCESS is a code-owned metric the model can't
   see or edit. Candidates rank by success, never by their own reward value.
3. **Anti-gaming detectors** — a candidate whose reward return is a strong positive outlier while its trusted
   success is below the field median is flagged GAMED and discarded even if it 'wins' on reward.

Pure-CPU + LLM-optional, so it is deterministically testable; the real trainer/gate plug into ``select_reward``.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass

# The ONLY names a reward expression may reference. Each is a per-step scalar the trainer/eval supplies. Keeping
# this list closed is the safety boundary: an expression referencing anything else fails to compile.
REWARD_FEATURES: tuple[str, ...] = (
    "forward_vel",       # signed base forward speed (m/s)
    "upright",           # world-up . body-up in [-1, 1]
    "height_ratio",      # base height / standing height in [0, 1]
    "foot_clearance",    # mean swing-foot lift (m)
    "contact_frac",      # fraction of feet grounded in [0, 1]
    "action_smooth",     # -mean((a_t - a_{t-1})^2)  (already negative-good)
    "energy",            # mean |torque * qvel| (cost; subtract it)
    "dist_to_goal",      # distance to a target (m); lower better
    "alive",             # 1.0 while upright, 0.0 after a fall
    "slip",              # grounded-foot horizontal speed (slide cost)
)

# Safe callables a reward may use (name -> impl). No builtins beyond these.
_SAFE_FUNCS = {
    "min": min, "max": max, "abs": abs,
    "exp": math.exp, "tanh": math.tanh, "sqrt": lambda x: math.sqrt(max(0.0, x)),
    "clip": lambda x, lo, hi: max(lo, min(hi, x)),
}


class RewardSyntaxError(ValueError):
    """Raised when a reward expression uses a disallowed construct or an unknown name."""


def _validate_ast(node: ast.AST) -> None:
    """Reject anything outside the allowlist. Whitelist of node types only; everything else (Attribute, Import,
    Lambda, comprehensions, Subscript, walrus, f-strings, ...) is a hard error."""
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load, ast.Call,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Mod,
               ast.Compare, ast.Gt, ast.Lt, ast.GtE, ast.LtE, ast.Eq, ast.NotEq,
               ast.IfExp, ast.BoolOp, ast.And, ast.Or)
    for sub in ast.walk(node):
        if not isinstance(sub, allowed):
            raise RewardSyntaxError(f"disallowed syntax: {type(sub).__name__}")
        if isinstance(sub, ast.Name) and sub.id not in REWARD_FEATURES and sub.id not in _SAFE_FUNCS:
            raise RewardSyntaxError(f"unknown name: {sub.id!r} (not a whitelisted feature or function)")
        if isinstance(sub, ast.Call) and not (isinstance(sub.func, ast.Name) and sub.func.id in _SAFE_FUNCS):
            raise RewardSyntaxError("only whitelisted functions may be called")
        # B0b: the RCE surface is closed, but `9**9**9` / `(10**9)**(10**9)` pass the allowlist and hang eval
        # building a ~370M-digit int (the float() OverflowError guard never reaches). All 10 features are in
        # [-1,1], so a constant exponent above ~8 or a huge constant base is never legitimate — reject it. This
        # bounds compute so an agent-authored (BYOK) reward can't DoS the loop.
        if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Pow):
            exp = sub.right
            if isinstance(exp, ast.Constant) and isinstance(exp.value, (int, float)) and abs(exp.value) > 8:
                raise RewardSyntaxError(f"exponent {exp.value} too large (max 8) — features are in [-1,1]")
            base = sub.left
            if isinstance(base, ast.Constant) and isinstance(base.value, (int, float)) and abs(base.value) > 1e6:
                raise RewardSyntaxError(f"constant base {base.value} too large (max 1e6)")


def compile_reward(expr: str):
    """Parse + validate a reward expression, return ``fn(features: dict) -> float``. Raises RewardSyntaxError on
    any disallowed construct. The fn substitutes missing features with 0.0 and returns 0.0 (not NaN) on a math
    domain error, so a single bad step never poisons a whole rollout."""
    tree = ast.parse(expr.strip(), mode="eval")
    _validate_ast(tree)
    code = compile(tree, "<reward>", "eval")

    def fn(features: dict) -> float:
        env = {k: float(features.get(k, 0.0)) for k in REWARD_FEATURES}
        env.update(_SAFE_FUNCS)
        try:
            v = eval(code, {"__builtins__": {}}, env)  # noqa: S307 - AST-allowlisted, no builtins, closed env
            return float(v) if math.isfinite(v) else 0.0
        except (ValueError, ZeroDivisionError, OverflowError):
            return 0.0

    return fn


def _reduce_binop(op, args):
    out = args[0]
    for a in args[1:]:
        out = op(out, a)
    return out


def compile_reward_batched(expr: str, *, xp=None):
    """R1 (MJX port): the SAME AST-validated reward, evaluated over ARRAYS with an array math backend (``numpy``
    by default, or ``jax.numpy`` for the on-GPU trainer). Returns ``fn(features: dict[str, array]) -> array`` so
    the MJX-PPO trainer can compute the 10 features per-env in JAX and score an LLM-authored reward NATIVELY —
    the CPU gait search and the GPU trainer share one reward definition, no per-step Python ``eval`` in the hot
    loop. Pure arithmetic + the whitelisted funcs (min/max/abs/exp/tanh/sqrt/clip), all with array equivalents,
    so the returned fn is ``jit``/``vmap``-friendly. Non-finite results are sanitized to 0 (never poison PPO)."""
    if xp is None:
        import numpy as xp
    tree = ast.parse(expr.strip(), mode="eval")
    _validate_ast(tree)
    code = compile(tree, "<reward_batched>", "eval")
    funcs = {
        "min": lambda *a: _reduce_binop(xp.minimum, a), "max": lambda *a: _reduce_binop(xp.maximum, a),
        "abs": xp.abs, "exp": xp.exp, "tanh": xp.tanh,
        "sqrt": lambda x: xp.sqrt(xp.maximum(0.0, x)),
        "clip": lambda x, lo, hi: xp.clip(x, lo, hi),
    }

    def fn(features: dict):
        env = {k: features.get(k, 0.0) for k in REWARD_FEATURES}   # arrays (or scalars) — no float() coercion
        env.update(funcs)
        v = eval(code, {"__builtins__": {}}, env)  # noqa: S307 - AST-allowlisted, no builtins, closed env
        return xp.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    return fn


@dataclass
class RewardCandidate:
    name: str
    expr: str
    compiled: object = None
    trusted_success: float = 0.0
    reward_return: float = 0.0
    gamed: bool = False
    error: str = ""


# Heuristic reward templates (the LLM-off fallback + the seed pool the LLM mutates). Each is a legal expression
# over REWARD_FEATURES capturing a different hypothesis about what a good gait rewards.
_TEMPLATES: dict[str, str] = {
    "velocity_track": "exp(-(forward_vel - 0.4)**2 / 0.25) * alive + 0.2*upright - 0.1*energy",
    "progress_upright": "max(0.0, forward_vel) * alive + 0.3*upright + 0.1*foot_clearance - 0.2*slip",
    "smooth_march": "max(0.0, forward_vel)*alive + 0.2*height_ratio + 0.3*action_smooth - 0.15*slip",
    "reach_goal": "-dist_to_goal + 0.3*upright*alive - 0.1*energy",
    "clearance_gait": "max(0.0, forward_vel)*alive + 0.5*foot_clearance*contact_frac - 0.2*slip - 0.05*energy",
}


def propose_rewards(task: str = "walk forward", *, n: int = 4, llm=None, reflection: str = "") -> list[RewardCandidate]:
    """Propose ``n`` candidate rewards. With ``llm`` (an object with ``.complete(prompt) -> str``) the model
    proposes expressions over the whitelisted API; each is validated + only compilable ones are kept, and the
    heuristic templates backfill to guarantee ``n`` legal candidates. LLM-off returns the templates directly.
    ``reflection`` (an R2 reflection block from the previous round) is injected into the LLM prompt so the next
    proposal CORRECTS the last run's failure instead of guessing afresh."""
    cands: list[RewardCandidate] = []
    if llm is not None:
        try:
            text = llm.complete(_reward_prompt(task, n, reflection))
            for i, line in enumerate([l.strip() for l in text.splitlines() if l.strip()][:n]):
                expr = line.split("#", 1)[0].strip().rstrip(",")
                try:
                    cands.append(RewardCandidate(name=f"llm_{i}", expr=expr, compiled=compile_reward(expr)))
                except RewardSyntaxError as e:
                    cands.append(RewardCandidate(name=f"llm_{i}", expr=expr, error=str(e)))
        except Exception:  # noqa: BLE001 - LLM failures fall back to templates, never crash the loop
            pass
    good = [c for c in cands if c.compiled is not None]
    for name, expr in _TEMPLATES.items():                    # backfill / seed pool
        if len(good) >= n:
            break
        good.append(RewardCandidate(name=name, expr=expr, compiled=compile_reward(expr)))
    return good[:n]


def _reward_prompt(task: str, n: int, reflection: str = "") -> str:
    feats = ", ".join(REWARD_FEATURES); funcs = ", ".join(_SAFE_FUNCS)
    fb = (f"\n\nFEEDBACK from the previous round (fix these specifically):\n{reflection}\n" if reflection else "")
    return (f"Write {n} candidate reward functions for the task: {task}.\n"
            f"Each MUST be a single Python arithmetic expression using ONLY these per-step features: {feats}.\n"
            f"You may call ONLY these functions: {funcs}. No other names, no attributes, no imports.\n"
            f"One expression per line, no prose. Reward forward progress while upright; penalize slip + energy."
            f"{fb}")


def select_reward(candidates: list[RewardCandidate], rollout_fn, *, gaming_reward_z: float = 2.0) -> dict:
    """Rank candidates by TRUSTED success and detect gaming. ``rollout_fn(reward_fn) -> (success, reward_return)``
    trains/evaluates a short probe with the candidate reward and returns the trusted success metric AND the
    accumulated reward. A candidate is GAMED if its reward_return is a strong positive outlier while its trusted
    success is below the field median — it racked up reward without succeeding.

    Returns ``{"best", "ranked", "n_gamed"}``. The winner is the highest trusted success among NON-gamed
    candidates."""
    import numpy as np

    live = [c for c in candidates if c.compiled is not None]
    for c in live:
        try:
            succ, ret = rollout_fn(c.compiled)
            c.trusted_success = float(succ); c.reward_return = float(ret)
        except Exception as e:  # noqa: BLE001
            c.trusted_success = 0.0; c.reward_return = 0.0; c.error = str(e)
    if not live:
        return {"best": None, "ranked": [], "n_gamed": 0}

    succ_med = float(np.median([c.trusted_success for c in live]))
    rets = np.array([c.reward_return for c in live], dtype=float)
    r_mean, r_std = float(rets.mean()), float(rets.std() + 1e-9)
    by_reward = sorted(live, key=lambda c: c.reward_return, reverse=True)
    for c in live:
        # anti-gaming — two complementary detectors (the z-score needs many candidates; the rank rule handles the
        # small-K night-probe regime): a candidate is GAMED if it earns lots of reward but little SUCCESS, i.e.
        # (a) it's a reward z-outlier while below the success median, OR (b) it's THE reward winner yet its
        # success is below median and its reward clearly beats the runner-up (optimizing it didn't optimize
        # success -> the reward is hackable).
        z = (c.reward_return - r_mean) / r_std
        z_flag = z > gaming_reward_z
        rank_flag = (len(live) >= 3 and c is by_reward[0]
                     and c.reward_return > 1.4 * by_reward[1].reward_return)
        c.gamed = bool((z_flag or rank_flag) and c.trusted_success < succ_med)
    ranked = sorted(live, key=lambda c: (not c.gamed, c.trusted_success), reverse=True)
    honest = [c for c in ranked if not c.gamed]
    return {"best": (honest[0] if honest else None), "ranked": ranked,
            "n_gamed": sum(c.gamed for c in live)}
