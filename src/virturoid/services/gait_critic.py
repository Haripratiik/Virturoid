"""AI gait critic — the 'AI-assisted training' reasoning step. Reads a GAIT-QUALITY diagnosis (gait_diagnostics)
plus the current reward weights, DIAGNOSES why the body isn't walking (sliding/dragging/hopping/leaning), and
REDESIGNS the reward weights to induce a real alternating stepping gait — the Eureka / Language-to-Rewards loop
applied to locomotion. Scored later by gait_quality (a process metric), never by raw forward velocity (which is
gameable: a slide scores high fwd_vel without taking a step). Offline-safe: no LLM backend -> None.

The reward DESIGN SPACE is a small bounded set of weights the MJX trainer exposes, so the LLM works in a
constrained, validated space (no free-form reward code) and every proposal is clamped to safe bounds.
"""

from __future__ import annotations

# knob -> (min, max, default). These are the reward-shaping weights the MJX trainer accepts as flags.
GAIT_REWARD_KNOBS = {
    "vtgt":     (0.1, 0.9, 0.3),    # target forward SPEED (m/s); too high invites a fall-forward lunge/slide
    "clear_w":  (0.0, 10.0, 2.5),   # MEAN foot-clearance reward (lift feet a little)
    "swing_w":  (0.0, 10.0, 0.0),   # HIGH-swing reward (one foot lifted HIGH = a real step, not a shuffle)
    "slip_w":   (0.0, 10.0, 0.0),   # anti-SLIP penalty (horizontal foot motion while in stance = sliding/dragging)
    "alt_w":    (0.0, 6.0, 0.0),    # PARTIAL-support reward (some feet down + some up = a real step; anti-stand + anti-bound)
    "smooth_w": (0.0, 2.0, 0.3),    # control smoothness penalty (high -> no violent leaps)
    "prog_w":   (0.0, 12.0, 4.0),   # forward-PROGRESS reward (linear, 0 at standstill); too high -> lunge/dive/faceplant
}

_SCHEMA = {
    "type": "object",
    "properties": {
        "failure_diagnosis": {"type": "string"},
        "key_change": {"type": "string"},
        "rationale": {"type": "string"},
        "weights": {
            "type": "object",
            "properties": {k: {"type": "number"} for k in GAIT_REWARD_KNOBS},
            "required": list(GAIT_REWARD_KNOBS),
            "additionalProperties": False,
        },
    },
    "required": ["failure_diagnosis", "key_change", "rationale", "weights"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You are Virturoid's locomotion reward engineer. A legged robot was trained with PPO and is NOT walking — "
    "you are given a GAIT-QUALITY diagnosis (cadence = foot steps/sec, alternation = left/right anti-phase, "
    "duty_factor = fraction of time feet are planted, clearance = how high feet lift, plus forward speed and "
    "uprightness) and the CURRENT reward weights. Diagnose the failure mode and REDESIGN the reward WEIGHTS to "
    "induce a real ALTERNATING STEPPING gait.\n"
    "CRITICAL — reward the PROCESS, not forward velocity (Goodhart/Eureka anti-hacking): high forward speed with "
    "cadence~0, clearance~0, duty~1.0 is a SLIDE/DRAG, not a walk; the policy games fwd_vel by sliding. To fix a "
    "slide: RAISE slip_w (penalize feet moving while planted) and swing_w (pay for lifting a foot HIGH), keep "
    "vtgt MODEST (a high target invites a lunge), and use alt_w to force left/right stepping. For a hop "
    "(alternation low but cadence>0): raise alt_w. For falling/leaning: don't over-raise vtgt; favor clearance + "
    "smoothness. Output ONLY the JSON: failure_diagnosis (what the gait is doing wrong), key_change (the single "
    "biggest lever you're pulling and why), rationale, and weights (ALL of: "
    + ", ".join(GAIT_REWARD_KNOBS) + "). Change weights meaningfully but stay within sane ranges."
)


def critique_gait(diagnosis: dict, config: dict, llm, *, history: list | None = None,
                  prior_knowledge: str = "") -> dict | None:
    """Given a gait ``diagnosis`` (from gait_diagnostics.diagnose_gait) and the current reward ``config``
    (weights), return ``{weights, failure_diagnosis, key_change, rationale, changes}`` with NEW, clamped reward
    weights — or None if no LLM backend. ``history`` = prior ``[{weights, gait_quality, summary}]`` rounds so the
    LLM reflects on what helped/hurt (Eureka reflection). ``prior_knowledge`` (Phase 2 RAG) = a retrieved
    PRIOR-KNOWLEDGE block (trainer tips + nearest walking skill's recipe for this body), prepended."""
    if llm is None:
        return None

    cur = {k: float(config.get(k, GAIT_REWARD_KNOBS[k][2])) for k in GAIT_REWARD_KNOBS}
    hist_txt = ""
    if history:
        hist_txt = "\n\nPrior rounds (weights -> resulting gait_quality):\n" + "\n".join(
            f"  round {i}: gait_quality={h.get('gait_quality')}, walking={h.get('is_walking')}, "
            f"weights={ {k: round(float(h['weights'].get(k, 0)), 2) for k in GAIT_REWARD_KNOBS} }, "
            f"gait='{h.get('summary', '')[:90]}'"
            for i, h in enumerate(history))
    user = (
        (f"{prior_knowledge}\n\n" if prior_knowledge else "")
        + f"GAIT DIAGNOSIS: {diagnosis}\n"
        f"CURRENT reward weights: {cur}\n"
        f"Bounds (min,max): { {k: (v[0], v[1]) for k, v in GAIT_REWARD_KNOBS.items()} }"
        f"{hist_txt}\n\n"
        "Redesign the reward weights to make it take real alternating steps (raise cadence, alternation, and "
        "foot clearance; kill any slide). Return the JSON."
    )
    try:
        raw = llm.complete_json(_SYSTEM, user, _SCHEMA, max_tokens=700)
    except Exception as exc:  # noqa: BLE001
        return {"weights": None, "error": str(exc), "valid": False}

    w = raw.get("weights", {}) or {}
    new = {}
    for k, (lo, hi, dflt) in GAIT_REWARD_KNOBS.items():
        try:
            new[k] = float(max(lo, min(hi, float(w.get(k, cur[k])))))   # CLAMP to safe bounds
        except (TypeError, ValueError):
            new[k] = cur[k]
    changes = {k: (round(cur[k], 2), round(new[k], 2)) for k in GAIT_REWARD_KNOBS if abs(new[k] - cur[k]) > 1e-6}
    return {"weights": new, "failure_diagnosis": raw.get("failure_diagnosis", ""),
            "key_change": raw.get("key_change", ""), "rationale": raw.get("rationale", ""),
            "changes": changes, "valid": True, "backend": getattr(llm, "name", "?")}
