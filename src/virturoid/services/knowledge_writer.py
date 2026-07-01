"""Knowledge write-back — bank a VERIFIED tip after a build passes its honesty gate (Phase 4, close the loop).

Phase 4 of the robotics-native-AI plan (``docs/robotics_native_ai_plan.md``): the flywheel actually turning.
The tip-writer was the under-used role — ``add_species_tip`` had one writer and no automatic trigger. Here it
fires on every honesty-gate PASS: given a *verified* build outcome, reflect and record a concise reusable tip
keyed to the species (which Phase 1's ``index_tips`` then makes retrievable by morphology for the next build).

The discipline is the whole point — **"LLM proposes, physics disposes."** We record ONLY verified wins (the
caller invokes this in the success branch, and we re-check the outcome met its bar), so the store never fills
with hopeful-but-false claims. Offline-safe: with no LLM, we write a grounded heuristic tip from the real
outcome numbers; the write is idempotent (``add_species_tip`` dedups).
"""

from __future__ import annotations

from virturoid.schemas.gene import RobotGene

_TIP_SCHEMA = {
    "type": "object",
    "properties": {"tip": {"type": "string"}, "audience": {"type": "string"}},
    "required": ["tip"], "additionalProperties": False,
}
_SYSTEM = (
    "You are Virturoid's reflector. A robot build just PASSED its honest physics evaluation. Write ONE "
    "concise, reusable tip (<= 30 words) a future designer/trainer of a SIMILAR body should know — grounded "
    "ONLY in the given outcome, no speculation. Output JSON {tip, audience} where audience is 'builder' or "
    "'trainer'."
)


def _heuristic_tip(gene: RobotGene, task_type: str, success: float, outcome: dict | None) -> str:
    """A grounded tip from the real outcome numbers (the offline / fallback reflector)."""
    base = f"{gene.robot_class} '{gene.species}' reached {success:.0%} on {task_type or 'its task'}"
    bits = []
    for key in ("cadence", "forward_m", "upright_frac", "blocks_placed", "controller"):
        if outcome and outcome.get(key) is not None:
            bits.append(f"{key}={outcome[key]}")
    extra = f" ({', '.join(bits)})" if bits else ""
    return f"{base}{extra} - reuse this body + its recipe to warm-start similar {task_type or 'task'} builds."


def _reflect_tip(gene: RobotGene, task_type: str, success: float, outcome: dict | None, llm) -> tuple[str, str]:
    """Return ``(tip, audience)``. LLM-authored when a backend is present + valid, else the heuristic."""
    heuristic = _heuristic_tip(gene, task_type, success, outcome)
    if llm is None:
        return heuristic, "builder"
    try:
        user = (f"Body: {gene.robot_class} '{gene.species}'. Task: {task_type}. "
                f"Outcome: success={success:.2f}, {outcome or {}}. Write the tip.")
        raw = llm.complete_json(_SYSTEM, user, _TIP_SCHEMA)
        tip = (raw.get("tip") or "").strip()
        aud = raw.get("audience") if raw.get("audience") in ("builder", "trainer") else "builder"
        return (tip or heuristic), aud
    except Exception:  # noqa: BLE001 - reflection is best-effort; fall back to the grounded heuristic
        return heuristic, "builder"


def record_verified_knowledge(db, gene: RobotGene, species_pattern: str, *, task_type: str,
                              success: float, target: float, outcome: dict | None = None,
                              llm=None, min_success: float | None = None) -> dict:
    """Bank a verified tip for ``species_pattern`` IFF the build actually passed its bar (verified-only).

    ``db`` is a MemoryDB. Returns ``{wrote_tip, tip?, reason?}``. Called from the build's success branch, but
    we re-gate on ``success >= (min_success or target)`` so a caller can't accidentally record a non-win — the
    anti-Goodhart discipline that keeps the knowledge store trustworthy.
    """
    threshold = target if min_success is None else min_success
    if success <= 0.0 or success < threshold:
        return {"wrote_tip": False, "reason": "outcome did not pass its bar (verified-only write-back)"}
    tip, audience = _reflect_tip(gene, task_type, success, outcome, llm)
    wrote = bool(db.add_species_tip(species_pattern, tip, audience=audience))
    return {"wrote_tip": wrote, "tip": tip, "audience": audience, "species": species_pattern}
