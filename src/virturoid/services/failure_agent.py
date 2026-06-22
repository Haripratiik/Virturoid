"""Failure Analyst agent (plan §8, Phase D): reason over real failures → targeted fix.

After a real-physics evaluation, the pick-and-place run produces task-grounded failure
clusters (missed_grasp, wrong_bin, dropped, ...). This agent (routed to Claude for the
reasoning) reads those clusters and proposes a single, actionable fix — strengthen the
body, regenerate scenes, or tune the controller — with a hypothesis. The proposal is
validated against a schema and against the observed labels (it must target a failure that
actually happened and pick an action Virturoid can execute), then fed back on mismatch.

With no LLM backend it returns None and the caller uses the rule-based regression mapping
already in ``physics_evaluator`` — offline-safe.
"""

from __future__ import annotations

_ACTIONS = {
    "co_design_hardware": "search a longer/stronger body (reach or torque limited)",
    "regenerate_scenes": "regenerate scenes around the failing configuration",
    "tune_controller": "co-optimize controller params (engage radius, hover, gains)",
    "retrain_policy": "retrain the learned policy on the failing cases",
}

FAILURE_SCHEMA = {
    "type": "object",
    "properties": {
        "primary_failure": {"type": "string"},
        "root_cause": {"type": "string"},
        "recommended_action": {"type": "string", "enum": list(_ACTIONS)},
        "detail": {"type": "string"},
    },
    "required": ["primary_failure", "root_cause", "recommended_action"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You are Virturoid's failure analyst. You are given task failure clusters from a real "
    "MuJoCo pick-and-place evaluation. Diagnose the single most impactful failure and recommend "
    "ONE action Virturoid can execute. Output ONLY JSON: primary_failure (must be one of the "
    "observed cluster labels), root_cause (one sentence), recommended_action (one of: "
    + ", ".join(_ACTIONS) + "), and an optional detail. "
    "missed_grasp usually means reach/approach; wrong_bin means placement/scene; dropped means "
    "lift/transport; instability means controller gains."
)


def analyze_failures(clusters: list[dict], llm, max_repairs: int = 1) -> dict | None:
    """Diagnose failures and recommend an executable action. None if no backend."""
    if llm is None or not clusters:
        return None

    observed = {c.get("label") or c.get("failure_label") for c in clusters}
    observed.discard(None)
    summary = ", ".join(f"{c.get('label') or c.get('failure_label')}×{c.get('count', '?')}" for c in clusters)
    user = (
        f"Observed failure clusters: {summary}.\n"
        f"Valid primary_failure values: {sorted(observed)}.\n"
        "Return JSON with primary_failure, root_cause, recommended_action, detail."
    )

    attempts = 0
    last_issues: list[str] = []
    for _ in range(max_repairs + 1):
        attempts += 1
        try:
            raw = llm.complete_json(_SYSTEM, user, FAILURE_SCHEMA)
        except Exception as exc:  # noqa: BLE001
            return {"diagnosis": None, "backend": getattr(llm, "name", "?"), "attempts": attempts, "valid": False, "error": str(exc)}
        issues = _validate(raw, observed)
        if not issues:
            return {
                "diagnosis": {
                    "primary_failure": raw["primary_failure"],
                    "root_cause": raw["root_cause"],
                    "recommended_action": raw["recommended_action"],
                    "detail": raw.get("detail", ""),
                },
                "backend": getattr(llm, "name", "?"),
                "attempts": attempts,
                "valid": True,
            }
        last_issues = issues
        user = f"{user}\n\nYour previous diagnosis {raw} was rejected: {'; '.join(issues)}. Return corrected JSON."

    return {"diagnosis": None, "backend": getattr(llm, "name", "?"), "attempts": attempts, "valid": False, "issues": last_issues}


def _validate(raw: dict, observed: set) -> list[str]:
    issues: list[str] = []
    if raw.get("primary_failure") not in observed:
        issues.append(f"primary_failure {raw.get('primary_failure')!r} is not an observed cluster {sorted(observed)}")
    if raw.get("recommended_action") not in _ACTIONS:
        issues.append(f"recommended_action {raw.get('recommended_action')!r} is not executable {list(_ACTIONS)}")
    if not raw.get("root_cause"):
        issues.append("root_cause is required")
    return issues
