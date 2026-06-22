"""Reward Translator agent (plan Phase 1 / Language-to-Rewards): NL task -> RewardSpec.

The highest-leverage step toward task diversity (per the research): turn a natural-language
task ("lift 2 kg boxes onto a shelf", "spray-paint this panel", "carry the box across the
room") into a structured, trainable objective — a sparse success criterion + bounded dense
shaping terms over a fixed quantity vocabulary + scene/end-effector needs + DR ranges. The LLM
works in this constrained parameter space (L2R), and every proposal is validated against the
schema + vocabulary; infeasible specs are fed back (self-repair). Scored later by the SPARSE
metric only, never the dense reward (Eureka anti-hacking).

Offline-safe: no backend -> None, and the caller falls back to the deterministic task graph.
"""

from __future__ import annotations

from virturoid.schemas.reward import (
    ALLOWED_QUANTITIES,
    END_EFFECTORS,
    TERM_KINDS,
    Criterion,
    RewardSpec,
    RewardTerm,
)

_TERM_SCHEMA = {
    "type": "object",
    "properties": {
        "quantity": {"type": "string", "enum": sorted(ALLOWED_QUANTITIES)},
        "kind": {"type": "string", "enum": sorted(TERM_KINDS)},
        "weight": {"type": "number"},
        "target": {"type": ["number", "null"]},
    },
    "required": ["quantity", "kind", "weight"],
    "additionalProperties": False,
}
_CRIT_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "expression": {"type": "string"}},
    "required": ["name", "expression"],
    "additionalProperties": False,
}

REWARD_SCHEMA = {
    "type": "object",
    "properties": {
        "task_type": {"type": "string"},
        "sparse_success": _CRIT_SCHEMA,
        "failure": {"type": "array", "items": _CRIT_SCHEMA},
        "dense_terms": {"type": "array", "items": _TERM_SCHEMA},
        "scene_requirements": {"type": "array", "items": {"type": "string"}},
        "end_effector": {"type": "string", "enum": sorted(END_EFFECTORS)},
        "domain_randomization": {"type": "object"},
        "rationale": {"type": "string"},
    },
    "required": ["task_type", "sparse_success", "failure", "dense_terms"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You are Virturoid's reward translator. Turn the user's robot task into a STRUCTURED training "
    "objective (do NOT write control code). Output ONLY JSON with: task_type (snake_case), "
    "sparse_success ({name, expression}) — the unambiguous condition that defines task completion; "
    "failure (list, MUST include an episode timeout); dense_terms (bounded shaping aids, each "
    "{quantity, kind, weight, target}); scene_requirements (objects/zones the task needs); "
    "end_effector; domain_randomization. dense_terms.quantity must be one of: "
    + ", ".join(sorted(ALLOWED_QUANTITIES))
    + ". kind is one of attract/repel/bonus/penalty and |weight| <= 10. The sparse_success is the "
    "metric; dense_terms are only optimization hints (keep them bounded so they can't be gamed).\n"
    "CRITICAL — reward the PROCESS, not just the outcome (avoid reward hacking / Goodhart): a "
    "pure outcome metric is gameable. 'object_in_zone == 1' is satisfied by a box that was KICKED "
    "or thrown into the zone exactly as well as one placed under control — but only the controlled "
    "behaviour is the real skill, and only it transfers to reality. So for ANY task that moves an "
    "object, sparse_success MUST include a process gate (e.g. 'object_in_zone == 1 and "
    "object_settled == 1', or add 'object_speed < 0.1', or require 'grasp_contact' held), and "
    "dense_terms SHOULD include a bounded process penalty (object_speed or energy) so the policy is "
    "rewarded for controlled manipulation, not launching. A validator checks the vocabulary and "
    "flags hacking risks; correct them when asked."
)


def translate_task_to_reward(prompt: str, requirements, llm, max_repairs: int = 2) -> dict | None:
    """NL task -> validated RewardSpec dict. None if no LLM backend (offline-safe)."""
    if llm is None:
        return None

    user = (
        f"Task: {prompt}\n"
        f"Robot class: {getattr(requirements, 'robot_class', '') or 'unknown'}. "
        f"Payload: {getattr(requirements, 'payload_kg', None)} kg. Reach: {getattr(requirements, 'reach_m', None)} m.\n"
        f"Allowed dense-term quantities: {sorted(ALLOWED_QUANTITIES)}.\n"
        "Return the RewardSpec JSON (sparse_success is the metric; include a timeout in failure)."
    )

    attempts = 0
    last_issues: list[str] = []
    for _ in range(max_repairs + 1):
        attempts += 1
        try:
            raw = llm.complete_json(_SYSTEM, user, REWARD_SCHEMA)
        except Exception as exc:  # noqa: BLE001
            return {"spec": None, "backend": getattr(llm, "name", "?"), "attempts": attempts, "valid": False, "error": str(exc)}
        spec = _to_spec(raw)
        issues = spec.validate()
        if not issues:
            # Spec is schema-valid; now reason about whether it's GAMEABLE and self-repair if so
            # (the kick-vs-place problem). We harden the success gate before accepting, but never
            # block forever on advisories — surface any residual risks to the caller.
            risks = spec.hacking_risks()
            if risks and attempts <= max_repairs:
                user = (f"{user}\n\nYour RewardSpec {raw} is valid but GAMEABLE: {'; '.join(risks)} "
                        "Return corrected JSON with a process-aware success gate and a bounded process penalty.")
                continue
            return {"spec": spec, "raw": raw, "backend": getattr(llm, "name", "?"),
                    "attempts": attempts, "valid": True, "hacking_risks": risks}
        last_issues = issues
        user = f"{user}\n\nYour previous RewardSpec {raw} was rejected: {'; '.join(issues)}. Return corrected JSON."

    return {"spec": None, "backend": getattr(llm, "name", "?"), "attempts": attempts, "valid": False, "issues": last_issues}


def _to_spec(raw: dict) -> RewardSpec:
    def crit(d):
        d = d or {}
        return Criterion(name=d.get("name", ""), expression=d.get("expression", ""))

    return RewardSpec(
        task_type=raw.get("task_type", ""),
        sparse_success=crit(raw.get("sparse_success")),
        failure=[crit(c) for c in raw.get("failure", [])],
        dense_terms=[RewardTerm(quantity=t.get("quantity", ""), kind=t.get("kind", ""),
                                weight=float(t.get("weight", 0.0)), target=t.get("target"))
                     for t in raw.get("dense_terms", [])],
        scene_requirements=list(raw.get("scene_requirements", [])),
        end_effector=raw.get("end_effector", "gripper"),
        domain_randomization=dict(raw.get("domain_randomization", {})),
        rationale=raw.get("rationale", ""),
    )
