"""Transfer Agent (plan §8, §8.6, Phase E): reuse prior designs/policies across robots.

A reasoning agent (routed to a cloud model — Claude or ChatGPT) that, given a new
task's requirements and a set of candidate records retrieved from memory, decides
whether to transfer a prior solution and which assets to carry over (design, scenes,
learned policy). The decision is not trusted blindly: the chosen candidate must be
one that was actually offered, and its converged design must be physically feasible
for the *new* robot (its reach envelope must cover the new task's reach). Infeasible
or hallucinated choices are fed back to the model (self-repair).

With no LLM backend it returns None and the caller uses the deterministic
``memory_store.find_similar_design`` transfer — offline-safe.
"""

from __future__ import annotations

_ASSETS = {"design", "scenes", "policy"}

TRANSFER_SCHEMA = {
    "type": "object",
    "properties": {
        "reuse": {"type": "boolean"},
        "source_key": {"type": "string"},
        "assets": {"type": "array", "items": {"type": "string", "enum": sorted(_ASSETS)}},
        "rationale": {"type": "string"},
    },
    "required": ["reuse"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You are Virturoid's transfer agent. Given a new task and a list of prior build records "
    "(each with a key, robot_class, task_type, converged_design, and success_rate), decide "
    "whether to warm-start from one of them. Output ONLY JSON: reuse (bool), source_key (the "
    "key of the chosen prior record, required if reuse=true), assets (subset of design, scenes, "
    "policy), and rationale. Prefer a high success_rate record whose design can physically cover "
    "the new task's reach. A validator checks feasibility on the new robot and may correct you."
)


def recommend_transfer(prompt: str, requirements, candidates: list[dict], llm, max_repairs: int = 2) -> dict | None:
    """Decide whether/what to transfer from memory; validate feasibility. None if no backend."""
    if llm is None:
        return None
    if not candidates:
        return {"decision": {"reuse": False}, "backend": getattr(llm, "name", "?"), "attempts": 0, "valid": True}

    required_reach = _required_reach(requirements)
    listing = "\n".join(
        f"- key={c.get('key')} class={c.get('robot_class')} task={c.get('task_type')} "
        f"success={c.get('success_rate')} design={c.get('converged_design')}"
        for c in candidates
    )
    by_key = {c.get("key"): c for c in candidates}
    user = (
        f"New task: {prompt}\n"
        f"New robot reach requirement: {required_reach:.3f} m.\n"
        f"Candidate prior records:\n{listing}\n"
        "Return JSON: {reuse, source_key, assets, rationale}."
    )

    attempts = 0
    last_issues: list[str] = []
    for _ in range(max_repairs + 1):
        attempts += 1
        try:
            raw = llm.complete_json(_SYSTEM, user, TRANSFER_SCHEMA)
        except Exception as exc:  # noqa: BLE001
            return {"decision": None, "backend": getattr(llm, "name", "?"), "attempts": attempts, "valid": False, "error": str(exc)}
        issues = _validate(raw, by_key, required_reach)
        if not issues:
            decision = {
                "reuse": bool(raw["reuse"]),
                "source_key": raw.get("source_key") if raw.get("reuse") else None,
                "assets": [a for a in (raw.get("assets") or []) if a in _ASSETS] if raw.get("reuse") else [],
                "rationale": raw.get("rationale", ""),
            }
            return {"decision": decision, "backend": getattr(llm, "name", "?"), "attempts": attempts, "valid": True}
        last_issues = issues
        user = f"{user}\n\nYour previous decision {raw} was rejected: {'; '.join(issues)}. Return corrected JSON."

    return {"decision": None, "backend": getattr(llm, "name", "?"), "attempts": attempts, "valid": False, "issues": last_issues}


def _validate(raw: dict, by_key: dict, required_reach: float) -> list[str]:
    issues: list[str] = []
    if not raw.get("reuse"):
        return issues  # declining to transfer is always valid
    key = raw.get("source_key")
    if key not in by_key:
        issues.append(f"source_key {key!r} is not one of the offered candidates {sorted(by_key)}")
        return issues
    design = (by_key[key] or {}).get("converged_design") or {}
    forearm = design.get("forearm_mm")
    wrist = design.get("wrist_mm")
    if forearm is None or wrist is None:
        issues.append(f"candidate {key!r} has no usable converged_design to transfer")
    else:
        envelope = (forearm + wrist) / 1000.0 + 0.07
        if envelope < required_reach + 0.02:
            issues.append(
                f"candidate {key!r} reach envelope {envelope:.3f} m cannot cover the new task's "
                f"required {required_reach + 0.02:.3f} m — not transferable"
            )
    for a in raw.get("assets") or []:
        if a not in _ASSETS:
            issues.append(f"asset {a!r} is not one of {sorted(_ASSETS)}")
    return issues


def _required_reach(requirements) -> float:
    try:
        from virturoid.services.design_agent import _required_reach_for_task
        from virturoid.services.task_builder import build_task_graph

        return _required_reach_for_task(build_task_graph(requirements))
    except Exception:  # noqa: BLE001
        return float(getattr(requirements, "reach_m", None) or 0.46)
