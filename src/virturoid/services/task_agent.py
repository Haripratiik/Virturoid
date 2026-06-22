"""Task Agent (plan §8, Phase D): natural language → a richer Task Graph.

A reasoning agent (routed to a cloud model — Claude or ChatGPT) that turns a
free-form task description into a structured Task Graph: the objects in the scene,
measurable success/failure criteria, safety constraints, and metrics. Every
proposal is validated against a schema and against semantic gates the simulator
relies on — there must be measurable success and failure criteria and an explicit
timeout — then fed back to the model on mismatch (self-repair).

With no LLM backend it returns None and the caller uses the deterministic
``build_task_graph`` — so the product runs fully offline. When the agent succeeds,
``to_task_graph`` converts the validated proposal into a real ``TaskGraph`` that
passes ``TaskGraph.validate()``.
"""

from __future__ import annotations

import re

from virturoid.schemas.tasks import Criterion, TaskGraph

_CRITERION_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "expression": {"type": "string"}},
    "required": ["name", "expression"],
    "additionalProperties": False,
}

TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "task_type": {"type": "string"},
        "objects": {"type": "array", "items": {"type": "string"}},
        "success_criteria": {"type": "array", "items": _CRITERION_SCHEMA},
        "failure_criteria": {"type": "array", "items": _CRITERION_SCHEMA},
        "safety_constraints": {"type": "array", "items": {"type": "string"}},
        "metrics": {"type": "array", "items": {"type": "string"}},
        "required_skills": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "task_type", "objects", "success_criteria", "failure_criteria", "metrics"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You are Virturoid's task planner. Turn the user's robot task into a structured Task Graph "
    "that a physics simulator can score. Output ONLY JSON with: name, task_type (snake_case, e.g. "
    "pick_place_sort or navigation), objects (the things in the scene), success_criteria and "
    "failure_criteria (each a {name, expression} where expression is a checkable condition, e.g. "
    "'red_block inside red_bin'), safety_constraints, metrics (must include success_rate), and "
    "required_skills. Every criterion must be objectively measurable. The failure_criteria MUST "
    "include an explicit episode timeout (e.g. 'episode_time > 60'). A validator checks your graph "
    "and may ask you to correct it."
)

_TIMEOUT_RE = re.compile(r"(episode_time|elapsed|time)\s*[><]=?\s*\d|timeout", re.IGNORECASE)


def synthesize_task_graph(prompt: str, requirements, llm, max_repairs: int = 2) -> dict | None:
    """Propose a Task Graph from NL; validate + self-repair. None if no backend."""
    if llm is None:
        return None

    user = (
        f"Task: {prompt}\n"
        f"Robot class hint: {getattr(requirements, 'robot_class', '') or 'unknown'}. "
        f"Reach: {getattr(requirements, 'reach_m', None)} m.\n"
        "Return JSON: {name, task_type, objects, success_criteria, failure_criteria, "
        "safety_constraints, metrics, required_skills}. Include an episode timeout in failure_criteria."
    )

    attempts = 0
    last_issues: list[str] = []
    for _ in range(max_repairs + 1):
        attempts += 1
        try:
            raw = llm.complete_json(_SYSTEM, user, TASK_SCHEMA)
        except Exception as exc:  # noqa: BLE001
            return {"graph": None, "backend": getattr(llm, "name", "?"), "attempts": attempts, "valid": False, "error": str(exc)}
        issues = _validate(raw)
        if not issues:
            return {"graph": raw, "backend": getattr(llm, "name", "?"), "attempts": attempts, "valid": True}
        last_issues = issues
        user = f"{user}\n\nYour previous task graph {raw} was rejected: {'; '.join(issues)}. Return corrected JSON."

    return {"graph": None, "backend": getattr(llm, "name", "?"), "attempts": attempts, "valid": False, "issues": last_issues}


def _validate(raw: dict) -> list[str]:
    issues: list[str] = []
    if not raw.get("name"):
        issues.append("name is required")
    if not raw.get("task_type"):
        issues.append("task_type is required")
    if not raw.get("objects"):
        issues.append("objects must be non-empty")
    for field_name in ("success_criteria", "failure_criteria"):
        crits = raw.get(field_name) or []
        if not crits:
            issues.append(f"{field_name} must be non-empty")
            continue
        for i, c in enumerate(crits):
            if not isinstance(c, dict) or not c.get("name") or not c.get("expression"):
                issues.append(f"{field_name}[{i}] needs a name and a measurable expression")
    failure_exprs = " ".join(c.get("expression", "") for c in (raw.get("failure_criteria") or []) if isinstance(c, dict))
    if not _TIMEOUT_RE.search(failure_exprs):
        issues.append("failure_criteria must include an explicit episode timeout (e.g. 'episode_time > 60')")
    metrics = raw.get("metrics") or []
    if "success_rate" not in metrics:
        issues.append("metrics must include 'success_rate'")
    return issues


def to_task_graph(proposal: dict, requirements) -> TaskGraph:
    """Convert a validated agent proposal into a TaskGraph that passes validate()."""
    g = proposal["graph"]
    graph = TaskGraph(
        id=f"task_{requirements.id}",
        name=g["name"],
        prompt=requirements.prompt,
        task_type=g["task_type"],
        required_skills=list(g.get("required_skills", [])),
        objects=list(g["objects"]),
        success_criteria=[Criterion(name=c["name"], expression=c["expression"]) for c in g["success_criteria"]],
        failure_criteria=[Criterion(name=c["name"], expression=c["expression"]) for c in g["failure_criteria"]],
        safety_constraints=list(g.get("safety_constraints", [])),
        metrics=list(g["metrics"]),
    )
    return graph
