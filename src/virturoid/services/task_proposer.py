"""Task proposer: LLM compiles a free prompt into a verifiable TaskSpec over the capability registry,
with a deterministic single-skill fallback (offline / on verify failure). The LLM-Modulo loop: propose
→ verify against morphology + skills → on failure, fall back (never run an unverified task).
"""

from __future__ import annotations

from virturoid.schemas.task_spec import Entity, Predicate, PredicateOp, SkillCall, TaskSpec
from virturoid.services.capability_registry import REGISTRY, skill_menu


def _fallback_task(prompt: str, gene) -> TaskSpec:
    """A deterministic single-skill task from the prompt + the robot's structure — the offline stand-in for the
    LLM proposer. Infers the REQUESTED task from the prompt FIRST so the morphology verifier can honestly reject
    a mismatch (an arm asked to 'solve a maze' is reported INFEASIBLE, never silently re-run as pick-place); only
    when the prompt names no task does it fall back to the morphology's natural default."""
    from virturoid.services.task_matched_eval import robot_kind
    kind = robot_kind(gene)
    p = (prompt or "").lower()

    def has(*words):
        return any(w in p for w in words)

    # 1) the task the prompt ASKS FOR (verified against morphology downstream — mismatch -> honest infeasible)
    if has("maze"):
        skill, op = "solve_maze", PredicateOp.PATH_FOLLOWED
    elif has("navigate", "drive to", "go to", "reach the", "patrol", "to the goal"):
        skill, op = "navigate", PredicateOp.REACHED_GOAL
    elif has("walk", "run", "gait", "locomot", "trot", "gallop", "crawl", "stride"):
        skill, op = "locomote", PredicateOp.TRAVELED
    elif has("spray", "paint", "coat", "cover the"):
        skill, op = "spray_coverage", PredicateOp.SURFACE_COVERED
    elif has("grasp", "lift", "pick up") and not has("sort", "place", "bin", "into", "deliver"):
        skill, op = "grasp_lift", PredicateOp.OBJECT_LIFTED
    elif has("sort", "place", "pick", "bin", "deliver", "stack", "load", "carry"):
        skill, op = "pick_place", PredicateOp.OBJECTS_PLACED
    else:
        # 2) no explicit task in the prompt -> the morphology's natural default task
        skill, op = {"mobile": ("navigate", PredicateOp.REACHED_GOAL),
                     "legged": ("locomote", PredicateOp.TRAVELED),
                     "spray": ("spray_coverage", PredicateOp.SURFACE_COVERED)}.get(
            kind, ("pick_place", PredicateOp.OBJECTS_PLACED))
    return TaskSpec(name=skill, prompt=prompt, steps=[SkillCall("s0", skill, {"prompt": prompt})],
                    goal=[Predicate(op)], source="heuristic")


_SYSTEM_TMPL = (
    "Compile a natural-language robot task into a TaskSpec JSON, sequencing ONLY these skills:\n{menu}\n\n"
    "Goal/failure predicate ops (use exactly these): {ops}.\n"
    "Rules: every step.skill MUST be one of the listed skills; pick skills the robot's morphology can run; "
    "list the goal as the predicate ops the task requires; declare any named entities the predicates use. "
    "Prefer a short linear sequence. Output ONLY the JSON."
)
_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "entities": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "kind": {"type": "string"}}}},
        "steps": {"type": "array", "items": {"type": "object", "properties": {
            "step_id": {"type": "string"}, "skill": {"type": "string"},
            "params": {"type": "object"}}, "required": ["skill"]}},
        "goal": {"type": "array", "items": {"type": "object", "properties": {
            "op": {"type": "string"}, "args": {"type": "array"}}, "required": ["op"]}},
        "failure": {"type": "array", "items": {"type": "object", "properties": {
            "op": {"type": "string"}, "args": {"type": "array"}}}},
    },
    "required": ["steps", "goal"],
}


def propose_task(prompt: str, gene, *, llm="auto") -> TaskSpec:
    """Propose a verified TaskSpec for ``prompt`` on ``gene``. Falls back to a single-skill task offline
    or when the LLM proposal fails verification."""
    if llm == "auto":
        from virturoid.services.llm_client import get_llm
        llm = get_llm("planner")
    if llm is not None:
        try:
            from virturoid.services.task_verifier import verify_task
            system = _SYSTEM_TMPL.format(menu=skill_menu(),
                                         ops=", ".join(o.value for o in PredicateOp))
            raw = llm.complete_json(system, f"Task: {prompt}", _SCHEMA, max_tokens=2000)
            spec = TaskSpec.from_dict(raw)
            spec.prompt, spec.source = prompt, "llm"
            if verify_task(spec, gene)["ok"]:            # only accept a feasible proposal
                return spec
        except Exception:  # noqa: BLE001 - any LLM/parse/verify failure falls back, never blocks
            pass
    return _fallback_task(prompt, gene)
