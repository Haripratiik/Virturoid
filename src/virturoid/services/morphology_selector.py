from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.morphology import MorphologyTemplate
from virturoid.schemas.requirements import RequirementsRecord
from virturoid.schemas.tasks import TaskGraph


@dataclass
class MorphologySelection:
    selected_template: MorphologyTemplate          # ALWAYS buildable (safe to dispatch)
    candidate_template_ids: list[str] = field(default_factory=list)
    rationale: str = ""
    fallback_used: bool = False
    # Species-tree honesty: the morphology the prompt actually maps to (may be unbuilt),
    # vs. selected_template (the nearest buildable relative). Equal on an exact match.
    requested_template: MorphologyTemplate | None = None
    requested_species: str = ""
    requested_robot_class: str = ""
    species_exact: bool = True
    species_note: str = ""


def select_morphology_template(
    requirements: RequirementsRecord,
    task: TaskGraph,
    templates: list[MorphologyTemplate],
) -> MorphologySelection:
    if not templates:
        raise ValueError("At least one morphology template is required.")

    scored = sorted(
        ((_score_template(requirements, task, template), template) for template in templates),
        key=lambda item: item[0],
        reverse=True,
    )
    selected_score, requested = scored[0]

    # The prompt may map to a species with no implemented builder yet (e.g. humanoid).
    # Trace the species tree to the nearest buildable relative so EVERY caller dispatches
    # a real builder, while we keep (and report) what was actually requested — never a
    # crash, never a silent mislabel. Imported lazily to avoid an import cycle.
    from virturoid.services.species_tree import resolve_buildable

    resolution = resolve_buildable(requested, templates)
    selected = resolution.template
    candidates = [template.id for score, template in scored if score > 0]
    return MorphologySelection(
        selected_template=selected,
        candidate_template_ids=candidates or [selected.id],
        rationale=_rationale(requirements, task, requested, selected_score),
        fallback_used=selected_score <= 0,
        requested_template=requested,
        requested_species=resolution.requested_species,
        requested_robot_class=requested.robot_class,
        species_exact=resolution.exact,
        species_note=resolution.note,
    )


def _score_template(requirements: RequirementsRecord, task: TaskGraph, template: MorphologyTemplate) -> int:
    score = 0
    prompt = requirements.prompt.lower()
    # Explicit robot-class intent is decisive and outweighs the generic task match, so
    # "humanoid robot arm" is recognized as a humanoid request (even with no humanoid
    # builder yet — the species tree then traces to the nearest buildable relative and
    # reports it honestly) rather than being silently flattened into a tabletop arm.
    explicit = _explicit_class(prompt)
    if explicit is not None and template.robot_class == explicit:
        score += 10  # explicit class intent dominates
    # Task match adds weight, but only for the explicit class (or any class if none named),
    # so a generic task can't pull a humanoid request back to the manipulator.
    if (explicit is None or template.robot_class == explicit) and task.task_type in template.supported_task_types:
        score += 5
    if requirements.environment in {"tabletop", "warehouse"} and template.robot_class == "manipulator" and explicit in (None, "manipulator"):
        score += 1
    return score


def _explicit_class(prompt: str) -> str | None:
    """The robot class explicitly named in the prompt, if any (most-specific first).

    Reuses the build planner's broad, word-boundary keyword vocabulary
    (``intent_planner._CLASS_WORDS`` / ``_kw``) instead of a narrow hand-list. The old hand-list was
    missing bare "dog" and the animal nouns, so "a dog like robot made to traverse a maze" matched no
    class and the navigation task pulled it to a wheeled ``mobile_base`` -- it built a rover, not a dog.
    The shared list maps "dog"/"cat"/"spider"/"traverse"/"walk" etc. to a quadruped/legged body, and the
    priority order checks quadruped BEFORE mobile_base so "dog ... maze" is a dog, not a rover. Imported
    lazily (same pattern as ``resolve_buildable`` above) to avoid any import cycle.
    """
    from virturoid.services.intent_planner import _CLASS_WORDS, _kw
    for cls in ("humanoid", "quadruped", "mobile_base", "manipulator"):
        if any(_kw(w, prompt) for w in _CLASS_WORDS[cls]):
            return cls
    return None


def _rationale(
    requirements: RequirementsRecord,
    task: TaskGraph,
    selected: MorphologyTemplate,
    score: int,
) -> str:
    if score <= 0:
        return f"No strong template match; selected {selected.id} as deterministic fallback for {task.task_type}."
    return (
        f"Selected {selected.id} because task_type={task.task_type}, "
        f"environment={requirements.environment}, and prompt='{requirements.prompt}' matched robot_class={selected.robot_class}."
    )
