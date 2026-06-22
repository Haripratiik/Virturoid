"""Robot species tree — the "tree of life" for robots (startup plan §7.2).

Virturoid organizes robot morphologies into a taxonomy (kingdom → ... → species),
where each node is a ``MorphologyTemplate`` carrying its parent, morphology tags, and a
build recipe ("gene"). Two responsibilities live here:

1. **Buildability** — not every species in the tree has an implemented builder yet
   (humanoid/quadruped are declared but ``builder_service`` points at a planned module).
   ``is_buildable`` tells them apart by checking the builder module actually imports.

2. **Tracing** — given the species a request maps to, ``resolve_buildable`` finds the
   nearest *buildable* relative (by taxonomic lineage, then morphology-tag similarity) so
   the builder can always produce something real. Crucially it reports BOTH the requested
   species and the one actually built, so Virturoid is honest: it never silently mislabels
   a humanoid as a tabletop arm, and never crashes on a not-yet-implemented class.

The Species Agent (``species_agent.py``) grows this tree by proposing new species nodes;
this module is the read/trace side the builder uses.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field

from virturoid.schemas.morphology import MorphologyTemplate


@dataclass
class BuildableResolution:
    requested_species: str          # what the request mapped to (may be unbuilt)
    built_as_species: str           # the buildable species actually used
    template: MorphologyTemplate    # the buildable template to build with
    exact: bool                     # True if requested == built (already buildable)
    lineage: list[str] = field(default_factory=list)  # requested -> ... -> built
    note: str = ""


def is_buildable(template: MorphologyTemplate) -> bool:
    """True if the template's builder_service points at an importable module.

    Real builders are dotted ``module.func`` paths under an importable package;
    placeholders like ``planned.humanoid_builder`` are not importable -> not buildable.
    """
    svc = (template.builder_service or "").strip()
    if not svc or svc.startswith("planned"):
        return False
    module_path = svc.rsplit(".", 1)[0]  # drop the function name
    try:
        return importlib.util.find_spec(module_path) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def buildable_templates(templates: list[MorphologyTemplate]) -> list[MorphologyTemplate]:
    return [t for t in templates if is_buildable(t)]


def resolve_buildable(selected: MorphologyTemplate, templates: list[MorphologyTemplate]) -> BuildableResolution:
    """Resolve the selected species to the nearest buildable one, honestly.

    If the selected species is already buildable, returns it as an exact match.
    Otherwise finds the nearest buildable relative — first up the taxonomic lineage
    (shared ``parent_species`` / ancestor), then by morphology-tag similarity within the
    same robot_class, then by global tag similarity — and records the lineage + a note.
    """
    requested = selected.species_pattern or selected.id
    if is_buildable(selected):
        return BuildableResolution(
            requested_species=requested, built_as_species=requested,
            template=selected, exact=True, lineage=[requested],
            note=f"'{requested}' is buildable.",
        )

    candidates = buildable_templates(templates)
    if not candidates:
        # Degenerate: nothing buildable at all. Return the selection unchanged so the
        # caller's existing "foundation package only" path handles it (never crash).
        return BuildableResolution(
            requested_species=requested, built_as_species=requested,
            template=selected, exact=False, lineage=[requested],
            note=f"No buildable species exist yet; '{requested}' produces a validated foundation package only.",
        )

    best = _nearest(selected, candidates)
    lineage = _lineage(selected, best)
    note = (
        f"No builder for requested species '{requested}' yet. Built as its nearest buildable "
        f"relative '{best.species_pattern}' ({best.robot_class}). Lineage: {' -> '.join(lineage)}. "
        f"The requested species is recorded in the tree; implementing its builder later upgrades it in place."
    )
    return BuildableResolution(
        requested_species=requested, built_as_species=best.species_pattern,
        template=best, exact=False, lineage=lineage, note=note,
    )


def _nearest(selected: MorphologyTemplate, candidates: list[MorphologyTemplate]) -> MorphologyTemplate:
    sel_tags = set(selected.morphology_tags)

    def score(c: MorphologyTemplate) -> tuple:
        same_class = 1 if c.robot_class == selected.robot_class else 0
        shared_ancestor = 1 if (selected.parent_species and selected.parent_species == c.parent_species) else 0
        tag_sim = _jaccard(sel_tags, set(c.morphology_tags))
        return (same_class, shared_ancestor, tag_sim)

    return max(candidates, key=score)


def _lineage(selected: MorphologyTemplate, built: MorphologyTemplate) -> list[str]:
    chain = [selected.species_pattern or selected.id]
    if selected.parent_species and selected.parent_species not in chain:
        chain.append(selected.parent_species)
    if built.parent_species and built.parent_species not in chain:
        chain.append(built.parent_species)
    if (built.species_pattern or built.id) not in chain:
        chain.append(built.species_pattern or built.id)
    return chain


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
