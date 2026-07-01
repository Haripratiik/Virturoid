"""Species Agent (startup plan §7.2): the model that grows the robot species tree.

This is the dedicated agent for *class and gene creation*. Given a request and the
existing taxonomy, it either (a) classifies the request into an existing species, or
(b) proposes a NEW species node — a taxonomy id (``humanoid.arm.single``), its parent,
robot class, morphology tags, and a parametric "gene" (a kinematic encoding: links +
joint chain + end effectors). Over many builds this continuously extends the tree, the
way the tree of life branches from kingdoms down to species.

Routed to the small local model (high-volume, cheap). Every proposal is validated:
taxonomy-id shape, a known parent (existing node or a known robot-class root), tags, and
— for a new gene — a valid kinematic tree (reuses the Robot Parser's tree checks: single
root, no cycles, connected, known links). Bad proposals are fed back (self-repair).

"LLM proposes, physics disposes": creating a species node does NOT mean it is buildable.
``species_tree.resolve_buildable`` decides what actually gets built (the nearest buildable
relative), and the build is always honest about requested-vs-built. With no LLM backend
this returns None and the deterministic ``morphology_selector`` is used — offline-safe.
"""

from __future__ import annotations

import re

from virturoid.services.robot_parser_agent import _validate as _validate_kinematic_tree

_PATTERN_RE = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)*$")

_GENE_SCHEMA = {
    "type": "object",
    "properties": {
        "links": {"type": "array", "items": {"type": "string"}},
        "joints": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"}, "joint_type": {"type": "string"},
                "parent_link": {"type": "string"}, "child_link": {"type": "string"},
            },
            "required": ["name", "joint_type", "parent_link", "child_link"],
            "additionalProperties": False,
        }},
        "end_effectors": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["links", "joints", "end_effectors"],
    "additionalProperties": False,
}

SPECIES_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["existing", "new"]},
        "species_pattern": {"type": "string"},
        "parent_species": {"type": "string"},
        "robot_class": {"type": "string"},
        "morphology_tags": {"type": "array", "items": {"type": "string"}},
        "genes": _GENE_SCHEMA,
        "rationale": {"type": "string"},
    },
    "required": ["decision", "species_pattern", "robot_class"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You are Virturoid's species taxonomist. Virturoid keeps a tree of robot species, like "
    "biology's tree of life (kingdom -> ... -> species). Given a request and the existing "
    "species, either CLASSIFY it into an existing species or CREATE a new species node. Output "
    "ONLY JSON: decision ('existing'|'new'), species_pattern (dotted lowercase taxonomy id, e.g. "
    "'humanoid.arm.single' or 'fixed_arm.three_dof.tabletop'), parent_species (an existing node "
    "or a known robot-class root), robot_class, morphology_tags, and — for a NEW species — genes "
    "(a kinematic encoding: links, joints [{name,joint_type,parent_link,child_link}] forming a "
    "valid tree with one root and no cycles, and end_effectors). Reuse an existing species when "
    "the request fits one; only branch a new node when the morphology is genuinely distinct. A "
    "validator checks the taxonomy id, parent, and kinematic tree and may ask you to correct it."
)


def classify_or_create_species(prompt: str, requirements, existing_nodes: list[dict],
                               known_roots: list[str], llm, max_repairs: int = 2,
                               prior_knowledge: str = "") -> dict | None:
    """Classify into an existing species or propose a new one. None if no backend.

    ``prior_knowledge`` (Phase 2 RAG): a retrieved PRIOR-ART block (similar past builds + reusable skills)
    prepended to the prompt so the designer reuses/amends proven morphology instead of proposing cold.
    """
    if llm is None:
        return None

    existing_patterns = {n.get("species_pattern") for n in existing_nodes if n.get("species_pattern")}
    listing = "\n".join(
        f"- {n['species_pattern']} (class={n.get('robot_class')}, parent={n.get('parent_species')}, "
        f"tags={n.get('morphology_tags')}, buildable={n.get('buildable')})"
        for n in existing_nodes
    ) or "(none yet)"
    user = (
        (f"{prior_knowledge}\n\n" if prior_knowledge else "")
        + f"Request: {prompt}\n"
        f"Existing species:\n{listing}\n"
        f"Known robot-class roots: {sorted(known_roots)}\n"
        "Classify into an existing species_pattern, or create a new one with a parent and genes. "
        "Return the JSON object."
    )

    attempts = 0
    last_issues: list[str] = []
    valid_parents = existing_patterns | set(known_roots)
    for _ in range(max_repairs + 1):
        attempts += 1
        try:
            raw = llm.complete_json(_SYSTEM, user, SPECIES_SCHEMA)
        except Exception as exc:  # noqa: BLE001
            return {"species": None, "backend": getattr(llm, "name", "?"), "attempts": attempts, "valid": False, "error": str(exc)}
        issues = _validate(raw, existing_patterns, valid_parents)
        if not issues:
            return {
                "species": {
                    "decision": raw["decision"],
                    "species_pattern": raw["species_pattern"],
                    "parent_species": raw.get("parent_species"),
                    "robot_class": raw["robot_class"],
                    "morphology_tags": raw.get("morphology_tags", []),
                    "genes": raw.get("genes"),
                    "rationale": raw.get("rationale", ""),
                },
                "backend": getattr(llm, "name", "?"), "attempts": attempts, "valid": True,
            }
        last_issues = issues
        user = f"{user}\n\nYour previous proposal {raw} was rejected: {'; '.join(issues)}. Return corrected JSON."

    return {"species": None, "backend": getattr(llm, "name", "?"), "attempts": attempts, "valid": False, "issues": last_issues}


def _validate(raw: dict, existing_patterns: set, valid_parents: set) -> list[str]:
    issues: list[str] = []
    decision = raw.get("decision")
    pattern = raw.get("species_pattern")
    if decision not in ("existing", "new"):
        issues.append("decision must be 'existing' or 'new'")
    if not pattern or not _PATTERN_RE.match(pattern):
        issues.append(f"species_pattern {pattern!r} must be a dotted lowercase taxonomy id")
    if not raw.get("robot_class"):
        issues.append("robot_class is required")

    if decision == "existing":
        if pattern not in existing_patterns:
            issues.append(f"species_pattern {pattern!r} is not an existing species {sorted(existing_patterns)}")
    elif decision == "new":
        if pattern in existing_patterns:
            issues.append(f"species_pattern {pattern!r} already exists; use decision='existing'")
        parent = raw.get("parent_species")
        if not parent:
            issues.append("a new species must name a parent_species")
        elif parent not in valid_parents:
            issues.append(f"parent_species {parent!r} is not a known node or root {sorted(valid_parents)}")
        if not raw.get("morphology_tags"):
            issues.append("a new species must include morphology_tags")
        genes = raw.get("genes")
        if genes:
            # Reuse the Robot Parser's kinematic-tree validation on the gene.
            tree_issues = _validate_kinematic_tree(genes)
            issues.extend(f"genes: {t}" for t in tree_issues)
    return issues
