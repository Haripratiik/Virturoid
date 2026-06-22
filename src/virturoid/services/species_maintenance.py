"""Ontology maintenance: keep the robot species tree from growing uncontrollably.

We cannot add every small variation as its own species, so the tree is periodically
*pruned*: near-duplicate siblings are **merged** (canonicalize → dedup, per the taxonomy
research), and **redundant unused leaves** are **trimmed**. This keeps "our memory of
robots" bounded and clean.

Two layers, "LLM proposes, schema disposes":

- ``prune_tree`` — the deterministic engine. It finds merge/trim candidates and applies
  only the safe ones, behind hard gates: never merge across robot_class; never merge two
  buildable species into one; never trim/merge a node that has recorded builds or live
  children; only collapse genuinely near-duplicate siblings.
- ``OntologyMaintenanceAgent`` (gated by an LLM backend) — proposes which nodes to merge
  or trim with a rationale; every proposal is then re-checked by the same deterministic
  gates before anything is changed. With no backend, the deterministic engine runs alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Sibling similarity at/above this (Jaccard over morphology tags) is a merge candidate.
_MERGE_SIM_THRESHOLD = 0.8


@dataclass
class PruneResult:
    merged: list[tuple[str, str]] = field(default_factory=list)   # (dropped, kept)
    trimmed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    note: str = ""


def prune_tree(db, llm=None, sim_threshold: float = _MERGE_SIM_THRESHOLD, dry_run: bool = False) -> PruneResult:
    """Merge near-duplicate siblings and trim redundant unused leaves. Returns what changed.

    If ``llm`` is provided, it proposes operations first; each proposal is validated by the
    same gates as the deterministic pass, so a bad suggestion can never corrupt the tree.
    With ``dry_run`` the tree is not modified — the result lists what *would* change.
    """
    result = PruneResult()
    nodes = {n["species_pattern"]: n for n in db.species_tree_nodes()}

    proposals = _agent_proposals(db, list(nodes.values()), llm) if llm is not None else None

    # 1) Merge near-duplicate siblings.
    merged_away: set[str] = set()
    merge_targets: set[str] = set()  # survivors of a merge — protect them from same-pass trim
    for drop, keep, reason in (proposals["merges"] if proposals else _merge_candidates(nodes, sim_threshold)):
        if drop in merged_away or keep in merged_away:
            continue
        ok, why = _can_merge(db, nodes.get(drop), nodes.get(keep))
        if not ok:
            result.skipped.append(f"merge {drop}->{keep}: {why}")
            continue
        if dry_run or db.merge_species(drop, keep):
            result.merged.append((drop, keep))
            merged_away.add(drop)
            merge_targets.add(keep)

    # 2) Trim redundant unused leaves (but never a node we just merged into).
    if not dry_run:
        nodes = {n["species_pattern"]: n for n in db.species_tree_nodes()}  # refresh after merges
    for pattern, reason in (proposals["trims"] if proposals else _trim_candidates(db, nodes)):
        if pattern in merged_away or pattern in merge_targets:
            continue
        ok, why = _can_trim(db, nodes.get(pattern))
        if not ok:
            result.skipped.append(f"trim {pattern}: {why}")
            continue
        if dry_run or db.trim_species(pattern):
            result.trimmed.append(pattern)

    result.note = f"merged {len(result.merged)}, trimmed {len(result.trimmed)}, skipped {len(result.skipped)}"
    return result


# ---------------------------------------------------------------- safety gates
def _can_merge(db, drop: dict | None, keep: dict | None) -> tuple[bool, str]:
    if drop is None or keep is None:
        return False, "missing node"
    if drop["species_pattern"] == keep["species_pattern"]:
        return False, "same node"
    if drop.get("robot_class") != keep.get("robot_class"):
        return False, "different robot_class"
    if drop.get("buildable") and keep.get("buildable"):
        return False, "both buildable (distinct implemented species)"
    if db.species_usage(drop["species_pattern"]) > 0:
        return False, "dropped node has recorded builds"
    return True, ""


def _can_trim(db, node: dict | None) -> tuple[bool, str]:
    if node is None:
        return False, "missing node"
    if node.get("buildable"):
        return False, "buildable species are never trimmed"
    if db.species_children(node["species_pattern"]):
        return False, "not a leaf"
    if db.species_usage(node["species_pattern"]) > 0:
        return False, "has recorded builds"
    if node.get("tips"):
        return False, "carries agent tips worth keeping"
    return True, ""


# ---------------------------------------------------------------- deterministic candidates
def _morph_distance(a: dict, b: dict) -> float | None:
    """Morphology vector distance between two species (if both carry a gene), else None."""
    from virturoid.schemas.gene import RobotGene
    from virturoid.services.morphology_embedding import distance, embed_gene
    ga, gb = a.get("genes"), b.get("genes")
    if not (isinstance(ga, dict) and ga.get("segments") and isinstance(gb, dict) and gb.get("segments")):
        return None
    try:
        return distance(embed_gene(RobotGene.from_dict(ga)), embed_gene(RobotGene.from_dict(gb)))
    except (KeyError, TypeError, ValueError):
        return None


def _merge_candidates(nodes: dict, threshold: float, dist_threshold: float = 0.6
                      ) -> list[tuple[str, str, str]]:
    """Sibling pairs (same parent + class) that are near-duplicates. Prefers the morphology VECTOR
    distance (two species within ``dist_threshold`` are the same species — consistent with
    ``species_discovery.CLASSIFY_THRESHOLD``); falls back to morphology-tag Jaccard when a gene is
    missing. So tree maintenance dedupes in the same vector space that discovery places into."""
    out = []
    by_parent: dict[tuple, list[dict]] = {}
    for n in nodes.values():
        by_parent.setdefault((n.get("parent_species"), n.get("robot_class")), []).append(n)
    for siblings in by_parent.values():
        for i in range(len(siblings)):
            for j in range(i + 1, len(siblings)):
                a, b = siblings[i], siblings[j]
                d = _morph_distance(a, b)
                if d is not None:
                    if d > dist_threshold:
                        continue
                    reason = f"morphology distance {d:.2f}"
                else:
                    sim = _jaccard(set(a.get("morphology_tags") or []), set(b.get("morphology_tags") or []))
                    if sim < threshold:
                        continue
                    reason = f"tag similarity {sim:.2f}"
                keep, drop = _pick_keep(a, b)   # keep the buildable/used one; drop the other
                out.append((drop["species_pattern"], keep["species_pattern"], reason))
    return out


def _trim_candidates(db, nodes: dict) -> list[tuple[str, str]]:
    """Trim only leaves that are *subsumed* by a sibling (their morphology adds nothing).

    A genuinely novel species — unique morphology tags, no dominating sibling — is a real
    branch of the tree and is kept even if unused. We only cut dominated duplicates, so the
    tree stops accumulating tiny variations without losing distinct robot kinds.
    """
    out = []
    by_parent: dict = {}
    for n in nodes.values():
        by_parent.setdefault((n.get("parent_species"), n.get("robot_class")), []).append(n)
    for n in nodes.values():
        ok, _ = _can_trim(db, n)
        if not ok:
            continue
        tags = set(n.get("morphology_tags") or [])
        siblings = [s for s in by_parent.get((n.get("parent_species"), n.get("robot_class")), [])
                    if s["species_pattern"] != n["species_pattern"]]
        # Subsumed = some sibling's tags are a superset of this node's tags.
        if any(tags and set(s.get("morphology_tags") or []) >= tags for s in siblings):
            out.append((n["species_pattern"], "subsumed by a sibling (redundant variation)"))
    return out


def _pick_keep(a: dict, b: dict) -> tuple[dict, dict]:
    if a.get("buildable") and not b.get("buildable"):
        return a, b
    if b.get("buildable") and not a.get("buildable"):
        return b, a
    # Prefer the one with more tips, then the shorter (more general) pattern.
    a_score = (len(a.get("tips") or []), -len(a["species_pattern"]))
    b_score = (len(b.get("tips") or []), -len(b["species_pattern"]))
    return (a, b) if a_score >= b_score else (b, a)


def _jaccard(x: set, y: set) -> float:
    if not x or not y:
        return 0.0
    return len(x & y) / len(x | y)


# ---------------------------------------------------------------- LLM proposer (gated)
_AGENT_SCHEMA = {
    "type": "object",
    "properties": {
        "merges": {"type": "array", "items": {
            "type": "object",
            "properties": {"drop": {"type": "string"}, "keep": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["drop", "keep"], "additionalProperties": False,
        }},
        "trims": {"type": "array", "items": {
            "type": "object",
            "properties": {"species_pattern": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["species_pattern"], "additionalProperties": False,
        }},
    },
    "required": ["merges", "trims"], "additionalProperties": False,
}

_AGENT_SYSTEM = (
    "You are Virturoid's robotics ontology maintainer. Given the current robot species tree, "
    "propose which near-duplicate species to MERGE (collapse tiny variations) and which "
    "redundant unused leaf species to TRIM, so the tree stays compact like a clean biological "
    "taxonomy. Output ONLY JSON: merges [{drop, keep, reason}] and trims [{species_pattern, reason}]. "
    "Never merge across robot_class; prefer keeping buildable species and the more general node. "
    "A validator re-checks every proposal and will discard unsafe ones."
)


def _agent_proposals(db, nodes: list[dict], llm) -> dict | None:
    listing = "\n".join(
        f"- {n['species_pattern']} (class={n.get('robot_class')}, parent={n.get('parent_species')}, "
        f"buildable={n.get('buildable')}, tags={n.get('morphology_tags')}, "
        f"builds={db.species_usage(n['species_pattern'])})"
        for n in nodes
    ) or "(empty)"
    user = f"Current species tree:\n{listing}\nPropose merges and trims as JSON."
    try:
        raw = llm.complete_json(_AGENT_SYSTEM, user, _AGENT_SCHEMA)
    except Exception:  # noqa: BLE001 - fall back to deterministic pass
        return None
    merges = [(m.get("drop"), m.get("keep"), m.get("reason", "")) for m in raw.get("merges", []) if m.get("drop") and m.get("keep")]
    trims = [(t.get("species_pattern"), t.get("reason", "")) for t in raw.get("trims", []) if t.get("species_pattern")]
    return {"merges": merges, "trims": trims}
