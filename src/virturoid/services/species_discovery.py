"""Autonomous species discovery: place ANY built/imported robot into the species tree by morphology.

The agents compose robots from building blocks (``GeneSegment``s); this is what makes the resulting
tree self-organizing instead of hand-curated. Every gene is embedded (``morphology_embedding``) and
matched against the existing species by nearest-neighbour in that vector space:

  * a near-duplicate of an existing node  -> CLASSIFIED as a parameter variant of that species;
  * anything else                         -> a NEW species is created and parented under its closest
                                             morphological relative (so the tree grows by similarity,
                                             like clustering word embeddings — no hand-written taxonomy).

Deterministic and LLM-free (an LLM may later *rename* a node, but discovery never needs one). The new
node carries the serialized gene + readable morphology tags, so it joins the flywheel and the
Obsidian-style vault immediately.
"""

from __future__ import annotations

from virturoid.schemas.gene import RobotGene
from virturoid.services.morphology_embedding import embed_gene, distance

# distance below which a gene is "the same species" (a parameter variant), not a new node. Calibrated
# from real genes: arm<->arm+gripper ~1.2, humanoid<->spray ~1.4 (distinct species); pure rescales <~0.5.
CLASSIFY_THRESHOLD = 0.6


def _node_gene(node: dict) -> RobotGene | None:
    g = node.get("genes")
    if isinstance(g, dict) and g.get("segments") and isinstance(g["segments"][0], dict):
        try:
            return RobotGene.from_dict(g)
        except (KeyError, TypeError, ValueError):
            return None
    return None


def nearest_species(gene: RobotGene, nodes: list[dict], limit: int = 5) -> list[dict]:
    """Rank existing species by morphology distance to ``gene`` (the vector-space retrieval seam)."""
    vec = embed_gene(gene)
    scored = []
    for node in nodes:
        ng = _node_gene(node)
        if ng is None:
            continue
        scored.append({"species_pattern": node["species_pattern"], "robot_class": node.get("robot_class"),
                       "distance": round(distance(vec, embed_gene(ng)), 4), "buildable": node.get("buildable")})
    scored.sort(key=lambda d: d["distance"])
    return scored[:limit]


def morphology_tags(gene: RobotGene) -> list[str]:
    """Readable tags derived from morphology (also feed the legacy tag/Jaccard paths + the vault)."""
    dof = len(gene.actuated_joints())
    children = {s.parent for s in gene.segments}
    branched = any(sum(1 for s in gene.segments if s.parent == p) >= 2 for p in children)
    n_fingers = sum(1 for s in gene.segments
                    if s.joint_type == "prismatic" and not any(c.parent == s.name for c in gene.segments))
    tags = [gene.robot_class, f"dof{dof}", f"links{len(gene.segments)}",
            f"ee_{gene.end_effector_type}", f"base_{gene.base_mount}",
            "branched" if branched else "serial"]
    if n_fingers:
        tags.append(f"fingers{n_fingers}")
    return tags


def _auto_pattern(gene: RobotGene, taken: set[str]) -> str:
    """A readable, unique taxonomy id from the morphology when the gene's species is unset/taken."""
    dof = len(gene.actuated_joints())
    base = f"{gene.robot_class}.dof{dof}.{gene.end_effector_type}"
    if base not in taken:
        return base
    i = 2
    while f"{base}.v{i}" in taken:
        i += 1
    return f"{base}.v{i}"


def transfer_candidates(gene: RobotGene, db, *, task_type: str | None = None, limit: int = 3) -> list[dict]:
    """The nearest prior robots in morphology space, each with what's reusable — the vector-space moat.

    For a freshly-composed/imported gene, returns ranked neighbors carrying the stored gene (to amend or
    warm-start the BODY) and, if a ``task_type`` is given, that class's banked skill (to warm-start the
    POLICY). Lets a new robot reuse the closest prior robot's design + skill instead of starting cold,
    even across hand-named species — the flywheel made semantic by the embedding (startup plan §8.6/§13).
    """
    nodes = db.species_tree_nodes()
    by_pattern = {n["species_pattern"]: n for n in nodes}
    out = []
    for nb in nearest_species(gene, nodes, limit=limit):
        node = by_pattern.get(nb["species_pattern"], {})
        skill = None
        if task_type and node.get("robot_class"):
            try:
                skill = db.recall_skill(node["robot_class"], task_type)
            except Exception:  # noqa: BLE001 - skills table optional
                skill = None
        out.append({"species_pattern": nb["species_pattern"], "robot_class": nb["robot_class"],
                    "distance": nb["distance"], "buildable": nb.get("buildable"),
                    "gene": node.get("genes"), "banked_skill": skill})
    return out


def auto_place_species(gene: RobotGene, db, *, source: str = "discovered",
                       classify_threshold: float = CLASSIFY_THRESHOLD) -> dict:
    """Place ``gene`` in the species tree by morphology nearest-neighbour. Returns a placement dict
    ``{action, species_pattern, parent, distance, neighbors}`` where action is 'classified' (variant of
    an existing species) or 'created' (a new species node parented under its closest relative)."""
    nodes = db.species_tree_nodes()
    neighbors = nearest_species(gene, nodes, limit=5)
    nearest = neighbors[0] if neighbors else None

    if nearest and nearest["distance"] <= classify_threshold:
        return {"action": "classified", "species_pattern": nearest["species_pattern"],
                "parent": None, "distance": nearest["distance"], "neighbors": neighbors}

    # NEW species: parent = nearest relative if it shares the robot class, else the class root.
    taken = {n["species_pattern"] for n in nodes}
    pattern = gene.species if (gene.species and gene.species not in taken) else _auto_pattern(gene, taken)
    parent = None
    if nearest and nearest.get("robot_class") == gene.robot_class:
        parent = nearest["species_pattern"]
    else:
        parent = next((n["species_pattern"] for n in nodes
                       if n.get("robot_class") == gene.robot_class and n.get("parent_species") is None), None)
    db.upsert_species_node(
        pattern, parent_species=parent, robot_class=gene.robot_class,
        # A discovered species carries a full compiled gene, so it IS re-buildable from that gene -- mark it
        # buildable so flywheel reuse (find_gene_for_class needs buildable=1 AND genes) can amend it next time.
        morphology_tags=morphology_tags(gene), genes=gene.to_dict(), buildable=True, source=source)
    return {"action": "created", "species_pattern": pattern, "parent": parent,
            "distance": (nearest["distance"] if nearest else None), "neighbors": neighbors}
