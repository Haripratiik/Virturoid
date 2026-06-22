"""Obsidian-style species vault: the robot tree of life as linked markdown notes.

The species tree lives canonically in SQLite (queryable, fast), but the user wants it
browsable and annotatable like an Obsidian vault — one markdown note per species, linked
parent<->child with ``[[wikilinks]]``, each carrying *tips* the building and training
agents can read and append. This module renders the SQLite tree into such a vault (and an
index note), so a human can open it in Obsidian and the agents have a place to leave
species-specific knowledge.

Note names are the species pattern (``humanoid.arm.single.md``). Frontmatter carries the
structured fields; the body carries lineage links, children links, agent tips, and the
known tasks/failures pulled from memory. This mirrors how Virturoid's own assistant memory
works (markdown + frontmatter + ``[[links]]`` + an index), applied to robots.
"""

from __future__ import annotations

from pathlib import Path

from virturoid.services.memory_db import MemoryDB


def export_species_vault(db: MemoryDB, vault_dir: Path) -> int:
    """Render the species tree to an Obsidian vault of linked notes. Returns note count."""
    vault_dir = Path(vault_dir)
    vault_dir.mkdir(parents=True, exist_ok=True)
    nodes = db.species_tree_nodes()
    by_parent: dict[str, list[dict]] = {}
    for n in nodes:
        by_parent.setdefault(n.get("parent_species") or "", []).append(n)
    neighbors = _morphology_neighbors(nodes)   # vector-space nearest species per node

    for n in nodes:
        (vault_dir / f"{_safe(n['species_pattern'])}.md").write_text(
            _render_note(db, n, by_parent.get(n["species_pattern"], []),
                         neighbors.get(n["species_pattern"], [])), encoding="utf-8"
        )
    (vault_dir / "Species Tree.md").write_text(_render_index(nodes, by_parent), encoding="utf-8")
    return len(nodes)


def _morphology_neighbors(nodes: list[dict], k: int = 3) -> dict[str, list[tuple[str, float]]]:
    """Per species, its nearest other species in the morphology vector space — so the Obsidian graph's
    links mirror the embedding neighbourhood (the 'vector space categorizes meaning' idea, for robots)."""
    from virturoid.schemas.gene import RobotGene
    from virturoid.services.morphology_embedding import distance, embed_gene

    embedded = []
    for n in nodes:
        g = n.get("genes")
        if isinstance(g, dict) and g.get("segments"):
            try:
                embedded.append((n["species_pattern"], embed_gene(RobotGene.from_dict(g))))
            except (KeyError, TypeError, ValueError):
                continue
    out: dict[str, list[tuple[str, float]]] = {}
    for sp, vec in embedded:
        ranked = sorted(((other, round(distance(vec, ov), 3)) for other, ov in embedded if other != sp),
                        key=lambda kv: kv[1])
        out[sp] = ranked[:k]
    return out


def _render_note(db: MemoryDB, node: dict, children: list[dict],
                 neighbors: list[tuple[str, float]] | None = None) -> str:
    sp = node["species_pattern"]
    tags = node.get("morphology_tags") or []
    parent_link = f"[[{_safe(node['parent_species'])}]]" if node.get("parent_species") else "(root)"
    lines = [
        "---",
        f"species: {sp}",
        f"robot_class: {node.get('robot_class', '')}",
        f"parent_species: {node.get('parent_species') or ''}",
        f"buildable: {bool(node.get('buildable'))}",
        f"source: {node.get('source', '')}",
        f"morphology_tags: [{', '.join(tags)}]",
        "---",
        "",
        f"# {sp}",
        "",
        f"**Robot class:** {node.get('robot_class', '?')}  ",
        f"**Buildable:** {'yes' if node.get('buildable') else 'no (traces to nearest buildable relative when built)'}",
        "",
        "## Lineage",
        f"- Parent: {parent_link}",
    ]
    if children:
        lines.append("- Children: " + ", ".join(f"[[{_safe(c['species_pattern'])}]]" for c in children))
    else:
        lines.append("- Children: (leaf)")

    # Morphology neighbours: the vector-space nearest species (links the graph by embedding, not lineage).
    if neighbors:
        lines += ["", "## Morphology neighbors (vector space)"]
        lines += [f"- [[{_safe(sp)}]] (distance {d})" for sp, d in neighbors]

    # Tips the agents have left for this species.
    lines += ["", "## Tips for builder / trainer agents"]
    tips = node.get("tips") or []
    if tips:
        lines += [f"- *({t.get('audience', 'agent')})* {t.get('tip')}" for t in tips]
    else:
        lines.append("- _(no tips yet — agents append what they learn here)_")

    # Grounded knowledge pulled from memory for this species' class/tasks.
    best = _best_for_species(db, sp)
    if best:
        lines += ["", "## What worked", f"- Best converged design: `{best.get('converged_design')}` "
                  f"(success {best.get('success_rate')}, task {best.get('task_type')})"]
    fails = _failures_for_species(db, node.get("robot_class"))
    if fails:
        lines += ["", "## Known failures (avoid)"]
        lines += [f"- {f['label']} ×{f['total']}" + (f" — {f['recommended_action']}" if f.get('recommended_action') else "") for f in fails]

    if node.get("genes"):
        g = node["genes"]
        lines += ["", "## Gene (kinematic encoding)",
                  f"- links: {g.get('links')}", f"- joints: {len(g.get('joints', []))}", f"- end_effectors: {g.get('end_effectors')}"]
    lines.append("")
    return "\n".join(lines)


def _render_index(nodes: list[dict], by_parent: dict[str, list[dict]]) -> str:
    lines = ["# Species Tree", "", "The robot tree of life. Each note is a species; links are lineage.", ""]
    roots = sorted({(n.get("parent_species") or "") for n in nodes} - {n["species_pattern"] for n in nodes})

    def walk(pattern: str, depth: int) -> None:
        for c in sorted(by_parent.get(pattern, []), key=lambda x: x["species_pattern"]):
            mark = "" if c.get("buildable") else "  _(not buildable yet)_"
            lines.append(f"{'  ' * depth}- [[{_safe(c['species_pattern'])}]]{mark}")
            walk(c["species_pattern"], depth + 1)

    for root in roots:
        lines.append(f"- **{root or '(root)'}** _(class root)_")
        walk(root, 1)
    # Also list any nodes whose parent is itself a node (covered by walk), plus orphans.
    listed = set()
    for root in roots:
        _collect(root, by_parent, listed)
    orphans = [n for n in nodes if n["species_pattern"] not in listed]
    if orphans:
        lines += ["", "## Other"]
        lines += [f"- [[{_safe(n['species_pattern'])}]]" for n in orphans]
    lines.append("")
    return "\n".join(lines)


def _collect(pattern: str, by_parent: dict, acc: set) -> None:
    for c in by_parent.get(pattern, []):
        if c["species_pattern"] not in acc:
            acc.add(c["species_pattern"])
            _collect(c["species_pattern"], by_parent, acc)


def _best_for_species(db: MemoryDB, species_pattern: str) -> dict | None:
    row = db.conn.execute(
        "SELECT converged_design, success_rate, task_type FROM runs WHERE species=? "
        "AND converged_design IS NOT NULL ORDER BY success_rate DESC LIMIT 1",
        (species_pattern,),
    ).fetchone()
    if not row:
        return None
    import json
    d = dict(row)
    try:
        d["converged_design"] = json.loads(d["converged_design"])
    except (json.JSONDecodeError, TypeError):
        pass
    return d


def _failures_for_species(db: MemoryDB, robot_class) -> list[dict]:
    if not robot_class:
        return []
    rows = db.conn.execute(
        "SELECT label, SUM(count) AS total, MAX(recommended_action) AS recommended_action "
        "FROM failures WHERE robot_class=? GROUP BY label ORDER BY total DESC LIMIT 5",
        (robot_class,),
    ).fetchall()
    return [dict(r) for r in rows]


def _safe(name: str) -> str:
    """Obsidian note names can't contain path separators or some specials."""
    return (name or "").replace("/", "_").replace("\\", "_").replace(":", "_")
