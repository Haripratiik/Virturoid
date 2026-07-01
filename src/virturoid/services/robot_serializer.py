"""Robot serializer — turn a robot gene into something an LLM/AI agent can *read*.

This is Phase 0 (the foundation) of the robotics-native-AI plan (``docs/robotics_native_ai_plan.md``):
the bridge that makes a robot legible to a language model. An LLM understands words because a *tokenizer*
turns text into tokens it can reason over; here we turn a ``RobotGene`` into (a) the per-joint **token
list** — the robot's "words", the same actuated-joint unit ``morph_graph.encode_robot`` feeds the RL
policy — and (b) an LLM-readable **natural-language summary** grounded entirely in the gene (no
hallucination). Every other seam (RAG-injecting the LLM design roles, the ``describe_robot`` agent tool,
the understanding surface) consumes this.

Each token carries the **TopoPE topology path** (BodyGen's edit-invariant positional code: the sequence of
"which child of my parent" indices from the root down to this part) — so this is a genuine down payment on
the tokenizer vision, not just a pretty-printer: structurally-similar limbs across different robots get
*aligned* paths.

Pure-Python and dependency-free (lives beside ``morphology_embedding``); no MuJoCo, no GPU. "LLM proposes,
physics disposes" — this is the read side that lets the LLM propose *about a specific body* instead of blind.
"""

from __future__ import annotations

from virturoid.schemas.gene import GeneSegment, RobotGene

_AXIS_LABELS = ("x", "y", "z")


def _dominant_axis(axis: tuple[float, float, float]) -> str:
    """The axis the joint mostly rotates/slides about (for a human-readable label)."""
    return _AXIS_LABELS[max(range(3), key=lambda i: abs(axis[i]))]


def _children_map(gene: RobotGene) -> dict[str | None, list[str]]:
    """parent name -> ordered child names (segment order is stable => a stable child index)."""
    kids: dict[str | None, list[str]] = {}
    for s in gene.segments:
        kids.setdefault(s.parent, []).append(s.name)
    return kids


def _topo_path(gene: RobotGene, seg: GeneSegment, kids: dict[str | None, list[str]]) -> list[int]:
    """TopoPE path: the sequence of child-indices from the root down to ``seg`` (edit-invariant).

    At each step we record which child ``seg`` (or its ancestor) is among its parent's children. Adding a
    limb *elsewhere* in the tree does not shift this path — the property that lets structurally-similar
    limbs on different robots share an aligned positional code (BodyGen / TopoPE).
    """
    by_name = {s.name: s for s in gene.segments}
    chain: list[int] = []
    node: GeneSegment | None = seg
    guard = 0
    while node is not None and node.parent is not None and guard < len(gene.segments) + 1:
        siblings = kids.get(node.parent, [])
        chain.append(siblings.index(node.name) if node.name in siblings else 0)
        node = by_name.get(node.parent)
        guard += 1
    chain.reverse()
    return chain


def _depth(gene: RobotGene, seg: GeneSegment) -> int:
    """Number of ancestors up to the root (root = 0)."""
    by_name = {s.name: s for s in gene.segments}
    d, node, guard = 0, seg, 0
    while node is not None and node.parent is not None and guard < len(gene.segments) + 1:
        d += 1
        node = by_name.get(node.parent)
        guard += 1
    return d


def robot_tokens(gene: RobotGene) -> list[dict]:
    """The robot's tokens: one per **actuated joint** (the canonical unit ``morph_graph`` feeds the policy).

    Non-actuated links (the welded base, a fixed gripper) are structural — summarized in ``describe_robot``'s
    stats, not tokenized — matching the RL tokenizer's contract so this serializes *the same* vocabulary.
    """
    kids = _children_map(gene)
    tokens: list[dict] = []
    for s in gene.actuated_joints():
        tokens.append({
            "name": s.name,
            "joint_type": s.joint_type,                       # revolute | prismatic
            "axis": _dominant_axis(s.joint_axis),             # dominant axis label
            "axis_vec": [round(float(v), 4) for v in s.joint_axis],
            "range": ([round(s.joint_lower, 4), round(s.joint_upper, 4)]
                      if s.joint_lower is not None and s.joint_upper is not None else None),
            "torque_nm": s.actuator_torque_nm,
            "mass_kg": round(s.mass_kg, 4),
            "length_m": round(s.length_m, 4),
            "radius_m": round(s.radius_m, 4),
            "parent": s.parent,
            "depth": _depth(gene, s),
            "topo_path": _topo_path(gene, s, kids),           # TopoPE positional code (edit-invariant)
            "is_end_effector": s.is_end_effector,
        })
    return tokens


def robot_stats(gene: RobotGene) -> dict:
    """Grounded structured aggregate of the body (every number comes straight from the gene)."""
    segs = gene.segments
    actuated = gene.actuated_joints()
    revolute = [s for s in actuated if s.joint_type == "revolute"]
    prismatic = [s for s in actuated if s.joint_type == "prismatic"]
    kids = _children_map(gene)
    # branches off the base = kinematic chains (legs of a quadruped, arms of a humanoid, the single arm of a
    # manipulator). Described generically as "chains" to stay honest across every morphology.
    root = gene.root()
    n_chains = len(kids.get(root.name, [])) if root else len(kids.get(None, []))
    depths = [_depth(gene, s) for s in segs]
    axis_hist = {"x": 0, "y": 0, "z": 0}
    for s in actuated:
        axis_hist[_dominant_axis(s.joint_axis)] += 1
    ee = gene.end_effector()
    return {
        "robot_class": gene.robot_class,
        "species": gene.species,
        "dof": len(actuated),
        "n_links": len(segs),
        "n_revolute": len(revolute),
        "n_prismatic": len(prismatic),
        "n_chains": n_chains,
        "tree_depth": max(depths) if depths else 0,
        "total_mass_kg": round(sum(s.mass_kg for s in segs), 3),
        "axis_mix": axis_hist,
        "end_effector_type": gene.end_effector_type,
        "end_effector_link": ee.name if ee else None,
        "base_mount": gene.base_mount,
        "parent_species": gene.parent_species,
    }


def _summary_text(gene: RobotGene, stats: dict, tokens: list[dict]) -> str:
    """A concise, honest natural-language description an LLM can reason over — every clause is grounded."""
    lines: list[str] = []
    chains = stats["n_chains"]
    chain_word = "chain" if chains == 1 else "chains"
    lines.append(
        f"{stats['robot_class']} '{stats['species']}': {stats['dof']}-DOF, {stats['n_links']} links, "
        f"{stats['total_mass_kg']} kg total, {chains} kinematic {chain_word} off a {stats['base_mount']} base."
    )
    jmix = []
    if stats["n_revolute"]:
        jmix.append(f"{stats['n_revolute']} revolute")
    if stats["n_prismatic"]:
        jmix.append(f"{stats['n_prismatic']} prismatic")
    ax = stats["axis_mix"]
    ax_str = ", ".join(f"{v}x-{k}" for k, v in ax.items() if v)
    lines.append(
        f"Joints: {' + '.join(jmix) or 'none actuated'}; rotation/slide axes {ax_str or 'n/a'}; "
        f"tree depth {stats['tree_depth']}."
    )
    ee = stats["end_effector_type"]
    lines.append(
        f"End effector: {ee}" + (f" (link '{stats['end_effector_link']}')." if stats["end_effector_link"] else ".")
        + (f" Amended from species '{stats['parent_species']}'." if stats["parent_species"] else "")
    )
    # per-joint detail, but only when small enough to be legible (else the stats above carry it)
    if 0 < len(tokens) <= 14:
        for t in tokens:
            rng = f" range {t['range']}" if t["range"] else ""
            tq = f", {t['torque_nm']} N-m" if t["torque_nm"] else ""
            lines.append(f"  - {t['name']}: {t['joint_type']} about {t['axis']}{rng}{tq} (depth {t['depth']}).")
    return "\n".join(lines)


def describe_robot(gene: RobotGene) -> dict:
    """Serialize a robot gene into an LLM-readable form: ``{tokens, summary_text, stats}``.

    The single seam that makes a robot legible to a language model — the read side every downstream role
    (RAG-injected design/reward/planner roles, the ``describe_robot`` agent tool, ``recall_knowledge``)
    builds on. Pure-Python, deterministic, grounded entirely in the gene.
    """
    tokens = robot_tokens(gene)
    stats = robot_stats(gene)
    return {"tokens": tokens, "summary_text": _summary_text(gene, stats, tokens), "stats": stats}


def structural_diagnosis(gene: RobotGene) -> dict:
    """A fast, PHYSICS-FREE structural read of a body: ``{stats, reach_m, observations}``.

    Grounded, unambiguous facts an agent can reason over to understand what the body *is* and its structural
    limits (DOF, reach, end effector, limb count) — the cheap "understand this robot" tool. Not a physics
    evaluation: it makes no claim about whether the body will actually succeed at a task (that needs the
    simulator); every observation here is read straight off the gene.
    """
    stats = robot_stats(gene)
    by_name = {s.name: s for s in gene.segments}
    kids = _children_map(gene)

    def _chain_len(name: str, guard: int = 0) -> float:
        s = by_name.get(name)
        if s is None or guard > len(gene.segments):
            return 0.0
        ch = kids.get(name, [])
        return s.length_m + (max((_chain_len(k, guard + 1) for k in ch), default=0.0))

    root = gene.root()
    reach = _chain_len(root.name) if root else sum(s.length_m for s in gene.segments)

    obs: list[str] = [f"{stats['dof']} actuated DOF"
                      + (" - low articulation (<3 DOF), limited dexterity" if stats["dof"] < 3 else "")]
    obs.append(f"reach ~{reach:.2f} m along the longest kinematic chain")
    if stats["end_effector_type"] in (None, "none"):
        obs.append("no end effector - a locomotion/mobility body, not a manipulator")
    else:
        obs.append(f"end effector: {stats['end_effector_type']} (link '{stats['end_effector_link']}')")
    if stats["n_chains"] >= 3:
        obs.append(f"{stats['n_chains']} kinematic chains off the base (legged / multi-limb)")
    if stats["n_prismatic"] and not stats["n_revolute"]:
        obs.append("prismatic-only articulation (linear stages, no rotation)")
    return {"stats": stats, "reach_m": round(reach, 3), "observations": obs}
