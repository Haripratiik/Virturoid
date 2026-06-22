"""Morphology embedding: a robot gene -> a fixed-length feature vector (the "vector space" for robots).

The platform's premise is LEGO-like: a few parametric building blocks (``GeneSegment``s — links, joints,
shapes) compose into ANY robot, and novel robots are auto-categorized into a self-organizing species
tree. Categorizing by hand-written taxonomy strings (token Jaccard) is brittle; this gives every gene a
deterministic numeric embedding that captures its morphology *meaning* — degrees of freedom, joint mix,
kinematic shape, geometry, end-effector, base — so "near in vector space" means "morphologically
similar", exactly how word embeddings cluster meaning. Species discovery (``species_discovery.py``) uses
nearest-neighbour in this space to place a new robot under its closest relative or detect it as novel.

Pure-Python and dependency-free (lives beside the SQLite memory layer); the embedding is fixed-length
and stable, so it can be cached on a species node and compared cheaply.
"""

from __future__ import annotations

import math

from virturoid.schemas.gene import RobotGene

# end-effector / base vocabularies (one-hot blocks of the embedding)
_EE = ("gripper", "suction", "spray_nozzle", "hook", "none")
_BASE = ("floor", "table", "torso")

# Per-feature scales so heterogeneous features (counts ~1-10, lengths ~0.3 m, mass ~1 kg) land at ~O(1),
# making Euclidean distance meaningful across the whole vector. Tuned to typical tabletop/mobile scales.
_SCALES = {
    "n_segments": 6.0, "dof": 6.0, "n_revolute": 6.0, "n_prismatic": 4.0, "n_fixed": 3.0,
    "depth": 6.0, "max_branching": 3.0, "n_branch_points": 3.0, "n_leaves": 4.0,
    "total_length": 1.0, "max_reach": 1.0, "mean_length": 0.3, "std_length": 0.3,
    "mean_radius": 0.1, "total_mass": 3.0, "mean_mass": 1.0, "n_fingers": 3.0, "symmetry": 3.0,
}

FEATURE_NAMES = (
    list(_SCALES.keys())
    + ["axis_x", "axis_y", "axis_z"]
    + [f"ee_{e}" for e in _EE]
    + [f"base_{b}" for b in _BASE]
)


def _depth_and_branching(gene: RobotGene):
    children: dict[str, list[str]] = {}
    for s in gene.segments:
        children.setdefault(s.parent, []).append(s.name)
    child_counts = [len(children.get(s.name, [])) for s in gene.segments]
    n_leaves = sum(1 for c in child_counts if c == 0)
    n_branch_points = sum(1 for c in child_counts if c >= 2)
    max_branching = max(child_counts) if child_counts else 0

    by_name = {s.name: s for s in gene.segments}

    def chain_len(seg) -> float:
        # longest reach from this segment outward (sum of link lengths along the deepest branch)
        kids = children.get(seg.name, [])
        return seg.length_m + (max((chain_len(by_name[k]) for k in kids), default=0.0))

    def chain_depth(seg) -> int:
        kids = children.get(seg.name, [])
        return 1 + (max((chain_depth(by_name[k]) for k in kids), default=0))

    root = gene.root()
    max_reach = chain_len(root) if root else sum(s.length_m for s in gene.segments)
    depth = chain_depth(root) if root else len(gene.segments)
    return depth, max_branching, n_branch_points, n_leaves, max_reach, children


def embed_gene(gene: RobotGene) -> list[float]:
    """Deterministic morphology feature vector (length ``len(FEATURE_NAMES)``), scaled to ~O(1)."""
    segs = gene.segments
    n = len(segs)
    revolute = [s for s in segs if s.joint_type == "revolute"]
    prismatic = [s for s in segs if s.joint_type == "prismatic"]
    fixed = [s for s in segs if s.joint_type in (None, "fixed")]
    actuated = revolute + prismatic
    depth, max_branching, n_branch_points, n_leaves, max_reach, children = _depth_and_branching(gene)

    lengths = [s.length_m for s in segs] or [0.0]
    mean_length = sum(lengths) / len(lengths)
    std_length = (sum((x - mean_length) ** 2 for x in lengths) / len(lengths)) ** 0.5
    masses = [s.mass_kg for s in segs] or [0.0]
    radii = [s.radius_m for s in segs] or [0.0]

    # joint-axis distribution over the ACTUATED joints (which way the robot articulates)
    ax = [0.0, 0.0, 0.0]
    for s in actuated:
        a = s.joint_axis
        k = max(range(3), key=lambda i: abs(a[i]))  # dominant axis
        ax[k] += 1.0
    tot_ax = sum(ax) or 1.0
    ax = [v / tot_ax for v in ax]

    # parallel-jaw-style fingers: prismatic leaves; symmetry: pairs with opposite-sign y mount offset
    n_fingers = sum(1 for s in prismatic if not children.get(s.name))
    ys = [round(s.mount_offset[1], 4) for s in segs if abs(s.mount_offset[1]) > 1e-6]
    symmetry = sum(1 for y in ys if -y in ys) / 2.0

    raw = {
        "n_segments": n, "dof": len(actuated), "n_revolute": len(revolute),
        "n_prismatic": len(prismatic), "n_fixed": len(fixed), "depth": depth,
        "max_branching": max_branching, "n_branch_points": n_branch_points, "n_leaves": n_leaves,
        "total_length": sum(lengths), "max_reach": max_reach, "mean_length": mean_length,
        "std_length": std_length, "mean_radius": sum(radii) / len(radii), "total_mass": sum(masses),
        "mean_mass": sum(masses) / len(masses), "n_fingers": n_fingers, "symmetry": symmetry,
    }
    vec = [raw[k] / _SCALES[k] for k in _SCALES]
    vec += ax
    vec += [1.0 if gene.end_effector_type == e else 0.0 for e in _EE]
    vec += [1.0 if gene.base_mount == b else 0.0 for b in _BASE]
    return vec


def distance(a: list[float], b: list[float]) -> float:
    """Euclidean distance in the scaled morphology space (smaller = more similar)."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)
