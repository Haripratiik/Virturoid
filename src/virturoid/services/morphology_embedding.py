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

#: MASS IS LOG-COMPRESSED, and it is the only feature that has to be. ``_SCALES`` above states the contract —
#: every feature lands at ~O(1) so no single one owns the distance — and mass is the one that breaks it: a
#: linear divisor cannot hold 0.2 kg and 100 kg at O(1) at the same time, so the heaviest robot in any
#: comparison sets the direction of the whole vector.
#:
#: MEASURED 2026-08-08 on the live composer: "a small quadruped robot dog" (2.09 kg, anatomy route) against
#: "a large quadruped robot" (14.92 kg, template route) — two quadrupeds — cosine 0.813, of which
#: ``total_mass`` alone supplied 0.293 of the 0.375 squared distance, i.e. **78%**. Every other feature agreed
#: they were the same kind of machine. And the ordering was inverted, not merely blurred: the same large
#: quadruped scored 0.886 against a two-legged HUMANOID, so the nearest verified precedent to a quadruped was
#: a biped. That is the moat's own key, and ``_robotics_grounding`` gates warm-start at 0.85 — so the flywheel
#: answered "no close verified precedent" for a quadruped standing next to a banked quadruped.
#:
#: The root cause is mass being a pre-grounding PLACEHOLDER the two builders disagree about ~3x on at equal
#: size (1.31 vs 4.35 kg per metre of link) — re-ground both to the same physical band and the same pair reads
#: 0.994. A key that only works on already-grounded bodies is not a key: an ingested 15 kg Go2 must land next
#: to our composed dog, and no amount of fixing OUR builders can arrange that. So the key stops reading raw
#: kilograms. log1p over the same scale keeps the feature's meaning and its sign of variation (heavier is
#: still further) while bounding its span: the pair above moves 7.1x -> 2.5x apart and lands at 0.964, and
#: quadruped-vs-quadruped (0.964) now correctly outranks quadruped-vs-biped (0.907).
#:
#: ``embed_gene_rich`` below already did exactly this and named the same defect ("total_mass alone is 55% of
#: its variance"); it just never shipped, because it is gated behind a learned metric that has to prove itself
#: first. This is the same fix on the vector that actually ships.
#:
#: ONE FEATURE, and that is a measured choice, not caution. Swept against the 26-body physics-verified transfer
#: corpus (``embedding_eval``), compressing ``total_mass`` alone beat every wider variant on every axis --
#: kendall 0.4794 / pearson 0.5593 (vs 0.4756 / 0.5583 adding ``mean_mass``, 0.4782 / 0.5570 adding the length
#: magnitudes) -- and it opens the widest correct gap between quadruped-quadruped (0.9640) and quadruped-biped
#: (0.9073); the wider variants close that gap to +0.037 and +0.017.
#:
#: HONEST COST, on the same corpus: ``triplet_ranking_acc`` falls 0.9036 -> 0.8635 while the two
#: proximity-vs-measured-transfer CORRELATIONS both rise (kendall 0.4506 -> 0.4794, pearson 0.5050 -> 0.5593).
#: The corpus's own bodies come from our two builders, which disagree ~3x about placeholder mass at equal size,
#: so raw mass is partly a BUILDER label there and the triplet metric was partly scoring that. The correlations
#: are the less thresholded read and they improve; the class ordering above was flatly wrong before and is
#: right now. Both numbers are recorded here so the trade is visible rather than claimed as a clean win.
_LOG_SCALED = ("total_mass",)

#: Bumped whenever the map above changes, so a PERSISTED body index built by an older version is detected and
#: rebuilt instead of being silently compared against fresh vectors (see ``RoboticsVectorMemory``).
BODY_EMBED_VERSION = 2


def _scaled(name: str, raw: float) -> float:
    s = _SCALES[name]
    if name in _LOG_SCALED:
        return math.log1p(max(0.0, raw)) / math.log1p(s)
    return raw / s


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
    vec = [_scaled(k, raw[k]) for k in _SCALES]
    vec += ax
    vec += [1.0 if gene.end_effector_type == e else 0.0 for e in _EE]
    vec += [1.0 if gene.base_mount == b else 0.0 for b in _BASE]
    return vec


# --------------------------------------------------------------------------- rich embedding (flywheel path)
# The 29-D vector above is a super-class + SIZE detector: measured, `total_mass` alone is 55% of its variance
# (a rescaled body looks as far as a different class) and ~18 of 29 dims are dead, so it has near-zero resolution
# INSIDE the legged family — exactly where gait-hint transfer happens (cosine 0.91-1.0 for dog/hexapod/octopus).
# `embed_gene_rich` fixes the ROOT: log-compress the size magnitudes (kill the mass tyranny) and add the
# gait-relevant structure that already varies but was swamped (limb count, DoF-per-limb, serial-vs-branched,
# shape mix, cross-section anisotropy, joint ROM, wheels). Paired with `morphology_whiten` (mean-center + unit
# variance) so cosine escapes the positive orthant, this is what makes distance discriminate — measured by
# `embedding_eval`, not asserted. Used by `robotics_vector_memory.embed_body`; the 29-D `embed_gene` above stays
# the species-tree space until Wave 4 unifies them.

RICH_FEATURE_NAMES = (
    ["n_segments", "dof", "n_revolute", "n_prismatic", "n_fixed", "depth", "max_branching",
     "n_branch_points", "n_leaves", "log_total_length", "log_max_reach", "mean_length", "std_length",
     "mean_radius", "log_total_mass", "log_mean_mass", "n_fingers", "symmetry",
     # new discriminators (topology + shape + actuation) that separate the legged cone
     "limb_count", "dof_per_limb", "serial_ratio", "branch_ratio", "shape_box_frac", "shape_capsule_frac",
     "shape_cylinder_frac", "cross_section_aniso", "authored_geom_frac", "mean_joint_rom", "wheel_frac"]
    + ["axis_x", "axis_y", "axis_z"]
    + [f"ee_{e}" for e in _EE]
    + [f"base_{b}" for b in _BASE]
)

def embed_gene_rich(gene: RobotGene) -> list[float]:
    """Improved morphology feature vector (length ``len(RICH_FEATURE_NAMES)``): size magnitudes LOG-compressed so
    mass/length stop dominating, plus limb/shape/actuation features that resolve the legged family. Deterministic,
    pure-python. Best paired with the fitted whitener (``morphology_whiten``) before comparison."""
    segs = gene.segments
    n = len(segs) or 1
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

    ax = [0.0, 0.0, 0.0]
    for s in actuated:
        a = s.joint_axis
        ax[max(range(3), key=lambda i: abs(a[i]))] += 1.0
    tot_ax = sum(ax) or 1.0
    ax = [v / tot_ax for v in ax]

    n_fingers = sum(1 for s in prismatic if not children.get(s.name))
    ys = [round(s.mount_offset[1], 4) for s in segs if abs(s.mount_offset[1]) > 1e-6]
    symmetry = sum(1 for y in ys if -y in ys) / 2.0

    # --- new discriminators ---
    root = gene.root()
    limb_count = len(children.get(root.name, [])) if root else 0          # appendages off the torso (leg count)
    dof_per_limb = len(actuated) / max(1, limb_count)
    serial_ratio = depth / n                                              # ~1 for a snake, low for a branched quad
    branch_ratio = n_branch_points / n
    shapes = [(getattr(s, "shape", "capsule") or "capsule") for s in segs]
    box_frac = sum(1 for s in shapes if s == "box") / n
    cap_frac = sum(1 for s in shapes if s == "capsule") / n
    cyl_frac = sum(1 for s in shapes if s == "cylinder") / n
    xs_aniso = [abs(cs[0] - cs[1]) / (abs(cs[0]) + abs(cs[1]) + 1e-6)
                for s in segs if (cs := getattr(s, "cross_section", None))]
    cross_section_aniso = (sum(xs_aniso) / len(xs_aniso)) if xs_aniso else 0.0
    authored_geom_frac = sum(1 for s in segs if getattr(s, "geometry", None)) / n
    roms = [abs(s.joint_upper - s.joint_lower) for s in actuated
            if s.joint_lower is not None and s.joint_upper is not None]
    mean_joint_rom = min(1.0, (sum(roms) / len(roms) / 3.1416)) if roms else 0.0
    # wheels: cylinder leaves driven by a revolute joint (the rover's homeless-body signal)
    wheel_frac = sum(1 for s in segs if getattr(s, "shape", "") == "cylinder"
                     and s.joint_type == "revolute" and not children.get(s.name)) / n

    raw = {
        "n_segments": n, "dof": len(actuated), "n_revolute": len(revolute), "n_prismatic": len(prismatic),
        "n_fixed": len(fixed), "depth": depth, "max_branching": max_branching,
        "n_branch_points": n_branch_points, "n_leaves": n_leaves, "mean_length": mean_length,
        "std_length": std_length, "mean_radius": sum(radii) / len(radii), "n_fingers": n_fingers,
        "symmetry": symmetry, "limb_count": limb_count, "dof_per_limb": dof_per_limb,
    }
    # features in RICH_FEATURE_NAMES order; size magnitudes LOG-compressed (the mass-tyranny fix)
    vec = [
        raw["n_segments"] / 6.0, raw["dof"] / 6.0, raw["n_revolute"] / 6.0, raw["n_prismatic"] / 4.0,
        raw["n_fixed"] / 3.0, raw["depth"] / 6.0, raw["max_branching"] / 3.0, raw["n_branch_points"] / 3.0,
        raw["n_leaves"] / 4.0, math.log1p(sum(lengths)), math.log1p(max_reach), mean_length / 0.3,
        std_length / 0.3, sum(radii) / len(radii) / 0.1, math.log1p(sum(masses)),
        math.log1p(sum(masses) / len(masses)), n_fingers / 3.0, symmetry / 3.0,
        limb_count / 6.0, dof_per_limb / 3.0, serial_ratio, branch_ratio, box_frac, cap_frac, cyl_frac,
        cross_section_aniso, authored_geom_frac, mean_joint_rom, wheel_frac,
    ]
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
