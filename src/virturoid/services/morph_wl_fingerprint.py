"""Weisfeiler-Lehman fingerprint of a robot's kinematic tree — the graph channel the aggregate counts destroy.

The rich feature vector still collapses TREE WIRING to scalars (depth, branching, leaf counts), so two bodies with
the same counts but a different topology (a serial snake vs a star-shaped spider vs a branched quadruped) can land
close. WL relabeling is the standard zero-training, deterministic whole-graph fingerprint (graph2vec lineage, the
SOTA sweep's day-1 baseline): each node starts with a coarse structural label, then repeatedly absorbs its
neighbours' labels; the multiset of labels over all rounds, feature-hashed into a fixed vector, distinguishes
topologies. Pure-python + md5 (process-independent), so the fingerprint is stable across runs and machines.

The node label is PURELY STRUCTURAL and must stay that way — see ``_initial_label`` for the measurement that
forced it. A WL round hashes a node's own label together with its neighbours', so the kernel only credits
EXACT label matches: one mismatched attribute anywhere propagates outward and zeroes every later round. Feed
it dimensions and it stops being a similarity measure and becomes a near-equality test — which is what it had
become, ranking a quadruped nearer a snake than another quadruped. Dimensions belong to the other channels.
"""
from __future__ import annotations

import hashlib

from virturoid.schemas.gene import RobotGene

_DIM = 32


def _bucket(x: float, edges: tuple[float, ...]) -> int:
    return sum(1 for e in edges if x >= e)


#: Where a part sits along its own chain, as a FRACTION of the tree's own depth (root / near / mid / far).
_DEPTH_EDGES = (0.001, 0.34, 0.67)
#: How much of the whole body hangs off this part, as a FRACTION of the segment count. A serial spine carries
#: nearly the entire body on its second link (~0.9); a quadruped's thigh carries one leg (~0.15). This one
#: number is what tells a CHAIN apart from a BRANCH, and it needs no dimensions to do it.
#:
#: The ladder is TIP (a foot / head / hand) -> LIMB SEGMENT -> MAJOR APPENDAGE -> TRUNK, and it is spaced
#: roughly logarithmically on purpose. An evenly-spaced ladder makes the label depend on how many OTHER limbs
#: the body has: the same calf is 2/18 of a quadruped and 2/26 of a hexapod, which a linear edge at 0.10 puts
#: in DIFFERENT buckets. Measured on the sweep, that single edge crossing dropped quadruped-vs-hexapod to
#: +0.087 while quadruped-vs-biped sat at +0.410 -- a quadruped further from a hexapod than from a biped.
#: Log spacing absorbs the limb-count ratio, and the same sweep then reads +0.285 (hexapod) > +0.236 (biped)
#: > +0.118 (spider) > +0.015 (snake), with the quadruped-cluster margin over every non-quadruped rising from
#: +0.262 to +0.408.
_SUBTREE_EDGES = (0.03, 0.15, 0.55)


def _initial_label(seg, *, role: str, degree: int, depth_frac: float, subtree_frac: float) -> str:
    """PURELY STRUCTURAL label for a node: actuation kind + tree role + degree + where it sits in the tree.

    This is the GRAPH channel, so the label carries graph facts only. It used to carry the segment's own
    length and radius as hard-edged buckets, and that is what destroyed it — measured 2026-08-08 on the
    live composer:

      * The two quadruped builders decompose a leg differently (three ~equal links vs a 25 mm hip stub +
        thigh + calf), so NOT ONE leg-link label matched between them: round-0 cosine -0.008, round-1
        cosine EXACTLY 0.000 (zero shared labels), and the 0.087 that reached the caller was round-2 hash
        collision noise. Meanwhile quad-vs-SNAKE scored 0.140 out of the same noise — the channel ranked a
        quadruped nearer a serpent than another quadruped.
      * Same-builder pairs pinned at the top for the same reason: the template quadruped scored 0.984
        against a HEXAPOD and the anatomy dog scored 1.000 against a cat. The channel had become a
        BUILDER-IDENTITY detector, which is the exact inverse of a morphology key.

    Proportion is a DIMENSIONAL fact and it is already carried, un-bucketed, by ``embed_gene_rich``
    (mean_length / std_length / mean_radius / cross_section_aniso / the shape fractions). Re-encoding it here
    through hard bucket edges bought nothing and cost the only channel that can see wiring: WL rounds hash
    ``own label + sorted neighbour labels``, so a single mismatched bucket anywhere propagates and zeroes
    every later round. A graph kernel has to be fed graph labels.

    ``depth_frac`` and ``subtree_frac`` are fractions of the body's OWN depth and size, so the label stays
    scale-free (the property the previous docstring was right to insist on) while now also being
    decomposition-free: it does not care how many pieces a builder cut the same limb into, only where those
    pieces sit in the tree.
    """
    kind = "act" if (seg.joint_type or "fixed") in ("revolute", "prismatic", "continuous") else "weld"
    return (f"{kind}|{role}|d{min(degree, 8)}"
            f"|h{_bucket(depth_frac, _DEPTH_EDGES)}|s{_bucket(subtree_frac, _SUBTREE_EDGES)}")


def _tree_stats(segs, children: dict, by_name: dict) -> tuple[dict, dict]:
    """(depth from the root, subtree size) for every segment — iterative, so a 100-link spine cannot recurse
    out, and cycle-safe, so a malformed parent chain degrades instead of hanging."""
    roots = [s.name for s in segs if s.parent is None or s.parent not in by_name]
    depth: dict[str, int] = {}
    order: list[str] = []
    stack = [(r, 0) for r in roots]
    while stack:
        name, d = stack.pop()
        if name in depth:
            continue
        depth[name] = d
        order.append(name)
        stack.extend((k, d + 1) for k in children.get(name, ()))
    subtree: dict[str, int] = {s.name: 1 for s in segs}
    for name in reversed(order):                       # children are always visited after their parent
        for k in children.get(name, ()):
            subtree[name] = subtree.get(name, 1) + subtree.get(k, 1)
    for s in segs:                                     # orphaned by a broken parent link -> its own root
        depth.setdefault(s.name, 0)
    return depth, subtree


def _hash_to(vec: list[float], token: str) -> None:
    h = hashlib.md5(token.encode("utf-8")).digest()
    b = int.from_bytes(h[:4], "big") % len(vec)
    s = 1.0 if (h[4] & 1) else -1.0
    vec[b] += s


def wl_fingerprint(gene: RobotGene, *, iterations: int = 2, dim: int = _DIM) -> list[float]:
    """Deterministic WL fingerprint (length ``dim``, L2-normalized). Captures the tree topology — a snake (deep
    chain), a spider (shallow star) and a quadruped (branched) get distinct fingerprints even at equal counts."""
    segs = gene.segments
    if not segs:
        return [0.0] * dim
    children: dict[str, list[str]] = {}
    for s in segs:
        if s.parent is not None:
            children.setdefault(s.parent, []).append(s.name)
    by_name = {s.name: s for s in segs}
    neighbours: dict[str, list[str]] = {s.name: [] for s in segs}
    for s in segs:
        if s.parent is not None and s.parent in by_name:
            neighbours[s.name].append(s.parent)
            neighbours[s.parent].append(s.name)
    # A CLOSED LOOP is a neighbour relation too, and WL exists to capture exactly that. Without it a gantry whose
    # bridge is braced at both columns and one that cantilevers off a single column produced IDENTICAL
    # fingerprints — two robots with genuinely different load paths and different control problems, indexed to
    # the same key. That undermines the whole point of a morphology-keyed memory: the flywheel would hand a
    # braced machine the hints it learned from a cantilever.
    for lc in (getattr(gene, "loop_closures", None) or []):
        a, b = (lc or {}).get("a"), (lc or {}).get("b")
        if a in by_name and b in by_name and a != b:
            neighbours[a].append(b)
            neighbours[b].append(a)

    depth, subtree = _tree_stats(segs, children, by_name)
    max_depth = max(depth.values()) or 1
    n_segs = len(segs)
    labels = {}
    for s in segs:
        kids = len(children.get(s.name, ()))
        is_root = s.parent is None or s.parent not in by_name
        role = "root" if is_root else ("leaf" if not kids else ("branch" if kids >= 2 else "link"))
        labels[s.name] = _initial_label(
            s, role=role, degree=len(neighbours[s.name]),
            depth_frac=depth[s.name] / max_depth, subtree_frac=subtree.get(s.name, 1) / n_segs)
    vec = [0.0] * dim
    for name, lab in labels.items():
        _hash_to(vec, f"0:{lab}")                                # round-0 multiset
    for it in range(iterations):
        new = {}
        for s in segs:
            nb = sorted(labels[m] for m in neighbours[s.name])
            new[s.name] = hashlib.md5((labels[s.name] + "||" + "|".join(nb)).encode("utf-8")).hexdigest()[:12]
        labels = new
        for name, lab in labels.items():
            _hash_to(vec, f"{it + 1}:{lab}")
    n = sum(x * x for x in vec) ** 0.5 or 1.0
    return [x / n for x in vec]
