"""Weisfeiler-Lehman fingerprint of a robot's kinematic tree — the graph channel the aggregate counts destroy.

The rich feature vector still collapses TREE WIRING to scalars (depth, branching, leaf counts), so two bodies with
the same counts but a different topology (a serial snake vs a star-shaped spider vs a branched quadruped) can land
close. WL relabeling is the standard zero-training, deterministic whole-graph fingerprint (graph2vec lineage, the
SOTA sweep's day-1 baseline): each node starts with a coarse structural label, then repeatedly absorbs its
neighbours' labels; the multiset of labels over all rounds, feature-hashed into a fixed vector, distinguishes
topologies. Pure-python + md5 (process-independent), so the fingerprint is stable across runs and machines.
"""
from __future__ import annotations

import hashlib

from virturoid.schemas.gene import RobotGene

_DIM = 32


def _bucket(x: float, edges: tuple[float, ...]) -> int:
    return sum(1 for e in edges if x >= e)


def _initial_label(seg, is_leaf: bool, n_children: int, scale: float = 1.0) -> str:
    """Coarse, SCALE-INVARIANT structural label for a node (joint kind + shape + proportion/role buckets).

    ``scale`` is the body's own characteristic segment length, so the buckets encode PROPORTION (how long/thick
    this part is relative to its own robot) rather than absolute metres. This is what makes the fingerprint
    match its documented contract: a robot and a uniformly scaled copy of it have the SAME topology and now
    get the SAME fingerprint.

    Measured 2026-07-21: with absolute buckets ``(0.08, 0.2, 0.5)`` a 0.72x rescale pushed segments across
    bucket edges and produced a near-ORTHOGONAL vector for the same body plan — cos(small dog, medium quad)
    = -0.13 while cos(small dog, SNAKE) = +0.67, i.e. the "topology" channel ranked a quadruped closer to a
    snake than to another quadruped purely because of size. Absolute size still reaches the embedding through
    the other (dimensional) channels; this one is the graph channel and must be size-free.
    """
    jt = seg.joint_type or "fixed"
    shape = getattr(seg, "shape", "capsule") or "capsule"
    s = scale if scale > 1e-9 else 1.0
    lb = _bucket(seg.length_m / s, (0.55, 1.45, 3.4))       # this part vs the body's typical part
    rb = _bucket(seg.radius_m / s, (0.2, 0.42, 0.85))       # slenderness, likewise relative
    role = "leaf" if is_leaf else ("branch" if n_children >= 2 else "link")
    return f"{jt}|{shape}|l{lb}|r{rb}|{role}"


def _body_scale(segs) -> float:
    """The body's characteristic segment length (median) — a robust per-robot unit for the buckets above.
    Median rather than mean/max so one outsized torso or tail cannot set the scale for every limb."""
    lens = sorted(float(s.length_m or 0.0) for s in segs if (s.length_m or 0.0) > 0.0)
    if not lens:
        return 1.0
    mid = len(lens) // 2
    return lens[mid] if len(lens) % 2 else 0.5 * (lens[mid - 1] + lens[mid])


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

    scale = _body_scale(segs)
    labels = {s.name: _initial_label(s, not children.get(s.name), len(children.get(s.name, [])), scale)
              for s in segs}
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
