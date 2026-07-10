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


def _initial_label(seg, is_leaf: bool, n_children: int) -> str:
    """Coarse, size-robust structural label for a node (joint kind + shape + size/role buckets)."""
    jt = seg.joint_type or "fixed"
    shape = getattr(seg, "shape", "capsule") or "capsule"
    lb = _bucket(seg.length_m, (0.08, 0.2, 0.5))
    rb = _bucket(seg.radius_m, (0.03, 0.06, 0.12))
    role = "leaf" if is_leaf else ("branch" if n_children >= 2 else "link")
    return f"{jt}|{shape}|l{lb}|r{rb}|{role}"


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

    labels = {s.name: _initial_label(s, not children.get(s.name), len(children.get(s.name, []))) for s in segs}
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
