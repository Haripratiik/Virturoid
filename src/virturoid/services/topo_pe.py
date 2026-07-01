"""Topology positional encoding + kinematic hop-distance — the tokenizer's structural signals (Phase 5).

Phase 5 of the robotics-native-AI plan (``docs/robotics_native_ai_plan.md``): two pure-NumPy-free structural
primitives the SOTA robot tokenizer needs, validated CPU-first before any GPU (Flax) port.

* ``topo_path_vector`` — **TopoPE** (BodyGen): map a part's root->part *child-index path* to a fixed vector so
  the code is **edit-invariant** (adding a limb ELSEWHERE doesn't renumber existing parts — the failure mode
  of the 1-D traversal index at ``morph_graph.py``) and structurally-similar limbs on *different* robots get
  *aligned* codes (shared path prefix -> shared components). This is the positional encoding that lets
  knowledge/control transfer across morphologies.

* ``hop_distance_matrix`` — shortest-path hop distance between tokens on the kinematic tree, the input to the
  **topology-aware attention bias** (2603.00182: ``B_{i,j} = theta_{d(i,j)}``) so attention is structure-aware
  without hard GNN locality (dodging the Amorpheus multi-hop bottleneck).

Deterministic (md5, not the per-process-salted ``hash``) so a persisted/relearned representation is stable.
"""

from __future__ import annotations

import hashlib
from collections import deque


def topo_path_vector(path: list[int], dim: int = 4) -> list[float]:
    """Edit-invariant positional vector for a part, from its root->part child-index ``path`` (e.g. ``[0, 1]``
    = the 2nd child of the 1st child of the root). Each path *prefix* contributes a hashed, depth-decayed
    component, so parts in the same sub-tree share early components and a part's code depends only on its own
    path (siblings added elsewhere leave it unchanged). ``path == []`` (the root) -> a zero vector."""
    vec = [0.0] * dim
    for depth in range(len(path)):
        prefix = ",".join(str(x) for x in path[: depth + 1])       # the edge chain root..this-part
        h = hashlib.md5(prefix.encode("utf-8")).digest()
        bucket = int.from_bytes(h[:4], "big") % dim
        sign = 1.0 if (h[4] & 1) else -1.0
        vec[bucket] += sign * (0.5 ** depth)                       # root structure dominates; deeper edges decay
    return vec


def hop_distance_matrix(parent: list[int], *, virtual_base: bool = True) -> list[list[int]]:
    """Pairwise shortest-path hop distances between the ``n`` tokens on the kinematic tree.

    ``parent[i]`` = token i's parent token index, or ``-1`` if it attaches to the base (as in
    ``morph_graph.MorphGraph.parent``). With ``virtual_base`` (default), base-attached tokens connect through
    a shared virtual base node, so two legs off the torso are 2 hops apart (not disconnected). Returns an
    ``n x n`` integer matrix (0 on the diagonal).
    """
    n = len(parent)
    if n == 0:
        return []
    base = n                                                        # virtual base node index
    adj: list[list[int]] = [[] for _ in range(n + 1)]
    for i, p in enumerate(parent):
        if 0 <= p < n:
            adj[i].append(p)
            adj[p].append(i)
        elif virtual_base:
            adj[i].append(base)
            adj[base].append(i)
    inf = n + 1
    dist = [[0 if i == j else inf for j in range(n)] for i in range(n)]
    for s in range(n):
        seen = {s: 0}
        dq = deque([s])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if v not in seen:
                    seen[v] = seen[u] + 1
                    dq.append(v)
        for v, d in seen.items():
            if v < n:
                dist[s][v] = d
    return dist
