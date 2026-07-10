"""Grade a body embedding by the ONLY thing that matters: does distance predict transfer?

Reads the physics-verified transfer matrix (``embedding_transfer_corpus``) and scores any ``embed_fn(gene)->vec``
against it — no training, milliseconds, deterministic. These are the metrics the research settled on (PSE/Task2Vec/
alignment-uniformity lineage), so an embedding change is PROVEN by a measured before/after, never asserted:

  * triplet_ranking_acc — over (query j, source i+ that transfers to j, source i- that doesn't): fraction with
    d(j,i+) < d(j,i-). THE single "distance-predicts-transfer" number (chance = 0.5).
  * precision@k / recall@k — for query j, do its k nearest source bodies actually transfer to it? (the production KPI)
  * kendall_tau / pearson — correlation of (-distance) with graded forward_m over source→target pairs.
  * alignment / uniformity — transferring pairs land close (low align) without the space collapsing (uniformity).
  * within_legged_cos — the pathology tracker: cosine spread inside the legged family (baseline was 0.91-1.0).

Pure-python (no numpy dependency required, but uses it if present for speed). Also ``nearest_report`` for a
qualitative homeless-body check.
"""
from __future__ import annotations

import math

_LEGGED_CLASSES = {"quadruped", "legged", "hexapod", "octopod", "humanoid", "biped", "legged6"}


def _l2(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _cos(a, b):
    m = min(len(a), len(b))
    return sum(a[i] * b[i] for i in range(m))            # a,b assumed L2-normalized


def _embed_all(embed_fn, corpus):
    return [_l2(list(embed_fn(b["_gene"]))) for b in corpus["bodies"]]


def evaluate_embedding(embed_fn, corpus, *, ks=(1, 2, 3)) -> dict:
    """Score ``embed_fn`` against the cached transfer matrix. Returns the metric dict (all CPU, deterministic)."""
    bodies = corpus["bodies"]
    T = corpus["transfer"]
    F = corpus["forward"]
    n = len(bodies)
    Z = _embed_all(embed_fn, corpus)
    # cosine distance matrix (1 - cos); sources = bodies with a credible own gait (the only meaningful transfer rows)
    D = [[1.0 - _cos(Z[i], Z[j]) for j in range(n)] for i in range(n)]
    sources = [i for i, b in enumerate(bodies) if b.get("self_credible")]

    # 1) triplet ranking accuracy: for each query j, positives transfer to j, negatives (credible sources) do not
    tw = tt = 0.0
    for j in range(n):
        pos = [i for i in sources if i != j and T[i][j] == 1]
        neg = [i for i in sources if i != j and T[i][j] == 0]
        for p in pos:
            for q in neg:
                tt += 1
                dp, dq = D[j][p], D[j][q]
                tw += 1.0 if dp < dq else (0.5 if dp == dq else 0.0)
    triplet_acc = (tw / tt) if tt else float("nan")

    # 2) precision@k / recall@k over queries that HAVE at least one transferring source
    prec = {k: [] for k in ks}
    rec = {k: [] for k in ks}
    for j in range(n):
        cand = sorted([i for i in sources if i != j], key=lambda i: D[j][i])
        pos = {i for i in cand if T[i][j] == 1}
        if not pos:
            continue
        for k in ks:
            topk = cand[:k]
            hit = sum(1 for i in topk if i in pos)
            prec[k].append(hit / max(1, len(topk)))
            rec[k].append(hit / len(pos))
    precision_at = {k: (sum(v) / len(v) if v else float("nan")) for k, v in prec.items()}
    recall_at = {k: (sum(v) / len(v) if v else float("nan")) for k, v in rec.items()}

    # 3) correlation of proximity with graded forward over source->target pairs (exclude self)
    xs, ys = [], []
    for i in sources:
        for j in range(n):
            if i == j:
                continue
            xs.append(-D[i][j])                          # nearer = larger -> should correlate with more forward
            ys.append(F[i][j])
    pear = _pearson(xs, ys)
    tau = _kendall_tau(xs, ys)

    # 4) alignment (positive pairs close) / uniformity (space not collapsed)
    pos_pairs = [(i, j) for i in sources for j in range(n) if i != j and T[i][j] == 1]
    align = (sum(D[i][j] ** 2 for i, j in pos_pairs) / len(pos_pairs)) if pos_pairs else float("nan")
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    uni = (math.log(sum(math.exp(-2.0 * (D[i][j] ** 2)) for i, j in all_pairs) / len(all_pairs))
           if all_pairs else float("nan"))

    # 5) within-legged cosine spread (the pathology): smaller mean / larger std = better resolution
    leg = [i for i, b in enumerate(bodies) if (b.get("robot_class") or "").lower() in _LEGGED_CLASSES]
    lc = [_cos(Z[i], Z[j]) for a, i in enumerate(leg) for j in leg[a + 1:]]
    within_legged = {"n": len(lc), "min": _mn(lc), "max": _mx(lc), "mean": _mean(lc), "std": _std(lc)}

    return {
        "n_bodies": n, "n_sources": len(sources), "n_credible_transfers": sum(sum(r) for r in T),
        "triplet_ranking_acc": round(triplet_acc, 4) if triplet_acc == triplet_acc else None,
        "precision_at": {k: round(v, 4) for k, v in precision_at.items()},
        "recall_at": {k: round(v, 4) for k, v in recall_at.items()},
        "pearson_prox_forward": round(pear, 4), "kendall_tau_prox_forward": round(tau, 4),
        "alignment": round(align, 4) if align == align else None,
        "uniformity": round(uni, 4) if uni == uni else None,
        "within_legged_cos": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in within_legged.items()},
    }


def nearest_report(embed_fn, corpus, *, k: int = 3) -> list[dict]:
    """Each body's top-k nearest neighbours by cosine — the qualitative homeless-body check (snake->? rover->?)."""
    bodies = corpus["bodies"]
    Z = _embed_all(embed_fn, corpus)
    n = len(bodies)
    out = []
    for i in range(n):
        order = sorted([j for j in range(n) if j != i], key=lambda j: 1.0 - _cos(Z[i], Z[j]))
        out.append({"id": bodies[i]["id"], "class": bodies[i].get("robot_class"),
                    "nearest": [(bodies[j]["id"], round(_cos(Z[i], Z[j]), 3)) for j in order[:k]]})
    return out


# --------------------------------------------------------------------- tiny stats (no scipy)
def _mean(v):
    return sum(v) / len(v) if v else float("nan")


def _std(v):
    if not v:
        return float("nan")
    m = _mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))


def _mn(v):
    return min(v) if v else float("nan")


def _mx(v):
    return max(v) if v else float("nan")


def _pearson(x, y):
    if len(x) < 2:
        return float("nan")
    mx, my = _mean(x), _mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) or 1.0
    return num / den


def _kendall_tau(x, y):
    """Kendall tau-a (no ties handling beyond sign); O(n^2), fine for our pair counts."""
    n = len(x)
    if n < 2:
        return float("nan")
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            sx = (x[i] > x[j]) - (x[i] < x[j])
            sy = (y[i] > y[j]) - (y[i] < y[j])
            p = sx * sy
            if p > 0:
                conc += 1
            elif p < 0:
                disc += 1
    tot = conc + disc
    return (conc - disc) / tot if tot else float("nan")
