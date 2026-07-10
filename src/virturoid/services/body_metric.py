"""The learned transfer metric — the keystone that makes body distance PREDICT control transfer.

Unsupervised feature engineering can't beat the hand-crafted baseline (measured: whitening/rich features regress
transfer-ranking because our transfer structure aligns with size/class similarity the baseline already encodes).
The only thing that can beat it is SUPERVISION from the physics-verified transfer matrix — the PSE/DBC lesson:
learn a per-feature reweighting so that (weighted) distance reproduces measured transfer. Here that is a DIAGONAL
metric (per-feature relevance weights, `w`), fit by how well each feature's difference separates transferring
pairs from non-transferring ones — deterministic, pure-python, no overfitting-prone deep net at our data scale.

HONESTY GATE: `fit_body_metric` measures the metric with LEAVE-ONE-BODY-OUT (fit on the rest, rank an UNSEEN
body's transfers — exactly the product scenario) and the bundle is marked ``proven`` ONLY if held-out triplet
ranking BEATS the baseline. ``embed_metric`` uses the weights only when proven; otherwise it returns the baseline
29-D vector unchanged, so the flywheel can NEVER regress by adopting a metric that didn't earn it.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from virturoid.schemas.gene import RobotGene

DEFAULT_METRIC_PATH = Path("build/models/body_metric.json")
_cache: dict | None = None
_cache_mtime: float | None = None
_TRIED = False


def _features(gene: RobotGene, space: str) -> list[float]:
    from virturoid.services.morphology_embedding import embed_gene, embed_gene_rich
    if space.startswith("dyn"):                                  # behavioral fingerprint ONLY (probe features)
        from virturoid.services.behavioral_fingerprint import z_dyn
        return z_dyn(gene)
    base = embed_gene_rich(gene) if space.startswith("rich") else embed_gene(gene)
    if space.endswith("_wl"):                                    # append the WL topology channel
        from virturoid.services.morph_wl_fingerprint import wl_fingerprint
        base = base + wl_fingerprint(gene)
    if space.endswith("_dyn"):                                   # append the behavioral (probe) channel — the
        from virturoid.services.behavioral_fingerprint import z_dyn   # dynamics geometry can't express
        base = base + z_dyn(gene)
    return base


def _load(path: Path | None = None) -> dict | None:
    global _cache, _cache_mtime
    path = path or DEFAULT_METRIC_PATH
    if not path.exists():
        return None
    mt = path.stat().st_mtime
    if _cache is not None and _cache_mtime == mt:
        return _cache
    _cache = json.loads(path.read_text())
    _cache_mtime = mt
    return _cache


def embed_metric(gene: RobotGene) -> list[float]:
    """The body vector the flywheel indexes. Baseline 29-D unless a PROVEN transfer metric is on disk, in which
    case: features (rich) -> whiten -> diagonal reweight `w`. Never regresses (unproven/absent -> baseline)."""
    from virturoid.services.morphology_embedding import embed_gene
    bundle = _load()
    if not bundle or not bundle.get("proven"):
        return embed_gene(gene)
    from virturoid.services.morphology_whiten import apply_whiten
    v = _features(gene, bundle.get("feature_space", "rich"))
    v = apply_whiten(v, bundle.get("whiten"))
    w = bundle.get("weights") or [1.0] * len(v)
    return [v[i] * w[i] for i in range(min(len(v), len(w)))]


# ------------------------------------------------------------------ fit (relevance-weighted diagonal metric)
def _relevance_weights(vecs: list[list[float]], T, sources: list[int], train: set[int]) -> list[float]:
    """Per-feature weight = how well |a_k - b_k| separates transferring from non-transferring source->target pairs
    (a Fisher-style score on pairwise feature differences). Feature whose difference is SMALL when a gait
    transfers and LARGE when it doesn't -> high weight. Only pairs within the ``train`` body set are used."""
    dim = len(vecs[0])
    pos, neg = [], []                                            # squared per-feature diffs, split by transfer
    for i in sources:
        if i not in train:
            continue
        for j in train:
            if i == j:
                continue
            d = [(vecs[i][k] - vecs[j][k]) ** 2 for k in range(dim)]
            (pos if T[i][j] == 1 else neg).append(d)
    if not pos or not neg:
        return [1.0] * dim
    w = []
    for k in range(dim):
        pk = [d[k] for d in pos]
        nk = [d[k] for d in neg]
        mp, mn = sum(pk) / len(pk), sum(nk) / len(nk)
        vp = sum((x - mp) ** 2 for x in pk) / len(pk)
        vn = sum((x - mn) ** 2 for x in nk) / len(nk)
        # high weight when non-transfer diff (mn) >> transfer diff (mp), normalized by spread (Fisher score)
        score = max(0.0, (mn - mp)) / math.sqrt(vp + vn + 1e-9)
        w.append(score)
    mx = max(w) or 1.0
    # keep a small floor so no feature is fully zeroed (robustness); normalize to ~[0.1, 1]
    return [0.1 + 0.9 * (x / mx) for x in w]


def _margin_weights(vecs, T, sources, train: set[int], *, iters: int = 200, lr: float = 0.2,
                    l2: float = 0.02) -> list[float]:
    """Learn diagonal weights by gradient ascent on a smooth triplet margin over training pairs: for each
    (target j, positive p that transfers, negative q that doesn't) push weighted-cosine(j,p) above (j,q). The
    supervised PSE-style fit; L2-regularized toward 1 to resist overfitting the tiny corpus. Weights kept >=0."""
    dim = len(vecs[0])
    w = [1.0] * dim
    triples = []
    for j in train:
        pos = [i for i in sources if i in train and i != j and T[i][j] == 1]
        neg = [i for i in sources if i in train and i != j and T[i][j] == 0]
        for p in pos:
            for q in neg:
                triples.append((j, p, q))
    if not triples:
        return w

    def wcos(a, b, wv):
        ua = [a[k] * wv[k] for k in range(dim)]
        ub = [b[k] * wv[k] for k in range(dim)]
        na = math.sqrt(sum(x * x for x in ua)) or 1e-9
        nb = math.sqrt(sum(x * x for x in ub)) or 1e-9
        return sum(ua[k] * ub[k] for k in range(dim)) / (na * nb)

    for _ in range(iters):
        grad = [0.0] * dim
        for (j, p, q) in triples:
            spos, sneg = wcos(vecs[j], vecs[p], w), wcos(vecs[j], vecs[q], w)
            if spos - sneg > 0.3:                                # margin satisfied -> no push (hinge)
                continue
            # finite-difference gradient of (spos - sneg) per feature (cheap: dim ~40, triples small)
            for k in range(dim):
                wk = w[k]
                w[k] = wk + 1e-3
                d = (wcos(vecs[j], vecs[p], w) - wcos(vecs[j], vecs[q], w)) - (spos - sneg)
                w[k] = wk
                grad[k] += d / 1e-3
        w = [max(0.0, w[k] + lr * (grad[k] / len(triples) - l2 * (w[k] - 1.0))) for k in range(dim)]
    return w


def _triplet_acc_for(query_targets, vecs, T, sources, w, train: set[int]) -> tuple[float, float]:
    """Weighted (and baseline) triplet ranking accuracy for the given held-out target bodies: positives are TRAIN
    sources that transfer to the target, negatives are train sources that don't; count d_w(pos) < d_w(neg)."""
    def dist(a, b, weights):
        # weighted cosine distance (matches the flywheel's cosine retrieval)
        ua = [a[k] * weights[k] for k in range(len(a))]
        ub = [b[k] * weights[k] for k in range(len(b))]
        na = math.sqrt(sum(x * x for x in ua)) or 1.0
        nb = math.sqrt(sum(x * x for x in ub)) or 1.0
        return 1.0 - sum(ua[k] * ub[k] for k in range(len(ua))) / (na * nb)

    ones = [1.0] * len(vecs[0])
    ww = wt = bw = bt = 0.0
    for j in query_targets:
        pos = [i for i in sources if i in train and i != j and T[i][j] == 1]
        neg = [i for i in sources if i in train and i != j and T[i][j] == 0]
        for p in pos:
            for q in neg:
                wt += 1; bt += 1
                ww += 1.0 if dist(vecs[j], vecs[p], w) < dist(vecs[j], vecs[q], w) else 0.0
                bw += 1.0 if dist(vecs[j], vecs[p], ones) < dist(vecs[j], vecs[q], ones) else 0.0
    return (ww / wt if wt else float("nan")), (bw / bt if bt else float("nan"))


def _fit_weights(method, vecs, T, sources, train):
    return _margin_weights(vecs, T, sources, train) if method == "margin" \
        else _relevance_weights(vecs, T, sources, train)


def fit_body_metric(corpus: dict, *, feature_space: str = "rich", whiten: bool = False, method: str = "relevance",
                    save: bool = False, path: Path | None = None) -> dict:
    """Fit the diagonal transfer metric and MEASURE it leave-one-body-out against the TRUE 29-D baseline (not the
    same-space unweighted vector — the honest bar is 'beat what ships today'). ``proven`` iff held-out weighted
    ranking beats the true baseline's held-out ranking. Saves iff proven -> ``embed_metric`` never regresses."""
    from virturoid.services.morphology_embedding import embed_gene
    from virturoid.services.morphology_whiten import apply_whiten, fit_whitener
    bodies = corpus["bodies"]
    genes = [b.get("_gene") for b in bodies]
    T = corpus["transfer"]
    sources = [i for i, b in enumerate(bodies) if b.get("self_credible")]
    raw = [_features(g, feature_space) for g in genes]
    wh = fit_whitener(genes, embed_fn=lambda g: _features(g, feature_space)) if whiten else None
    vecs = [apply_whiten(v, wh) for v in raw] if wh else raw
    base_vecs = [embed_gene(g) for g in genes]                    # the TRUE baseline (what ships today)

    n = len(bodies)
    ones = [1.0] * len(base_vecs[0])
    ww = bw = cnt = 0.0
    for h in range(n):                                            # leave-one-BODY-out: rank an UNSEEN body's transfers
        train = set(range(n)) - {h}
        w = _fit_weights(method, vecs, T, sources, train)
        a, _ = _triplet_acc_for([h], vecs, T, sources, w, train)
        b, _ = _triplet_acc_for([h], base_vecs, T, sources, ones, train)
        if a == a and b == b:
            ww += a; bw += b; cnt += 1
    held_metric = (ww / cnt) if cnt else float("nan")
    held_base = (bw / cnt) if cnt else float("nan")

    w_full = _fit_weights(method, vecs, T, sources, set(range(n)))
    proven = bool(held_metric == held_metric and held_base == held_base and held_metric > held_base + 1e-9)
    bundle = {"feature_space": feature_space, "whiten": wh, "weights": w_full, "method": method, "proven": proven,
              "held_out_triplet_acc": round(held_metric, 4) if held_metric == held_metric else None,
              "baseline_held_out_triplet_acc": round(held_base, 4) if held_base == held_base else None,
              "n_bodies": n, "n_sources": len(sources)}
    if save and proven:
        (path or DEFAULT_METRIC_PATH).parent.mkdir(parents=True, exist_ok=True)
        (path or DEFAULT_METRIC_PATH).write_text(json.dumps(bundle))
    return bundle
