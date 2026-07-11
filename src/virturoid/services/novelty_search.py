"""The corpus-factory selection substrate — Novelty × Learnability × Value (master_plan_v6 §10.1 / WS-C.1).

The factory must not grow randomly. Three research anchors decide *which body next*:

  * **Novelty** — two complementary signals: (1) a **MAP-Elites niche grid** over interpretable structural
    descriptors (kind, DOF, limb-chains, size) so we can see empty/thin cells and enforce class balance; and
    (2) **PATA-EC** (POET) — a body is novel iff it induces a *never-before-seen rank-order* of existing
    controllers' performance on it. PATA-EC is computed straight from our transfer ledger's forward matrix and is
    sharper than raw embedding distance (it measures behavioural, not just structural, novelty).
  * **Learnability** — POET's minimal criterion: neither trivial (every gait already walks it) nor currently
    unsolvable (no gait is credible). A peaked function of the credible-fraction; ACCEL's trick (mutate high-regret
    elites) is expressed as ``regret``.
  * **ExpectedCorpusValue** — for transfer *pairs*: uncertainty × diversity selection (~5× the label efficiency of
    pure uncertainty), never all N², never random.

All pure/CPU/deterministic and testable without a DB; ``ledger_*`` helpers read the transfer ledger when present.
Nothing here designs a body — it SCORES candidate bodies the LLM proposes, so the factory spends effort well.
"""
from __future__ import annotations

import math

# ------------------------------------------------------------------ MAP-Elites niche grid (structural descriptors)
_DOF_BINS = (0, 3, 6, 10)          # buckets: [0,3) [3,6) [6,10) [10,inf)
_LIMB_BINS = (0, 2, 4, 6)          # [0,2) [2,4) [4,6) [6,inf)
_SIZE_BINS = (0.0, 0.8, 2.0)       # small / medium / large (total length m)


def _bucket(value: float, edges) -> int:
    b = 0
    for e in edges:
        if value >= e:
            b += 1
    return b - 1                    # index of the bin whose lower edge value cleared


def niche_key(gene) -> tuple:
    """The structural MAP-Elites cell a body lands in: (kind, dof-bucket, limb-chain-bucket, size-bucket).
    Interpretable niches (not raw embedding bins) so 'thin niche' and 'class balance' are legible."""
    from virturoid.services.heldout_set import _leg_chain_count
    from virturoid.services.task_matched_eval import robot_kind
    dof = len(gene.actuated_joints())
    limbs = _leg_chain_count(gene)
    size = sum(float(getattr(s, "length_m", 0.0)) for s in gene.segments)
    return (robot_kind(gene), _bucket(dof, _DOF_BINS), _bucket(limbs, _LIMB_BINS), _bucket(size, _SIZE_BINS))


class MapElitesGrid:
    """A niche-occupancy grid over a corpus of bodies — coverage, per-cell counts, and per-candidate novelty."""

    def __init__(self, genes=()):
        self.cells: dict[tuple, int] = {}
        for g in genes:
            self.add(g)

    def add(self, gene) -> tuple:
        key = niche_key(gene)
        self.cells[key] = self.cells.get(key, 0) + 1
        return key

    def count(self, gene) -> int:
        return self.cells.get(niche_key(gene), 0)

    def is_empty_cell(self, gene) -> bool:
        return self.count(gene) == 0

    def novelty(self, gene) -> float:
        """Niche novelty in [0,1]: 1.0 for an empty cell, decaying as the cell fills (1/(1+count))."""
        return 1.0 / (1.0 + self.count(gene))

    def coverage(self) -> dict:
        filled = len(self.cells)
        counts = sorted(self.cells.values(), reverse=True)
        return {"filled_cells": filled, "total_bodies": sum(self.cells.values()),
                "max_cell": counts[0] if counts else 0,
                "class_balance": self._class_balance()}

    def _class_balance(self) -> dict:
        out: dict[str, int] = {}
        for (kind, *_), n in self.cells.items():
            out[kind] = out.get(kind, 0) + n
        return out

    def thinnest_classes(self) -> list[str]:
        cb = self._class_balance()
        return [k for k, _ in sorted(cb.items(), key=lambda kv: kv[1])]


# ------------------------------------------------------------------ embedding novelty (structural distance)
def embedding_novelty(gene, corpus_genes) -> float:
    """Min cosine DISTANCE (1 - similarity) to the nearest corpus body — 1.0 when the corpus is empty, →0 for a
    near-duplicate. Complements the niche grid (continuous vs discrete)."""
    from virturoid.services.morphology_embedding import embed_gene
    corpus_genes = [g for g in corpus_genes if g is not None]
    if not corpus_genes:
        return 1.0
    q = embed_gene(gene)

    def _sim(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na > 0 and nb > 0 else 0.0
    best = max(_sim(q, embed_gene(g)) for g in corpus_genes)
    return max(0.0, 1.0 - best)


# ------------------------------------------------------------------ PATA-EC (rank-order novelty) — the sharp signal
def rank_vector(scores) -> list[int]:
    """Ordinal ranks (0 = lowest score) with ties broken by index — a stable rank-order of controllers."""
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    ranks = [0] * len(scores)
    for r, i in enumerate(order):
        ranks[i] = r
    return ranks


def rank_distance(a: list[int], b: list[int]) -> float:
    """Normalised Spearman footrule (mean |rank_a - rank_b| / max-possible) in [0,1]."""
    n = len(a)
    if n <= 1:
        return 0.0
    max_disp = (n * n) / 2.0 if n % 2 == 0 else (n * n - 1) / 2.0
    return sum(abs(x - y) for x, y in zip(a, b)) / max_disp if max_disp else 0.0


def pata_ec_novelty(candidate_scores, existing_score_vectors, *, k: int = 5) -> float:
    """PATA-EC: novelty = mean rank-order distance to the k NEAREST already-seen rank-orderings. ``candidate_scores``
    is how each existing controller performed on the candidate body; ``existing_score_vectors`` are the same-length
    vectors for already-banked bodies. High = the candidate reorders controllers in a way we've never seen."""
    if not candidate_scores:
        return 1.0
    if not existing_score_vectors:
        return 1.0
    cr = rank_vector(candidate_scores)
    dists = sorted(rank_distance(cr, rank_vector(e)) for e in existing_score_vectors if len(e) == len(cr))
    if not dists:
        return 1.0
    kk = min(k, len(dists))
    return sum(dists[:kk]) / kk


def ledger_pata_ec(db) -> dict:
    """Compute each banked body's PATA-EC novelty from the transfer ledger's forward matrix (novelty of the column
    'how did every source gait score on this body' vs all other columns). Returns {body_id: novelty} or {}."""
    from virturoid.services.transfer_ledger import corpus_from_ledger
    corpus = corpus_from_ledger(db, min_trials=1)
    if not corpus:
        return {}
    F = corpus["forward"]
    bodies = corpus["bodies"]
    n = len(bodies)
    cols = [[F[i][j] for i in range(n)] for j in range(n)]   # column j = scores of all sources on body j
    out: dict[str, float] = {}
    for j in range(n):
        others = [cols[m] for m in range(n) if m != j]
        out[bodies[j]["id"]] = round(pata_ec_novelty(cols[j], others), 4)
    return out


# ------------------------------------------------------------------ learnability (minimal criterion / regret)
def learnability(credible_fraction: float, *, low: float = 0.05, high: float = 0.85) -> float:
    """POET minimal criterion as a peaked score in [0,1]: ~0 when trivial (everything already walks it) or
    unsolvable (nothing does), peaking in the mid band where there's genuine learning progress to bank."""
    f = max(0.0, min(1.0, credible_fraction))
    if f <= low or f >= high:
        return 0.0
    mid = 0.5 * (low + high)
    half = 0.5 * (high - low)
    return round(1.0 - abs(f - mid) / half, 4)


def regret(best_achieved: float, reference: float) -> float:
    """ACCEL: the gap between a reference performance and the best achieved on a body — high-regret elites are the
    cheapest source of high-value new bodies (mutate them). Clamped to [0,1] against the reference scale."""
    if reference <= 0:
        return 0.0
    return round(max(0.0, min(1.0, (reference - best_achieved) / reference)), 4)


# ------------------------------------------------------------------ combined body score + selection
def body_score(gene, *, grid: MapElitesGrid, corpus_genes=(), credible_fraction: float | None = None,
               class_boost: float | None = None) -> dict:
    """score = Novelty × Learnability (× class-balance boost). Novelty blends niche + embedding novelty. When
    learnability is unknown (no trials yet), it is treated as neutral (the minimal criterion applies after a
    verify-build). Returns the score + its factors (auditable, never a black box)."""
    niche_nov = grid.novelty(gene)
    emb_nov = embedding_novelty(gene, corpus_genes)
    novelty = 0.5 * niche_nov + 0.5 * emb_nov
    learn = 1.0 if credible_fraction is None else learnability(credible_fraction)
    boost = 1.0 if class_boost is None else class_boost
    score = round(novelty * (learn if credible_fraction is not None else 1.0) * boost, 4)
    return {"score": score, "novelty": round(novelty, 4), "niche_novelty": round(niche_nov, 4),
            "embedding_novelty": round(emb_nov, 4), "learnability": round(learn, 4),
            "niche": niche_key(gene), "empty_cell": grid.is_empty_cell(gene)}


def select_next_bodies(candidates, *, corpus_genes=(), k: int = 10) -> list[dict]:
    """Rank candidate genes by body_score against the current corpus, boosting the thinnest classes for balance.
    Returns the top-k as [{gene, ...score-factors}]. The factory's 'which body next' — not random, not a list."""
    grid = MapElitesGrid(corpus_genes)
    cb = grid._class_balance()
    total = sum(cb.values()) or 1
    scored = []
    for g in candidates:
        from virturoid.services.task_matched_eval import robot_kind
        kind = robot_kind(g)
        share = cb.get(kind, 0) / total
        boost = 1.0 + max(0.0, 0.5 - share)              # under-represented classes get up to +0.5
        s = body_score(g, grid=grid, corpus_genes=corpus_genes, class_boost=boost)
        s["gene"] = g
        scored.append(s)
    scored.sort(key=lambda s: s["score"], reverse=True)
    return scored[:k]


# ------------------------------------------------------------------ transfer-pair value (uncertainty × diversity)
def select_transfer_pairs(body_ids, *, embed_by_id, tested_pairs=frozenset(), uncertainty_by_pair=None,
                          k: int = 20) -> list[tuple]:
    """Pick the ~k highest-value UNTESTED transfer pairs by uncertainty × diversity — never all N², never random
    (§10.1). ``embed_by_id`` maps id→vector (for diversity); ``uncertainty_by_pair`` optionally scores metric
    uncertainty for a pair (defaults to max uncertainty when unknown — an unseen pair is maximally uncertain)."""
    def _sim(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na > 0 and nb > 0 else 0.0
    cand = []
    for i, a in enumerate(body_ids):
        for b in body_ids[i + 1:]:
            if (a, b) in tested_pairs or (b, a) in tested_pairs:
                continue
            va, vb = embed_by_id.get(a), embed_by_id.get(b)
            diversity = (1.0 - _sim(va, vb)) if (va and vb) else 1.0
            unc = 1.0 if uncertainty_by_pair is None else float(uncertainty_by_pair.get((a, b), 1.0))
            cand.append((unc * diversity, a, b))
    cand.sort(reverse=True)
    return [(a, b) for _, a, b in cand[:k]]
