"""VIRT-Bench-Transfer (P6): the standing scorecard for the moat's headline claim — does embedding distance
predict physics-verified control transfer? Nobody else benchmarks this (the Dec-2025 co-design survey lists
controller-transfer-between-morphologies as "largely unexplored"), so it is both our internal regression gate
and, later, a public artifact. Runs the un-gameable eval (``embedding_eval``) over a physics-verified transfer
corpus and returns the coarse (class-level) transfer-ranking numbers + the gated-metric state, with a floor gate
so a real regression BLOCKS. Pure-python + deterministic given the (cached/fixture) corpus.
"""
from __future__ import annotations

from pathlib import Path

# floor well below the measured baseline (~0.90 coarse) so noise passes but a real regression (toward chance 0.5)
# blocks. The corpus's own class structure sets the achievable ceiling; we gate on beating chance clearly.
TRIPLET_FLOOR = 0.65


def embedding_transfer_scorecard(corpus_path: str | Path | None = None, *, floor: float = TRIPLET_FLOOR) -> dict:
    """Score the SHIPPED embedding (embed_body) on the transfer corpus. Returns the coarse transfer-ranking
    metrics + adoption state + a pass/fail gate. Honest 'no corpus' status when none is cached (CI builds it
    once); never raises."""
    from virturoid.services.embedding_eval import evaluate_embedding
    from virturoid.services.embedding_transfer_corpus import DEFAULT_CACHE, load_corpus
    from virturoid.services.robotics_vector_memory import embed_body

    path = Path(corpus_path) if corpus_path else DEFAULT_CACHE
    corpus = load_corpus(path)
    if not corpus or not corpus.get("bodies"):
        return {"ok": True, "status": "no_corpus", "gate_pass": True,
                "note": f"no transfer corpus at {path}; run embedding_transfer_corpus.build_transfer_corpus to populate"}

    m = evaluate_embedding(embed_body, corpus)
    triplet = m.get("triplet_ranking_acc")
    try:
        from virturoid.services.body_metric import _load
        bundle = _load()
        active = "learned_metric" if (bundle and bundle.get("proven")) else "baseline_29d"
    except Exception:  # noqa: BLE001
        active = "baseline_29d"
    gate_pass = bool(triplet is not None and triplet >= floor)
    return {
        "ok": True, "status": "scored",
        "n_bodies": m.get("n_bodies"), "n_sources": m.get("n_sources"),
        "n_credible_transfers": m.get("n_credible_transfers"),
        "coarse_triplet_ranking_acc": triplet,
        "precision_at": m.get("precision_at"),
        "kendall_tau_prox_forward": m.get("kendall_tau_prox_forward"),
        "within_legged_cos": m.get("within_legged_cos"),
        "embedding_active": active,
        "floor": floor, "gate_pass": gate_pass,
        "headline": (f"distance-predicts-transfer (coarse) = {triplet} on {m.get('n_bodies')} bodies / "
                     f"{m.get('n_credible_transfers')} verified transfers; embedding={active}; "
                     f"gate {'PASS' if gate_pass else 'FAIL'} (floor {floor})"),
    }
