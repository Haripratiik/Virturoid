"""Transfer ledger — the auto-growing ground truth that lets the embedding UPGRADE ITSELF with usage.

The gated metric (``body_metric``) can only beat the baseline when it has fine-grained, physics-verified transfer
labels to learn from (measured: on coarse class-level data the baseline is already ~0.90; the fine structure is
where learning can win). Those labels are cheap to manufacture at the moment a gait is BANKED: replay the new
gait on the K embedding-nearest banked bodies, and their gaits on the new body — a bounded 2·K rollouts — and
record each verified outcome. This module owns that ledger and the RATCHET that periodically re-fits the metric
against it, adopting a learned metric ONLY when it provably beats the shipped baseline held-out (the fit's own
LOO gate). So the moat compounds automatically: more usage → denser matrix → the first honest win flips the
metric on, and a failed fit changes nothing.

CPU, deterministic, bounded; genes are stored inline (JSON) so trials survive species-vault churn.
"""
from __future__ import annotations

import json
import time

_SCHEMA = """
CREATE TABLE IF NOT EXISTS transfer_trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    src_skill_id TEXT,                 -- the banked gait that was replayed
    src_gene_id  TEXT NOT NULL,        -- body the gait was learned ON
    dst_gene_id  TEXT NOT NULL,        -- body it was replayed ON
    src_gene     TEXT NOT NULL,        -- JSON RobotGene (inline: survives vault churn)
    dst_gene     TEXT NOT NULL,
    gait_params  TEXT NOT NULL,        -- JSON params replayed
    credible     INTEGER NOT NULL,     -- un-gameable verdict: survived + credible walk + forward >= floor
    forward_m    REAL NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_transfer_src ON transfer_trials(src_gene_id);
CREATE INDEX IF NOT EXISTS idx_transfer_dst ON transfer_trials(dst_gene_id);
"""

_CRED_FORWARD = 0.25


def _conn(db):
    conn = db.conn if hasattr(db, "conn") else db
    conn.executescript(_SCHEMA)
    return conn


def record_transfer_trial(db, *, src_gene, dst_gene, gait_params: dict, result: dict,
                          src_skill_id: str | None = None) -> None:
    """Record one physics-verified replay outcome (``result`` from ``evaluate_gait``)."""
    conn = _conn(db)
    cred = bool(result.get("survived") and result.get("credible")
                and float(result.get("forward", 0.0)) >= _CRED_FORWARD)
    conn.execute(
        "INSERT INTO transfer_trials (src_skill_id, src_gene_id, dst_gene_id, src_gene, dst_gene, gait_params,"
        " credible, forward_m, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (src_skill_id, getattr(src_gene, "id", "?"), getattr(dst_gene, "id", "?"),
         json.dumps(src_gene.to_dict()), json.dumps(dst_gene.to_dict()), json.dumps(gait_params),
         1 if cred else 0, round(float(result.get("forward", 0.0)), 4), time.strftime("%Y-%m-%dT%H:%M:%S")))
    conn.commit()


def cross_evaluate_on_bank(db, gene, gait_params: dict, *, skill_id: str | None = None, k: int = 3,
                           steps: int = 500) -> int:
    """The bounded corpus-densifier run at bank time: replay THIS gait on the K embedding-nearest banked bodies,
    and THEIR banked gaits on this body (2·K rollouts). Every outcome — walk or fall — is a verified label.
    Returns the number of trials recorded. Best-effort: any failure records nothing and raises nothing."""
    try:
        from virturoid.schemas.gene import RobotGene
        from virturoid.services.gait_flywheel import LOCOMOTION, _gait_params_for_skill
        from virturoid.services.gait_search import evaluate_gait
        from virturoid.services.robotics_vector_memory import RoboticsVectorMemory
        vm = RoboticsVectorMemory(db)
        species_genes = vm._species_genes()
        n = 0
        for hit in vm.nearest_skills(gene, LOCOMOTION, k=k, min_sim=0.0):
            sid = str(hit.get("obj_id", ""))
            if sid == skill_id:
                continue                                          # don't trial a gait against itself
            other_params = _gait_params_for_skill(db, sid)
            meta = hit.get("meta") or {}
            # the neighbour's BODY: inline gene in the vector meta (self-contained), else the species vault
            other_gene = None
            if isinstance(meta.get("gene"), dict):
                try:
                    other_gene = RobotGene.from_dict(meta["gene"])
                except Exception:  # noqa: BLE001
                    other_gene = None
            if other_gene is None:
                other_gene = species_genes.get(meta.get("species"))
            # a) the NEW gait on the neighbour's body
            if other_gene is not None:
                r = evaluate_gait(other_gene, gait_params, steps=steps)
                record_transfer_trial(db, src_gene=gene, dst_gene=other_gene, gait_params=gait_params,
                                      result=r, src_skill_id=skill_id)
                n += 1
            # b) the neighbour's gait on the NEW body
            if isinstance(other_params, dict) and other_gene is not None:
                r = evaluate_gait(gene, other_params, steps=steps)
                record_transfer_trial(db, src_gene=other_gene, dst_gene=gene, gait_params=other_params,
                                      result=r, src_skill_id=sid)
                n += 1
        return n
    except Exception:  # noqa: BLE001 - the ledger is an accelerant; banking must never fail because of it
        return 0


def corpus_from_ledger(db, *, min_trials: int = 8) -> dict | None:
    """Assemble a ``body_metric``-fit corpus from the accumulated trials (bodies = every distinct gene seen;
    transfer[i][j] = did i's gait walk credibly on j, where trialed). None when the ledger is too thin."""
    from virturoid.schemas.gene import RobotGene
    conn = _conn(db)
    rows = conn.execute("SELECT * FROM transfer_trials").fetchall()
    if len(rows) < min_trials:
        return None
    genes: dict[str, RobotGene] = {}
    for r in rows:
        for key_id, key_gene in (("src_gene_id", "src_gene"), ("dst_gene_id", "dst_gene")):
            gid = r[key_id]
            if gid not in genes:
                try:
                    genes[gid] = RobotGene.from_dict(json.loads(r[key_gene]))
                except Exception:  # noqa: BLE001
                    continue
    ids = sorted(genes)
    idx = {g: i for i, g in enumerate(ids)}
    n = len(ids)
    T = [[0] * n for _ in range(n)]
    F = [[0.0] * n for _ in range(n)]
    seen = [[False] * n for _ in range(n)]
    src_ok = set()
    for r in rows:
        i, j = idx.get(r["src_gene_id"]), idx.get(r["dst_gene_id"])
        if i is None or j is None:
            continue
        # keep-best across repeat trials of the same pair
        if not seen[i][j] or r["forward_m"] > F[i][j]:
            T[i][j] = int(r["credible"])
            F[i][j] = float(r["forward_m"])
            seen[i][j] = True
        src_ok.add(r["src_gene_id"])
    bodies = [{"id": g, "robot_class": getattr(genes[g], "robot_class", None), "_gene": genes[g],
               "self_credible": g in src_ok} for g in ids]
    return {"bodies": bodies, "transfer": T, "forward": F, "n_trials": len(rows)}


def ratchet_metric(db, *, min_trials: int = 8) -> dict:
    """Re-fit the gated transfer metric against the ledger. Adopts (saves) ONLY when the fit's own leave-one-
    body-out gate says it beats the shipped baseline — otherwise nothing changes. Returns a status dict either
    way (for the readiness ledger / night shift log)."""
    from virturoid.services.body_metric import fit_body_metric
    corpus = corpus_from_ledger(db, min_trials=min_trials)
    if corpus is None:
        return {"ok": True, "action": "skipped", "reason": f"ledger too thin (<{min_trials} trials)"}
    bundle = fit_body_metric(corpus, feature_space="rich", whiten=False, method="relevance", save=True)
    reindexed = False
    if bundle["proven"]:
        # adoption CHANGES embed_body's dimension -> the existing body index is now stale (the nearest() dim-guard
        # would skip every vector). Re-index the BODY sub-space so retrieval works at the new dim immediately.
        try:
            import virturoid.services.body_metric as _bm
            from virturoid.services.robotics_vector_memory import RoboticsVectorMemory
            _bm._cache = None                                    # force embed_metric to pick up the just-saved bundle
            _bm._cache_mtime = None
            RoboticsVectorMemory(db).index_species_bodies()
            reindexed = True
        except Exception:  # noqa: BLE001 - re-index is best-effort; the dim-guard keeps retrieval safe meanwhile
            pass
    return {"ok": True, "action": "adopted" if bundle["proven"] else "kept_baseline",
            "held_out_triplet_acc": bundle["held_out_triplet_acc"],
            "baseline_held_out_triplet_acc": bundle["baseline_held_out_triplet_acc"],
            "n_bodies": bundle["n_bodies"], "n_trials": corpus["n_trials"], "reindexed_bodies": reindexed}
