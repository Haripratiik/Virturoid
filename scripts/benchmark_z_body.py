"""Benchmark: does the LEARNED z_body retrieve success-similar designs better than the hand-crafted vector?

The honest Move-2 acceptance criterion (NOT agreement-with-deterministic). Splits a labeled gene dataset,
trains the GNN on the TRAIN split only (fair held-out), and compares k-NN success-correlation on the TEST
split for the learned latent vs the deterministic ``embed_gene`` vector. Higher = the embedding places
similar-success designs nearer = better for warm-start-by-quality.

    python scripts/benchmark_z_body.py --data build/data/multiclass.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Held-out embedding-quality benchmark for the learned z_body.")
    ap.add_argument("--data", default="build/data/multiclass.jsonl")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    from virturoid.schemas.gene import RobotGene
    from virturoid.services.embedding_benchmark import knn_success_correlation, pearson
    from virturoid.services.gene_surrogate_nn import GeneGNN
    from virturoid.services.morphology_embedding import embed_gene

    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line.strip()]
    genes = [RobotGene.from_dict(r["gene"]) for r in rows]
    ys = [float(r["success_rate"]) for r in rows]
    rng = random.Random(args.seed)
    idx = list(range(len(genes)))
    rng.shuffle(idx)
    cut = int(len(idx) * 0.8)
    tr, te = idx[:cut], idx[cut:]
    ytr, yte = [ys[i] for i in tr], [ys[i] for i in te]
    print(f"dataset {len(genes)} rows -> train {len(tr)} / test {len(te)}  (k={args.k})")

    det = knn_success_correlation([embed_gene(genes[i]) for i in tr], ytr,
                                  [embed_gene(genes[i]) for i in te], yte, k=args.k)

    gnn = GeneGNN(hidden=32, device="cpu")
    gnn.fit([genes[i] for i in tr], ytr, epochs=args.epochs)   # train on TRAIN only -> fair held-out
    learned = knn_success_correlation([gnn.embed(genes[i]) for i in tr], ytr,
                                      [gnn.embed(genes[i]) for i in te], yte, k=args.k)
    head = round(pearson([gnn.predict(genes[i]) for i in te], yte), 4)   # the GNN's own success head (reference)

    print(f"  deterministic embed_gene : pearson={det['pearson']:+.4f}  spearman={det['spearman']:+.4f}")
    print(f"  learned GNN latent (kNN) : pearson={learned['pearson']:+.4f}  spearman={learned['spearman']:+.4f}")
    print(f"  GNN success head (direct): pearson={head:+.4f}")
    margin = learned["pearson"] - det["pearson"]
    verdict = ("LEARNED z_body wins -> activate for warm-start-by-quality"
               if margin > 0.05 else
               "DETERMINISTIC z_body wins/ties -> keep the hand-crafted embedding (honest)")
    print(f"\nVERDICT (learned-minus-deterministic pearson = {margin:+.4f}): {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
