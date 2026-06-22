"""Train the GNN fitness surrogate on the 3060 (Phase C).

Loads a (gene -> measured success) JSONL produced by generate_surrogate_data.py and trains the
message-passing GNN, using CUDA when available (the RTX 3060) and falling back to CPU. Saves
the model so the co-design inner loop can screen candidate genes cheaply.

    python scripts/train_gene_gnn.py --data build/data/humanoid_surrogate.jsonl \
        --out build/models/gene_gnn.pt --epochs 400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train the gene GNN surrogate.")
    ap.add_argument("--data", required=True, help="JSONL of {gene, success_rate}")
    ap.add_argument("--out", default="build/models/gene_gnn.pt")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--hidden", type=int, default=32)
    args = ap.parse_args(argv)

    import torch

    from virturoid.services.gene_surrogate_nn import train_from_rows_jsonl

    print(f"CUDA available: {torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else " — training on CPU"))
    model, metrics = train_from_rows_jsonl(Path(args.data), epochs=args.epochs)
    model.save(Path(args.out))
    print(f"device={metrics['device']} loss {metrics['first_loss']:.4f} -> {metrics['final_loss']:.4f}")
    print(f"saved GNN -> {args.out}")
    print("Use it as a cheap inner-loop screen: GeneGNN.load(path).rank(candidate_genes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
