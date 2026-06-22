"""Build the SFT dataset to distill a local proposer (Nemotron) from the flywheel.

Plan Phase E. This assembles supervised (prompt -> physics-verified gene/design) examples
from the memory flywheel (+ the seed genes) into JSONL. The actual LoRA/SFT training is
external GPU work — this only builds the data and prints the recipe; we don't fake a model.

    python scripts/distill_proposer.py --memory build/webapp/memory --min-success 0.8 \
        --out build/memory/proposer_sft.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

_RECIPE = """
Next (external GPU): LoRA-SFT a small base (e.g. Nemotron) on this JSONL with trl SFTTrainer,
merge the adapter, then serve via vLLM or export GGUF -> Ollama. Set VIRTUROID_LLM_BACKEND=local
+ VIRTUROID_LOCAL_LLM_URL/MODEL to route the proposer to your distilled model. Re-run the eval
harness to confirm it matches/beats the cloud before promoting it.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the proposer-distillation SFT dataset.")
    ap.add_argument("--memory", default="build/webapp/memory", help="memory dir with virturoid_memory.db")
    ap.add_argument("--min-success", type=float, default=0.8)
    ap.add_argument("--out", default="build/memory/proposer_sft.jsonl")
    ap.add_argument("--include-seed-genes", action="store_true",
                    help="also include the seed-gene library as (species-prompt -> gene) examples")
    args = ap.parse_args(argv)

    from virturoid.services.proposer_distill import dataset_from_memory, export_sft_jsonl, sft_example

    examples = dataset_from_memory(Path(args.memory), min_success=args.min_success)
    if args.include_seed_genes:
        from virturoid.fixtures.gene_library import seed_genes
        examples += [sft_example(f"Build a {g.robot_class} for: {g.species}", g) for g in seed_genes()]

    n = export_sft_jsonl(examples, Path(args.out))
    print(f"wrote {n} SFT examples -> {args.out}")
    if n == 0:
        print("(no verified rows yet — run builds to grow the flywheel, then re-run)")
    print(_RECIPE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
