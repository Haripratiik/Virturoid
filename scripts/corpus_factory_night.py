"""Run one corpus-factory night (master_plan_v6 WS-C.2).

    PYTHONPATH=src python scripts/corpus_factory_night.py [--bodies 20] [--memory build/memory] [--grow-ledger]

Proposer: OFFLINE by default (heuristic composition over a diverse prompt bank + verified-design jitter — zero
tokens). In production the proposer is an offline batch LLM agent (explicit dev tokens, never the customer hot
path); pass --strict-llm to route composition through the live design model. The night runs the ordered gate
stack, checkpoints after every admit, banks admitted bodies into the retrievable corpus, and ratchets the metric.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# A diverse prompt bank spanning structural families (breadth, not depth — GenBot-1K). The proposer biases toward
# the thinnest classes reported in the night context. NONE overlaps the held-out set (the guard also enforces it).
_PROMPT_BANK = {
    "legged": ["a sturdy four-legged walking robot", "a low wide six-legged crawler",
               "a tall long-legged striding quadruped", "a compact short-legged trotting robot",
               "a heavy broad-bodied walking robot", "a light nimble quadruped runner"],
    "mobile": ["a four-wheeled flat-deck rover", "a compact two-wheel differential robot",
               "a wide six-wheeled hauler", "a small indoor delivery cart on four wheels"],
    "manipulator": ["a 4-joint tabletop arm with a gripper", "a 6-joint industrial arm",
                    "a long slender inspection arm", "a compact 3-joint pick arm"],
}


def _offline_proposer(strict_llm: bool):
    """Yield (gene, prompt) candidates, biased toward the thinnest classes in the night context."""
    from virturoid.services.morphology_composer import compose_robot
    counters = {k: 0 for k in _PROMPT_BANK}

    def propose(context):
        thin = [c for c in context.get("thinnest_classes", []) if c in _PROMPT_BANK]
        order = thin + [c for c in _PROMPT_BANK if c not in thin] or list(_PROMPT_BANK)
        cls = order[0] if order else "legged"
        bank = _PROMPT_BANK.get(cls) or next(iter(_PROMPT_BANK.values()))
        prompt = bank[counters[cls] % len(bank)]
        counters[cls] += 1
        try:
            gene = compose_robot(prompt, llm="auto", ensure_walkable=(cls == "legged"), strict_llm=strict_llm)
        except Exception:  # noqa: BLE001 - a failed composition is skipped; the night moves on
            return None
        return gene, prompt
    return propose


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bodies", type=int, default=20)
    ap.add_argument("--memory", default="build/memory")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--grow-ledger", action="store_true")
    ap.add_argument("--strict-llm", action="store_true")
    ap.add_argument("--deep-verify", action="store_true",
                    help="VERIFY-BUILD via a CPU gait search (minutes/body) instead of the fast scripted verdict")
    args = ap.parse_args()

    from virturoid.services.corpus_factory import (FactoryConfig, default_bank_fn, gait_search_verify,
                                                   run_factory_night)
    mem = Path(args.memory)
    mem.mkdir(parents=True, exist_ok=True)
    cfg = FactoryConfig(max_bodies=args.bodies, grow_ledger=args.grow_ledger)
    res = run_factory_night(_offline_proposer(args.strict_llm), config=cfg,
                            manifest_path=args.manifest or (mem / "corpus_factory.json"),
                            memory_dir=mem, bank_fn=default_bank_fn,
                            verify_fn=(gait_search_verify if args.deep_verify else None))
    print(json.dumps(res.to_dict(), indent=2, default=str))
    print(f"\nadmitted {len(res.admitted)} · ANNECS {res.annecs} · rejected {dict(res.rejected)} · "
          f"mean-sim {res.mean_pairwise_similarity} · {res.wall_s}s")


if __name__ == "__main__":
    main()
