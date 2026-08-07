"""Run one corpus-factory night (master_plan_v6 WS-C.2).

    PYTHONPATH=src python scripts/corpus_factory_night.py --memory build/memory [--bodies 20] [--grow-ledger]

Proposer: OFFLINE by default (heuristic composition over a diverse prompt bank + verified-design jitter — zero
tokens). In production the proposer is an offline batch LLM agent (explicit dev tokens, never the customer hot
path); pass --strict-llm to route composition through the live design model. The night runs the ordered gate
stack, checkpoints after every admit, banks admitted bodies into the retrievable corpus, and ratchets the metric.

``--memory`` IS REQUIRED AND IT IS NOT A FORMALITY. This script is a DELIBERATE WRITER: growing the bank is the
whole point of running it, so it is the one entry point that must never guess. It used to default to
``build/memory``, which meant ``python scripts/corpus_factory_night.py`` — the obvious thing to type when
experimenting — grew the developer's real corpus.

AND IT DID NOT ACTUALLY HONOUR THE FLAG END TO END. Measured 2026-08-07 with the redirect pointed at an empty
directory: the PROPOSER's own ``compose_robot(ensure_walkable=True)`` reaches ``ensure_walkable_quad`` ->
``fit_gait_for_body(db=None)`` -> ``gait_flywheel._open_db_and_learn``, which resolves its destination with
``agent_tools.safe_build_path(None, "memory")`` — a second default rule that reads ``<cwd>/build/memory`` and has
never read ``VIRTUROID_MEMORY_DIR``. It created a 122 KB database there while the night's own directory stayed
empty, and it does that with ``bank=True``. So a night aimed at a fresh corpus was warm-starting AND banking
against the very bank it was supposed to be replacing.

The fix is to say the destination ONCE, in the environment, BEFORE any virturoid module is imported — the same
mechanism ``tests/conftest.py`` uses, and the reason ``memory_db`` now rewrites the conventional path when a
redirect is in force. Passing ``memory_dir=`` down the call chain cannot work: the leaking call sites are three
levels below this file and take no destination argument at all.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# A diverse prompt bank spanning structural families (breadth, not depth — GenBot-1K). The proposer biases toward
# the thinnest classes reported in the night context. NONE overlaps the held-out set (the guard also enforces it).
#
# The legged bank is DELIBERATELY the longest and varies the axes a gait actually cares about — leg length, body
# width, mass, and stance — because a gait corpus mined for param regions needs bodies that DIFFER in the
# dimensions the parameters address. It stays inside the held-out partition: ``_niche_many_limb`` reserves any
# body with >=6 limb chains wholesale, so no hexapod/octopod prompt belongs here (the one that was here,
# "a low wide six-legged crawler", was burning a proposal slot on a body the guard rejects by construction).
_PROMPT_BANK = {
    "legged": ["a sturdy four-legged walking robot", "a tall long-legged striding quadruped",
               "a compact short-legged trotting robot", "a heavy broad-bodied walking robot",
               "a light nimble quadruped runner", "a wide-stance four-legged carrier robot",
               "a narrow-bodied four-legged patrol robot", "a small lightweight four-legged inspection robot",
               "a large heavy four-legged load-carrying robot", "a mid-size four-legged robot with long shins",
               "a low-slung four-legged robot with short thick legs",
               "a four-legged robot with a long body and short legs",
               "a four-legged robot with a short body and long legs",
               "a slender four-legged walking robot with thin limbs",
               "a squat wide four-legged robot with a broad chest",
               "a four-legged robot built for slow steady walking",
               "a four-legged robot built for fast light-footed running",
               "a medium four-legged robot with evenly proportioned legs"],
    "mobile": ["a four-wheeled flat-deck rover", "a compact two-wheel differential robot",
               "a wide six-wheeled hauler", "a small indoor delivery cart on four wheels"],
    "manipulator": ["a 4-joint tabletop arm with a gripper", "a 6-joint industrial arm",
                    "a long slender inspection arm", "a compact 3-joint pick arm"],
}


def _offline_proposer(strict_llm: bool, classes: tuple[str, ...] | None = None):
    """Yield (gene, prompt) candidates, biased toward the thinnest classes in the night context.

    ``classes`` restricts the bank to named families. A GAIT-corpus night passes ``("legged",)``: a wheeled base
    and an arm cost the same verify budget as a quadruped and can contribute no gait row at all, so spending a
    third of the night's proposals on them is spending it on nothing the gait bank will ever retrieve.
    """
    from virturoid.services.morphology_composer import compose_robot
    bank_by_class = {k: v for k, v in _PROMPT_BANK.items() if not classes or k in classes} or _PROMPT_BANK
    counters = {k: 0 for k in bank_by_class}

    def propose(context):
        thin = [c for c in context.get("thinnest_classes", []) if c in bank_by_class]
        order = thin + [c for c in bank_by_class if c not in thin] or list(bank_by_class)
        cls = order[0] if order else next(iter(bank_by_class))
        bank = bank_by_class.get(cls) or next(iter(bank_by_class.values()))
        prompt = bank[counters[cls] % len(bank)]
        counters[cls] += 1
        try:
            gene = compose_robot(prompt, llm="auto", ensure_walkable=(cls == "legged"), strict_llm=strict_llm)
        except Exception:  # noqa: BLE001 - a failed composition is skipped; the night moves on
            return None
        return gene, prompt
    return propose


def claim_destination(memory: str) -> Path:
    """Make ``--memory`` the destination for EVERY default in this process, before virturoid is imported.

    Returns the resolved directory. Refuses if a conflicting ``VIRTUROID_MEMORY_DIR`` is already exported, because
    silently preferring either one is how a night lands somewhere nobody asked for — the exact failure this whole
    change exists to prevent, just with the two sides swapped.
    """
    target = Path(memory).resolve()
    already = os.environ.get("VIRTUROID_MEMORY_DIR")
    if already and Path(already).resolve() != target:
        raise SystemExit(f"refusing to run: --memory says {target} but VIRTUROID_MEMORY_DIR says "
                         f"{Path(already).resolve()}. Pick one.")
    target.mkdir(parents=True, exist_ok=True)
    os.environ["VIRTUROID_MEMORY_DIR"] = str(target)
    return target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bodies", type=int, default=20)
    ap.add_argument("--memory", required=True,
                    help="REQUIRED destination for this night's corpus. Pass build/memory to grow the real bank; "
                         "pass anything else to build a fresh one. Exported as VIRTUROID_MEMORY_DIR before any "
                         "virturoid import, so nested defaults land here too.")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--grow-ledger", action="store_true")
    ap.add_argument("--strict-llm", action="store_true")
    ap.add_argument("--deep-verify", action="store_true",
                    help="VERIFY-BUILD via a CPU gait search (minutes/body) instead of the fast scripted verdict")
    ap.add_argument("--gait-corpus", action="store_true",
                    help="grow the GAIT bank: legged proposals only, each body fitted with an operating point of "
                         "its own at the settling horizon, and banked ONLY if that point survives a perturbation "
                         "of itself. This is the write path `default_bank_fn` never made (task #207).")
    ap.add_argument("--classes", default=None,
                    help="comma-separated prompt families to propose from (default: all)")
    args = ap.parse_args()

    # BEFORE the import below, not after: ``memory_db`` binds DEFAULT_DB_PATH at import time and
    # ``memory_store``/``autonomous_build`` bind their default arguments at def time.
    mem = claim_destination(args.memory)
    print(f"corpus-factory night writing to {mem} (VIRTUROID_MEMORY_DIR)", file=sys.stderr)

    from virturoid.services.corpus_factory import (FactoryConfig, default_bank_fn, gait_bank_fn,
                                                   gait_fit_verify_fn, gait_search_verify, held_out_aware,
                                                   run_factory_night)
    cfg = FactoryConfig(max_bodies=args.bodies, grow_ledger=args.grow_ledger)
    classes = tuple(c.strip() for c in args.classes.split(",")) if args.classes else \
        (("legged",) if args.gait_corpus else None)
    # v7-C1: wrap the proposer so held-out-niche bodies (the measured live-LLM waste, 25/30) don't burn the budget
    proposer = held_out_aware(_offline_proposer(args.strict_llm, classes))
    verify_fn = (gait_fit_verify_fn(mem) if args.gait_corpus else
                 (gait_search_verify if args.deep_verify else None))
    res = run_factory_night(proposer, config=cfg,
                            manifest_path=args.manifest or (mem / "corpus_factory.json"),
                            memory_dir=mem,
                            bank_fn=(gait_bank_fn if args.gait_corpus else default_bank_fn),
                            verify_fn=verify_fn)
    print(json.dumps(res.to_dict(), indent=2, default=str))
    print(f"\nadmitted {len(res.admitted)} · ANNECS {res.annecs} · rejected {dict(res.rejected)} · "
          f"mean-sim {res.mean_pairwise_similarity} · {res.wall_s}s")


if __name__ == "__main__":
    main()
