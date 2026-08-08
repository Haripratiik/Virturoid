"""Run one corpus-factory night (master_plan_v6 WS-C.2).

    python scripts/corpus_factory_night.py --memory build/memory [--bodies 20] [--grow-ledger]

Proposer: a target grid over structural families, composed through the PRODUCT's composer. "Zero tokens" was
claimed here and was not true: ``compose_robot(llm="auto")`` resolves whatever ``VIRTUROID_LLM_BACKEND`` names,
so with a backend configured every proposal has always been spending dev tokens. Pass ``--llm off`` for a
genuinely deterministic, zero-token night; ``--strict-llm`` routes composition through the live design model and
refuses to substitute a template. The night runs the ordered gate stack, checkpoints after every admit, banks
admitted bodies into the retrievable corpus, and ratchets the metric.

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

# Runnable as `python scripts/corpus_factory_night.py` from the repo root, which is how the docstring above and
# every task note describe it. 57 of the 67 scripts here already do this; this one did not, so the documented
# invocation died on ModuleNotFoundError before it reached a single robot. A long overnight run that fails in the
# first second is a cheap failure, but it fails at the exact moment nobody is watching.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# A diverse prompt bank spanning structural families (breadth, not depth — GenBot-1K). The proposer biases toward
# the thinnest classes reported in the night context. NONE overlaps the held-out set (the guard also enforces it).
#
# THE LEGGED BANK IS BUILT, NOT LISTED (task #281). The hand-written list that used to sit here varied adjectives
# — "sturdy", "nimble", "slender", "squat" — across eighteen entries, and MEASURED 2026-08-08 the composer
# collapsed all eighteen to THREE distinct bodies (16 of them byte-identical in structure): the only phrase it
# actually reads off a functional walker prompt is the SIZE word. Eighteen prompts, three bodies, and a night
# that then rejects the repeats for novelty after paying 39 s to fit a controller for each.
#
# So the bank is generated from a TARGET GRID over the axes the composer DEMONSTRABLY honours, and only those.
# Measured 2026-08-08 by composing each candidate axis and comparing the resulting ``body_key``/niche/DOF:
#
#   MOVES THE BODY      size word (small|medium|large)          3 cells on each quadruped route
#                       body-plan route: a "walking robot" goes to the parametric template, a "walking creature"
#                       to the general anatomy compiler, a "two-legged" one to the anthropometric biped —
#                       different builders, different DOF (14 vs 8) and different MAP-Elites niches
#                       stature in metres, BUT ONLY ON THE BIPED, and only above ~1.03 m (the builder clamps
#                       head-height to [0.12, 0.32] = stature/8.55, so 0.5 m and 1.0 m are the same robot)
#   INERT               leg count 3 or 5 (the generic body builds legs in PAIRS, so 3→2 and 5→4)
#                       "long slender legs" / "short thick legs" / "wide stance" / "long body" — every
#                       proportion adjective composed to a byte-identical body
#                       stature on a quadruped, ``payload_kg=``, ``reach_m=`` — all byte-identical
#
# So the grid is ten targets, not forty-eight, and that is the HONEST ceiling of the offline composer: it can
# make about ten structurally distinct legged bodies and no more. A 20-body night will run out of new structures
# and stop, which is the correct behaviour — see the note on ``_offline_proposer``'s dedup. The bottleneck for
# #274 (distinct bodies) is the COMPOSER's expressiveness, not the factory's proposal budget, and no prompt
# engineering inside this file can move it further.
_SIZE_WORDS = ("small", "medium", "large")
_BIPED_STATURES_M = (1.2, 1.6, 2.0, 2.4)       # above the builder's lower clamp, so each is a different robot


def _legged_targets() -> list[tuple[str, dict]]:
    """The legged target grid, as (prompt, target) pairs, INTERLEAVED so the earliest slots of a short night
    span the widest structural range rather than three sizes of one body plan.

    ``target`` records what we ASKED for, so a diagnostic can say which cell a draft came from."""
    quad_tpl = [(f"a {w} four-legged walking robot", {"route": "template", "size": w}) for w in _SIZE_WORDS]
    quad_ana = [(f"a {w} four-legged walking creature", {"route": "anatomy", "size": w}) for w in _SIZE_WORDS]
    biped = [(f"a two-legged walking robot {h:.2f} m tall", {"route": "biped", "height_m": h})
             for h in _BIPED_STATURES_M]
    out, pools = [], [quad_tpl, quad_ana, biped]
    for i in range(max(len(p) for p in pools)):
        for pool in pools:
            if i < len(pool):
                out.append(pool[i])
    return out


_PROMPT_BANK = {
    "legged": [p for p, _ in _legged_targets()],
    "mobile": ["a four-wheeled flat-deck rover", "a compact two-wheel differential robot",
               "a wide six-wheeled hauler", "a small indoor delivery cart on four wheels"],
    "manipulator": ["a 4-joint tabletop arm with a gripper", "a 6-joint industrial arm",
                    "a long slender inspection arm", "a compact 3-joint pick arm"],
}


def _offline_proposer(strict_llm: bool, classes: tuple[str, ...] | None = None, llm="auto"):
    """Yield (gene, prompt) candidates, biased toward the thinnest classes in the night context.

    ``classes`` restricts the bank to named families. A GAIT-corpus night passes ``("legged",)``: a wheeled base
    and an arm cost the same verify budget as a quadruped and can contribute no gait row at all, so spending a
    third of the night's proposals on them is spending it on nothing the gait bank will ever retrieve.

    TWO-STAGE, SO A REJECTED BODY COSTS A DRAFT AND NOT A SLOT. The proposal is composed CHEAP first
    (``ensure_walkable=False``) and shown to the guard; only a body the guard admits pays for the walkability
    pass. It also refuses to hand up a body whose ``body_key`` the corpus already holds — the night's own
    structural dedup, done here for ~0 s instead of downstream for the price of a gait fit.
    """
    from virturoid.services.heldout_set import body_key, explain
    from virturoid.services.morphology_composer import compose_robot
    bank_by_class = {k: v for k, v in _PROMPT_BANK.items() if not classes or k in classes} or _PROMPT_BANK
    counters = {k: 0 for k in bank_by_class}
    burned: set[str] = set()               # prompts whose DRAFT the guard reserved — never drawn twice
    seen_keys: set[str] = set()            # body_keys this proposer has already handed up
    stats = {"drafts_composed": 0, "reserved_drafts": 0, "duplicate_drafts": 0, "failed_drafts": 0,
             "finalized": 0}

    def propose(context):
        thin = [c for c in context.get("thinnest_classes", []) if c in bank_by_class]
        order = thin + [c for c in bank_by_class if c not in thin] or list(bank_by_class)
        cls = order[0] if order else next(iter(bank_by_class))
        bank = bank_by_class.get(cls) or next(iter(bank_by_class.values()))
        corpus_keys = set(context.get("corpus_keys") or ()) | seen_keys
        for _ in range(len(bank)):                        # walk the grid; never re-draw a burned target
            prompt = bank[counters[cls] % len(bank)]
            counters[cls] += 1
            if prompt in burned:
                continue
            try:
                stats["drafts_composed"] += 1
                draft = compose_robot(prompt, llm=llm, ensure_walkable=False, strict_llm=strict_llm)
            except Exception:  # noqa: BLE001 - a failed composition is skipped; the night moves on
                stats["failed_drafts"] += 1
                burned.add(prompt)
                continue
            why = explain(draft, prompt=prompt)
            if why["held_out"]:
                stats["reserved_drafts"] += 1
                burned.add(prompt)                        # this target is reserved BY CONSTRUCTION -> retire it
                continue
            if body_key(draft) in corpus_keys:
                stats["duplicate_drafts"] += 1
                burned.add(prompt)                        # a structural repeat; the grid has better cells left
                continue
            gene = draft
            if cls == "legged" and not strict_llm:
                try:                                       # the EXPENSIVE stage, paid only for an admissible body
                    from virturoid.services.anatomy_compiler import ensure_walkable_quad
                    walked = ensure_walkable_quad(draft, prompt)
                    plan = (draft.metadata or {}).get("build_plan")
                    if plan and not (walked.metadata or {}).get("build_plan"):
                        # ensure_walkable_quad may hand back a substituted body; the request routing is
                        # provenance and must survive the swap (compose_robot re-attaches it for the same reason)
                        walked.metadata = {**(walked.metadata or {}), "build_plan": plan}
                    gene = walked
                except Exception:  # noqa: BLE001 - best-effort, exactly as compose_robot(ensure_walkable=True)
                    gene = draft
            # ...and dedup AGAIN on the finalized body. The walkability pass can substitute a template, so two
            # different drafts can converge on one shipped body: the first gated night after the guard fix drew
            # 20 slots and 11 of them were rejected at the novelty gate for exactly this, each having been let
            # through here because its DRAFT key was still novel.
            key = body_key(gene)
            if key in corpus_keys:
                stats["duplicate_drafts"] += 1
                burned.add(prompt)
                continue
            stats["finalized"] += 1
            seen_keys.add(key)
            return gene, prompt
        return None                                        # the grid holds no body this corpus does not have
    propose.stats = stats
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
    ap.add_argument("--llm", choices=("auto", "off"), default="auto",
                    help="'auto' resolves the configured design backend (the product path); 'off' composes "
                         "deterministically with zero tokens. 'auto' DEGRADES to 'off' if the preflight probe "
                         "finds the backend unreachable or capped — see --no-llm-preflight.")
    ap.add_argument("--no-llm-preflight", action="store_true",
                    help="skip the one-call reachability probe and let every composition discover the backend "
                         "is down on its own (measured 45 s of retry backoff per call when it is)")
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
    llm = "auto" if args.llm == "auto" else None
    if llm == "auto" and not args.no_llm_preflight and not args.strict_llm:
        ok, note = llm_preflight()
        print(f"llm preflight: {note}", file=sys.stderr)
        if not ok:
            llm = None       # compose deterministically instead of paying the backoff on every call, silently
    # v7-C1 / #281: the guard now STEERS the proposer (see `_offline_proposer`), and this wrapper is the
    # belt-and-braces second check for any proposer that does not self-check.
    raw = _offline_proposer(args.strict_llm, classes, llm=llm)
    proposer = held_out_aware(raw)
    verify_fn = (gait_fit_verify_fn(mem) if args.gait_corpus else
                 (gait_search_verify if args.deep_verify else None))
    res = run_factory_night(proposer, config=cfg,
                            manifest_path=args.manifest or (mem / "corpus_factory.json"),
                            memory_dir=mem,
                            bank_fn=(gait_bank_fn if args.gait_corpus else default_bank_fn),
                            verify_fn=verify_fn)
    out = res.to_dict()
    out["proposer"] = {**out.get("proposer", {}), **getattr(raw, "stats", {}), "llm": llm or "off"}
    print(json.dumps(out, indent=2, default=str))
    rows = gait_rows(mem) if args.gait_corpus else None
    hours = max(res.wall_s, 1e-9) / 3600.0
    print(f"\nadmitted {len(res.admitted)} · ANNECS {res.annecs} · rejected {dict(res.rejected)} · "
          f"mean-sim {res.mean_pairwise_similarity} · {res.wall_s}s")
    print(f"proposer {out['proposer']}")
    if rows is not None:
        print(f"gait rows banked this night: {rows} · {rows / hours:.1f} rows/hour · "
              f"{len(res.admitted) / hours:.1f} admits/hour")


def llm_preflight() -> tuple[bool, str]:
    """One cheap call against the configured design backend, so the night learns ONCE whether it has an LLM.

    MEASURED 2026-08-08 on this checkout: with the account's daily request cap exhausted, every design call
    spends ~45 s — four rate-limit retries with 3/6/9 s backoff, then a fallback-model attempt that 429s too —
    and then the composer quietly falls back to the deterministic body it would have built for free. A night
    makes several such calls per proposal. That is the difference between a corpus night that costs minutes and
    one that costs hours, and NOTHING in the output said the LLM was absent: the bodies looked the same.
    """
    try:
        from virturoid.services.llm_client import get_llm
        client = get_llm("morphology")
        if client is None:
            return False, "no backend configured (VIRTUROID_LLM_BACKEND off) -> deterministic composition"
        client.complete_json("You are terse.", 'Return {"ok": true}.',
                             {"type": "object", "properties": {"ok": {"type": "boolean"}}}, max_tokens=32)
        return True, f"{type(client).__name__} reachable -> LLM-designed bodies"
    except Exception as exc:  # noqa: BLE001
        return False, f"backend UNUSABLE ({type(exc).__name__}: {str(exc)[:110]}) -> deterministic composition"


def gait_rows(mem: Path) -> int:
    """Locomotion rows in this night's bank — the number a gait-corpus night exists to move."""
    try:
        import sqlite3
        with sqlite3.connect(mem / "virturoid_memory.db") as conn:
            return int(conn.execute("SELECT COUNT(*) FROM skills WHERE task_type='locomotion'").fetchone()[0])
    except Exception:  # noqa: BLE001
        return -1


if __name__ == "__main__":
    main()
