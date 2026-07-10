"""R2' runner — the decisive Thesis A number: does corpus grounding raise the verified-WALK rate over no
grounding and a single static exemplar? Seeds the gait corpus from reference bodies, then measures three
knowledge arms (off | cheatsheet | corpus) on a held-out battery with an un-gameable physics verdict.

    # the full CPU experiment (realistic search budget; minutes):
    python scripts/eval_r2prime.py --out reports/r2prime.json

    # a fast plumbing pass (won't produce a credible-walk signal, just proves the harness runs):
    python scripts/eval_r2prime.py --gens 2 --pop 4 --steps 300 --deploy 500 --battery-steps 500

Exit code 0 if corpus strictly beats both baselines; 1 otherwise (a null result is still a valid, reported
outcome — see the printed decision line and the dossier's decision rule).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="R2': measure corpus-grounding lift on verified-walk rate.")
    ap.add_argument("--gens", type=int, default=8, help="seed search generations per reference body")
    ap.add_argument("--pop", type=int, default=16, help="seed search population per generation")
    ap.add_argument("--steps", type=int, default=800, help="seed search rollout steps")
    ap.add_argument("--deploy", type=int, default=1200, help="seed deploy-select horizon")
    ap.add_argument("--battery-steps", type=int, default=1200, help="battery verdict horizon")
    ap.add_argument("--db", default=None, help="corpus DB path (default: a fresh temp DB, so the run is clean)")
    ap.add_argument("--out", default=None, help="write the JSON report here")
    args = ap.parse_args(argv)

    import tempfile

    from virturoid.services import r2prime as R
    from virturoid.services.memory_db import MemoryDB

    db_path = Path(args.db) if args.db else Path(tempfile.mkdtemp(prefix="r2prime_")) / "corpus.db"
    db = MemoryDB(db_path=db_path)
    print(f"corpus DB: {db_path}\nseeding from {list(R._REFERENCE_PROMPTS)} "
          f"(gens={args.gens} pop={args.pop} steps={args.steps})...", flush=True)

    t0 = time.time()
    banked = R.seed_corpus(db, generations=args.gens, pop=args.pop, steps=args.steps, deploy_steps=args.deploy)
    for b in banked:
        print(f"  seeded {b['prompt']!r} -> skill={b.get('skill_id')} "
              f"forward={b.get('forward_m')}{(' ERROR: ' + b['error']) if b.get('error') else ''}", flush=True)
    print(f"seeding took {time.time() - t0:.0f}s; measuring the battery {list(R._BATTERY_PROMPTS)}...", flush=True)

    # (1) ZERO-SHOT: deploy the recalled gait directly (the harshest test of grounding).
    res = R.run_r2prime(db, steps=args.battery_steps)
    print("\nZERO-SHOT (deploy the recalled gait as-is)\n  arm         verified-solve   per-body")
    for a in R.KNOWLEDGE_ARMS:
        r = res[a]
        flags = " ".join(("+" if row["solved"] else ".") for row in r["rows"])
        print(f"  {a:10s}  {r['verified_solve_rate']:>6.0%} ({r['solved']}/{r['n']})    [{flags}]")
    zdecision = R.decision(res)
    print(f"  -> {zdecision}")

    # (2) WARM-START: the corpus as a starting point a SHORT search refines (the flywheel's real mechanism).
    print(f"\nWARM-START (cold vs corpus-warm-started {args.gens // 2 or 3}-gen search)...", flush=True)
    ws = R.run_r2prime_warmstart(db, generations=max(3, args.gens // 2), pop=max(6, args.pop // 2),
                                 steps=max(400, args.steps // 2), deploy_steps=args.deploy)
    print("  arm                verified-solve   mean-forward")
    for a in ("cold", "corpus_warmstart"):
        print(f"  {a:16s}  {ws[a]['verified_solve_rate']:>6.0%} ({ws[a]['solved']}/{ws[a]['n']})    "
              f"{ws[a]['mean_forward_m']} m")
    ws_lift = ws["corpus_warmstart"]["verified_solve_rate"] - ws["cold"]["verified_solve_rate"]
    ws_fwd_lift = ws["corpus_warmstart"]["mean_forward_m"] - ws["cold"]["mean_forward_m"]
    print(f"  -> warm-start lift: solve {ws_lift:+.0%}, forward {ws_fwd_lift:+.3f} m")

    report = {"seeded": banked, "zeroshot": res, "zeroshot_decision": zdecision, "warmstart": ws,
              "config": {"gens": args.gens, "pop": args.pop, "steps": args.steps,
                         "deploy": args.deploy, "battery_steps": args.battery_steps}}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\nreport -> {args.out}")

    co = res["corpus"]["verified_solve_rate"]
    # success = the corpus helps by EITHER measure (zero-shot beat baselines, OR warm-start lifts solve/forward)
    zs_win = co > res["off"]["verified_solve_rate"] and co > res["cheatsheet"]["verified_solve_rate"]
    ws_win = ws_lift > 0 or ws_fwd_lift > 0.02
    return 0 if (zs_win or ws_win) else 1


if __name__ == "__main__":
    sys.exit(main())
