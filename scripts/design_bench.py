"""Run Design-Bench over the committed cassette and print/save the funnel (master_plan_v6 WS-A).

    PYTHONPATH=src python scripts/design_bench.py [--no-verify] [--fragility] [--out reports/design_bench.json]
    PYTHONPATH=src python scripts/design_bench.py --record            # re-record the cassette offline first
    PYTHONPATH=src python scripts/design_bench.py --record --strict-llm  # re-record with the live design model

The default (no flags) replays the committed cassette token-free and runs the real physics verdicts — the
deterministic CI baseline. ``--record`` regenerates the cassette (offline heuristic, or the live model under
``--strict-llm``) and catches model drift; that path spends design tokens only when ``--strict-llm`` is set.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-verify", action="store_true", help="structural funnel only (no physics rollouts)")
    ap.add_argument("--fragility", action="store_true", help="also probe verdict fragility (mild mass DR)")
    ap.add_argument("--record", action="store_true", help="(re)record the cassette before benching")
    ap.add_argument("--strict-llm", action="store_true", help="record via the live design model (spends tokens)")
    ap.add_argument("--model", default=None, help="label for this run's row in the per-model matrix")
    ap.add_argument("--out", default="reports/design_bench.json")
    args = ap.parse_args()

    from virturoid.services import design_battery as B
    from virturoid.services.design_bench import bench_from_cassette
    from virturoid.services.design_cassette import DesignCassette, design_from_prompt

    cas = DesignCassette()
    if args.record:
        ok = fail = 0
        for rec in B.battery():
            g, _ = design_from_prompt(rec["prompt"], prompt_id=B.prompt_id(rec), cassette=cas,
                                      allow_generate=True, record=True, strict_llm=args.strict_llm)
            ok, fail = (ok + 1, fail) if g is not None else (ok, fail + 1)
        cas.save(battery_version=B.BATTERY_VERSION)
        print(f"[record] cassette {'(strict-llm)' if args.strict_llm else '(offline)'}: ok={ok} fail={fail}")

    model = args.model or ("live_llm_v1" if args.strict_llm else "offline_heuristic_v1")
    t0 = time.time()
    out = bench_from_cassette(cassette=cas, verify=not args.no_verify, fragility=args.fragility, model=model)
    out["wall_s"] = round(time.time() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    print(f"\n=== Design-Bench {out['battery_version']} · model={model} · n={out['n_attempts']} ===")
    print(f"  schema_valid@1 {out['schema_valid@1']}   compile@1 {out['compile@1']}   "
          f"verdict@1 {out['verdict@1']}  (HEADLINE)")
    print(f"  fitness(ref-norm) {out['fitness']['ref_norm_mean']}   diversity(uniq) "
          f"{out['diversity']['unique_ratio']}   spec-faithful {out['spec_faithfulness']['rate']}")
    print(f"  verdict-fragility {out['verdict_fragility']['rate']}   quality/physics-eval "
          f"{out['quality_per_physics_eval']}")
    print(f"  by family: { {k: v['verdict@1'] for k, v in out['by_family'].items()} }")
    print(f"  by phrasing: { {k: v['verdict@1'] for k, v in out['by_phrasing'].items()} }")
    print(f"  -> {args.out}   ({out['wall_s']}s)")


if __name__ == "__main__":
    main()
