"""Run Design-Bench over a cassette and print/save the funnel (master_plan_v6 WS-A, provenance #208).

    PYTHONPATH=src python scripts/design_bench.py [--no-verify] [--fragility] [--out reports/design_bench.json]
    PYTHONPATH=src python scripts/design_bench.py --record            # re-record the cassette OFFLINE
    PYTHONPATH=src python scripts/design_bench.py --live --cassette reports/cassette_live.json --limit 5
    PYTHONPATH=src python scripts/design_bench.py --diff reports/design_bench.json reports/live.json

The default (no flags) replays the committed cassette token-free and runs the real physics verdicts — the
deterministic CI baseline.

MODES, and why the label is no longer a flag
--------------------------------------------
This script used to set ``model = "live_llm_v1" if args.strict_llm else "offline_heuristic_v1"`` — a label
derived from a flag on the CURRENT run, not from the rows being scored. So ``--strict-llm`` WITHOUT
``--record`` printed ``live_llm_v1`` over a replay of the offline fixture. The label now comes from
``DesignCassette.mode()``, which reads each row's ``design_source``, and ``bench_from_cassette`` rejects a
contradicting ``--model``.

  * ``--record``          offline re-record (no tokens): the deterministic builders author every design.
  * ``--live``            record through the PRODUCTION lane (``strict_llm=True`` → the anatomy-graph path),
                          spending DESIGN-PATH tokens. Requires an explicit ``--cassette`` that is not the
                          committed CI fixture, so a live run can never silently move the regression floor.
  * ``--limit N``         record only the first N battery prompts — the affordable live arm. The resulting
                          funnel is marked ``is_subset`` and must be compared with ``--diff``, never by
                          eyeballing two verdict@1 values over different denominators.

TOKEN BOUNDARY: ``--live`` is a developer/design-path action. The customer hot path (``create_robot`` /
``verify_robot``) never imports this script and is unaffected by anything here.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _print_funnel(out: dict, path: str) -> None:
    prov = out.get("provenance") or {}
    print(f"\n=== Design-Bench {out['battery_version']} · model={out['model']} · n={out['n_attempts']} ===")
    print(f"  MODE {out.get('mode')}   authored-by-model {prov.get('n_authored_by_model')}/{prov.get('n_rows')}"
          f"   sources {prov.get('design_sources')}")
    if out.get("infrastructure_warning"):
        print(f"  !! {out['infrastructure_warning']}")
    if out.get("subset_warning"):
        print(f"  !! {out['subset_warning']}")
    print(f"  schema_valid@1 {out['schema_valid@1']}   compile@1 {out['compile@1']}   "
          f"verdict@1 {out['verdict@1']}  (HEADLINE)")
    # verdict@1 is CAPABILITY (did the customer get a robot); correct@1 is HONESTY (did the product do the
    # right thing, which includes declining to build a body whose rest pose topples). Printed together so
    # neither can be quoted for the other -- the offline floor was read as capability for exactly that reason.
    print(f"  correct@1 {out.get('correct@1')}  = verdict@1 + refused@1 {out.get('refused@1')}   "
          f"(refusal = CORRECT, not failure)")
    for pid, why in sorted((out.get("refusals") or {}).items()):
        print(f"     refused  {pid:28s} {str(why)[:96]}")
    print(f"  fitness(ref-norm) {out['fitness']['ref_norm_mean']}   diversity(uniq) "
          f"{out['diversity']['unique_ratio']}   spec-faithful {out['spec_faithfulness']['rate']}")
    print(f"  verdict-fragility {out['verdict_fragility']['rate']}   quality/physics-eval "
          f"{out['quality_per_physics_eval']}")
    print(f"  by family: { {k: v['verdict@1'] for k, v in out['by_family'].items()} }")
    print(f"  by phrasing: { {k: v['verdict@1'] for k, v in out['by_phrasing'].items()} }")
    print(f"  -> {path}   ({out.get('wall_s')}s)")


def _do_diff(a_path: str, b_path: str, out_path: str) -> None:
    from virturoid.services.design_bench import diff_funnels
    a = json.loads(Path(a_path).read_text(encoding="utf-8"))
    b = json.loads(Path(b_path).read_text(encoding="utf-8"))
    d = diff_funnels(a, b)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")
    print(f"\n=== cassette-vs-live diff · {d['baseline_model']} ({d['baseline_mode']}) "
          f"-> {d['candidate_model']} ({d['candidate_mode']}) ===")
    print(f"  shared prompts: {d['n_shared']}   {d['shared_prompt_ids']}")
    print(f"  verdict@1 on the SHARED subset: {d['verdict@1_baseline_shared']} -> "
          f"{d['verdict@1_candidate_shared']}   delta {d['delta_verdict@1_shared']}")
    print(f"  gained: {d['gained']}")
    print(f"  lost:   {d['lost']}")
    # 'lost' is a boolean word for two different things. Split them: a prompt the candidate correctly DECLINED
    # is not a broken body. Neither list is a gate verdict -- two generators are never graded against each
    # other -- but conflating them is exactly how the offline floor came to be read as a capability claim.
    print(f"  of those, candidate DECLINED (correct, unserved): {d.get('declined_by_candidate')}")
    print(f"  of those, candidate BUILT and it does not work:   {d.get('built_but_broken_in_candidate')}")
    print(f"  -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-verify", action="store_true", help="structural funnel only (no physics rollouts)")
    ap.add_argument("--fragility", action="store_true", help="also probe verdict fragility (mild mass DR)")
    ap.add_argument("--record", action="store_true", help="(re)record the cassette before benching")
    ap.add_argument("--live", action="store_true",
                    help="record via the LIVE design model on the production lane (spends design tokens); "
                         "implies --record and requires a non-fixture --cassette")
    ap.add_argument("--limit", type=int, default=None, help="record only the first N battery prompts")
    ap.add_argument("--prompt-ids", default=None,
                    help="comma-separated prompt ids to record (e.g. lynx__appearance,rover__appearance). "
                         "Prefer this over --limit for a small live arm: --limit 5 takes the first five, "
                         "which are all one family, and a one-family sample cannot speak for the battery.")
    ap.add_argument("--cassette", default=None, help="cassette path (default: the committed CI fixture)")
    ap.add_argument("--model", default=None,
                    help="label for this run's row in the per-model matrix; rejected if it contradicts the "
                         "cassette's measured provenance")
    ap.add_argument("--only-recorded", action="store_true",
                    help="score only the prompts this cassette holds (for a subset/live cassette)")
    ap.add_argument("--diff", nargs=2, metavar=("BASELINE_JSON", "CANDIDATE_JSON"),
                    help="compare two saved funnels per case instead of running a bench")
    ap.add_argument("--out", default="reports/design_bench.json")
    args = ap.parse_args()

    if args.diff:
        _do_diff(args.diff[0], args.diff[1], args.out)
        return

    from virturoid.services import design_battery as B
    from virturoid.services.design_bench import bench_from_cassette
    from virturoid.services.design_cassette import DEFAULT_CASSETTE_PATH, DesignCassette, design_from_prompt

    cassette_path = Path(args.cassette) if args.cassette else DEFAULT_CASSETTE_PATH
    if args.live and cassette_path.resolve() == DEFAULT_CASSETTE_PATH.resolve():
        # The committed fixture IS the regression floor. A live re-record over it would move the floor and the
        # baseline gate in the same commit, which is exactly how an unmeasured claim gets normalised.
        raise SystemExit("--live refuses to write the committed CI fixture. Pass --cassette <other path>; "
                         "promote it deliberately once the diff has been reviewed.")

    cas = DesignCassette(cassette_path)
    if args.record or args.live:
        if args.prompt_ids:
            want = [s.strip() for s in args.prompt_ids.split(",") if s.strip()]
            by_id = {B.prompt_id(r): r for r in B.battery()}
            unknown = [w for w in want if w not in by_id]
            if unknown:
                raise SystemExit(f"unknown prompt ids {unknown}; battery has {sorted(by_id)}")
            prompts = [by_id[w] for w in want]
        else:
            prompts = B.battery()[: args.limit] if args.limit else B.battery()
        # An INFRASTRUCTURE failure is not a recorded answer, it is a missing one (#280). ``design_from_prompt``
        # replays any prompt the cassette already holds, so without this a single rate-limit window would
        # poison those rows PERMANENTLY: every later re-record would skip them and keep replaying the outage.
        stale = [B.prompt_id(r) for r in prompts
                 if cas.has(B.prompt_id(r)) and cas.entry_provenance(B.prompt_id(r))["infrastructure_failure"]]
        for pid in stale:
            cas.drop(pid)
        if stale:
            print(f"[record] retrying {len(stale)} row(s) whose last outcome was a transport failure: "
                  f"{', '.join(stale)}")
        ok = fail = 0
        t_rec = time.time()
        for rec in prompts:
            t_one = time.time()
            g, src = design_from_prompt(rec["prompt"], prompt_id=B.prompt_id(rec), cassette=cas,
                                        allow_generate=True, record=True, strict_llm=args.live)
            ok, fail = (ok + 1, fail) if g is not None else (ok, fail + 1)
            print(f"  [{'ok ' if g is not None else 'FAIL'}] {B.prompt_id(rec):28s} source={src} "
                  f"({round(time.time() - t_one, 1)}s)", flush=True)
            # Save after EVERY prompt, not once at the end. A live full-battery record is ~37 minutes of
            # billed completions; saving only on completion means one dropped connection discards all of it.
            # Replay reads whatever is on disk, so a partial cassette is a usable subset, not a corrupt file.
            cas.save(battery_version=B.BATTERY_VERSION)
        print(f"[record] {'LIVE (production lane)' if args.live else 'offline'}: ok={ok} fail={fail} "
              f"n={len(prompts)} wall={round(time.time() - t_rec, 1)}s -> {cassette_path}")
        print(f"[record] measured provenance: {cas.provenance()}")

    t0 = time.time()
    out = bench_from_cassette(cassette=cas, verify=not args.no_verify, fragility=args.fragility,
                              model=args.model,
                              only_recorded=args.only_recorded or bool(args.limit) or bool(args.prompt_ids))
    out["wall_s"] = round(time.time() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    _print_funnel(out, args.out)


if __name__ == "__main__":
    main()
