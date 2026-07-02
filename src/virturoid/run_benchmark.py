"""VIRT-Bench reproducibility capsule (plan v3 M4) — one command that makes "beats Claude+MCP" externally
checkable.

    python -m virturoid.run_benchmark --split held_out --seeds 20260701 20260702 20260703
    python -m virturoid.run_benchmark --with-baseline           # add Arm 0 (the literal Claude+MCP control)

It (1) FREEZES a pre-registration manifest (task registry + verifier + arms SHA256, seed table, arm defs,
predictions) with a deterministic ``manifest_hash`` -- re-running before any code edit reproduces the hash, so a
non-matching later run is NOT a replication; (2) runs the multi-seed 3-arm (or 4-arm with Arm 0) head-to-head,
each scored by the SAME independent verifier at the frozen horizon; (3) writes a Markdown + JSON report with the
per-task table, the deltas (harness / transfer / baseline) as mean±ci95 (IQM), the per-arm honesty (claimed vs
verified), and the grader SHAs. The report closes with the exact ``git tag`` command to stamp the run (we print
it rather than tagging automatically -- tagging is an explicit, user-owned action).

The report generator ``build_report`` is a pure function over the head-to-head result, so it unit-tests without
running physics; the manifest is deterministic, so its reproducibility unit-tests directly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _pct(n, d):
    return f"{(100.0 * n / d):.0f}%" if d else "n/a"


def _delta_line(name, agg):
    a = agg.get(name) or {}
    return f"- **{name}** = {a.get('mean', 0):+.2f} ± {a.get('ci95', 0):.2f} (IQM {a.get('iqm', 0):+.2f}, n={a.get('n', 0)} seeds)"


def build_report(h2h: dict, prereg: dict, *, with_baseline: bool = False) -> tuple[str, dict]:
    """Turn a ``run_head_to_head_multiseed`` result + a prereg manifest into (markdown, report_dict). Pure — no
    physics, no clock — so it unit-tests. Aggregates per-task solve rates across seeds and reports the deltas
    with uncertainty and the per-arm honesty (over-claim)."""
    seeds = h2h.get("seeds", [])
    per_seed = h2h.get("per_seed", [])
    agg = h2h.get("aggregate", {})
    n_seeds = max(1, len(per_seed))

    # per-task solve fraction across seeds, per arm
    arms = (["A0"] if with_baseline else []) + ["A", "Aplus", "B"]
    task_ids = [r["task"] for r in (per_seed[0]["rows"] if per_seed else [])]
    per_task = {}
    for tid in task_ids:
        row = {"task": tid}
        for arm in arms:
            solved = sum(1 for run in per_seed for r in run["rows"] if r["task"] == tid and r.get(f"{arm}_pass"))
            row[arm] = solved
        per_task[tid] = row

    # honesty: total claimed vs verified per arm across all seeds
    honesty = {}
    for arm in arms:
        claimed = sum(run["honesty"].get(_arm_key(arm), {}).get("claimed", 0) for run in per_seed)
        verified = sum(run["honesty"].get(_arm_key(arm), {}).get("verified", 0) for run in per_seed)
        honesty[arm] = {"claimed": claimed, "verified": verified, "overclaim": claimed - verified}

    report = {
        "benchmark": "VIRT-Bench",
        "manifest_hash": prereg.get("manifest_hash"),
        "verifier_sha256": prereg.get("verifier_sha256"),
        "arms_sha256": prereg.get("arms_sha256"),
        "task_registry_sha256": prereg.get("task_registry_sha256"),
        "seeds": seeds,
        "aggregate": agg,
        "per_task": per_task,
        "honesty": honesty,
        "with_baseline": with_baseline,
    }

    lines = ["# VIRT-Bench head-to-head report", ""]
    lines.append(f"Seeds: {seeds}  ·  manifest `{report['manifest_hash']}`")
    lines.append(f"Grader SHAs — verifier `{(report['verifier_sha256'] or '')[:12]}`, "
                 f"arms `{(report['arms_sha256'] or '')[:12]}` (the grader is code, and it is versioned).")
    lines.append("")
    lines.append("## Solved (fraction of seeds each arm passed the task)")
    header = "| task | " + " | ".join(_arm_label(a) for a in arms) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(arms) + 1))
    for tid in task_ids:
        cells = [_pct(per_task[tid][a], n_seeds) for a in arms]
        lines.append(f"| {tid} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Measured deltas (mean ± 95% CI, IQM)")
    lines.append(_delta_line("harness_delta", agg) + "  — value of the agentic search over a stock build")
    lines.append(_delta_line("transfer_delta", agg) + "  — value of the flywheel memory (the moat)")
    if with_baseline:
        lines.append("- **baseline_delta** = B_solved − A0_solved  — the headline *beats Claude+MCP* number "
                     "(per-seed; see aggregate).")
    lines.append("")
    lines.append("## Honesty (claimed vs verified — over-claim is the deploy/hallucination gap)")
    for a in arms:
        h = honesty[a]
        lines.append(f"- {_arm_label(a)}: claimed {h['claimed']}, verified {h['verified']}, "
                     f"over-claim {h['overclaim']:+d}")
    lines.append("")
    lines.append("## Reproduce / stamp")
    lines.append("- A second run before any code edit reproduces `manifest_hash` (the replication check).")
    lines.append("- Verify-the-verifier: human-spot-check ~10% of verdicts (SWE-bench-Verified discipline).")
    lines.append(f"- Stamp this run:  `git tag virtbench-{report['manifest_hash'][:8]} && "
                 "git push --tags`  (explicit, user-owned).")
    return "\n".join(lines) + "\n", report


def _arm_key(arm):
    return {"Aplus": "A+"}.get(arm, arm)


def _arm_label(arm):
    return {"A0": "A0 (Claude+MCP)", "A": "A (fixed)", "Aplus": "A+ (search)", "B": "B (search+memory)"}.get(arm, arm)


def run_benchmark(*, split: str = "held_out", seeds=(20260701, 20260702, 20260703), with_baseline: bool = False,
                  baseline_llm=None, use_gpu: bool = False, max_evals: int = 12, steps: int = 600,
                  families=("locomotion", "manipulation"), out_dir: str = "build/benchmark") -> dict:
    """Run the multi-seed head-to-head + write the prereg manifest, the Markdown report, and the JSON report to
    ``out_dir``. Returns the report dict (includes ``manifest_hash``). CPU by default (``use_gpu`` opts into the
    GPU rung)."""
    from virturoid.services.prereg import build_prereg_manifest
    from virturoid.services.virt_bench_arms import run_head_to_head_multiseed
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arms = ("A0", "A", "A+", "B") if with_baseline else ("A", "A+", "B")
    prereg = build_prereg_manifest(arms=arms, seeds=list(seeds), split=split)
    (out / "prereg.json").write_text(json.dumps(prereg, indent=2, sort_keys=True), encoding="utf-8")

    h2h = run_head_to_head_multiseed(seeds=tuple(seeds), split=split, steps=steps, max_evals=max_evals,
                                     families=families, use_gpu=use_gpu, with_baseline=with_baseline,
                                     baseline_llm=baseline_llm)
    md, report = build_report(h2h, prereg, with_baseline=with_baseline)
    (out / "report.md").write_text(md, encoding="utf-8")
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["_paths"] = {"report_md": str(out / "report.md"), "report_json": str(out / "report.json"),
                        "prereg": str(out / "prereg.json")}
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="VIRT-Bench reproducibility capsule (plan v3 M4)")
    ap.add_argument("--split", default="held_out", choices=["dev", "held_out", "all"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[20260701, 20260702, 20260703])
    ap.add_argument("--with-baseline", action="store_true", help="add Arm 0, the literal Claude+MCP control")
    ap.add_argument("--use-gpu", action="store_true")
    ap.add_argument("--max-evals", type=int, default=12)
    ap.add_argument("--families", nargs="+", default=["locomotion", "manipulation"])
    ap.add_argument("--out", default="build/benchmark")
    args = ap.parse_args(argv)
    split = None if args.split == "all" else args.split
    report = run_benchmark(split=split, seeds=tuple(args.seeds), with_baseline=args.with_baseline,
                           use_gpu=args.use_gpu, max_evals=args.max_evals, families=tuple(args.families),
                           out_dir=args.out)
    print(Path(report["_paths"]["report_md"]).read_text(encoding="utf-8"))
    print(f"manifest_hash={report['manifest_hash']}")
    print(f"report -> {report['_paths']['report_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
