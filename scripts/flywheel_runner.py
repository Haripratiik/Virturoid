"""Run the self-improving flywheel across several build cycles and show the memory compounding.

Drives a sequence of autonomous builds through ONE shared ``--memory-dir`` so designs/skills banked by
earlier cycles are available to later ones, then prints a per-cycle dashboard + a JSON summary. This is
Virturoid's moat made measurable (startup plan §13 memory, §32 training): each build should make the
next one better, and "compounding": true is the honest verdict that banked assets actually grew.

    PYTHONPATH=src python scripts/flywheel_runner.py \
        --prompt "sort red and blue blocks into bins" \
        --prompt "sort red, blue and green blocks into bins" \
        --prompt "sort blocks by colour into three bins" \
        [--train] [--memory-dir build/flywheel_demo/memory] [--out build/flywheel_demo/runs] [--json]

``--train`` exercises the full skill-flywheel write loop (trains + banks a control policy per cycle); it
is slow. Without it the design/body flywheel still compounds and the run is fast.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", action="append", dest="prompts", default=[],
                    help="A build prompt. Repeat for each cycle (>=2 to show compounding).")
    ap.add_argument("--train", action="store_true", help="Train + bank a control policy each cycle (slow).")
    ap.add_argument("--target", type=float, default=0.8, help="Target task success rate.")
    ap.add_argument("--memory-dir", default="build/flywheel_demo/memory", help="Shared memory dir across cycles.")
    ap.add_argument("--out", default="build/flywheel_demo/runs", help="Root output dir (one subdir per cycle).")
    ap.add_argument("--json", action="store_true", help="Print the full JSON report instead of the dashboard.")
    ap.add_argument("--report", default=None, help="Write the full JSON report to this path (the shareable artifact).")
    args = ap.parse_args()

    prompts = args.prompts or [
        "Build a tabletop robot arm that sorts red and blue blocks into matching bins.",
        "Build a tabletop arm that sorts red, blue and green blocks into three bins.",
        "Build a tabletop sorting arm that places coloured blocks into colour-matched bins.",
    ]

    from virturoid.services.flywheel_runner import run_flywheel

    def progress(ev):
        if not args.json:
            print(f"  [{ev['stage']}] {ev['message']}", flush=True)

    report = run_flywheel(prompts, memory_dir=Path(args.memory_dir), out_root=Path(args.out),
                          train=args.train, target_success_rate=args.target, progress=progress)

    if args.report:
        rp = Path(args.report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not args.json:
            print(f"  wrote report -> {rp}", flush=True)

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["compounding"] else 1

    t = report["totals"]
    print("\n=== Flywheel run ===")
    print(f"  cycles            : {t['cycles']}")
    print(f"  designs banked    : {t['designs_banked']}")
    print(f"  skills banked     : {t['skills_banked']}")
    print(f"  warm-start cycles : {t['warm_start_cycles']}/{t['cycles']} (had prior memory to reuse)")
    print(f"  succeeded         : {t['succeeded_cycles']}/{t['cycles']}")
    print(f"  mean success      : {t['mean_success']:.0%}")
    print(f"  COMPOUNDING       : {report['compounding']}")
    return 0 if report["compounding"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
