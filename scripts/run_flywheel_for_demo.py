"""Run a short flywheel and write the moat's COMPOUNDING CURVE artifact (YC demo plan item 6).

Builds several robots against ONE shared memory and measures whether banked assets + warm-start COMPOUND across
cycles -- the moat made measurable (growth-rate beats a static species tree). Writes
build/ui_verify/flywheel_compounding.json (the per-cycle series + the COMPOUNDING verdict) for the Memory panel,
and prints the dependency-free ASCII chart. Deterministic offline. Run:  python scripts/run_flywheel_for_demo.py
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("VIRTUROID_LLM_BACKEND", "off")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virturoid.services.flywheel_chart import flywheel_compounding_chart, render_ascii_chart  # noqa: E402
from virturoid.services.flywheel_runner import run_flywheel  # noqa: E402
from virturoid.services.install_paths import anchored

# similar bodies so later cycles warm-start from earlier ones (that is what makes the curve compound)
DEMO_PROMPTS = [
    "build a quadruped robot that walks",
    "build a sturdy quadruped robot that walks",
    "build a fast four-legged walking robot",
]


def main(out_root: str = None, n: int = 3) -> int:
    out_root = out_root if out_root is not None else anchored("build/ui_verify")  # tracked demo set; anchored
    prompts = DEMO_PROMPTS[: max(2, n)]
    with tempfile.TemporaryDirectory() as mem, tempfile.TemporaryDirectory() as builds:
        report = run_flywheel(prompts, memory_dir=Path(mem), out_root=Path(builds))
    chart = flywheel_compounding_chart(report)
    print(render_ascii_chart(chart))
    out = Path(out_root) / "flywheel_compounding.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(chart, indent=2), encoding="utf-8")
    print(f"-> {out}  ({'COMPOUNDING' if chart['compounding'] else 'flat'})")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run a short flywheel and write the compounding-curve artifact.")
    ap.add_argument("--out-root", default="build/ui_verify")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    raise SystemExit(main(args.out_root, args.n))
