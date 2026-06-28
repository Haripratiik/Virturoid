"""Re-bake the demo package set with the FULL honest reports (YC demo plan, item 1).

The build/ui_verify packages the demo shows were built BEFORE the honesty-report wiring landed, so
reports/spec_sheet.json, reports/bom_sim_fidelity.json, reports/spec_compliance.json and
reports/honesty_scorecard.json were all missing -- which hid the Spec Sheet card, the task-success number,
the "Build it for real" deployment button, and the honesty wedge (the gate catching the product's own limits).

This rebuilds the demo IDs through the gene path (build_robot_package_from_prompt), which now writes ALL of
those reports consistently, so the demo's claims are visible in the exact tabs it opens. Deterministic offline
(VIRTUROID_LLM_BACKEND=off) so it is reproducible. Run:  python scripts/rebake_demo_set.py [--only arm_sort ...]
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("VIRTUROID_LLM_BACKEND", "off")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virturoid.services.autonomous_builder import build_robot_package_from_prompt  # noqa: E402

# id (the autodemo references these) -> the prompt that regenerates it
DEMO_SET = {
    "arm_sort": "a tabletop robot arm that sorts red and blue blocks into matching bins",
    "build_a_quadruped_robot_that_walks": "build a quadruped robot that walks",
    "build_a_dog_robot_that_walks": "build a dog robot that walks",
    "hexapod_walk": "build a six-legged walking robot",
    "build_a_mobile_base_that_delivers_parts_indoors": "build a mobile base that delivers parts indoors",
    "humanoid": "a humanoid robot",
}
# the reports whose presence the demo depends on (the scorecard is the wedge)
WANT = ["reports/spec_sheet.json", "reports/bom_sim_fidelity.json", "reports/honesty_scorecard.json",
        "reports/product_readiness_ledger.json", "robot/bill_of_materials.json"]


def _ensure_honest_reports(out: Path, prompt: str) -> None:
    """Make every package -- whether built on the gene path or the legacy TEMPLATE path (arms) -- carry the spec
    sheet, the BOM<->sim fidelity report, and the unified honesty scorecard. The template path doesn't write
    these, so we add them post-build (spec sheet reads on-disk BOM/genome; fidelity needs a gene, recomposed
    deterministically from the prompt; the scorecard aggregates whatever honest reports are present)."""
    try:
        from virturoid.services.gene_build import _maybe_write_summaries
        _maybe_write_summaries(out)                      # spec_sheet + deployment_guide (from on-disk BOM/genome)
    except Exception as exc:  # noqa: BLE001
        print(f"  (summaries) {exc}")
    if not (out / "reports" / "bom_sim_fidelity.json").exists():
        try:
            from virturoid.services.fidelity_report import bom_sim_fidelity
            from virturoid.services.morphology_composer import compose_robot
            gene = compose_robot(prompt, llm=None)
            (out / "reports").mkdir(parents=True, exist_ok=True)
            (out / "reports" / "bom_sim_fidelity.json").write_text(
                json.dumps(bom_sim_fidelity(gene), indent=2, default=str), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            print(f"  (fidelity) {exc}")
    try:
        from virturoid.services.honesty_scorecard import scorecard_from_package
        (out / "reports").mkdir(parents=True, exist_ok=True)
        (out / "reports" / "honesty_scorecard.json").write_text(
            json.dumps(scorecard_from_package(out), indent=2, default=str), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"  (scorecard) {exc}")


def rebake(root: str = "build/ui_verify", only=None) -> int:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    failures = 0
    for pid, prompt in DEMO_SET.items():
        if only and pid not in only:
            continue
        out = root / pid
        print(f"[rebake] {pid}: {prompt!r}")
        try:
            build_robot_package_from_prompt(prompt, output_dir=out)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED to build: {exc}")
            failures += 1
            continue
        _ensure_honest_reports(out, prompt)
        missing = [w for w in WANT if not (out / w).exists()]
        sc_path = out / "reports" / "honesty_scorecard.json"
        head = "(no scorecard)"
        if sc_path.exists():
            try:
                head = json.loads(sc_path.read_text(encoding="utf-8")).get("headline", "?")
            except (OSError, json.JSONDecodeError):
                head = "(unreadable scorecard)"
        status = "ALL reports present" if not missing else "MISSING: " + ", ".join(missing)
        print(f"  {status}  |  scorecard: {head}")
        if missing:
            failures += 1
    print(f"\n[rebake] done; {failures} package(s) with problems.")
    return failures


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Re-bake the demo package set with the full honest reports.")
    ap.add_argument("--root", default="build/ui_verify")
    ap.add_argument("--only", nargs="*", help="rebake only these package ids")
    args = ap.parse_args()
    raise SystemExit(1 if rebake(args.root, set(args.only) if args.only else None) else 0)
