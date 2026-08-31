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
from virturoid.services.install_paths import anchored

# id (the autodemo references these) -> the prompt that regenerates it
DEMO_SET = {
    "arm_sort": "a tabletop robot arm that sorts red and blue blocks into matching bins",
    "build_a_quadruped_robot_that_walks": "build a quadruped robot that walks",
    "dog_walk": "build a dog robot that walks",
    "hexapod_walk": "build a six-legged walking robot",
    "build_a_mobile_base_that_delivers_parts_indoors": "build a mobile base that delivers parts indoors",
    "humanoid": "a humanoid robot",
}
# the reports whose presence the demo depends on (the scorecard is the wedge)
WANT = ["reports/spec_sheet.json", "reports/bom_sim_fidelity.json", "reports/honesty_scorecard.json",
        "reports/product_readiness_ledger.json", "robot/bill_of_materials.json"]


def _maybe_write_sim2sim(out: Path, prompt: str) -> None:
    """For WALKERS, write reports/sim2sim_report.json -- how robustly the banked gait survives DOMAIN
    RANDOMIZATION (randomized actuator gain / mass / damping / friction). DR-robustness is the recognized
    sim2real-trust signal for locomotion, so this is the honest BEAT-4 number; it also feeds the scorecard's
    sim2sim row. Best-effort (skips silently if there's no exactly-matching banked policy)."""
    if not any(w in prompt.lower() for w in ("walk", "quadruped", "dog", "legged", "hexapod", "six-leg")):
        return
    try:
        from virturoid.services.morph_graph import encode_robot
        from virturoid.services.morph_policy import compiled_model, recipe_robustness, robot_mjcf
        from virturoid.services.morphology_composer import compose_robot
        from virturoid.services.viewer_sim import _load_locomotion_policy
        gene = compose_robot(prompt, llm=None)
        fd = encode_robot(compiled_model(robot_mjcf(gene))).feature_dim
        pol = _load_locomotion_policy(out, fd, "models")
        if pol is None:
            return                                            # no matching banked gait -> nothing honest to report
        rob = recipe_robustness(gene, pol, n=8)
        surv = float(rob.get("survival_rate", 0.0))
        grade = "STRONG" if surv >= 0.7 else "MODERATE" if surv >= 0.4 else "WEAK"
        rep = {"verdict": f"transfer {grade}: banked gait survives {int(surv * 100)}% of randomized-dynamics worlds",
               "survival_rate": surv, "mean_forward_m": rob.get("mean_forward"),
               "min_forward_m": rob.get("min_forward"), "n_trials": rob.get("n"),
               "method": "recipe_robustness: the banked policy under randomized actuator gain / mass / damping / friction"}
        (out / "reports").mkdir(parents=True, exist_ok=True)
        (out / "reports" / "sim2sim_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"  sim2sim: {rep['verdict']}")
    except Exception as exc:  # noqa: BLE001
        print(f"  (sim2sim) {exc}")


def _maybe_write_grasp_dataset(out: Path, prompt: str) -> None:
    """For MANIPULATORS, write reports/grasp_dataset_summary.json -- the MimicGen data engine (1 scripted grasp
    -> N verified demos via randomization + rejection sampling), so the demo's data-bottleneck-escape claim is a
    concrete artifact. Best-effort."""
    if not any(w in prompt.lower() for w in ("arm", "grasp", "sort", "pick", "manipulat", "gripper", "block")):
        return
    try:
        from virturoid.services.data_factory import generate_grasp_demos
        from virturoid.services.morphology_composer import compose_robot
        gene = compose_robot("grasp and lift a box on a table", llm=None)
        ds = generate_grasp_demos(gene, n=32)
        summary = {k: v for k, v in ds.items() if k != "demos"}
        summary["headline"] = (f"{ds['augmentation_x']} verified demos from 1 scripted grasp "
                               f"({int(ds['yield'] * 100)}% yield) - the in-sim data engine")
        (out / "reports").mkdir(parents=True, exist_ok=True)
        (out / "reports" / "grasp_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"  data-factory: {summary['headline']}")
    except Exception as exc:  # noqa: BLE001
        print(f"  (data-factory) {exc}")


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
    _maybe_write_sim2sim(out, prompt)                        # sim2sim BEFORE the scorecard so the row is included
    _maybe_write_grasp_dataset(out, prompt)                  # data-engine artifact for manipulators
    try:
        from virturoid.services.honesty_scorecard import scorecard_from_package
        (out / "reports").mkdir(parents=True, exist_ok=True)
        (out / "reports" / "honesty_scorecard.json").write_text(
            json.dumps(scorecard_from_package(out), indent=2, default=str), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"  (scorecard) {exc}")


def rebake(root: str = None, only=None) -> int:
    # ANCHORED: this REGENERATES the git-tracked demo set (incident 3's own destination). A CWD-relative
    # default meant running it from the wrong directory silently rebaked a stray tree while `git status`
    # stayed clean -- which reads as "already up to date", the exact wrong conclusion.
    root = root if root is not None else anchored("build/ui_verify")
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
