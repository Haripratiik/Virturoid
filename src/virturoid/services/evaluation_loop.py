"""The product's core loop, on REAL physics (startup plan §26, §11.2, §31.5, Milestone E):

    Task -> Scene tiers -> Run -> cluster Failures -> failure-conditioned Regression -> re-Evaluate -> Report+Gate

This runs the scripted pick-place controller under real MuJoCo across NAMED evaluation tiers (smoke / baseline /
randomized / stress), surfaces and CLUSTERS the real failures, generates FAILURE-CONDITIONED regression scenes
(one-variable counterfactuals around each failing config — reusing scene_generator._condition_scene_on_failure),
re-evaluates them to CONFIRM the diagnosed cause, and returns a structured report with an export-readiness gate.

The stress tier deliberately pushes objects past the reliable envelope (reach edge + heavy + clutter) so the loop
has real failures to chew on — otherwise a strong scripted controller passes every easy scene and the failure
loop is vacuous. The robust per-arm reach controller is unchanged; only the SCENE difficulty is escalated.
"""

from __future__ import annotations

import collections
import math

from virturoid.schemas.scenes import SceneGraph
from virturoid.services.scene_generator import _condition_scene_on_failure
from virturoid.services.task_runtime import evaluate_gene_on_task, generate_task_scenes, select_task_spec

# Named evaluation tiers (plan §11.2). count = scenes per tier. 'stress' is escalated below; it is useful for
# ANALYSIS and is NOT required to pass for export (the export gate is baseline+randomized).
_TIERS = {"smoke": 2, "baseline": 4, "randomized": 6, "stress": 6}
_BASE_PARAMS = {"kp": 14.0, "kd": 2.0, "phase_steps": 500, "engage_radius": 0.15, "grasp_z": 0.05}


def _stress_scenes(gene, spec, count: int, *, seed: int) -> list[SceneGraph]:
    """STRESS tier: take robot-matched graspable scenes and push them PAST the reliable envelope — blocks to
    ~0.80 of max reach (beyond the ~0.72 reliable-grasp fraction) and heavy (0.35 kg) — so the controller
    genuinely fails (missed_grasp at the reach edge, dropped on heavy). This is what gives the failure loop
    real material to analyze (plan §11.2 stress eval / §10.3 intentional negatives)."""
    reach = sum(s.length_m for s in gene.actuated_joints()) + 0.05
    scenes = generate_task_scenes(gene, spec, count, seed=seed + 1000)
    out: list[SceneGraph] = []
    for i, sc in enumerate(scenes):
        for o in sc.objects:
            if o.object_type == "cube":
                x, y, z, *rest = o.pose_xyz_rpy
                r = math.hypot(x, y) or 1.0
                s = (0.80 * reach) / r
                o.pose_xyz_rpy = (round(x * s, 4), round(y * s, 4), z, *rest)
                o.mass_kg = 0.35
        sc.id = f"stress_{gene.id}_{i}"
        sc.variation_parameters = {**sc.variation_parameters, "tier": "stress", "reach_frac": 0.80, "mass_kg": 0.35}
        out.append(sc)
    return out


def _tier_scenes(gene, spec, tier: str, count: int, seed: int) -> list[SceneGraph]:
    if tier == "stress":
        return _stress_scenes(gene, spec, count, seed=seed)
    # baseline = fixed seed (deterministic comparisons); randomized/smoke = varied seed (robustness)
    tier_seed = seed if tier == "baseline" else seed + {"smoke": 3, "randomized": 7}.get(tier, 0)
    return generate_task_scenes(gene, spec, count, seed=tier_seed)


def run_pickplace_failure_loop(gene, *, prompt: str = "sort blocks into bins", seed: int = 0,
                               params: dict | None = None, export_threshold: float = 0.8,
                               tier_counts: dict | None = None) -> dict:
    """Run the full real-physics failure loop for a manipulator ``gene`` and return a structured report.

    Returns ``{tiers, failure_clusters, regression, export_ready, export_blockers, notes}`` where:
      - ``tiers``: {tier: {success_rate, episodes, failures}} from real MuJoCo runs.
      - ``failure_clusters``: [{failure_label, count, example_scene_ids, suggested_regression}] over ALL tiers.
      - ``regression``: failure-CONDITIONED counterfactual scenes re-run; ``cause_confirmed_rate`` = fraction of
        failures whose targeted one-variable change RESOLVED the failure (confirming the diagnosed cause).
      - ``export_ready`` / ``export_blockers``: the §15.3 gate — baseline & randomized must clear ``export_threshold``
        (stress is analysis-only, not a gate)."""
    params = {**_BASE_PARAMS, **(params or {})}
    spec = select_task_spec(prompt)

    tiers: dict[str, dict] = {}
    scene_by_id: dict[str, SceneGraph] = {}
    failing: list[dict] = []                       # episodes that failed, across tiers
    for tier, count in {**_TIERS, **(tier_counts or {})}.items():
        scenes = _tier_scenes(gene, spec, tier, count, seed)
        for sc in scenes:
            scene_by_id[sc.id] = sc
        r = evaluate_gene_on_task(gene, spec, scenes, params=params)
        # aggregate the per-episode manipulation metrics for this tier (plan §11.1)
        eps = r["episodes"]
        def _avg(key):
            vals = [e["metrics"][key] for e in eps if e.get("metrics", {}).get(key) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None
        tiers[tier] = {"success_rate": r["success_rate"], "episodes": len(eps),
                       "failures": sum(1 for e in eps if e["status"] != "success"),
                       "metrics": {"grasp_rate": _avg("grasp_rate"), "place_rate": _avg("place_rate"),
                                   "mean_ee_path_m": _avg("ee_path_len_m"),
                                   "drops": sum(e.get("metrics", {}).get("drop_count", 0) for e in eps)}}
        for e in eps:
            if e["status"] != "success":
                failing.append({"tier": tier, **e})

    # cluster failures by label (plan §11/§31.5)
    counts = collections.Counter(e["failure_label"] or "unknown" for e in failing)
    examples: dict[str, list[str]] = collections.defaultdict(list)
    for e in failing:
        examples[e["failure_label"] or "unknown"].append(e["scene_id"])
    failure_clusters = [{"failure_label": label, "count": n, "example_scene_ids": examples[label][:3],
                         "suggested_regression": _SUGGESTED.get(label, "regenerate harder variants around this failure")}
                        for label, n in counts.most_common()]

    # failure-CONDITIONED regression: rebuild each failing scene as a one-variable counterfactual + re-run it.
    # If the targeted change RESOLVES the failure, the diagnosed correlated variable is CONFIRMED as the cause.
    regression_results = []
    confirmed = 0
    for e in failing:
        src = scene_by_id.get(e["scene_id"])
        if src is None:
            continue
        objs, applied = _condition_scene_on_failure(src.objects, e["failure_label"])
        reg = SceneGraph(id=f"reg_{e['scene_id']}", name=f"regression {e['failure_label']}",
                         backend_targets=["mujoco"], robot_spawn_xyz_rpy=(0, 0, 0, 0, 0, 0), objects=objs,
                         variation_parameters={"failure_label": e["failure_label"], **applied})
        reg_params = {**params, **{k: v for k, v in (applied.get("controller_adjustment") or {}).items()}}
        # map our generic controller hints onto the controller's param names
        if "transport_speed" in reg_params:
            reg_params["phase_steps"] = int(params["phase_steps"] / max(0.3, reg_params.pop("transport_speed")))
        rr = evaluate_gene_on_task(gene, spec, [reg], params=reg_params)
        resolved = rr["success_rate"] >= 0.5
        confirmed += int(resolved)
        regression_results.append({"failure_label": e["failure_label"], "counterfactual": applied["counterfactual"],
                                   "resolved": resolved})

    n_reg = len(regression_results)
    regression = {"n": n_reg, "cause_confirmed": confirmed,
                  "cause_confirmed_rate": round(confirmed / n_reg, 3) if n_reg else 1.0,
                  "results": regression_results}

    # export gate (plan §15.3): baseline + randomized must clear the threshold (stress is analysis-only).
    blockers = [t for t in ("baseline", "randomized") if tiers[t]["success_rate"] < export_threshold]
    report = {
        "robot": gene.id, "task": prompt, "controller": "scripted_pick_place(auto-reach)",
        "tiers": tiers, "failure_clusters": failure_clusters, "regression": regression,
        "export_ready": not blockers,
        "export_blockers": [f"{t} success {tiers[t]['success_rate']} < {export_threshold}" for t in blockers],
        "notes": [
            "Real MuJoCo pick-place across named tiers (smoke/baseline/randomized/stress).",
            "Stress tier pushes objects past the reliable envelope so real failures surface for analysis.",
            "Regression scenes are failure-conditioned counterfactuals (one variable changed) that confirm the cause.",
        ],
    }
    return report


def gated_evaluation_run(gene, *, prompt: str = "sort blocks into bins", seed: int = 0,
                         params: dict | None = None, export_threshold: float = 0.8,
                         tier_counts: dict | None = None, package_dir=None) -> dict:
    """The §15.3 EvaluationRun coordinator + EXPORT GATE: run the real-physics tiered failure loop and emit a
    single, explicit export decision. When ``package_dir`` is given, persist ``reports/evaluation_run.json`` so
    the Product Readiness Ledger ENFORCES it — a robot that ran real physics but FAILED the task (baseline or
    randomized below ``export_threshold``) is then NOT ``safe_to_export`` (the ``physics_evaluated`` stage reads
    ``below_gate``, not ``attained``). This closes the loop: "physics ran" is no longer the same as "task passed".

    Returns the full failure-loop report plus a ``decision`` block (the compact, persisted gate)."""
    import json
    from pathlib import Path

    report = run_pickplace_failure_loop(gene, prompt=prompt, seed=seed, params=params,
                                        export_threshold=export_threshold, tier_counts=tier_counts)
    decision = {
        "schema": "evaluation_run/v1",
        "robot": report["robot"], "task": report["task"],
        "export_ready": report["export_ready"],
        "export_blockers": report["export_blockers"],
        "export_threshold": export_threshold,
        "gate_tiers": ["baseline", "randomized"],          # stress is analysis-only (plan §11.2)
        "tier_success": {t: report["tiers"][t]["success_rate"] for t in report["tiers"]},
        "regression_cause_confirmed_rate": report["regression"]["cause_confirmed_rate"],
    }
    if package_dir is not None:
        rep_dir = Path(package_dir) / "reports"
        rep_dir.mkdir(parents=True, exist_ok=True)
        (rep_dir / "evaluation_run.json").write_text(
            json.dumps({**decision, "full_report": report}, indent=2), encoding="utf-8")
    return {**report, "decision": decision}


_SUGGESTED = {
    "missed_grasp": "block centered within the reliable reach envelope",
    "wrong_bin": "wider bin spacing + clearer color separation",
    "dropped": "lower lift height + slower transport (and smaller object)",
    "instability": "lighter block + lower controller gains",
    "wrong_bin ": "wider bin spacing",
}
