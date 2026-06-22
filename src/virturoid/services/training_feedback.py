from __future__ import annotations

import json
from pathlib import Path

from virturoid.schemas.training_feedback import RedesignRecommendation, TrainingFeedbackReport


def analyze_training_feedback_from_export(package_dir: Path) -> TrainingFeedbackReport:
    training_config = _read_json(package_dir / "training" / "training_run_config.json")
    dry_run = _read_json(package_dir / "runs" / "mvp_training" / "dry_run_result.json")
    evaluation = _read_json(package_dir / "reports" / "mvp_evaluation_report.json")

    recommendations = [
        *_recommend_from_dry_run(dry_run),
        *_recommend_from_evaluation(evaluation),
    ]
    if not recommendations:
        recommendations.append(
            RedesignRecommendation(
                id="rec_continue_validation",
                target="training",
                priority="low",
                summary="Continue validation with harder scenes.",
                rationale="No major dry-run or mock-evaluation weakness was detected.",
                suggested_actions=["increase holdout scene count", "add unseen object and lighting variations"],
                source_artifacts=["training/training_run_config.json", "runs/mvp_training/dry_run_result.json"],
            )
        )

    report = TrainingFeedbackReport(
        id=f"feedback_{training_config['id']}",
        training_run_config_id=training_config["id"],
        dry_run_result_id=dry_run["id"],
        evaluation_run_id=evaluation["id"],
        overall_status="needs_iteration" if any(item.priority in {"high", "medium"} for item in recommendations) else "monitor",
        recommendations=recommendations,
        notes=[
            "MVP feedback combines dry-run success estimates with mock evaluation failure records.",
            "Future simulator adapters should attach richer trajectory, contact, and control diagnostics here.",
        ],
    )
    _raise_if_invalid(report.validate(), report.id)
    _write_report(package_dir, report)
    return report


def _recommend_from_dry_run(dry_run: dict) -> list[RedesignRecommendation]:
    recommendations: list[RedesignRecommendation] = []
    for group in dry_run.get("group_results", []):
        success_rate = group["simulated_success_rate"]
        if success_rate >= 0.7:
            continue
        recommendations.append(
            RedesignRecommendation(
                id=f"rec_improve_{group['purpose']}_training",
                target="policy",
                priority="high" if success_rate < 0.65 else "medium",
                summary=f"Improve policy behavior on {group['purpose']} scenes.",
                rationale=(
                    f"The {group['purpose']} group estimated success rate is {success_rate}, "
                    "below the MVP threshold of 0.70."
                ),
                suggested_actions=[
                    "add recovery waypoints to the policy plan",
                    "increase scene coverage around the low-success variation family",
                    "route failed episodes into regression scenes before the next run",
                ],
                source_artifacts=[
                    "runs/mvp_training/dry_run_result.json",
                    group["scene_artifact"],
                ],
            )
        )
    return recommendations


# Each REAL evaluation failure type -> a targeted redesign recommendation (plan §31.5: act on the corrected
# variable). 'policy' = retune/replan the controller; 'scene_and_policy' = also regenerate harder scenes.
_EVAL_REC = {
    "missed_grasp": ("scene_and_policy", "Recenter graspable objects and improve the approach.",
                     "objects were placed past the reliable grasp envelope and missed"),
    "wrong_bin": ("scene_and_policy", "Widen bin spacing and clarify colors to stop mis-sorts.",
                  "objects were released over the wrong bin"),
    "dropped": ("policy", "Lower the lift height and slow transport so grasped objects aren't dropped.",
                "objects were grasped but dropped during transport"),
    "instability": ("policy", "Reduce controller gains for heavy/large objects.",
                    "the controller went unstable on the object"),
    "collision": ("scene_and_policy", "Reduce collision failures found by evaluation.",
                  "evaluation failures reported collision risk"),
}


def _recommend_from_evaluation(evaluation: dict) -> list[RedesignRecommendation]:
    """One targeted recommendation per REAL failure type the physics evaluation reported (was collision-only;
    real MuJoCo eval surfaces missed_grasp / dropped / wrong_bin / instability, each with its own fix)."""
    recommendations: list[RedesignRecommendation] = []
    failures = evaluation.get("failures", [])
    if not failures:
        return recommendations
    by_type: dict[str, list] = {}
    for failure in failures:
        by_type.setdefault(failure.get("failure_type", "unknown"), []).append(failure)
    for ftype, group in sorted(by_type.items()):
        target, summary, why = _EVAL_REC.get(
            ftype, ("scene_and_policy", f"Address {ftype} failures found by evaluation.", f"evaluation reported {ftype}"))
        actions: list[str] = []
        for failure in group:
            actions.extend(failure.get("suggested_fixes", []))
        recommendations.append(
            RedesignRecommendation(
                id=f"rec_{ftype}_failures",
                target=target,
                priority="medium",
                summary=summary,
                rationale=f"{len(group)} evaluation failure(s): {why}.",
                suggested_actions=sorted(set(actions or [f"add failure-conditioned regression scenes for {ftype}"])),
                source_artifacts=["reports/physics_evaluation_report.json", "simulation/regression_scene_set.json"],
            )
        )
    return recommendations


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_report(package_dir: Path, report: TrainingFeedbackReport) -> Path:
    path = package_dir / "reports" / "training_feedback.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return path


def _raise_if_invalid(validation_result, entity_id: str) -> None:
    if validation_result.ok:
        return
    issues = ", ".join(issue.code for issue in validation_result.issues)
    raise ValueError(f"{entity_id} failed validation: {issues}")
