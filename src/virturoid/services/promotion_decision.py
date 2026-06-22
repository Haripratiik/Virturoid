from __future__ import annotations

import json
from pathlib import Path

from virturoid.schemas.base import ArtifactRef
from virturoid.schemas.promotion import ActiveTrainingInputs, PromotionDecision


PROMOTION_DECISION_URI = "reports/promotion_decision.json"
ACTIVE_INPUTS_URI = "training/active_training_inputs.json"


def write_promotion_decision_from_export(package_dir: Path) -> PromotionDecision:
    improvement = _read_json(package_dir / "reports" / "improvement_report.json")
    success_delta = _metric_delta(improvement, "target_group_success_rate")
    scene_count_delta = _metric_delta(improvement, "target_scene_count")
    episode_count_delta = _metric_delta(improvement, "target_planned_episode_count")

    gates = {
        "success_improved": success_delta > 0,
        "scene_coverage_preserved": scene_count_delta >= 0,
        "needs_more_revision_episodes": episode_count_delta < 0,
    }
    promote_revised = gates["success_improved"] and gates["scene_coverage_preserved"]
    decision_text = "promote_revised_with_episode_warning" if promote_revised else "hold_original"

    selected_training_config = ArtifactRef(
        uri="training/training_run_config_revised.json" if promote_revised else "training/training_run_config.json",
        media_type="application/json",
    )
    selected_policy_plan = ArtifactRef(
        uri="software/policy_plan_revised.json" if promote_revised else "software/policy_plan.json",
        media_type="application/json",
    )
    selected_scene_set = ArtifactRef(
        uri="simulation/revision_scene_set.json" if promote_revised else "simulation/regression_scene_set.json",
        media_type="application/json",
    )
    rationale = _rationale(improvement, gates, promote_revised)

    decision = PromotionDecision(
        id=f"promotion_{improvement['id']}",
        improvement_report_id=improvement["id"],
        decision=decision_text,
        selected_training_config=selected_training_config,
        selected_policy_plan=selected_policy_plan,
        selected_scene_set=selected_scene_set,
        active_inputs_artifact=ArtifactRef(uri=ACTIVE_INPUTS_URI, media_type="application/json"),
        rationale=rationale,
        gates=gates,
        notes=[
            "Promotion decision is deterministic and based on the MVP improvement report.",
            "A real simulator-backed pipeline should add safety gates before deployment.",
        ],
    )
    active_inputs = ActiveTrainingInputs(
        id=f"active_{decision.id}",
        decision_id=decision.id,
        training_config=selected_training_config,
        policy_plan=selected_policy_plan,
        scene_set=selected_scene_set,
        reason=rationale,
    )
    _raise_if_invalid(decision.validate(), decision.id)
    _raise_if_invalid(active_inputs.validate(), active_inputs.id)
    _write_json(package_dir / PROMOTION_DECISION_URI, decision.to_dict())
    _write_json(package_dir / ACTIVE_INPUTS_URI, active_inputs.to_dict())
    return decision


def _metric_delta(improvement: dict, metric_name: str) -> float:
    for metric in improvement.get("metrics", []):
        if metric["name"] == metric_name:
            return float(metric["delta"])
    raise ValueError(f"Improvement report {improvement['id']} has no {metric_name} metric.")


def _rationale(improvement: dict, gates: dict[str, bool], promote_revised: bool) -> str:
    if promote_revised and gates["needs_more_revision_episodes"]:
        return (
            f"{improvement['id']} improved the target success rate, so revised artifacts are selected, "
            "but the revised pass used fewer planned episodes and needs expanded rollout coverage."
        )
    if promote_revised:
        return f"{improvement['id']} improved the target success rate while preserving coverage."
    return f"{improvement['id']} did not clear promotion gates; keep original artifacts active."


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _raise_if_invalid(validation_result, entity_id: str) -> None:
    if validation_result.ok:
        return
    issues = ", ".join(issue.code for issue in validation_result.issues)
    raise ValueError(f"{entity_id} failed validation: {issues}")
