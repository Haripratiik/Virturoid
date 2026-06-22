from __future__ import annotations

import json
from pathlib import Path

from virturoid.schemas.improvement import ImprovementMetric, ImprovementReport


IMPROVEMENT_REPORT_URI = "reports/improvement_report.json"


def write_improvement_report_from_export(package_dir: Path) -> ImprovementReport:
    source = _read_json(package_dir / "runs" / "mvp_training" / "dry_run_result.json")
    revised = _read_json(package_dir / "runs" / "mvp_training_revision" / "dry_run_result.json")

    source_regression = _group_success_rate(source, "regression")
    revised_revision = _group_success_rate(revised, "revision")
    success_delta = round(revised_revision - source_regression, 3)
    scene_delta = revised["total_scenes"] - _group_scene_count(source, "regression")
    episode_delta = revised["total_planned_episodes"] - _group_episode_count(source, "regression")

    report = ImprovementReport(
        id=f"improvement_{revised['training_run_config_id']}",
        source_dry_run_id=source["id"],
        revised_dry_run_id=revised["id"],
        comparison_basis="original regression scene group vs revised scene group",
        outcome="improved" if success_delta > 0 else "needs_more_iteration",
        metrics=[
            ImprovementMetric(
                name="target_group_success_rate",
                before=source_regression,
                after=revised_revision,
                delta=success_delta,
                interpretation="Positive values mean the revised scene/policy pass improved the low-success regression target.",
            ),
            ImprovementMetric(
                name="target_scene_count",
                before=float(_group_scene_count(source, "regression")),
                after=float(revised["total_scenes"]),
                delta=float(scene_delta),
                interpretation="Shows whether the revised pass preserved target scene coverage.",
            ),
            ImprovementMetric(
                name="target_planned_episode_count",
                before=float(_group_episode_count(source, "regression")),
                after=float(revised["total_planned_episodes"]),
                delta=float(episode_delta),
                interpretation="Shows whether the revised pass changed planned rollout volume.",
            ),
        ],
        next_actions=_next_actions(success_delta),
        source_artifacts=[
            "runs/mvp_training/dry_run_result.json",
            "runs/mvp_training_revision/dry_run_result.json",
            "reports/redesign_revision.json",
        ],
    )
    _raise_if_invalid(report.validate(), report.id)
    _write_json(package_dir / IMPROVEMENT_REPORT_URI, report.to_dict())
    return report


def _group_success_rate(result: dict, purpose: str) -> float:
    for group in result.get("group_results", []):
        if group["purpose"] == purpose:
            return float(group["simulated_success_rate"])
    raise ValueError(f"Dry-run result {result['id']} has no {purpose} group.")


def _group_scene_count(result: dict, purpose: str) -> int:
    for group in result.get("group_results", []):
        if group["purpose"] == purpose:
            return int(group["scene_count"])
    raise ValueError(f"Dry-run result {result['id']} has no {purpose} group.")


def _group_episode_count(result: dict, purpose: str) -> int:
    for group in result.get("group_results", []):
        if group["purpose"] == purpose:
            return int(group["planned_episodes"])
    raise ValueError(f"Dry-run result {result['id']} has no {purpose} group.")


def _next_actions(success_delta: float) -> list[str]:
    if success_delta > 0:
        return [
            "promote revised policy and revision scene set into the next simulator-backed run",
            "increase revised-scene episode count before replacing the baseline policy",
        ]
    return [
        "generate another revision with larger clearance changes",
        "add contact-aware trajectory diagnostics before the next run",
    ]


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
