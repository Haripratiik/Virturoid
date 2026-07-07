"""Privileged-state leakage gate + perception-rung report — Phase 0 of the Training Improvement Plan.

The plan's grounding rule is "sensors observe, physics disposes": a training plan is only trustworthy if the
policy is fed the observations the *deployed* robot will actually have, not privileged simulator truth. The plan
names a concrete offender — ``policy_trainer.py`` still puts privileged ``target_block_xy`` in its observation
space — and lists the leaks that must reject a deployable plan (object poses as inputs, goal-truth given when the
task expects vision, contact truth bypassing contact sensors, segmentation in a RGB-only deploy, unlogged
camera randomization, train/held-out seed overlap).

This module is deterministic and backend-free: it inspects an :class:`ObservationContract` and returns a
:class:`ValidationResult`, plus a print-ready report that selects the perception rung and surfaces existing
range/vision demos — all *without running any training* (the Phase 0 "print the rungs" command).
"""

from __future__ import annotations

from virturoid.schemas.base import ValidationResult
from virturoid.schemas.observation_contract import (
    CAMERA_MODALITIES,
    LOWDIM_MODALITIES,
    ObservationContract,
    PerceptionRung,
    PerceptionTrainingPlan,
)

# Lowercased substrings that mark *privileged simulator truth* leaking into a POLICY input. Kept conservative:
# unambiguous truth markers (object/world pose, gt_/true_/oracle_/privileged_, contact truth) and the plan's
# concrete `target_block_xy`/`block_xy` case — NOT generic command-like inputs (a "goal_command" can be legit).
_PRIVILEGED_PATTERNS = (
    "target_block_xy", "block_xy", "block_pos", "block_position",
    "object_pose", "object_xy", "object_position", "object_com", "target_object_pose", "world_pose",
    "gt_", "true_", "ground_truth", "oracle_", "privileged_", "contact_truth", "sim_contact",
)

ALLOWED_PRIVILEGED_USES = (
    "teacher_trajectories", "supervised_labels", "success_detectors", "debugging_upper_bound", "offline_ablation",
)

# Existing perception capabilities already in the repo — surfaced so Phase 0 makes them visible as prior art.
DEFAULT_VISIBLE_DEMOS = (
    "exteroception: MuJoCo rangefinder ring -> perception_nav obstacle avoidance",
    "vision_train/vision_nav: tiny image encoder predicts goal bearing for navigation",
    "vision_grasp: overhead RGB box locator drives a real grasp attempt",
    "percept_pretrain: image features distilled from privileged rangefinder signals (teacher/student)",
)


def _matched_privileged_pattern(key: str) -> str | None:
    lowered = key.lower()
    for pattern in _PRIVILEGED_PATTERNS:
        if pattern in lowered:
            return pattern
    return None


def _is_segmentation_key(key: str) -> bool:
    lowered = key.lower()
    return "segmentation" in lowered or lowered in {"seg", "seg_mask", "segmask"}


def check_privileged_leakage(contract: ObservationContract) -> ValidationResult:
    """Reject a deployable training plan that lets the policy cheat with privileged simulator state.

    Errors block a deployable plan (``result.ok`` is False); randomization-not-logged is a warning under a
    permissive policy and an error under a strict one.
    """
    result = ValidationResult()
    privileged = set(contract.privileged_label_keys)
    deploy = {modality.lower() for modality in contract.deploy_modalities}

    for key in contract.policy_observation_keys:
        if key in privileged:
            result.add("privileged_label_in_policy",
                       f"'{key}' is declared a privileged label but is also a policy observation.", key)
            continue
        if _is_segmentation_key(key):
            if deploy and "segmentation" not in deploy:
                result.add("segmentation_leak",
                           f"Policy uses segmentation '{key}' but the deploy modalities have no segmentation.", key)
            continue
        pattern = _matched_privileged_pattern(key)
        if pattern is not None:
            result.add("privileged_state_in_policy",
                       f"Policy observation '{key}' looks like privileged simulator truth (matched '{pattern}').",
                       key)

    overlap = sorted(set(contract.train_scene_seeds) & set(contract.heldout_scene_seeds))
    if overlap:
        result.add("seed_overlap",
                   f"Train and held-out scene seeds overlap ({overlap}); held-out evaluation would be leaked.",
                   "heldout_scene_seeds")

    uses_camera = bool({m.lower() for m in contract.required_modalities} & CAMERA_MODALITIES)
    if uses_camera and not contract.randomization_logged:
        severity = "error" if contract.leakage_policy == "strict" else "warning"
        result.add("randomization_unlogged",
                   "Camera modalities are used but domain randomization is not logged; extrinsics/randomization "
                   "must be recorded for held-out generalization to be trustworthy.",
                   "randomization_logged", severity=severity)
    return result


def select_perception_rung(contract: ObservationContract) -> PerceptionRung:
    """Suggest the minimum perception rung implied by the contract's required modalities."""
    modalities = {modality.lower() for modality in contract.required_modalities}
    if modalities & CAMERA_MODALITIES:
        return PerceptionRung.RUNG_2_NATIVE_SENSORS
    if modalities & LOWDIM_MODALITIES:
        return PerceptionRung.RUNG_1_SYNTHETIC_ADAPTER
    return PerceptionRung.RUNG_0_PRIVILEGED


def training_plan_report(
    contract: ObservationContract,
    *,
    plan: PerceptionTrainingPlan | None = None,
    existing_demos: tuple[str, ...] | list[str] | None = None,
) -> dict:
    """Print-ready perception-plan summary WITHOUT running heavy training (Phase 0 deliverable).

    Surfaces declared-vs-suggested rung, the leakage-gate result, allowed privileged uses, and existing
    range/vision demos so a developer sees current capabilities and blockers at a glance.
    """
    leakage = check_privileged_leakage(contract)
    errors = [issue for issue in leakage.issues if issue.severity == "error"]
    warnings = [issue for issue in leakage.issues if issue.severity != "error"]
    suggested = select_perception_rung(contract)
    report = {
        "observation_contract_id": contract.id,
        "declared_rung": contract.perception_rung.value,
        "suggested_rung": suggested.value,
        "rung_matches": contract.perception_rung == suggested,
        "required_modalities": list(contract.required_modalities),
        "deploy_modalities": list(contract.deploy_modalities),
        "policy_observation_keys": list(contract.policy_observation_keys),
        "privileged_label_keys": list(contract.privileged_label_keys),
        "leakage": {
            "ok": leakage.ok,
            "errors": [{"code": i.code, "field": i.field, "message": i.message} for i in errors],
            "warnings": [{"code": i.code, "field": i.field, "message": i.message} for i in warnings],
        },
        "allowed_privileged_uses": list(ALLOWED_PRIVILEGED_USES),
        "visible_demos": list(existing_demos if existing_demos is not None else DEFAULT_VISIBLE_DEMOS),
        "ablation_matrix": (plan.ablation_matrix if plan is not None else []),
    }
    report["recommendation"] = (
        f"PASS: train at {suggested.value}; use privileged state only as teacher/label/upper-bound."
        if leakage.ok else
        "BLOCKED: resolve leakage errors before training a deployable policy "
        "(privileged state or held-out seeds are leaking into the policy)."
    )
    return report


def format_report(report: dict) -> str:
    """Human-readable one-screen rendering of :func:`training_plan_report` (the 'print the rungs' command)."""
    lines = [
        f"Observation contract: {report['observation_contract_id']}",
        f"  perception rung:   declared={report['declared_rung']}  suggested={report['suggested_rung']}"
        f"  {'(match)' if report['rung_matches'] else '(MISMATCH)'}",
        f"  policy observes:   {', '.join(report['policy_observation_keys']) or '(none)'}",
        f"  privileged labels: {', '.join(report['privileged_label_keys']) or '(none)'}",
        f"  leakage gate:      {'PASS' if report['leakage']['ok'] else 'FAIL'}",
    ]
    for err in report["leakage"]["errors"]:
        lines.append(f"    ! {err['code']}: {err['message']}")
    for warn in report["leakage"]["warnings"]:
        lines.append(f"    ~ {warn['code']}: {warn['message']}")
    lines.append("  existing perception demos:")
    for demo in report["visible_demos"]:
        lines.append(f"    - {demo}")
    lines.append(f"  => {report['recommendation']}")
    return "\n".join(lines)
