"""Failure -> curriculum repair mapping (dossier R7 + Training Improvement Plan sensor-failure taxonomy).

After every evaluation, failure clusters should turn directly into concrete curriculum / repair changes instead
of vague advice. This module is the deterministic table the dossier specifies (no_contact -> closer approach,
gripped_no_lift -> lift-phase reward, fell_over -> stability prior, reward_high_success_low -> reject the reward
candidate, ...), extended with the plan's perception failures. Pure data mapping, no backend, fully testable.
"""

from __future__ import annotations

_FAILURE_REPAIRS: dict[str, list[str]] = {
    # dossier R7 manipulation / locomotion clusters
    "no_contact": ["closer_approach_scenes", "perception_ik_calibration", "gripper_aperture_variants"],
    "gripped_no_lift": ["lift_phase_reward", "friction_mass_randomization", "finger_residual_control"],
    "lifted_then_dropped": ["hold_phase_scenes", "slip_penalty", "slow_trajectory_teacher"],
    "fell_over": ["stability_prior", "shorter_horizon", "higher_alive_gate", "terrain_simplification"],
    "upright_but_slow": ["speed_curriculum", "command_conditioning"],
    "reward_high_success_low": ["reject_reward_candidate"],
    # Training Improvement Plan perception failures
    "bad_depth_at_grasp": ["depth_dropout_curriculum", "grasp_depth_calibration", "wrist_camera_reposition"],
    "occlusion_failure": ["occluder_curriculum", "multi_view_camera", "viewpoint_randomization"],
    "out_of_fov_failure": ["fov_curriculum", "camera_pose_search"],
    "missed_detection": ["detection_hard_negatives", "lighting_randomization", "detector_retrain"],
    "false_positive_detection": ["detector_hard_negatives", "confidence_threshold_curriculum"],
    "latency_instability": ["latency_curriculum", "action_lowpass", "control_rate_match"],
    "segmentation_leak": ["remove_segmentation_from_policy", "rgb_only_ablation"],
    "pose_estimate_drift": ["pose_filter", "extrinsics_recalibration"],
    "sensor_saturation": ["exposure_randomization", "hdr_curriculum"],
    "contact_false_positive": ["contact_threshold_calibration", "force_noise_randomization"],
    "proprioception_bias_failure": ["proprioception_bias_randomization", "encoder_calibration"],
    "privileged_state_leakage": ["enforce_observation_contract", "rerun_leakage_gate"],
}

REWARD_HACK_FAILURE = "reward_high_success_low"


def curriculum_updates_for_failure(failure_type: str) -> list[str]:
    """Return the concrete curriculum/repair changes for a single failure type (empty if unmapped)."""
    return list(_FAILURE_REPAIRS.get(failure_type, []))


def is_mapped(failure_type: str) -> bool:
    return failure_type in _FAILURE_REPAIRS


def curriculum_from_failure_clusters(clusters: dict[str, int] | list[str]) -> dict:
    """Aggregate failure clusters into an ordered, de-duplicated curriculum update plan.

    ``clusters`` may be a ``{failure_type: count}`` mapping or a list of failure-type strings. Updates are ordered
    by failure frequency (then alphabetically), de-duplicated, with unmapped failures reported separately and a
    flag set when a reward-hacking failure demands rejecting the current reward candidate.
    """
    if isinstance(clusters, dict):
        counts = dict(clusters)
    else:
        counts = {}
        for failure in clusters:
            counts[failure] = counts.get(failure, 0) + 1

    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    updates: list[str] = []
    seen: set[str] = set()
    by_failure: dict[str, list[str]] = {}
    unmapped: list[str] = []
    reject_rewards = False

    for failure, _count in ordered:
        repairs = curriculum_updates_for_failure(failure)
        by_failure[failure] = repairs
        if not repairs:
            unmapped.append(failure)
            continue
        if failure == REWARD_HACK_FAILURE:
            reject_rewards = True
        for repair in repairs:
            if repair not in seen:
                seen.add(repair)
                updates.append(repair)

    return {
        "updates": updates,
        "by_failure": by_failure,
        "unmapped": unmapped,
        "reject_reward_candidates": reject_rewards,
    }
