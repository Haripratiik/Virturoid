"""Export a trained policy as an inference-ready controller bundle (#3).

The bundle is self-contained: a metadata JSON describing the interface
(observation inputs, action limits, control frequency, joint mappings, safety
clamps), the learned parameters, and a standalone ``controller.py`` that runs
inference with no Virturoid or MuJoCo dependency.
"""

from __future__ import annotations

import json
from pathlib import Path

from virturoid.schemas.trained_policy import TrainedPolicy

_CONTROLLER_SOURCE = '''"""Standalone inference controller exported by Virturoid.

Loads the learned linear reach policy and maps an observed target-block position
to safety-clamped joint position targets. Pure standard library; no MuJoCo or
Virturoid imports, so it can run inside a ROS2 node or a bare Python process.
"""

from __future__ import annotations

import json
from pathlib import Path


class ReachController:
    def __init__(self, params: dict):
        self.joint_names = params["joint_names"]
        self.weights = params["weights"]
        self.position_limits = params["safety_clamps"]["joint_position_limits"]
        self.control_frequency_hz = params["control_frequency_hz"]

    @classmethod
    def from_file(cls, path: str) -> "ReachController":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def infer(self, target_block_xy) -> dict:
        """Return safety-clamped joint position targets for the target block."""
        features = [float(target_block_xy[0]), float(target_block_xy[1]), 1.0]
        targets = []
        for row, (low, high) in zip(self.weights, self.position_limits):
            value = sum(w * f for w, f in zip(row, features))
            targets.append(max(low, min(high, value)))
        return dict(zip(self.joint_names, targets))


if __name__ == "__main__":
    controller = ReachController.from_file(str(Path(__file__).with_name("policy_params.json")))
    print(json.dumps(controller.infer([0.4, -0.1]), indent=2))
'''


def export_controller_bundle(
    package_dir: Path,
    policy: TrainedPolicy,
    bundle_subdir: str = "software/controller",
) -> Path:
    """Write controller.py, policy_params.json, and controller_bundle.json."""
    package_dir = Path(package_dir)
    bundle_dir = package_dir / bundle_subdir
    bundle_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "policy_id": policy.id,
        "policy_type": policy.policy_type,
        "joint_names": policy.joint_names,
        "weights": policy.weights,
        "input_features": policy.input_features,
        "control_frequency_hz": policy.control_frequency_hz,
        "safety_clamps": policy.safety_clamps,
    }
    (bundle_dir / "policy_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    (bundle_dir / "controller.py").write_text(_CONTROLLER_SOURCE, encoding="utf-8")

    bundle = {
        "id": f"controller_bundle_{policy.id}",
        "policy_id": policy.id,
        "policy_type": policy.policy_type,
        "robot_genome_id": policy.robot_genome_id,
        "entrypoint": f"{bundle_subdir}/controller.py",
        "inference_class": "ReachController",
        "parameters_uri": f"{bundle_subdir}/policy_params.json",
        "observation_inputs": [
            {"name": "target_block_xy", "shape": [2], "units": "m", "frame": "base_link"},
        ],
        "action_outputs": [
            {
                "name": dim.name,
                "type": "joint_position_target",
                "units": "rad",
                "lower": dim.lower,
                "upper": dim.upper,
                "velocity_limit_rad_s": dim.velocity_limit,
                "effort_limit_nm": dim.effort_limit,
            }
            for dim in policy.action_dimensions
        ],
        "control_frequency_hz": policy.control_frequency_hz,
        "low_level_control_frequency_hz": policy.low_level_control_frequency_hz,
        "joint_mapping": {name: index for index, name in enumerate(policy.joint_names)},
        "safety_clamps": policy.safety_clamps,
        "evaluation": {
            "eval_scene_count": policy.evaluation.eval_scene_count,
            "mean_reach_distance_m": policy.evaluation.mean_reach_distance_m,
            "success_rate": policy.evaluation.success_rate,
            "success_threshold_m": policy.evaluation.success_threshold_m,
        },
        "notes": [
            "Inference-ready: load policy_params.json into ReachController and call infer(target_block_xy).",
            "Outputs joint position targets in radians, already clamped to joint limits.",
            "Downstream PD / ros2_control tracks these targets at the low-level control frequency.",
        ],
    }
    bundle_path = bundle_dir / "controller_bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return bundle_path


def write_training_acceptance_report(
    package_dir: Path,
    policy: TrainedPolicy,
    bundle_path: Path,
    *,
    min_success_rate: float = 0.5,
    min_stable_rate: float = 0.9,
) -> Path:
    """Write the honest training acceptance verdict for an exported controller."""
    package_dir = Path(package_dir)
    bundle_path = Path(bundle_path)
    controller_dir = package_dir / "software" / "controller"
    policy_path = controller_dir / "trained_policy.json"
    params_path = controller_dir / "policy_params.json"
    entrypoint_path = controller_dir / "controller.py"
    metrics_path = package_dir / "training" / "policy_training_metrics.json"
    validation = policy.validate()
    files = {
        "trained_policy": policy_path.exists(),
        "controller_bundle": bundle_path.exists(),
        "policy_params": params_path.exists(),
        "controller_entrypoint": entrypoint_path.exists(),
        "training_metrics": metrics_path.exists(),
    }
    eval_success = float(policy.evaluation.success_rate)
    stable = float(policy.evaluation.stable_rate)
    improved = policy.training.best_reward >= policy.training.initial_reward
    accepted = (
        validation.ok
        and all(files.values())
        and policy.evaluation.eval_scene_count > 0
        and improved
        and eval_success >= min_success_rate
        and stable >= min_stable_rate
    )
    report = {
        "accepted": bool(accepted),
        "policy_id": policy.id,
        "method": policy.training.method,
        "thresholds": {
            "min_success_rate": min_success_rate,
            "min_stable_rate": min_stable_rate,
        },
        "training": {
            "iterations": policy.training.iterations,
            "candidates_per_iteration": policy.training.candidates_per_iteration,
            "training_scene_count": policy.training.training_scene_count,
            "initial_reward": policy.training.initial_reward,
            "best_reward": policy.training.best_reward,
            "improved_reward": bool(improved),
        },
        "evaluation": {
            "eval_scene_count": policy.evaluation.eval_scene_count,
            "success_rate": eval_success,
            "stable_rate": stable,
            "mean_reach_distance_m": policy.evaluation.mean_reach_distance_m,
            "best_reach_distance_m": policy.evaluation.best_reach_distance_m,
            "success_threshold_m": policy.evaluation.success_threshold_m,
        },
        "files": files,
        "validation_ok": validation.ok,
        "validation_issues": [issue.code for issue in validation.issues],
        "notes": [
            "Acceptance is based on held-out reach evaluation, controller bundle completeness, and policy schema validation.",
            "This reach controller is a first training proof, not a full pick/place/grasp policy.",
        ],
    }
    out = package_dir / "reports" / "training_acceptance_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out
