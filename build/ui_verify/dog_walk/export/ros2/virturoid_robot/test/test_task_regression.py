"""Regression test: config loads, joints agree, and (if present) the exported controller RUNS.

Runs without a ROS2 install — the ReachController is pure stdlib — so `colcon test` / `pytest` exercises
the actual exported policy: it must infer one joint position target per joint, each within its limits.
"""

import json
import importlib.util
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1]


def _config():
    return json.loads((_PKG / "config" / "robot.yaml").read_text())


def test_config_has_joints():
    config = _config()
    assert config["joints"], "robot config must list joints"
    assert config["control_frequency_hz"] > 0


def test_controller_runs_if_present():
    config = _config()
    pkg = _PKG / _PKG.name
    if not config.get("has_controller") or not (pkg / "controller.py").exists():
        return  # no controller bundle exported with this package
    spec = importlib.util.spec_from_file_location("vq_controller", pkg / "controller.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    if config.get("policy_type") == "trot_cpg_gait":
        controller = mod.GaitController.from_file(str(pkg / "policy_params.json"))
        for t in (0.0, 0.1, 0.25, 0.5):
            out = controller.infer(t)
            assert set(out) == set(controller.joint_names), "controller must output every joint"
            for j, limit in zip(controller.joint_names, controller.limits):
                assert limit[0] - 1e-6 <= out[j] <= limit[1] + 1e-6, f"{j} target out of limits"
    else:
        controller = mod.ReachController.from_file(str(pkg / "policy_params.json"))
        for target in config.get("target_positions", [[0.4, 0.0]]):
            out = controller.infer(target)
            assert set(out) == set(controller.joint_names), "controller must output every joint"
            for j, (low, high) in zip(controller.joint_names, controller.position_limits):
                assert low - 1e-6 <= out[j] <= high + 1e-6, f"{j} target out of limits"
