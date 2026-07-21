"""Standalone trot-gait controller exported by Virturoid.

Computes feed-forward joint POSITION TARGETS for a CPG trot gait. For joint j at time t (seconds):
    target[j] = default_pose[j] + amplitude[j] * sin(2*pi*frequency_hz*t + phase_offset[j])
clamped to the joint position limits. A downstream PD / ros2_control loop tracks these targets at the
low-level control frequency. Pure standard library; no MuJoCo or Virturoid imports, so it runs inside a
ROS2 node or a bare Python process.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


class GaitController:
    def __init__(self, params: dict):
        self.joint_names = params["joint_names"]
        self.default_pose = params["default_pose"]
        self.amplitude = params["amplitude"]
        self.phase_offset = params["phase_offset"]
        self.frequency_hz = float(params["frequency_hz"])
        self.limits = params["position_limits"]

    @classmethod
    def from_file(cls, path: str) -> "GaitController":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def infer(self, t_seconds: float) -> dict:
        """Return clamped joint position targets for the gait phase at time ``t_seconds``."""
        omega = 2.0 * math.pi * self.frequency_hz
        out = {}
        for name, q0, amp, ph, limit in zip(
            self.joint_names, self.default_pose, self.amplitude, self.phase_offset, self.limits
        ):
            value = q0 + amp * math.sin(omega * t_seconds + ph)
            out[name] = max(limit[0], min(limit[1], value))
        return out


if __name__ == "__main__":
    _here = Path(__file__).parent
    _params = _here / "policy_params.json"
    if not _params.exists():
        _params = _here / "control_program.json"
    controller = GaitController.from_file(str(_params))
    for _i in range(6):
        _t = _i * 0.1
        print(json.dumps({"t": round(_t, 2), "targets": controller.infer(_t)}))
