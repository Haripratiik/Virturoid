"""The deploy loop, proven in sim on CPU (plan §4.7) — the "sim controls the robot, safely" mechanism.

The ROS2 export gets a trained controller ONTO a robot, but the thesis "I build it -> the policy runs, and the
sim can drive/validate it over ROS" needs the CLOSED LOOP: read joint state -> controller -> SAFETY clamp ->
command motors -> read back. This module runs exactly that loop with MuJoCo standing in as the "real hardware"
(no ROS install needed), so the design->build->run path is demonstrable and testable today. On metal the same
shape runs through ros2_control + the hardware interface the exporter emits; here MuJoCo is the hardware behind
the same command/state interface.

``SafetyFilter`` is the CBF-lite gate (joint position + rate limits) that MUST sit between any policy and a real
motor — it is what makes "drop the policy onto the robot" safe rather than reckless.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SafetyFilter:
    """Clamp commanded joint-position targets to (1) hard joint limits and (2) a per-step velocity (rate) limit,
    so no command can drive a joint past its mechanical stop or slew faster than the actuator/safety budget. The
    minimal real-actuation gate; on hardware this is the last thing before the motor driver."""
    lower: list                       # per-joint position min (rad)
    upper: list                       # per-joint position max (rad)
    vel_limit: float = 8.0            # max joint slew (rad/s) -> per-step rate limit

    def clamp(self, target: list, q: list, dt: float) -> tuple[list, int]:
        """Return (clamped_targets, n_violations). A violation = the raw command had to be altered."""
        out: list = []
        violations = 0
        step = max(1e-6, self.vel_limit * dt)
        for i, t in enumerate(target):
            c = min(self.upper[i], max(self.lower[i], float(t)))          # joint-limit clamp
            c = min(q[i] + step, max(q[i] - step, c))                     # rate (velocity) limit
            if abs(c - float(t)) > 1e-6:
                violations += 1
            out.append(c)
        return out, violations


class SimHardwareBridge:
    """MuJoCo as the robot's hardware behind a ROS-style command/state loop. ``read_state()`` is the
    ``/joint_states`` publisher; ``step(command_fn)`` is one control tick: read state -> command_fn (the policy
    or controller) -> SafetyFilter -> PD to the commanded position target -> advance the 'hardware'. The same
    shape the exported ROS2 node runs against ros2_control on metal."""

    def __init__(self, gene, *, kp: float = 32.0, kd: float = 1.5, vel_limit: float = 8.0):
        import mujoco

        from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
        self._mj = mujoco
        xml = compile_gene_to_mjcf(gene, spawn_z=standing_spawn_z(gene, meshed=False))
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.kp, self.kd = float(kp), float(kd)
        self.dt = float(self.model.opt.timestep)
        nu = self.model.nu
        self._aj = [int(self.model.actuator_trnid[i, 0]) for i in range(nu)]      # actuator -> joint id
        self._qadr = [int(self.model.jnt_qposadr[j]) for j in self._aj]
        self._vadr = [int(self.model.jnt_dofadr[j]) for j in self._aj]
        self._frange = [tuple(float(x) for x in self.model.actuator_forcerange[i]) for i in range(nu)]
        self.joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"
                            for j in self._aj]
        lo = [float(self.model.jnt_range[j, 0]) for j in self._aj]
        hi = [float(self.model.jnt_range[j, 1]) for j in self._aj]
        self.lower = [a if b > a else -3.14159 for a, b in zip(lo, hi)]            # unlimited (0,0) -> wide bound
        self.upper = [b if b > a else 3.14159 for a, b in zip(lo, hi)]
        self.safety = SafetyFilter(self.lower, self.upper, vel_limit=vel_limit)
        self.q_default = self.read_state()["position"]

    def read_state(self) -> dict:
        """The /joint_states message: actuated-joint positions + velocities."""
        return {"position": [float(self.data.qpos[a]) for a in self._qadr],
                "velocity": [float(self.data.qvel[a]) for a in self._vadr]}

    def step(self, command_fn) -> dict:
        st = self.read_state()
        raw = list(command_fn(st))                                                # policy/controller targets
        if len(raw) != self.model.nu:
            raise ValueError(f"command_fn returned {len(raw)} targets, expected {self.model.nu}")
        target, violations = self.safety.clamp(raw, st["position"], self.dt)
        q, qd = st["position"], st["velocity"]
        for i in range(self.model.nu):
            tau = self.kp * (target[i] - q[i]) - self.kd * qd[i]                   # PD to the safe target
            lo, hi = self._frange[i]
            if hi > lo:
                tau = min(hi, max(lo, tau))                                        # actuator torque limit
            self.data.ctrl[i] = tau
        self._mj.mj_step(self.model, self.data)
        return {"target": target, "violations": violations}

    def run(self, command_fn, *, steps: int = 200) -> dict:
        """Run the closed loop for ``steps`` ticks; return a deploy summary (the sim-validated 'it runs' proof)."""
        import math
        total_viol = 0
        for _ in range(steps):
            total_viol += self.step(command_fn)["violations"]
        final = self.read_state()
        finite = all(math.isfinite(x) for x in final["position"] + final["velocity"])
        return {"steps": steps, "joint_names": list(self.joint_names), "n_joints": self.model.nu,
                "total_violations": total_viol, "final_state": final, "finite": finite}


def hold_pose_command(bridge: SimHardwareBridge):
    """A safe default 'policy': hold the standing/default pose. What a stable deploy looks like (no violations)."""
    q0 = list(bridge.q_default)
    return lambda state: q0


def run_sim_ros_demo(gene, *, command_fn=None, steps: int = 200) -> dict:
    """Convenience: instantiate the bridge for ``gene`` and run the closed loop (default = hold pose). The
    one-call "the sim drives the robot through the safe deploy loop" demonstration."""
    bridge = SimHardwareBridge(gene)
    return bridge.run(command_fn or hold_pose_command(bridge), steps=steps)
