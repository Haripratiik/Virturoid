"""Simulator backend adapter registry (plan §9).

MuJoCo is the implemented physics backend. Isaac Sim / Isaac Lab and Gazebo are
real product targets but require those engines (not installable in this offline
repo), so they are exposed as declared adapter interfaces with availability
detection. This keeps the multi-backend boundary explicit: the rest of the
system targets `BackendAdapter`, and an engine becomes usable by implementing
`compile_scene` / `run_episode` without changing callers.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Protocol


class BackendAdapter(Protocol):
    name: str
    available: bool

    def status(self) -> dict:
        ...


@dataclass
class BackendStatus:
    backend: str
    adapter: str
    available: bool
    status: str
    reason: str
    capabilities: list[str] = field(default_factory=list)


class MujocoBackend:
    name = "mujoco"

    @property
    def available(self) -> bool:
        return importlib.util.find_spec("mujoco") is not None

    def status(self) -> BackendStatus:
        ok = self.available
        return BackendStatus(
            backend="mujoco",
            adapter="mujoco.physics_v0",
            available=ok,
            status="implemented" if ok else "not_installed",
            reason="Compiles MJCF, steps physics, runs pick-place/navigation, returns real rollouts."
            if ok
            else "Install the `mujoco` package.",
            capabilities=["compile_scene", "run_episode", "render_rgbd", "metrics"],
        )


class IsaacBackend:
    name = "isaac"

    @property
    def available(self) -> bool:
        return importlib.util.find_spec("omni") is not None or importlib.util.find_spec("isaaclab") is not None

    def status(self) -> BackendStatus:
        ok = self.available
        return BackendStatus(
            backend="isaac",
            adapter="isaac.sim",
            available=ok,
            status="available" if ok else "interface_only",
            reason="Isaac Sim/Lab detected." if ok else "Requires NVIDIA Isaac Sim/Lab (GPU). Interface declared; not installed here.",
            capabilities=["usd_compile", "domain_randomization", "synthetic_data", "gpu_parallel_envs"],
        )


class GazeboBackend:
    name = "gazebo"

    @property
    def available(self) -> bool:
        return importlib.util.find_spec("gazebo") is not None

    def status(self) -> BackendStatus:
        ok = self.available
        return BackendStatus(
            backend="gazebo",
            adapter="gazebo.sdf",
            available=ok,
            status="available" if ok else "interface_only",
            reason="Gazebo detected." if ok else "Requires Gazebo + ROS2. Interface declared (SDF/world export); not installed here.",
            capabilities=["sdf_export", "ros2_bridge", "world_build"],
        )


_BACKENDS = {"mujoco": MujocoBackend(), "isaac": IsaacBackend(), "gazebo": GazeboBackend()}


def backend_registry() -> dict:
    """Return the status of every known simulator backend."""
    return {name: b.status() for name, b in _BACKENDS.items()}


def available_backends() -> list[str]:
    return [name for name, b in _BACKENDS.items() if b.available]
