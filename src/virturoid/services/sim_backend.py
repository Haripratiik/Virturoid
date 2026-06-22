"""Backend-agnostic simulation boundary (startup plan §9/§29; roadmap ADR-2, §9, milestone M-N9).

The product must never let callers reach a simulator directly — everything compiles through a strict
``SimulationBackend`` so MuJoCo (today) and Isaac/Gazebo (later) slot in without rewrites. This module
*defines that boundary* and provides a MuJoCo adapter that WRAPS the existing, working compiler/runner
functions. It is purely additive: current callers are unchanged; new code can depend on the interface.

Registering an Isaac adapter (USD / Isaac Lab env config) or a Gazebo adapter (SDF / ROS2) later means
implementing this same Protocol and adding it to ``_BACKENDS`` — the rest of the platform is unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# Geoms MJX-JAX cannot collide against box/mesh (advisory in validate_robot; see phase3_rl_mjx.md §0).
_MJX_UNSAFE_SHAPES = ("cylinder", "ellipsoid")


@dataclass
class BackendValidation:
    ok: bool
    backend: str
    issues: list[str] = field(default_factory=list)     # hard problems (gene invalid for this backend)
    warnings: list[str] = field(default_factory=list)   # advisories (compiles, but caveats)


@runtime_checkable
class SimulationBackend(Protocol):
    """The strict boundary every simulator adapter implements (startup plan §29.1)."""
    name: str

    def available(self) -> bool:
        """Is this backend usable in the current environment (deps installed)?"""

    def validate_robot(self, gene) -> BackendValidation:
        """Can this backend represent the gene? Issues = hard, warnings = advisory."""

    def compile_robot(self, gene, *, include_floor: bool = True) -> str:
        """Compile the gene to this backend's robot description (e.g. MJCF / USD / SDF)."""

    def compile_scene(self, gene, scene_objects) -> str:
        """Compile gene + scene objects to this backend's world description."""

    def evaluate(self, gene, spec, scenes, *, params=None) -> dict:
        """Run the task on the gene and return an evaluation result (success_rate, episodes, ...)."""


class MujocoBackend:
    """MuJoCo / MJX adapter — wraps the existing gene compiler + task runtime (no behavior change)."""

    name = "mujoco"

    def available(self) -> bool:
        try:
            from virturoid.services.mujoco_runner import mujoco_available
            return bool(mujoco_available())
        except Exception:  # noqa: BLE001
            return False

    def validate_robot(self, gene) -> BackendValidation:
        issues = list(gene.validate())
        warnings: list[str] = []
        if any(getattr(s, "shape", None) in _MJX_UNSAFE_SHAPES for s in getattr(gene, "segments", [])):
            warnings.append("cylinder/ellipsoid colliders are unsupported vs box/mesh in MJX-JAX; emit them "
                            "visual-only with a capsule/box collider, or route to the MJX-Warp backend.")
        if not self.available():
            warnings.append("mujoco not installed — compile works, but episodes can't run here.")
        return BackendValidation(ok=not issues, backend=self.name, issues=issues, warnings=warnings)

    def compile_robot(self, gene, *, include_floor: bool = True) -> str:
        from virturoid.services.gene_compiler import compile_gene_to_mjcf
        return compile_gene_to_mjcf(gene, include_floor=include_floor)

    def compile_scene(self, gene, scene_objects) -> str:
        from virturoid.services.gene_compiler import compile_gene_with_scene
        return compile_gene_with_scene(gene, scene_objects)

    def evaluate(self, gene, spec, scenes, *, params=None) -> dict:
        from virturoid.services.task_runtime import evaluate_gene_on_task
        return evaluate_gene_on_task(gene, spec, scenes, params=params)


# Adapter registry. Isaac (USD/Isaac Lab) and Gazebo (SDF/ROS2) register here when implemented (ADR-2).
_BACKENDS: dict[str, type] = {"mujoco": MujocoBackend}


def get_backend(name: str = "mujoco") -> SimulationBackend:
    if name not in _BACKENDS:
        raise KeyError(f"unknown backend {name!r}; available: {sorted(_BACKENDS)} "
                       "(isaac/gazebo adapters not yet implemented — see roadmap §9).")
    return _BACKENDS[name]()


def available_backends() -> dict[str, bool]:
    """Map of backend name -> usable-in-this-environment (for a UI/CLI backend picker)."""
    out = {}
    for n, cls in _BACKENDS.items():
        try:
            out[n] = bool(cls().available())
        except Exception:  # noqa: BLE001
            out[n] = False
    return out
