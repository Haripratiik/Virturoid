"""Compute provenance: prove what a build actually executed.

A fast build is suspicious unless you can see whether real physics ran. This
meter wraps a block of work, counts real MuJoCo calls (physics steps, IK forward
kinematics, models compiled) and wall time, so every real run can stamp its
artifacts with auditable evidence of the simulation/optimization it performed.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass


@dataclass
class ComputeProvenance:
    physics_executed: bool = False
    physics_steps: int = 0
    ik_evaluations: int = 0
    models_simulated: int = 0
    wall_time_seconds: float = 0.0
    backend: str = "mujoco"
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class RealComputeMeter:
    """Context manager that counts real MuJoCo work within the block.

    Monkeypatches the mujoco entry points used by Virturoid for the duration of
    the block only, then restores them. If mujoco is not installed the counts
    stay zero and ``physics_executed`` is False.
    """

    def __init__(self) -> None:
        self.provenance = ComputeProvenance()
        self._start = 0.0
        self._patched = []

    def __enter__(self) -> "RealComputeMeter":
        self._start = time.perf_counter()
        try:
            import mujoco
        except Exception:  # noqa: BLE001
            return self

        p = self.provenance
        orig_step = mujoco.mj_step
        orig_kin = mujoco.mj_kinematics
        orig_from_path = mujoco.MjModel.from_xml_path
        orig_from_str = mujoco.MjModel.from_xml_string

        def step(*a, **k):
            p.physics_steps += 1
            return orig_step(*a, **k)

        def kin(*a, **k):
            p.ik_evaluations += 1
            return orig_kin(*a, **k)

        def from_path(*a, **k):
            p.models_simulated += 1
            return orig_from_path(*a, **k)

        def from_str(*a, **k):
            p.models_simulated += 1
            return orig_from_str(*a, **k)

        mujoco.mj_step = step
        mujoco.mj_kinematics = kin
        mujoco.MjModel.from_xml_path = staticmethod(from_path)
        mujoco.MjModel.from_xml_string = staticmethod(from_str)
        self._patched = [
            (mujoco, "mj_step", orig_step),
            (mujoco, "mj_kinematics", orig_kin),
            (mujoco.MjModel, "from_xml_path", staticmethod(orig_from_path)),
            (mujoco.MjModel, "from_xml_string", staticmethod(orig_from_str)),
        ]
        return self

    def __exit__(self, *exc) -> None:
        for obj, name, original in self._patched:
            setattr(obj, name, original)
        self._patched = []
        p = self.provenance
        p.wall_time_seconds = round(time.perf_counter() - self._start, 3)
        p.physics_executed = p.physics_steps > 0
        p.note = (
            f"Real physics: {p.physics_steps} MuJoCo steps, {p.ik_evaluations} IK evals, "
            f"{p.models_simulated} models, {p.wall_time_seconds}s."
            if p.physics_executed
            else "No physics executed (MuJoCo not installed or no simulation requested)."
        )
