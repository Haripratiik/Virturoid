"""Falsifiable entry gate for morphology-conditioned generalist-control R&D.

This is deliberately not a trainer.  The product plan calls for a four-body CPU
smoke test, a simulator-throughput check, and a local-GPU requirement *before*
funding a multi-day generalist run.  Keeping that boundary executable prevents a
research aspiration from silently becoming a shipped product claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter


@dataclass(frozen=True)
class BodySmokeResult:
    index: int
    topology_key: str
    n_tokens: int
    feature_dim: int
    joint_description_dim: int
    finite: bool
    steps_per_second: float


@dataclass(frozen=True)
class GeneralistControlGateReport:
    passed: bool
    local_jax_gpu: bool
    required_bodies: int
    distinct_topologies: int
    min_steps_per_second: float
    bodies: list[BodySmokeResult] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def local_jax_gpu_available() -> bool:
    """Return true only when the training runtime can actually see a local JAX GPU."""
    try:
        import jax

        return any(getattr(device, "platform", "") == "gpu" for device in jax.devices())
    except Exception:  # noqa: BLE001 - unavailable/broken accelerator is a negative gate result
        return False


def run_generalist_control_gate(
    robots: list,
    *,
    steps: int = 400,
    required_bodies: int = 4,
    min_steps_per_second: float = 10_000.0,
    require_local_gpu: bool = True,
) -> GeneralistControlGateReport:
    """Exercise the shared token interface on structurally distinct bodies.

    ``robots`` may contain RobotGene objects or raw MJCF strings. Compilation is
    intentionally excluded from throughput: the falsifier asks whether repeated
    physics/control stepping is viable after a model has been prepared.
    """
    if steps <= 0 or required_bodies <= 0 or min_steps_per_second <= 0:
        raise ValueError("steps, required_bodies, and min_steps_per_second must be positive")

    import mujoco
    import numpy as np

    from virturoid.services.morph_graph import encode_robot
    from virturoid.services.morph_policy import robot_mjcf

    body_results: list[BodySmokeResult] = []
    for index, robot in enumerate(robots):
        model = mujoco.MjModel.from_xml_string(robot_mjcf(robot))
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        graph = encode_robot(model)
        topology = f"n{graph.n_tokens}:" + ",".join(str(parent) for parent in graph.parent)
        actions = np.zeros(graph.n_tokens, dtype=float)
        start = perf_counter()
        finite = True
        for _ in range(steps):
            observation = graph.observe(model, data)
            finite = finite and bool(np.all(np.isfinite(observation)))
            graph.apply(model, data, actions, alpha=0.0)
            mujoco.mj_step(model, data)
            if not np.all(np.isfinite(data.qpos)):
                finite = False
                break
        elapsed = max(perf_counter() - start, 1e-9)
        body_results.append(BodySmokeResult(
            index=index,
            topology_key=topology,
            n_tokens=graph.n_tokens,
            feature_dim=graph.feature_dim,
            joint_description_dim=int(graph.static.shape[1]),
            finite=finite,
            steps_per_second=round(steps / elapsed, 1),
        ))

    blockers: list[str] = []
    if len(body_results) < required_bodies:
        blockers.append(f"needs_at_least_{required_bodies}_bodies")
    distinct = len({body.topology_key for body in body_results})
    if distinct < required_bodies:
        blockers.append(f"needs_{required_bodies}_distinct_topologies")
    if len({body.feature_dim for body in body_results}) > 1:
        blockers.append("token_feature_width_varies_by_body")
    if any(body.joint_description_dim < 14 for body in body_results):
        blockers.append("joint_datasheet_is_under_14_dimensions")
    if any(body.n_tokens == 0 for body in body_results):
        blockers.append("body_has_no_actuated_joint_tokens")
    if any(not body.finite for body in body_results):
        blockers.append("non_finite_cpu_smoke_rollout")
    measured_min = min((body.steps_per_second for body in body_results), default=0.0)
    if measured_min < min_steps_per_second:
        blockers.append("simulator_throughput_below_threshold")
    gpu = local_jax_gpu_available()
    if require_local_gpu and not gpu:
        blockers.append("local_jax_gpu_unavailable")

    return GeneralistControlGateReport(
        passed=not blockers,
        local_jax_gpu=gpu,
        required_bodies=required_bodies,
        distinct_topologies=distinct,
        min_steps_per_second=round(measured_min, 1),
        bodies=body_results,
        blockers=blockers,
    )
