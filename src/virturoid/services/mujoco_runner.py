"""Real MuJoCo physics execution for generated Virturoid packages.

This module is the first non-synthetic simulator backend. It loads the compiled
MJCF scene XML produced by ``mujoco_exporter``, steps real physics under a
deterministic PD controller, and emits real rollout traces and physics-grounded
metrics. It implements the same ``SimulatorAdapter`` protocol as the dry-run
adapter, so it can replace the synthetic runner without reshaping the package.

``mujoco`` is imported lazily so the rest of the codebase (and the stdlib-only
test suite) keeps working when the backend is not installed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from virturoid.schemas.physics_result import (
    PhysicsEpisodeResult,
    PhysicsGroupResult,
    PhysicsRunResult,
)
from virturoid.schemas.training_result import TrainingEpisodeTrace
from virturoid.services.simulator_adapter import EpisodeBatchRequest


def mujoco_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("mujoco") is not None


class MujocoSimulatorAdapter:
    """Steps real MuJoCo physics for a batch of episodes.

    The controller is a deterministic per-joint PD law driving each actuated
    joint toward a seeded target within its range. This is a baseline scripted
    controller, not a learned policy, but every metric it reports comes from
    stepping a real model.
    """

    name = "mujoco.physics_v0"

    def __init__(
        self,
        package_dir: Path,
        rollout_dir: str = "runs/mvp_physics/rollouts",
        control_horizon: int = 600,
        kp: float = 3.0,
        kd: float = 0.6,
        torque_clamp: float = 2.0,
        record_every: int = 20,
    ) -> None:
        self.package_dir = Path(package_dir)
        self.rollout_dir = rollout_dir
        self.control_horizon = control_horizon
        self.kp = kp
        self.kd = kd
        self.torque_clamp = torque_clamp
        self.record_every = record_every
        self._model_cache: dict[str, object] = {}

    def run_episode_batch(self, request: EpisodeBatchRequest) -> list[TrainingEpisodeTrace]:
        traces: list[TrainingEpisodeTrace] = []
        for local_index in range(request.planned_episodes):
            scene = request.scenes[local_index % len(request.scenes)]
            scene_id = scene["id"]
            global_index = request.start_index + local_index
            xml_uri = request.compiled_scene_xml_by_scene_id.get(scene_id, "")
            episode_id = _episode_id(request.purpose, global_index)
            if not xml_uri:
                traces.append(
                    _error_trace(request, episode_id, scene_id, "missing_compiled_scene_xml")
                )
                continue
            result = self._rollout(request, episode_id, scene_id, xml_uri, seed=global_index)
            traces.append(
                TrainingEpisodeTrace(
                    episode_id=result.episode_id,
                    backend=request.backend,
                    adapter_name=self.name,
                    purpose=request.purpose,
                    scene_id=result.scene_id,
                    scene_artifact=request.scene_artifact,
                    policy_id=request.policy_id,
                    status=result.status,
                    reward=result.reward,
                    success_probability=result.success_probability,
                    compiled_scene_xml_uri=xml_uri,
                    observation_keys=list(request.observation_keys),
                    action_trace=list(request.action_trace),
                    notes=[
                        "Real MuJoCo physics rollout.",
                        f"{result.steps_simulated} steps over {result.sim_seconds:.3f} sim seconds.",
                        *([f"safety: {', '.join(result.safety_events)}"] if result.safety_events else []),
                    ],
                    metrics=result.metrics,
                    rollout_uri=result.rollout_uri,
                )
            )
        return traces

    # The detailed result objects are returned separately for the run summary.
    def rollout_results(self, request: EpisodeBatchRequest) -> list[PhysicsEpisodeResult]:
        results: list[PhysicsEpisodeResult] = []
        for local_index in range(request.planned_episodes):
            scene = request.scenes[local_index % len(request.scenes)]
            scene_id = scene["id"]
            global_index = request.start_index + local_index
            xml_uri = request.compiled_scene_xml_by_scene_id.get(scene_id, "")
            episode_id = _episode_id(request.purpose, global_index)
            if not xml_uri:
                results.append(
                    PhysicsEpisodeResult(
                        episode_id=episode_id,
                        purpose=request.purpose,
                        scene_id=scene_id,
                        compiled_scene_xml_uri="",
                        seed=global_index,
                        status="error",
                        reward=0.0,
                        success_probability=0.0,
                        steps_simulated=0,
                        sim_seconds=0.0,
                        safety_events=["missing_compiled_scene_xml"],
                    )
                )
                continue
            results.append(self._rollout(request, episode_id, scene_id, xml_uri, seed=global_index))
        return results

    def _load_model(self, xml_uri: str):
        import mujoco

        if xml_uri not in self._model_cache:
            xml_path = self.package_dir / xml_uri
            self._model_cache[xml_uri] = mujoco.MjModel.from_xml_path(str(xml_path))
        return self._model_cache[xml_uri]

    def _rollout(
        self,
        request: EpisodeBatchRequest,
        episode_id: str,
        scene_id: str,
        xml_uri: str,
        seed: int,
    ) -> PhysicsEpisodeResult:
        import mujoco
        import numpy as np

        model = self._load_model(xml_uri)
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)

        rng = np.random.default_rng(seed)
        actuated = _actuated_joint_map(model)
        clamps = _actuator_force_clamps(model)
        targets = _seeded_targets(actuated, rng)

        ee_site = _ee_site_id(model)
        block_geoms = _named_geoms(model, "block")

        max_joint_speed = 0.0
        max_contacts = 0
        limit_violation_steps = 0
        ee_path_length = 0.0
        prev_ee = _ee_position(data, ee_site)
        stable = True
        steps = 0
        samples: list[dict] = []

        for step in range(self.control_horizon):
            for slot, (qadr, vadr, jnt) in enumerate(actuated):
                error = targets[slot] - data.qpos[qadr]
                torque = self.kp * error - self.kd * data.qvel[vadr]
                clamp = min(self.torque_clamp, clamps[slot])
                data.ctrl[slot] = float(max(-clamp, min(clamp, torque)))

            mujoco.mj_step(model, data)
            steps += 1

            if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
                stable = False
                break

            speed = float(np.max(np.abs(data.qvel))) if data.qvel.size else 0.0
            max_joint_speed = max(max_joint_speed, speed)
            max_contacts = max(max_contacts, int(data.ncon))
            ee_now = _ee_position(data, ee_site)
            ee_path_length += float(np.linalg.norm(ee_now - prev_ee))
            prev_ee = ee_now

            for (qadr, vadr, jnt) in actuated:
                if model.jnt_limited[jnt]:
                    low, high = model.jnt_range[jnt]
                    if data.qpos[qadr] < low - 1e-3 or data.qpos[qadr] > high + 1e-3:
                        limit_violation_steps += 1
                        break

            if step % self.record_every == 0:
                samples.append(
                    {
                        "t": round(float(data.time), 4),
                        "qpos": [round(float(v), 5) for v in data.qpos],
                        "qvel_max": round(speed, 5),
                        "ctrl": [round(float(v), 5) for v in data.ctrl],
                        "ee_pos": [round(float(v), 5) for v in ee_now],
                        "ncon": int(data.ncon),
                    }
                )

        if max_joint_speed > 50.0:
            stable = False

        final_tracking_error = 0.0
        if actuated:
            final_tracking_error = float(
                np.mean([abs(targets[i] - data.qpos[adr]) for i, (adr, _, _) in enumerate(actuated)])
            )
        ee_block_distance = _nearest_block_distance(data, ee_site, block_geoms)

        safety_events: list[str] = []
        if not stable:
            safety_events.append("instability")
        if limit_violation_steps:
            safety_events.append("joint_limit_violation")

        reward = _score(stable, final_tracking_error, ee_block_distance)
        success_probability = round(reward / 100.0, 3)
        if not stable:
            status = "failure"
        elif reward >= 70.0:
            status = "success"
        else:
            status = "needs_review"

        rollout_uri = f"{self.rollout_dir}/{episode_id}.jsonl"
        self._write_rollout(rollout_uri, samples)

        metrics: dict[str, float | int | bool] = {
            "steps_simulated": steps,
            "sim_seconds": round(float(data.time), 4),
            "max_joint_speed_rad_s": round(max_joint_speed, 4),
            "final_tracking_error_rad": round(final_tracking_error, 4),
            "ee_path_length_m": round(ee_path_length, 4),
            "ee_nearest_block_distance_m": round(ee_block_distance, 4) if ee_block_distance is not None else -1.0,
            "max_contacts": max_contacts,
            "joint_limit_violation_steps": limit_violation_steps,
            "stable": stable,
        }

        return PhysicsEpisodeResult(
            episode_id=episode_id,
            purpose=request.purpose,
            scene_id=scene_id,
            compiled_scene_xml_uri=xml_uri,
            seed=seed,
            status=status,
            reward=reward,
            success_probability=success_probability,
            steps_simulated=steps,
            sim_seconds=round(float(data.time), 4),
            metrics=metrics,
            safety_events=safety_events,
            rollout_uri=rollout_uri,
        )

    def _write_rollout(self, rollout_uri: str, samples: list[dict]) -> None:
        path = self.package_dir / rollout_uri
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(sample, sort_keys=True) for sample in samples]
        path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def run_physics_rollout_from_export(
    package_dir: Path,
    config_uri: str = "training/training_run_config.json",
    output_subdir: str = "runs/mvp_physics",
    episodes_per_scene: int = 1,
    control_horizon: int = 600,
) -> PhysicsRunResult:
    """Execute real MuJoCo rollouts for an exported package and write artifacts.

    Writes ``<output_subdir>/physics_run_result.json``, an episode trace, and one
    rollout file per episode under ``<output_subdir>/rollouts/``.
    """
    if not mujoco_available():
        raise RuntimeError(
            "MuJoCo Python package is not installed; run `pip install mujoco` to execute physics rollouts."
        )

    package_dir = Path(package_dir)
    config = json.loads((package_dir / config_uri).read_text(encoding="utf-8"))
    compiled_index = json.loads(
        (package_dir / config["compiled_scene_artifact"]["uri"]).read_text(encoding="utf-8")
    )
    compiled_scene_xml_by_scene_id = {
        entry["scene_id"]: entry["mujoco_xml"] for entry in compiled_index.get("scenes", [])
    }
    policy_plan = _load_optional(package_dir, config.get("policy_artifact"))
    action_trace = [step["command"] for step in policy_plan.get("steps", [])] if policy_plan else []
    observation_keys = policy_plan.get("observation_keys", []) if policy_plan else []

    rollout_dir = f"{output_subdir}/rollouts"
    adapter = MujocoSimulatorAdapter(
        package_dir, rollout_dir=rollout_dir, control_horizon=control_horizon
    )

    group_results: list[PhysicsGroupResult] = []
    episodes: list[PhysicsEpisodeResult] = []
    timestep_s = 0.0
    robot_model_uri = config["robot_model"]["uri"]

    for scene_artifact in config["scene_artifacts"]:
        uri = scene_artifact["uri"]
        scene_set = json.loads((package_dir / uri).read_text(encoding="utf-8"))
        purpose = scene_set["purpose"]
        scenes = scene_set["scenes"]
        planned = max(1, len(scenes) * episodes_per_scene)
        request = EpisodeBatchRequest(
            backend=config["backend"],
            adapter_name=adapter.name,
            purpose=purpose,
            scene_artifact=uri,
            scenes=scenes,
            planned_episodes=planned,
            policy_id=config["policy_id"],
            start_index=len(episodes) + 1,
            target_success_rate=0.0,
            observation_keys=observation_keys,
            action_trace=action_trace,
            compiled_scene_xml_by_scene_id=compiled_scene_xml_by_scene_id,
        )
        group_episodes = adapter.rollout_results(request)
        episodes.extend(group_episodes)
        successes = sum(1 for episode in group_episodes if episode.status == "success")
        group_results.append(
            PhysicsGroupResult(
                purpose=purpose,
                scene_artifact=uri,
                scene_count=len(scenes),
                episodes=len(group_episodes),
                success_rate=round(successes / len(group_episodes), 3) if group_episodes else 0.0,
            )
        )

    if adapter._model_cache:
        any_model = next(iter(adapter._model_cache.values()))
        timestep_s = float(any_model.opt.timestep)

    total_episodes = len(episodes)
    measured_success = sum(1 for episode in episodes if episode.status == "success")
    stable_episodes = sum(1 for episode in episodes if episode.metrics.get("stable"))
    episode_trace_uri = f"{output_subdir}/logs/episode_trace.jsonl"

    result = PhysicsRunResult(
        id=f"physics_{config['id']}",
        training_run_config_id=config["id"],
        backend=config["backend"],
        adapter_name=adapter.name,
        robot_model_uri=robot_model_uri,
        compiled_scene_index_uri=config["compiled_scene_artifact"]["uri"],
        control_horizon=control_horizon,
        timestep_s=round(timestep_s, 6),
        total_scenes=sum(group.scene_count for group in group_results),
        total_episodes=total_episodes,
        measured_success_rate=round(measured_success / total_episodes, 3) if total_episodes else 0.0,
        stable_episode_rate=round(stable_episodes / total_episodes, 3) if total_episodes else 0.0,
        episode_trace_uri=episode_trace_uri,
        rollout_dir=rollout_dir,
        group_results=group_results,
        episodes=episodes,
        notes=[
            "Real MuJoCo physics: each episode stepped a compiled MJCF model.",
            "Controller is a deterministic per-joint PD baseline (not a learned policy).",
            "Success requires a stable rollout that tracks commanded joint targets and approaches a target block.",
            f"Per-episode rollouts (qpos/qvel/ctrl/ee) under {rollout_dir}/.",
        ],
    )
    validation = result.validate()
    if not validation.ok:
        issues = ", ".join(issue.code for issue in validation.issues)
        raise ValueError(f"{result.id} failed validation: {issues}")

    _write_episode_trace(package_dir, episode_trace_uri, episodes, result.backend, adapter.name)
    result_path = package_dir / output_subdir / "physics_run_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result


def _write_episode_trace(
    package_dir: Path,
    uri: str,
    episodes: list[PhysicsEpisodeResult],
    backend: str,
    adapter_name: str,
) -> None:
    path = package_dir / uri
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for episode in episodes:
        lines.append(
            json.dumps(
                {
                    "episode_id": episode.episode_id,
                    "backend": backend,
                    "adapter_name": adapter_name,
                    "purpose": episode.purpose,
                    "scene_id": episode.scene_id,
                    "compiled_scene_xml_uri": episode.compiled_scene_xml_uri,
                    "seed": episode.seed,
                    "status": episode.status,
                    "reward": episode.reward,
                    "success_probability": episode.success_probability,
                    "metrics": episode.metrics,
                    "safety_events": episode.safety_events,
                    "rollout_uri": episode.rollout_uri,
                },
                sort_keys=True,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_optional(package_dir: Path, artifact: dict | None) -> dict:
    if not artifact:
        return {}
    return json.loads((package_dir / artifact["uri"]).read_text(encoding="utf-8"))


def _actuated_joint_map(model) -> list[tuple[int, int, int]]:
    """Return (qpos_addr, dof_addr, joint_id) for each actuator's target joint."""
    mapping: list[tuple[int, int, int]] = []
    for actuator in range(model.nu):
        jnt = int(model.actuator_trnid[actuator, 0])
        mapping.append((int(model.jnt_qposadr[jnt]), int(model.jnt_dofadr[jnt]), jnt))
    return mapping


def _actuator_force_clamps(model, default: float = 8.0) -> list[float]:
    """Per-actuator torque clamp taken from the MJCF force range (the part's effort)."""
    clamps: list[float] = []
    for i in range(model.nu):
        limited = bool(model.actuator_forcelimited[i]) if hasattr(model, "actuator_forcelimited") else False
        upper = float(model.actuator_forcerange[i][1])
        clamps.append(upper if (limited and upper > 0) else default)
    return clamps


def _seeded_targets(actuated: list[tuple[int, int, int]], rng) -> list[float]:
    targets: list[float] = []
    for (_, _, jnt) in actuated:
        targets.append(float(rng.uniform(-0.5, 0.5)))
    return targets


def _named_geoms(model, needle: str) -> list[int]:
    import mujoco

    found: list[int] = []
    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
        if name and needle in name:
            found.append(geom)
    return found


def reach_rollout(
    model,
    target_angles,
    *,
    horizon: int = 250,
    kp: float = 8.0,
    kd: float = 1.2,
    record_every: int | None = None,
):
    """Step a PD-controlled rollout toward fixed joint targets.

    Shared by the physics adapter and the policy trainer. Returns the final
    end-effector position, nearest block distance, stability, and peak speed.
    When ``record_every`` is set, also returns downsampled per-step samples.
    """
    import mujoco
    import numpy as np

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    actuated = _actuated_joint_map(model)
    clamps = _actuator_force_clamps(model)
    ee_site = _ee_site_id(model)
    block_geoms = _named_geoms(model, "block")
    targets = list(target_angles)

    max_speed = 0.0
    stable = True
    samples: list[dict] = []
    for step in range(horizon):
        for slot, (qadr, vadr, jnt) in enumerate(actuated):
            target = targets[slot] if slot < len(targets) else 0.0
            if model.jnt_limited[jnt]:
                low, high = model.jnt_range[jnt]
                target = float(min(max(target, low), high))
            torque = kp * (target - data.qpos[qadr]) - kd * data.qvel[vadr]
            clamp = clamps[slot]
            data.ctrl[slot] = float(max(-clamp, min(clamp, torque)))
        mujoco.mj_step(model, data)
        if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
            stable = False
            break
        max_speed = max(max_speed, float(np.max(np.abs(data.qvel))) if data.qvel.size else 0.0)
        if record_every and step % record_every == 0:
            samples.append(
                {
                    "t": round(float(data.time), 4),
                    "qpos": [round(float(v), 5) for v in data.qpos],
                    "ctrl": [round(float(v), 5) for v in data.ctrl],
                    "ee_pos": [round(float(v), 5) for v in _ee_position(data, ee_site)],
                }
            )

    if max_speed > 50.0:
        stable = False
    ee_pos = _ee_position(data, ee_site)
    result = {
        "ee_pos": [float(v) for v in ee_pos],
        "nearest_block_distance_m": _nearest_block_distance(data, ee_site, block_geoms),
        "stable": stable,
        "max_joint_speed_rad_s": round(max_speed, 4),
    }
    if record_every:
        result["samples"] = samples
    return result


def _ee_site_id(model) -> int:
    import mujoco

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    return site_id  # -1 if the model has no ee_site


def _ee_position(data, ee_site: int):
    import numpy as np

    if ee_site >= 0:
        return np.array(data.site_xpos[ee_site], dtype=float)
    # Fall back to the last body origin if no end-effector site is defined.
    return np.array(data.xpos[-1], dtype=float)


def _nearest_block_distance(data, ee_site: int, block_geoms: list[int]):
    import numpy as np

    if not block_geoms:
        return None
    ee = _ee_position(data, ee_site)
    return float(min(np.linalg.norm(np.array(data.geom_xpos[g], dtype=float) - ee) for g in block_geoms))


def _score(stable: bool, tracking_error: float, ee_block_distance) -> float:
    if not stable:
        return 0.0
    reward = 100.0
    reward -= min(40.0, tracking_error * 40.0)
    if ee_block_distance is not None:
        reward -= min(30.0, ee_block_distance * 30.0)
    return round(max(0.0, min(100.0, reward)), 2)


def _episode_id(purpose: str, index: int) -> str:
    return f"episode_{purpose}_{index:04d}"


def _error_trace(
    request: EpisodeBatchRequest, episode_id: str, scene_id: str, reason: str
) -> TrainingEpisodeTrace:
    return TrainingEpisodeTrace(
        episode_id=episode_id,
        backend=request.backend,
        adapter_name=MujocoSimulatorAdapter.name,
        purpose=request.purpose,
        scene_id=scene_id,
        scene_artifact=request.scene_artifact,
        policy_id=request.policy_id,
        status="error",
        reward=0.0,
        success_probability=0.0,
        notes=[f"Could not run physics rollout: {reason}."],
        metrics={"stable": False},
    )
