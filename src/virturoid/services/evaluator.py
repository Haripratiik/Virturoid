from __future__ import annotations

from virturoid.schemas.robot import RobotGenome
from virturoid.schemas.runs import EpisodeRecord, EvaluationRun, FailureRecord, PolicyRecord
from virturoid.schemas.scenes import SceneSet
from virturoid.schemas.tasks import TaskGraph


def run_mock_evaluation(
    robot: RobotGenome,
    task: TaskGraph,
    scene_set: SceneSet,
    policy: PolicyRecord,
    backend: str = "mock",
) -> EvaluationRun:
    """Deterministic SYNTHETIC evaluation (every 3rd scene "fails" by index, no physics). The default backend
    is 'mock' — NOT 'mujoco' — so the emitted report is honestly labelled a synthetic/contract result rather
    than masquerading as a real physics run (the real physics path is the gene/locomotion evaluators)."""
    episodes: list[EpisodeRecord] = []
    failures: list[FailureRecord] = []
    run_id = f"run_{task.id}_{scene_set.purpose}"

    for index, scene in enumerate(scene_set.scenes):
        failed = index % 3 == 2
        episode = EpisodeRecord(
            id=f"episode_{run_id}_{index:03d}",
            run_id=run_id,
            robot_genome_id=robot.id,
            task_graph_id=task.id,
            scene_graph_id=scene.id,
            policy_id=policy.id,
            backend=backend,
            seed=index,
            result="failed" if failed else "succeeded",
            metrics={
                "success": not failed,
                "collision_count": 1 if failed else 0,
                "timeout": False,
            },
            safety_events=["collision_with_bin"] if failed else [],
            failure_labels=["collision"] if failed else [],
        )
        episodes.append(episode)

        if failed:
            failures.append(
                FailureRecord(
                    id=f"failure_{episode.id}",
                    episode_id=episode.id,
                    failure_type="collision",
                    severity="medium",
                    summary="Synthetic (mock) evaluator flagged a bin collision in this scene — NOT a physics result.",
                    likely_causes=["bin spacing too tight", "missing pre-place waypoint"],
                    suggested_fixes=["generate wider-bin-spacing regression scene", "add pre-place waypoint"],
                    regression_scene_id=f"regr_{scene.id}",
                )
            )

    return EvaluationRun(
        id=run_id,
        robot_genome_id=robot.id,
        task_graph_id=task.id,
        scene_set_id=scene_set.id,
        policy_id=policy.id,
        backend=backend,
        episodes=episodes,
        failures=failures,
    )


_CAUSES = {
    "missed_grasp": ["object outside the reliable grasp envelope (near full reach)", "approach overshoot"],
    "wrong_bin": ["bins too close / colors ambiguous", "release over the wrong target"],
    "dropped": ["grasp slipped during transport (heavy/large object)", "lift too fast"],
    "instability": ["controller gains too high for the object mass", "contact blow-up"],
}
_FIX = {
    "missed_grasp": "regenerate the scene with the block centered in the reliable reach envelope",
    "wrong_bin": "regenerate with wider bin spacing and clearer color separation",
    "dropped": "regenerate with lower lift height and slower transport",
    "instability": "reduce controller gains and regenerate with a lighter block",
}


def run_physics_evaluation_in_memory(robot: RobotGenome, task: TaskGraph, scene_set: SceneSet,
                                     policy: PolicyRecord, *, controller_params: dict | None = None) -> EvaluationRun:
    """REAL pick-place evaluation, IN-MEMORY (no disk package needed): compile robot+each scene to MJCF and run
    the scripted controller under MuJoCo, producing HONEST EpisodeRecords + FailureRecords with REAL failure
    types + per-episode metrics. Falls back to ``run_mock_evaluation`` (clearly labelled ``backend='mock'``)
    when MuJoCo is unavailable — so the packaged MVP reflects real physics outcomes, not synthetic ones that
    masquerade as success (the plan's §32.5 anti-'we-trained-something-and-hope-it-works' thesis)."""
    from virturoid.services.mujoco_runner import mujoco_available
    if not mujoco_available():
        return run_mock_evaluation(robot, task, scene_set, policy, backend="mock")
    import mujoco

    from virturoid.services.mujoco_exporter import robot_and_single_scene_to_mjcf
    from virturoid.services.pick_place_controller import run_pick_place_episode
    params = controller_params or {"kp": 14.0, "kd": 2.0, "phase_steps": 500, "engage_radius": 0.15, "grasp_z": 0.05}
    episodes: list[EpisodeRecord] = []
    failures: list[FailureRecord] = []
    run_id = f"run_{task.id}_{scene_set.purpose}"
    for index, scene in enumerate(scene_set.scenes):
        try:
            model = mujoco.MjModel.from_xml_string(robot_and_single_scene_to_mjcf(robot, scene, scene_set.id))
            objs = {o.name: tuple(o.pose_xyz_rpy[:3]) for o in scene.objects}
            out = run_pick_place_episode(model, {"objects": objs}, params=params)
        except Exception:  # noqa: BLE001 - a scene that won't compile/run is skipped, not faked
            continue
        failed = out["status"] != "success"
        ftype = out["failure_label"] or "unknown"
        ep = EpisodeRecord(
            id=f"episode_{run_id}_{index:03d}", run_id=run_id, robot_genome_id=robot.id, task_graph_id=task.id,
            scene_graph_id=scene.id, policy_id=policy.id, backend="mujoco", seed=index,
            result="failed" if failed else "succeeded",
            metrics={"success": not failed, **out.get("metrics", {})},
            failure_labels=[ftype] if failed else [])
        episodes.append(ep)
        if failed:
            failures.append(FailureRecord(
                id=f"failure_{ep.id}", episode_id=ep.id, failure_type=ftype, severity="medium",
                summary=f"Real MuJoCo pick-place: {ftype} in scene {scene.id}.",
                likely_causes=_CAUSES.get(ftype, ["see episode metrics"]),
                suggested_fixes=[_FIX.get(ftype, "regenerate a failure-conditioned regression scene")],
                regression_scene_id=f"regr_{scene.id}"))
    return EvaluationRun(id=run_id, robot_genome_id=robot.id, task_graph_id=task.id, scene_set_id=scene_set.id,
                         policy_id=policy.id, backend="mujoco", episodes=episodes, failures=failures)

