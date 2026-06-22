"""Train a reach controller against real MuJoCo rollouts.

This is the first real training loop. It optimizes a small linear policy that
maps the observed target-block position to joint position targets, using the
cross-entropy method (CEM) over real physics rollouts. No heavy ML dependency is
needed: the policy is tiny and learning is by evolutionary search over rewards
that come from stepping the generated MJCF models.

The trainer is deliberately scoped to a reach controller (move the end-effector
to the target block) rather than full grasping, which keeps the reward honest
and measurable while exercising the genome -> observation/action -> policy ->
evaluation -> artifact path end to end.
"""

from __future__ import annotations

import json
from pathlib import Path

from virturoid.schemas.trained_policy import (
    ActionDimension,
    PolicyEvalMetrics,
    PolicyTrainingMetrics,
    TrainedPolicy,
)
from virturoid.services.mujoco_runner import mujoco_available, reach_rollout

SUCCESS_THRESHOLD_M = 0.12


def train_arm_controller_from_export(
    package_dir: Path,
    training_scene_uri: str = "simulation/scene_set.json",
    eval_scene_uri: str = "simulation/holdout_scene_set.json",
    output_subdir: str = "runs/mvp_policy_eval",
    iterations: int = 14,
    candidates: int = 24,
    elite_fraction: float = 0.3,
    horizon: int = 320,
    seed: int = 7,
) -> TrainedPolicy:
    """Train and evaluate a reach controller; write the policy and eval episodes."""
    if not mujoco_available():
        raise RuntimeError("MuJoCo is not installed; run `pip install mujoco` to train a controller.")

    import mujoco
    import numpy as np

    package_dir = Path(package_dir)
    genome = json.loads((package_dir / "robot" / "robot_genome.json").read_text(encoding="utf-8"))
    compiled_index = json.loads(
        (package_dir / "simulation" / "mujoco" / "compiled_scene_index.json").read_text(encoding="utf-8")
    )
    xml_by_scene = {entry["scene_id"]: entry["mujoco_xml"] for entry in compiled_index.get("scenes", [])}

    joints = genome["joints"]
    joint_names = [joint["name"] for joint in joints]
    n_act = len(joint_names)

    train_scenes = _load_scene_targets(package_dir, training_scene_uri)
    eval_scenes = _load_scene_targets(package_dir, eval_scene_uri)

    model_cache: dict[str, object] = {}

    def get_model(scene_id: str):
        uri = xml_by_scene[scene_id]
        if uri not in model_cache:
            model_cache[uri] = mujoco.MjModel.from_xml_path(str(package_dir / uri))
        return model_cache[uri]

    ranges = _joint_ranges(joints)

    def policy_targets(weights, target_xy):
        features = np.array([target_xy[0], target_xy[1], 1.0])
        raw = weights @ features
        return [float(np.clip(raw[i], ranges[i][0], ranges[i][1])) for i in range(n_act)]

    def evaluate(weights, scenes, record_every=None):
        results = []
        for scene in scenes:
            if scene["scene_id"] not in xml_by_scene:
                continue
            model = get_model(scene["scene_id"])
            targets = policy_targets(weights, scene["target_xy"])
            rollout = reach_rollout(model, targets, horizon=horizon, record_every=record_every)
            ee = np.array(rollout["ee_pos"])
            block = np.array(scene["target_xyz"])
            distance = float(np.linalg.norm(ee - block))
            reward = -distance - (1.0 if not rollout["stable"] else 0.0)
            results.append({"scene": scene, "distance": distance, "reward": reward, "rollout": rollout, "targets": targets})
        return results

    rng = np.random.default_rng(seed)
    dim = (n_act, 3)
    mean = np.zeros(dim)
    std = np.full(dim, 1.0)
    n_elite = max(2, int(candidates * elite_fraction))

    def mean_reward(weights, scenes):
        results = evaluate(weights, scenes)
        return float(np.mean([r["reward"] for r in results])) if results else -1e9

    initial_reward = mean_reward(mean, train_scenes)
    reward_history: list[float] = []
    best_weights = mean.copy()
    best_reward = initial_reward
    for _ in range(iterations):
        samples = rng.normal(loc=mean, scale=std, size=(candidates, *dim))
        scored = sorted(
            ((mean_reward(w, train_scenes), w) for w in samples),
            key=lambda item: item[0],
            reverse=True,
        )
        elites = np.array([w for _, w in scored[:n_elite]])
        mean = elites.mean(axis=0)
        std = elites.std(axis=0) + 1e-3
        iteration_best = scored[0][0]
        reward_history.append(round(iteration_best, 4))
        if iteration_best > best_reward:
            best_reward = iteration_best
            best_weights = scored[0][1].copy()

    # Evaluate the best controller on held-out scenes and record replayable episodes.
    eval_results = evaluate(best_weights, eval_scenes, record_every=20)
    distances = [r["distance"] for r in eval_results]
    successes = sum(1 for d in distances if d < SUCCESS_THRESHOLD_M)
    stable = sum(1 for r in eval_results if r["rollout"]["stable"])
    episode_trace_uri = f"{output_subdir}/logs/episode_trace.jsonl"
    _write_eval_episodes(package_dir, output_subdir, episode_trace_uri, eval_results)

    policy = TrainedPolicy(
        id=f"trained_{genome['id']}",
        name="Learned linear reach controller",
        robot_genome_id=genome["id"],
        task_graph_id=compiled_index.get("robot_genome_id", ""),
        observation_space=["joint_positions", "joint_velocities", "target_block_xy"],
        joint_names=joint_names,
        action_dimensions=[
            ActionDimension(
                name=joint["name"],
                lower=ranges[i][0],
                upper=ranges[i][1],
                velocity_limit=(joint.get("limit") or {}).get("velocity"),
                effort_limit=(joint.get("limit") or {}).get("effort"),
            )
            for i, joint in enumerate(joints)
        ],
        control_frequency_hz=20.0,
        low_level_control_frequency_hz=1.0 / 0.002,
        weights=[[round(float(v), 6) for v in row] for row in best_weights],
        input_features=["target_block_x", "target_block_y", "bias"],
        safety_clamps={
            "joint_position_limits": [list(r) for r in ranges],
            "max_joint_velocity_rad_s": [(j.get("limit") or {}).get("velocity") for j in joints],
            "max_torque_nm": [(j.get("limit") or {}).get("effort") for j in joints],
        },
        training=PolicyTrainingMetrics(
            method="cross_entropy_method",
            iterations=iterations,
            candidates_per_iteration=candidates,
            training_scene_count=len(train_scenes),
            reward_history=reward_history,
            best_reward=round(best_reward, 4),
            initial_reward=round(initial_reward, 4),
        ),
        evaluation=PolicyEvalMetrics(
            eval_scene_count=len(eval_results),
            mean_reach_distance_m=round(sum(distances) / len(distances), 4) if distances else 0.0,
            best_reach_distance_m=round(min(distances), 4) if distances else 0.0,
            success_rate=round(successes / len(eval_results), 3) if eval_results else 0.0,
            stable_rate=round(stable / len(eval_results), 3) if eval_results else 0.0,
            success_threshold_m=SUCCESS_THRESHOLD_M,
        ),
        notes=[
            "Trained with CEM over real MuJoCo reach rollouts.",
            "Reward = negative end-effector-to-target-block distance, with an instability penalty.",
            f"Replayable evaluation episodes under {output_subdir}/.",
        ],
    )
    validation = policy.validate()
    if not validation.ok:
        issues = ", ".join(issue.code for issue in validation.issues)
        raise ValueError(f"{policy.id} failed validation: {issues}")

    policy_path = package_dir / "software" / "controller" / "trained_policy.json"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(json.dumps(policy.to_dict(), indent=2), encoding="utf-8")

    metrics_path = package_dir / "training" / "policy_training_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "policy_id": policy.id,
                "method": policy.training.method,
                "initial_reward": policy.training.initial_reward,
                "best_reward": policy.training.best_reward,
                "reward_history": policy.training.reward_history,
                "eval": policy.to_dict()["evaluation"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return policy


def _load_scene_targets(package_dir: Path, scene_uri: str) -> list[dict]:
    data = json.loads((package_dir / scene_uri).read_text(encoding="utf-8"))
    scenes = []
    for scene in data["scenes"]:
        target = _target_block(scene["objects"])
        if target is None:
            continue
        x, y = target[0], target[1]
        scenes.append(
            {
                "scene_id": scene["id"],
                "target_xy": (x, y),
                "target_xyz": (x, y, 0.05),
            }
        )
    return scenes


def _target_block(objects: list[dict]):
    for item in objects:
        if item["name"] == "red_block":
            return item["pose_xyz_rpy"]
    for item in objects:
        if item.get("object_type") in {"cube", "box"}:
            return item["pose_xyz_rpy"]
    return None


def _joint_ranges(joints: list[dict]) -> list[tuple[float, float]]:
    ranges = []
    for joint in joints:
        limit = joint.get("limit") or {}
        lower = limit.get("lower")
        upper = limit.get("upper")
        ranges.append((lower if lower is not None else -3.14, upper if upper is not None else 3.14))
    return ranges


def _write_eval_episodes(
    package_dir: Path,
    output_subdir: str,
    episode_trace_uri: str,
    eval_results: list[dict],
) -> None:
    rollout_dir = package_dir / output_subdir / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for index, result in enumerate(eval_results, start=1):
        episode_id = f"episode_eval_{index:04d}"
        rollout_uri = f"{output_subdir}/rollouts/{episode_id}.jsonl"
        samples = result["rollout"].get("samples", [])
        (package_dir / rollout_uri).write_text(
            ("\n".join(json.dumps(s, sort_keys=True) for s in samples) + "\n") if samples else "",
            encoding="utf-8",
        )
        lines.append(
            json.dumps(
                {
                    "episode_id": episode_id,
                    "scene_id": result["scene"]["scene_id"],
                    "adapter_name": "mujoco.physics_v0",
                    "status": "success" if result["distance"] < SUCCESS_THRESHOLD_M else "needs_review",
                    "reach_distance_m": round(result["distance"], 4),
                    "stable": result["rollout"]["stable"],
                    "joint_targets": [round(t, 5) for t in result["targets"]],
                    "rollout_uri": rollout_uri,
                },
                sort_keys=True,
            )
        )
    trace_path = package_dir / episode_trace_uri
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
