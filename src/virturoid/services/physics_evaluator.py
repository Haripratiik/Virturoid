"""Real pick-and-place evaluation across generated scenes (closes the MVP loop).

Runs the scripted pick-and-place controller on every scene in the chosen scene
sets under real MuJoCo physics, aggregates task-grounded success/failure
outcomes, clusters the failures, and proposes a regression focus for each
cluster. This is the honest replacement for the synthetic mock evaluation.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

from virturoid.schemas.physics_evaluation import FailureCluster, PhysicsEvaluationReport, PickPlaceEpisode
from virturoid.services.mujoco_runner import mujoco_available
from virturoid.services.pick_place_controller import run_pick_place_episode

_REGRESSION_FOR = {
    "missed_grasp": "regenerate scenes with the block nearer the reachable workspace center",
    "wrong_bin": "regenerate scenes with wider bin spacing and clearer color separation",
    "dropped": "regenerate scenes with lower lift height and slower transport",
    "instability": "reduce controller gains and regenerate with lighter blocks",
}

_DEFAULT_SCENE_URIS = [
    "simulation/scene_set.json",
    "simulation/holdout_scene_set.json",
]


def run_physics_pick_place_evaluation(
    package_dir: Path,
    scene_uris: list[str] | None = None,
    output_subdir: str = "runs/mvp_pick_place_eval",
    max_scenes: int | None = None,
    controller_params: dict | None = None,
    perception=None,
) -> PhysicsEvaluationReport:
    if not mujoco_available():
        raise RuntimeError("MuJoCo is not installed; run `pip install mujoco` to evaluate pick-and-place.")

    import mujoco

    package_dir = Path(package_dir)
    genome = json.loads((package_dir / "robot" / "robot_genome.json").read_text(encoding="utf-8"))
    compiled_index = json.loads(
        (package_dir / "simulation" / "mujoco" / "compiled_scene_index.json").read_text(encoding="utf-8")
    )
    xml_by_scene = {entry["scene_id"]: entry["mujoco_xml"] for entry in compiled_index.get("scenes", [])}
    scene_uris = scene_uris or _DEFAULT_SCENE_URIS

    episodes: list[PickPlaceEpisode] = []
    placed = blocks_total = 0
    purpose_totals: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])  # [success, total]
    cluster_examples: dict[str, list[str]] = collections.defaultdict(list)
    index = 0

    for uri in scene_uris:
        path = package_dir / uri
        if not path.exists():
            continue
        scene_set = json.loads(path.read_text(encoding="utf-8"))
        purpose = scene_set["purpose"]
        scenes = scene_set["scenes"]
        if max_scenes is not None:
            scenes = scenes[:max_scenes]
        for scene in scenes:
            xml_uri = xml_by_scene.get(scene["id"])
            if not xml_uri:
                continue
            model = mujoco.MjModel.from_xml_path(str(package_dir / xml_uri))
            objects = {o["name"]: tuple(o["pose_xyz_rpy"][:3]) for o in scene["objects"]}
            outcome = run_pick_place_episode(model, {"objects": objects}, params=controller_params, perception=perception)
            index += 1
            episodes.append(
                PickPlaceEpisode(
                    episode_id=f"pickplace_{index:04d}",
                    scene_id=scene["id"],
                    purpose=purpose,
                    status=outcome["status"],
                    failure_label=outcome["failure_label"],
                    placed_count=outcome["placed_count"],
                    block_count=outcome["block_count"],
                )
            )
            placed += outcome["placed_count"]
            blocks_total += outcome["block_count"]
            purpose_totals[purpose][1] += 1
            if outcome["status"] == "success":
                purpose_totals[purpose][0] += 1
            else:
                cluster_examples[outcome["failure_label"] or "unknown"].append(scene["id"])

    total = len(episodes)
    success_count = sum(1 for e in episodes if e.status == "success")
    counts = collections.Counter(e.failure_label for e in episodes if e.status != "success")
    clusters = [
        FailureCluster(
            failure_label=label,
            count=count,
            example_scene_ids=cluster_examples[label][:3],
            suggested_regression=_REGRESSION_FOR.get(label, "regenerate harder variants around this failure"),
        )
        for label, count in counts.most_common()
    ]

    report = PhysicsEvaluationReport(
        id=f"physics_eval_{genome['id']}",
        robot_genome_id=genome["id"],
        task_graph_id=compiled_index.get("robot_genome_id", ""),
        backend="mujoco",
        total_episodes=total,
        success_count=success_count,
        success_rate=round(success_count / total, 3) if total else 0.0,
        blocks_placed=placed,
        blocks_total=blocks_total,
        success_rate_by_purpose={
            purpose: round(s / t, 3) if t else 0.0 for purpose, (s, t) in purpose_totals.items()
        },
        failure_clusters=clusters,
        episodes=episodes,
        notes=[
            "Real MuJoCo pick-and-place: the arm sorted blocks into matching bins.",
            "Arm motion is real physics; grasp is an idealized tip engagement; release is gravity.",
            f"{placed}/{blocks_total} blocks placed in their matching bin across {total} episodes.",
            f"Perception: {getattr(perception, 'name', 'privileged_sim_pose')}.",
        ],
    )
    report.controller = "scripted_pick_place" + ("+perception" if perception is not None else "")
    validation = report.validate()
    if not validation.ok:
        issues = ", ".join(issue.code for issue in validation.issues)
        raise ValueError(f"{report.id} failed validation: {issues}")

    out_dir = package_dir / output_subdir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_lines = [json.dumps(_episode_dict(e), sort_keys=True) for e in episodes]
    (out_dir / "episode_trace.jsonl").write_text("\n".join(trace_lines) + "\n", encoding="utf-8")
    report_path = package_dir / "reports" / "physics_evaluation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report


def _episode_dict(episode: PickPlaceEpisode) -> dict:
    return {
        "episode_id": episode.episode_id,
        "scene_id": episode.scene_id,
        "purpose": episode.purpose,
        "status": episode.status,
        "failure_label": episode.failure_label,
        "placed_count": episode.placed_count,
        "block_count": episode.block_count,
    }
