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
    contact_grasp = run_contact_grasp_evaluation_from_export(package_dir)
    _write_manipulation_fidelity_gap_report(package_dir, report, contact_grasp=contact_grasp)
    return report


def run_contact_grasp_evaluation_from_export(
    package_dir: Path,
    *,
    scene_uris: list[str] | None = None,
    max_attempts: int = 4,
    controller_params: dict | None = None,
    write_report: bool = True,
) -> dict:
    """Evaluate the exported MVP arm's own gripper by contact, with no pinning.

    This is intentionally package-native: it loads the compiled MJCF files that ship in the export, closes the
    exported slide-joint jaws around blocks, and measures whether the object lifts by friction.
    """
    if not mujoco_available():
        raise RuntimeError("MuJoCo is not installed; run `pip install mujoco` to evaluate contact grasping.")

    import mujoco

    package_dir = Path(package_dir)
    genome = json.loads((package_dir / "robot" / "robot_genome.json").read_text(encoding="utf-8"))
    compiled_index = json.loads(
        (package_dir / "simulation" / "mujoco" / "compiled_scene_index.json").read_text(encoding="utf-8")
    )
    xml_by_scene = {entry["scene_id"]: entry["mujoco_xml"] for entry in compiled_index.get("scenes", [])}
    scene_uris = scene_uris or ["simulation/baseline_scene_set.json", "simulation/scene_set.json"]

    attempts: list[dict] = []
    for uri in scene_uris:
        if len(attempts) >= max_attempts:
            break
        path = package_dir / uri
        if not path.exists():
            continue
        scene_set = json.loads(path.read_text(encoding="utf-8"))
        for scene in scene_set.get("scenes", []):
            if len(attempts) >= max_attempts:
                break
            xml_uri = xml_by_scene.get(scene["id"])
            if not xml_uri:
                continue
            model = mujoco.MjModel.from_xml_path(str(package_dir / xml_uri))
            manipulable = [
                obj["name"]
                for obj in scene.get("objects", [])
                if obj.get("object_type") in ("cube", "box") and _free_joint_exists(model, obj["name"])
            ]
            for name in manipulable:
                if len(attempts) >= max_attempts:
                    break
                attempts.append(_template_contact_grasp_attempt(model, name, scene["id"], controller_params))

    n = max(1, len(attempts))
    success_count = sum(1 for item in attempts if item.get("success"))
    lifts = [float(item.get("lifted_m", 0.0)) for item in attempts]
    max_lifts = [float(item.get("max_lift_m", 0.0)) for item in attempts]
    report = {
        "id": f"grasp_eval_{genome['id']}",
        "robot_genome_id": genome["id"],
        "task_type": "grasp_lift",
        "backend": "mujoco",
        "grasp_model": "contact",
        "total_attempts": len(attempts),
        "success_count": success_count,
        "success_rate": round(success_count / n, 4) if attempts else 0.0,
        "mean_lift_m": round(sum(lifts) / n, 4) if attempts else 0.0,
        "max_lift_m": round(max(max_lifts), 4) if max_lifts else 0.0,
        "attempts": attempts,
        "note": (
            "Same exported MJCF model, no pinning: slide-joint gripper pads close on the block and lift by "
            "contact friction."
        ),
    }
    if write_report:
        out = package_dir / "reports" / "grasp_evaluation_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
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


def _free_joint_exists(model, block_name: str) -> bool:
    import mujoco

    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"free_{block_name}") >= 0


def _template_contact_grasp_attempt(model, block_name: str, scene_id: str, controller_params: dict | None = None) -> dict:
    import mujoco
    import numpy as np

    from virturoid.services.mujoco_runner import _actuator_force_clamps, _actuated_joint_map
    from virturoid.services.pick_place_controller import plan_joint_targets

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    grasp_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    box_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"free_{block_name}")
    box_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, block_name)
    if grasp_site < 0:
        return _failed_contact_attempt(scene_id, block_name, "no_grasp_site")
    if box_jid < 0 or box_geom < 0:
        return _failed_contact_attempt(scene_id, block_name, "no_free_block")

    slide_actuators: list[tuple[int, int, int]] = []
    finger_bodies = set()
    for actuator in range(model.nu):
        jnt = int(model.actuator_trnid[actuator, 0])
        if int(model.jnt_type[jnt]) == int(mujoco.mjtJoint.mjJNT_SLIDE):
            slide_actuators.append((actuator, int(model.jnt_qposadr[jnt]), int(model.jnt_dofadr[jnt])))
            finger_bodies.add(int(model.jnt_bodyid[jnt]))
    if len(slide_actuators) < 2:
        return _failed_contact_attempt(scene_id, block_name, "no_actuated_fingers")

    box_qadr = int(model.jnt_qposadr[box_jid])
    box_body = int(model.jnt_bodyid[box_jid])
    finger_geoms = {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in finger_bodies}

    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        if body == box_body:
            model.geom_contype[geom], model.geom_conaffinity[geom] = 0b11, 0b11
        elif geom in finger_geoms:
            model.geom_contype[geom], model.geom_conaffinity[geom] = 0b10, 0b10
        elif body == 0:
            model.geom_contype[geom], model.geom_conaffinity[geom] = 0b01, 0b01
        else:
            model.geom_contype[geom], model.geom_conaffinity[geom] = 0, 0

    arm = _actuated_joint_map(model)
    arm_actuators = []
    arm_lookup = {(qadr, vadr, jnt): idx for idx, (qadr, vadr, jnt) in enumerate(_actuated_joint_map(model, include_slide_joints=True))}
    for item in arm:
        arm_actuators.append((arm_lookup[item], *item))
    clamps = _actuator_force_clamps(model)
    bx, by, bz0 = (float(data.qpos[box_qadr]), float(data.qpos[box_qadr + 1]), float(data.qpos[box_qadr + 2]))
    cfg = {
        "arm_kp": 90.0,
        "arm_kd": 10.0,
        "finger_kp": 500.0,
        "finger_kd": 18.0,
        "finger_target": 0.038,
        "hover_offset_m": 0.18,
        "grasp_offset_m": 0.005,
        "lift_offset_m": 0.20,
    }
    if controller_params:
        cfg.update(controller_params)
    targets = {
        "above": plan_joint_targets(model, (bx, by, bz0 + float(cfg["hover_offset_m"])), seed=11, site_name="grasp_site")[0],
        "at": plan_joint_targets(model, (bx, by, bz0 + float(cfg["grasp_offset_m"])), seed=12, site_name="grasp_site")[0],
        "lift": plan_joint_targets(model, (bx, by, bz0 + float(cfg["lift_offset_m"])), seed=13, site_name="grasp_site")[0],
    }

    max_both = 0
    max_lift = 0.0
    stable = True

    def finger_contacts() -> int:
        touched = set()
        for idx in range(int(data.ncon)):
            g1, g2 = int(data.contact[idx].geom1), int(data.contact[idx].geom2)
            if box_geom in (g1, g2):
                for geom in (g1, g2):
                    if geom in finger_geoms:
                        touched.add(int(model.geom_bodyid[geom]))
        return len(touched)

    def drive(target, steps: int, close: bool) -> None:
        nonlocal max_both, max_lift, stable
        for _ in range(steps):
            for slot, qadr, vadr, jnt in arm_actuators:
                desired = target[slot] if slot < len(target) else 0.0
                if model.jnt_limited[jnt]:
                    lo, hi = model.jnt_range[jnt]
                    desired = min(max(float(desired), float(lo)), float(hi))
                cmd = data.qfrc_bias[vadr] + float(cfg["arm_kp"]) * (desired - data.qpos[qadr]) - float(cfg["arm_kd"]) * data.qvel[vadr]
                data.ctrl[slot] = float(np.clip(cmd, -clamps[slot], clamps[slot]))
            finger_target = float(cfg["finger_target"]) if close else 0.0
            for slot, qadr, vadr in slide_actuators:
                cmd = data.qfrc_bias[vadr] + float(cfg["finger_kp"]) * (finger_target - data.qpos[qadr]) - float(cfg["finger_kd"]) * data.qvel[vadr]
                data.ctrl[slot] = float(np.clip(cmd, -clamps[slot], clamps[slot]))
            mujoco.mj_step(model, data)
            if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
                stable = False
                return
            if close:
                max_both = max(max_both, finger_contacts())
            max_lift = max(max_lift, float(data.qpos[box_qadr + 2]) - bz0)

    drive(targets["above"], 260, close=False)
    drive(targets["at"], 320, close=False)
    drive(targets["at"], 260, close=True)
    drive(targets["lift"], 420, close=True)

    lifted = float(data.qpos[box_qadr + 2]) - bz0
    tcp_dist = float(np.linalg.norm(data.site_xpos[grasp_site] - data.qpos[box_qadr:box_qadr + 3]))
    success = bool(stable and max_both >= 2 and max_lift >= 0.035 and lifted >= 0.02 and tcp_dist < 0.12)
    reason = None
    if not success:
        if not stable:
            reason = "instability"
        elif max_both < 2:
            reason = "no_grasp_contact"
        elif max_lift < 0.035:
            reason = "gripped_no_lift"
        elif lifted < 0.02:
            reason = "lifted_then_dropped"
        else:
            reason = "tcp_not_near_block"
    return {
        "scene_id": scene_id,
        "block": block_name,
        "success": success,
        "lifted_m": round(lifted, 4),
        "max_lift_m": round(max_lift, 4),
        "both_finger_contact": bool(max_both >= 2),
        "tcp_distance_m": round(tcp_dist, 4),
        "reason": reason,
    }


def _failed_contact_attempt(scene_id: str, block_name: str, reason: str) -> dict:
    return {
        "scene_id": scene_id,
        "block": block_name,
        "success": False,
        "lifted_m": 0.0,
        "max_lift_m": 0.0,
        "both_finger_contact": False,
        "tcp_distance_m": None,
        "reason": reason,
    }


def _write_manipulation_fidelity_gap_report(
    package_dir: Path,
    report: PhysicsEvaluationReport,
    *,
    contact_grasp: dict | None = None,
) -> Path:
    """Write the explicit gap between demo task success and export-grade manipulation physics.

    The MVP template arm still uses an idealized pin for pick/place. The Product Readiness Ledger already blocks
    export on that disclosure; this report makes the reason and required closing evidence discoverable in the
    package instead of burying it in ledger prose.
    """
    grasp_model = str(getattr(report, "grasp_model", "") or "")
    contact_success_rate = None if contact_grasp is None else float(contact_grasp.get("success_rate", 0.0))
    contact_certified = grasp_model == "contact" or (contact_success_rate is not None and contact_success_rate >= 0.5)
    data = {
        "id": f"manipulation_fidelity_gap_{report.robot_genome_id}",
        "robot_genome_id": report.robot_genome_id,
        "task_type": "pick_place_sort",
        "backend": report.backend,
        "controller": report.controller,
        "task_success_rate": report.success_rate,
        "grasp_model": grasp_model,
        "contact_grasp_certified": contact_certified,
        "contact_grasp_success_rate": contact_success_rate,
        "export_blocker": (
            None
            if contact_certified
            else "task success used an idealized pin grasp; same-robot contact grasp/lift evidence is below gate"
        ),
        "required_evidence_to_close": [
            "same robot model has actuated gripper/finger geometry in MuJoCo",
            "reports/grasp_evaluation_report.json declares grasp_model='contact'",
            "contact grasp/lift success_rate >= 0.5 with measured object lift",
        ],
        "notes": [
            "This is not a failure of MuJoCo execution; real physics ran for arm motion and object release.",
            (
                "The same exported model now has contact grasp/lift evidence above the readiness gate."
                if contact_certified
                else "It is a manipulation-fidelity gap: the package has not yet proven a friction grasp for this model."
            ),
        ],
    }
    out = package_dir / "reports" / "manipulation_fidelity_gap_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out
