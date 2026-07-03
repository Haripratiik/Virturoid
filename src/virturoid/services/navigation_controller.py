"""Differential-drive navigation task loop for the mobile base (Phase 5).

Gives the second robot class a real, physics-grounded task loop (like pick-and-
place for the arm): drive from start to the goal zone while avoiding obstacles,
with task-grounded outcomes (reached / collision / timeout) and failure clusters.
"""

from __future__ import annotations

import collections
import json
import math
from pathlib import Path

from virturoid.services.mujoco_runner import mujoco_available

# Goal radius is forgiving on purpose: a fast skid-steer rover that has to register "reached" within a tiny
# radius overshoots it and then drives off a wall-less arena until timeout. ~the goal-zone footprint lets it
# stop AT the goal (a delivered package), which is both the right behaviour and what reads well on replay.
GOAL_RADIUS_M = 0.35
OBSTACLE_CLEARANCE_M = 0.17
REPULSE_RADIUS_M = 0.6                    # heading-repulsion reaction radius (P7: unstick the grazing stall)


def _corridor_waypoints(goal_xy, obstacle_xy: list, *, side_lane_m: float = 0.45, forward_offset_m: float = 0.2):
    """Return simple navigation waypoints around point obstacles.

    The generated indoor scenes place obstacles along the start->goal corridor. A skid-steer rover does better
    with a few straight side-lane targets than with pure reactive repulsion, which can settle into a local minimum
    in front of alternating obstacles. Each obstacle gets a waypoint one ``side_lane_m`` off-centre on the side
    AWAY from it; a light obstacle-repulsion in the control law (run_navigation_episode) unsticks the rover when
    a committed lane still grazes a close obstacle."""
    import numpy as np

    obstacles = sorted(obstacle_xy or [], key=lambda o: float(o[0]))
    if not obstacles:
        return [np.array(goal_xy, dtype=float)]
    waypoints = []
    for obstacle in obstacles:
        ox, oy = float(obstacle[0]), float(obstacle[1])
        pass_side = -1.0 if oy >= 0.0 else 1.0
        waypoints.append(np.array([ox + forward_offset_m, pass_side * side_lane_m], dtype=float))
    waypoints.append(np.array(goal_xy, dtype=float))
    return waypoints


def run_navigation_episode(model, goal_xy, obstacle_xy: list, horizon: int = 1500, record_frames=None, frame_every: int = 25) -> dict:
    """Drive the base toward the goal with waypoint heading control + obstacle avoidance."""
    import mujoco
    import numpy as np

    from virturoid.services.pick_place_controller import _capture_geom_frame

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    # Locate the floating base (any freejoint) — works for the composed rover (chassis_free) + legacy.
    jid = next((j for j in range(model.njnt) if int(model.jnt_type[j]) == 0), -1)
    if jid < 0:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "chassis_free")
    qadr = int(model.jnt_qposadr[jid])
    # Differential drive over ALL wheel actuators grouped by side (left: chassis-frame y>0, right: y<0),
    # not just ctrl[0]/ctrl[1] — so a composed 4-wheel rover drives all four, not two.
    base_y = float(data.xpos[int(model.jnt_bodyid[jid])][1])
    left_act, right_act = [], []
    for u in range(model.nu):
        ajid = int(model.actuator_trnid[u, 0])
        if int(model.jnt_type[ajid]) != 3:               # only revolute (wheel) actuators
            continue
        wy = float(data.xpos[int(model.jnt_bodyid[ajid])][1])
        (left_act if wy >= base_y else right_act).append(u)
    if not left_act and not right_act:                   # fallback: legacy ctrl[0]/ctrl[1]
        left_act, right_act = [0], [1] if model.nu > 1 else [0]
    goal = np.array(goal_xy, dtype=float)

    def _yaw():
        q = data.qpos[qadr + 3: qadr + 7]
        return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))

    # Auto-calibrate the drive signs (the torque→motion convention depends on wheel mounting): a brief
    # forward impulse tells us which torque sign moves the base along its +x; a differential impulse tells
    # us which way it yaws. Makes navigation robust to any rover the composer produces.
    def _calibrate():
        p0 = np.array(data.qpos[qadr:qadr + 2]); y0 = _yaw()
        for _ in range(80):
            for u in left_act + right_act:
                data.ctrl[u] = 5.0
            mujoco.mj_step(model, data)
        disp = np.array(data.qpos[qadr:qadr + 2]) - p0
        fsign = 1.0 if (disp[0] * math.cos(y0) + disp[1] * math.sin(y0)) >= 0 else -1.0
        mujoco.mj_resetData(model, data); mujoco.mj_forward(model, data)
        y1 = _yaw()
        for _ in range(80):
            for u in left_act:                           # MATCH the control law's turn pattern
                data.ctrl[u] = -5.0                      #   left = forward - turn  (turn>0 here)
            for u in right_act:
                data.ctrl[u] = 5.0                       #   right = forward + turn
            mujoco.mj_step(model, data)
        tsign = 1.0 if math.atan2(math.sin(_yaw() - y1), math.cos(_yaw() - y1)) >= 0 else -1.0
        mujoco.mj_resetData(model, data); mujoco.mj_forward(model, data)
        return fsign, tsign

    fwd_sign, turn_sign = _calibrate()
    waypoints = _corridor_waypoints(goal_xy, obstacle_xy)
    waypoint_index = 0
    waypoint_best_dist = 1e9
    waypoint_stale_steps = 0

    status = "timeout"
    min_goal_dist = 1e9
    for step in range(horizon):
        pos = np.array(data.qpos[qadr : qadr + 2], dtype=float)
        quat = data.qpos[qadr + 3 : qadr + 7]
        yaw = math.atan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]), 1 - 2 * (quat[2] ** 2 + quat[3] ** 2))
        to_goal = goal - pos
        dist = float(np.linalg.norm(to_goal))
        min_goal_dist = min(min_goal_dist, dist)
        if dist < GOAL_RADIUS_M:
            status = "reached"
            break

        # Collision check against obstacles.
        for ob in obstacle_xy:
            if float(np.linalg.norm(pos - np.array(ob[:2]))) < OBSTACLE_CLEARANCE_M:
                status = "collision"
                break
        if status == "collision":
            break

        while waypoint_index < len(waypoints) - 1 and float(np.linalg.norm(waypoints[waypoint_index] - pos)) < 0.35:
            waypoint_index += 1
            waypoint_best_dist = 1e9
            waypoint_stale_steps = 0
        to_waypoint = waypoints[waypoint_index] - pos
        waypoint_dist = float(np.linalg.norm(to_waypoint))
        if waypoint_dist < waypoint_best_dist - 0.02:
            waypoint_best_dist = waypoint_dist
            waypoint_stale_steps = 0
        else:
            waypoint_stale_steps += 1
        if waypoint_stale_steps > 550 and waypoint_index < len(waypoints) - 1:
            waypoint_index += 1
            waypoint_best_dist = 1e9
            waypoint_stale_steps = 0
            to_waypoint = waypoints[waypoint_index] - pos
            waypoint_dist = float(np.linalg.norm(to_waypoint))
        # OBSTACLE REPULSION: bend the heading away from any close obstacle so a committed side-lane that grazes
        # one doesn't stall the rover in front of it (measured: alternating obstacles stalled it at ~0.9 m). Sum
        # of away-vectors within a reaction radius, added to the (normalized) waypoint direction.
        repulse = np.zeros(2)
        for ob in obstacle_xy:
            d = pos - np.array(ob[:2], dtype=float)
            dn = float(np.linalg.norm(d))
            if 1e-6 < dn < REPULSE_RADIUS_M:
                repulse += (d / dn) * (REPULSE_RADIUS_M - dn) / REPULSE_RADIUS_M
        heading = to_waypoint / max(1e-6, waypoint_dist) + 1.3 * repulse
        desired_yaw = math.atan2(float(heading[1]), float(heading[0]))
        yaw_err = math.atan2(math.sin(desired_yaw - yaw), math.cos(desired_yaw - yaw))
        # Drive forward (slowing when mis-aligned) + a strong differential to steer. Calibrated signs
        # (fwd_sign/turn_sign) make this work for any wheel convention; the high turn gain rotates a
        # 4-wheel skid-steer rover against its lateral friction. Keep some forward during turns so
        # obstacle scenes that need continuous re-steering still make progress.
        # Let the rover ROTATE IN PLACE when badly mis-aligned (forward -> 0 for |yaw_err| >~ 0.77 rad) so it can
        # execute the SHARP turn an alternating-obstacle path demands, instead of the old 0.2 forward floor that
        # drove it into a stuck equilibrium at ~0.95 m (measured: scene_000 froze there for 12000 steps).
        forward = fwd_sign * 8.0 * min(waypoint_dist, 0.7) * max(0.0, 1.0 - 1.3 * abs(yaw_err))
        turn = turn_sign * 16.0 * math.tanh(1.6 * yaw_err)
        left = forward - turn
        right = forward + turn
        for u in left_act:
            data.ctrl[u] = float(np.clip(left, -model.actuator_forcerange[u, 1], model.actuator_forcerange[u, 1]))
        for u in right_act:
            data.ctrl[u] = float(np.clip(right, -model.actuator_forcerange[u, 1], model.actuator_forcerange[u, 1]))
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            status = "instability"
            break
        if record_frames is not None and step % frame_every == 0:
            record_frames.append(_capture_geom_frame(data, model))

    return {"status": status, "final_goal_distance_m": round(min_goal_dist, 4), "steps": step + 1}


def run_navigation_evaluation(package_dir: Path, scene_uri: str = "simulation/scene_set.json") -> dict:
    if not mujoco_available():
        raise RuntimeError("MuJoCo is not installed; run `pip install mujoco` to evaluate navigation.")

    import mujoco

    package_dir = Path(package_dir)
    compiled = json.loads((package_dir / "simulation" / "mujoco" / "compiled_scene_index.json").read_text(encoding="utf-8"))
    xml_by_scene = {e["scene_id"]: e["mujoco_xml"] for e in compiled.get("scenes", [])}
    scene_set = json.loads((package_dir / scene_uri).read_text(encoding="utf-8"))

    episodes = []
    labels = collections.Counter()
    for scene in scene_set["scenes"]:
        xml_uri = xml_by_scene.get(scene["id"])
        if not xml_uri:
            continue
        model = mujoco.MjModel.from_xml_path(str(package_dir / xml_uri))
        goal = next((o["pose_xyz_rpy"][:2] for o in scene["objects"] if o["name"] == "goal_zone"), None)
        obstacles = [o["pose_xyz_rpy"][:2] for o in scene["objects"] if o.get("object_type") == "obstacle"]
        if goal is None:
            continue
        # Scale the step budget to the goal distance: the rover maneuvers slowly around obstacles (~1 m per
        # ~1000 steps), so the fixed 1500-step horizon timed out before it could arrive.
        dist = (float(goal[0]) ** 2 + float(goal[1]) ** 2) ** 0.5
        # Turn-in-place trades speed for turn authority near obstacles; give the maneuver budget headroom.
        outcome = run_navigation_episode(model, goal, obstacles, horizon=max(2000, int(2600 * dist + 900)))
        episodes.append({"scene_id": scene["id"], **outcome})
        if outcome["status"] != "reached":
            labels[outcome["status"]] += 1

    total = len(episodes)
    reached = sum(1 for e in episodes if e["status"] == "reached")
    report = {
        "id": f"navigation_eval_{compiled.get('robot_genome_id', 'mobile')}",
        "task": "navigation",
        "backend": "mujoco",
        "controller": "corridor_waypoint_skid_steer",
        "route_gate_success_rate": 0.8,
        "total_episodes": total,
        "reached": reached,
        "success_rate": round(reached / total, 3) if total else 0.0,
        "failure_clusters": [{"label": k, "count": v} for k, v in labels.most_common()],
        "episodes": episodes,
        "notes": ["Real MuJoCo differential-drive navigation: start -> goal with obstacle avoidance."],
    }
    out = package_dir / "reports" / "navigation_evaluation_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
