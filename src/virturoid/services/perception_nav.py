"""Reactive PERCEPTION navigation: reach a goal in an UNKNOWN obstacle field using only what the robot
SENSES, not privileged obstacle positions.

Every other nav path is state-blind or map-aware: ``navigation_controller`` repels from the ground-truth
obstacle xy, and ``nav_learned.solve_maze_learned`` follows an A* path that already knows the maze. This one
is the perception keystone ([[perception-next-keystone]]): a ring of MuJoCo rangefinders (``exteroception``)
is the ONLY exteroceptive input -- each step the robot reads its 8 ray distances and steers by a
sensed-obstacle potential field (goal attraction + per-ray repulsion weighted by 1/distance). It never reads
where the obstacles actually are, so it generalizes to obstacle fields it was never told about.
"""
from __future__ import annotations

import math


def reactive_heading(goal_dir: float, ranges, *, n_rays: int = 8, safe_m: float = 0.85,
                     clearance_w: float = 2.2):
    """Gap-seeking steering from sensed ranges -> a desired body-frame heading (rad).

    ``goal_dir`` is the goal bearing in the body frame; ``ranges[i]`` is the measured distance along body
    yaw ``2*pi*i/n_rays``. Each candidate direction is scored by OPENNESS (range, normalized) traded off
    against ALIGNMENT with the goal (cos of the bearing gap). The robot heads for the most-open direction
    that still points roughly at the goal -- so a blocked goal direction makes it steer AROUND the obstacle
    instead of stalling in the potential-field local minimum directly in front of it.
    """
    best_score, best_dir = -1e9, goal_dir
    for i in range(min(n_rays, len(ranges))):
        a = 2 * math.pi * i / n_rays
        clearance = min(1.0, float(ranges[i]) / safe_m)   # 1.0 beyond safe_m; only CLOSE obstacles deflect
        align = math.cos(a - goal_dir)                     # -1 (away) .. 1 (toward goal)
        score = clearance_w * clearance + align
        if score > best_score:
            best_score, best_dir = score, a
    return math.atan2(math.sin(best_dir), math.cos(best_dir))


def run_perception_nav_episode(model, goal_xy, *, n_rays: int = 8, max_range: float = 3.0,
                               horizon: int = 3000, record_frames=None, frame_every: int = 25) -> dict:
    """Drive a wheeled base to ``goal_xy`` using ONLY the rangefinder ring (the model must already carry the
    ``rf{i}`` sensors -- see exteroception.with_rangefinders). Differential-drive steering toward the
    sensed-obstacle potential-field heading; stops on reach / collision (a ray reading ~0) / timeout."""
    import mujoco
    import numpy as np

    from virturoid.services.exteroception import read_ranges
    from virturoid.services.pick_place_controller import _capture_geom_frame

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    jid = next((j for j in range(model.njnt) if int(model.jnt_type[j]) == 0), -1)  # the floating base
    if jid < 0:
        return {"status": "no_base", "reached": False, "final_goal_distance_m": None, "steps": 0}
    qadr = int(model.jnt_qposadr[jid])
    base_y = float(data.xpos[int(model.jnt_bodyid[jid])][1])
    left_act, right_act = [], []
    for u in range(model.nu):
        ajid = int(model.actuator_trnid[u, 0])
        if int(model.jnt_type[ajid]) != 3:               # revolute (wheel) actuators only
            continue
        wy = float(data.xpos[int(model.jnt_bodyid[ajid])][1])
        (left_act if wy >= base_y else right_act).append(u)
    if not left_act and not right_act:
        left_act, right_act = [0], [1] if model.nu > 1 else [0]
    goal = np.asarray(goal_xy, dtype=float)

    def _yaw():
        q = data.qpos[qadr + 3:qadr + 7]
        return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))

    # Auto-calibrate drive signs (mounting-agnostic), same as navigation_controller.
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
            for u in left_act:
                data.ctrl[u] = -5.0
            for u in right_act:
                data.ctrl[u] = 5.0
            mujoco.mj_step(model, data)
        tsign = 1.0 if math.atan2(math.sin(_yaw() - y1), math.cos(_yaw() - y1)) >= 0 else -1.0
        mujoco.mj_resetData(model, data); mujoco.mj_forward(model, data)
        return fsign, tsign

    fwd_sign, turn_sign = _calibrate()
    status, min_goal, step, commit = "timeout", 1e9, 0, 0.0   # commit: 0 none / +1 round left / -1 round right
    react_m = 0.9
    for step in range(horizon):
        pos = np.asarray(data.qpos[qadr:qadr + 2], dtype=float)
        yaw = _yaw()
        dist = float(np.linalg.norm(goal - pos))
        min_goal = min(min_goal, dist)
        if dist < 0.35:
            status = "reached"; break
        ranges = read_ranges(model, data, n_rays=n_rays, max_range=max_range)
        if float(np.min(ranges)) < 0.06:                 # a ray basically touching a surface -> collision
            status = "collision"; break
        goal_dir = math.atan2(float(goal[1] - pos[1]), float(goal[0] - pos[0])) - yaw
        goal_dir = math.atan2(math.sin(goal_dir), math.cos(goal_dir))
        front = min(float(ranges[7]), float(ranges[0]), float(ranges[1]))   # forward arc (-45,0,+45)
        if commit == 0.0 and front < react_m and abs(goal_dir) < 1.2:
            lc = float(ranges[1] + ranges[2] + ranges[3]); rc = float(ranges[7] + ranges[6] + ranges[5])
            commit = 1.0 if lc >= rc else -1.0                  # commit to rounding the more-open side
        elif commit != 0.0 and front > react_m * 1.7 and abs(goal_dir) < 0.9:
            commit = 0.0                                        # ahead well clear AND goal roughly ahead -> release
        if commit != 0.0:
            desired = math.atan2(math.sin(goal_dir + commit * 1.2), math.cos(goal_dir + commit * 1.2))
        else:
            desired = reactive_heading(goal_dir, ranges, n_rays=n_rays)
        # Forward slows when something is sensed straight ahead (ray 0) or when steering hard.
        ahead = float(ranges[0])
        # Brake ONLY for an imminent frontal hit, and keep a forward floor with a MODERATE turn gain so the
        # robot CURVES around an obstacle (advancing while it turns) instead of pivoting in place in front of it
        # and re-aiming at the goal forever (the centered-obstacle limit cycle).
        slow = 1.0 if ahead > 0.5 else max(0.3, ahead / 0.5)
        forward = fwd_sign * 7.0 * min(dist, 0.7) * slow
        turn = turn_sign * 8.0 * math.tanh(1.5 * desired)
        for u in left_act:
            data.ctrl[u] = float(np.clip(forward - turn, -model.actuator_forcerange[u, 1], model.actuator_forcerange[u, 1]))
        for u in right_act:
            data.ctrl[u] = float(np.clip(forward + turn, -model.actuator_forcerange[u, 1], model.actuator_forcerange[u, 1]))
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            status = "instability"; break
        if record_frames is not None and step % frame_every == 0:
            record_frames.append(_capture_geom_frame(data, model))

    return {"status": status, "reached": status == "reached",
            "final_goal_distance_m": round(float(min_goal), 4), "steps": step + 1, "sensed_only": True}


def build_perception_scene(gene, obstacles, goal_xy, *, n_rays: int = 8) -> str:
    """Compile ``gene`` on open ground with cylinder ``obstacles`` (list of (x, y)), a goal marker, and a ring of
    rangefinders on its base -- a self-contained perception nav scene MJCF."""
    from virturoid.services.exteroception import inject_rangefinders
    from virturoid.services.gene_compiler import compile_gene_to_mjcf

    xml = compile_gene_to_mjcf(gene, include_floor=True)
    obs = "".join(f'<geom name="obs{i}" type="cylinder" pos="{x:.3f} {y:.3f} 0.3" size="0.16 0.3" '
                  f'rgba="0.82 0.32 0.30 1"/>' for i, (x, y) in enumerate(obstacles))
    goal = (f'<site name="goal_zone" pos="{goal_xy[0]:.3f} {goal_xy[1]:.3f} 0.05" size="0.12" '
            f'rgba="0.22 0.9 0.35 0.6"/>')
    xml = xml.replace("<worldbody>", "<worldbody>" + obs + goal, 1)
    return inject_rangefinders(xml, gene.root().name, n_rays=n_rays)


def evaluate_perception_nav(gene, *, n_scenes: int = 6, seed: int = 0, goal_x: float = 3.0,
                            n_rays: int = 8, horizon: int = 7000) -> dict:
    """Run the SENSED-ONLY reactive navigator across random scattered-obstacle scenes and report the reach rate.

    The robot only ever reads its rangefinder ring -- it is never told where the obstacles are -- so this
    measures real perception-driven navigation in environments it was given no map of. Honest: dense
    adversarial clutter is the reactive frontier (a learned/planned local planner would do better)."""
    import mujoco
    import numpy as np

    rng = np.random.default_rng(seed)
    goal = (float(goal_x), 0.0)
    episodes = []
    for s in range(n_scenes):
        k = int(rng.integers(1, 4))                       # 1-3 scattered obstacles
        obstacles = [(float(rng.uniform(1.0, goal_x - 0.6)), float(rng.uniform(-1.1, 1.1))) for _ in range(k)]
        try:
            model = mujoco.MjModel.from_xml_string(build_perception_scene(gene, obstacles, goal, n_rays=n_rays))
            res = run_perception_nav_episode(model, goal, n_rays=n_rays, horizon=horizon)
        except Exception as exc:  # noqa: BLE001
            res = {"status": f"error:{type(exc).__name__}", "reached": False, "final_goal_distance_m": None}
        episodes.append({"scene": s, "n_obstacles": k, **res})
    reached = sum(1 for e in episodes if e.get("reached"))
    return {"task": "perception_navigation", "controller": "sensed_reactive_potential_field", "sensed_only": True,
            "n_scenes": n_scenes, "reached": reached, "success_rate": round(reached / max(1, n_scenes), 3),
            "episodes": episodes}
