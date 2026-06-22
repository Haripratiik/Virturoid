"""Reactive sense-and-avoid navigation (perception Rung A, the "unknown environment" capability).

``nav_learned`` follows a PRE-GIVEN A* path — it assumes the maze is fully known. Real generality means
navigating an environment you can only SENSE. This drives a steerable, velocity-conditioned policy from a
rangefinder ring (``exteroception``) alone: each step it picks the body-relative heading that is most
OPEN (longest range) while biased toward the goal, and slows when the way ahead is blocked — a lightweight
vector-field-histogram / potential-field controller. No global plan, no wall map. The command geometry is
pure + unit-testable; the closed loop needs a trained steerable policy. Relates to [[perception-next-keystone]].
"""

from __future__ import annotations

import math


def reactive_command(ranges, ray_angles, goal_bearing, *, max_range: float = 3.0, cruise_v: float = 0.8,
                     wz_max: float = 0.8, turn_gain: float = 1.6, open_weight: float = 2.0,
                     goal_weight: float = 1.0, block_dist: float = 0.6):
    """From a range ring → a body-velocity command ``(vx_cmd, wz_cmd)``.

    ``ranges[i]`` is the sensed distance along body-relative yaw ``ray_angles[i]``; ``goal_bearing`` is the
    body-relative direction to the goal (rad). Picks the heading maximizing openness + goal alignment,
    steers toward it, and throttles forward speed by how clear the chosen heading is (creeps when blocked).
    """
    import numpy as np

    ranges = np.asarray(ranges, dtype=float)
    ang = np.asarray(ray_angles, dtype=float)
    # score each candidate heading: prefer open (long range) AND aligned with the goal
    score = open_weight * np.clip(ranges / max_range, 0, 1) + goal_weight * np.cos(ang - goal_bearing)
    best = int(np.argmax(score))
    desired = math.atan2(math.sin(ang[best]), math.cos(ang[best]))     # wrap to [-pi, pi]
    wz_cmd = float(np.clip(turn_gain * desired, -wz_max, wz_max))
    # vx moves the body along its CURRENT heading, so gate forward speed on the FORWARD ray's clearance
    # (slow/turn-in-place when the way directly ahead is blocked), and also throttle during sharp turns.
    fwd = int(np.argmin(np.abs(np.arctan2(np.sin(ang), np.cos(ang)))))
    clearance = float(np.clip(ranges[fwd] / max(block_dist, 1e-6), 0.0, 1.0))
    vx_cmd = float(cruise_v * clearance * max(0.15, math.cos(desired)))
    return vx_cmd, wz_cmd


def navigate_sensing(gene, policy, goal_xy, *, walls=None, n_rays: int = 8, max_range: float = 3.0,
                     max_steps: int = 6000, goal_radius: float = 0.4, start_xy=(0.0, 0.0),
                     record_frames=None, frame_every: int = 25) -> dict:
    """Drive a legged ``gene`` toward ``goal_xy`` using ONLY sensed ranges (no path), with a steerable
    velocity-conditioned MorphPolicy. ``walls`` is an optional ``maze.Maze`` to inject as obstacles."""
    import mujoco
    import numpy as np

    from virturoid.services.exteroception import read_ranges, with_rangefinders
    from virturoid.services.morph_graph import encode_robot
    from virturoid.services.pick_place_controller import _capture_geom_frame

    model = mujoco.MjModel.from_xml_string(with_rangefinders(gene, walls=walls, n_rays=n_rays))
    model.opt.iterations = 20
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    graph = encode_robot(model)
    if graph.base_jid < 0 or graph.n_tokens == 0:
        return {"status": "not_legged", "reached": False}
    bq = graph.base_qadr
    data.qpos[bq] = start_xy[0]; data.qpos[bq + 1] = start_xy[1]
    mujoco.mj_forward(model, data)
    ray_ang = np.array([2 * math.pi * i / n_rays for i in range(n_rays)])
    goal = np.asarray(goal_xy, float)

    def _yaw():
        q = data.qpos[bq + 3:bq + 7]
        return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))

    status, min_goal, step = "timeout", 1e9, 0
    for step in range(max_steps):
        pos = np.asarray(data.qpos[bq:bq + 2], float)
        gd = float(np.linalg.norm(goal - pos)); min_goal = min(min_goal, gd)
        if gd < goal_radius:
            status = "reached"; break
        to = goal - pos
        goal_bearing = math.atan2(math.sin(math.atan2(to[1], to[0]) - _yaw()),
                                  math.cos(math.atan2(to[1], to[0]) - _yaw()))
        ranges = read_ranges(model, data, n_rays=n_rays, max_range=max_range)
        vx_cmd, wz_cmd = reactive_command(ranges, ray_ang, goal_bearing, max_range=max_range)
        # Feed the sensed ranges + the goal-velocity command through the policy's GLOBAL perception token
        # (keeping per-token obs at feature_dim=24) rather than concatenating to every token's features.
        obs = graph.observe(model, data)
        graph.apply(model, data, policy.act(obs, ranges=ranges, cmd=[vx_cmd, wz_cmd]), alpha=0.6)
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            status = "instability"; break
        if record_frames is not None and step % frame_every == 0:
            record_frames.append(_capture_geom_frame(data, model))
    return {"status": status, "reached": status == "reached", "steps": step + 1,
            "final_goal_distance_m": round(float(min_goal), 4)}
