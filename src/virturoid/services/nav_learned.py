"""Learned steerable legged navigation (gap-closure W3): a LEGGED robot follows a planned path via a
velocity-command-conditioned MorphPolicy — the legged analogue of the wheeled ``maze.run_maze_episode``.

The two-layer Nav2 pattern, legged: a global A* planner (``maze.astar``) routes through the walls; a
local **pure-pursuit** controller turns the path into a body-velocity command (forward v_x, yaw rate
w_z); the low-level **learned** policy (``scripts/mjx_morph_vel``) tracks that command. This is what
scripted gaits could not do — steerable legged locomotion — so the frog (or any composed legged body)
can solve a maze. Pure-pursuit geometry is unit-testable without the policy; the closed loop needs a
trained velocity policy. Relates to [[morph-policy-theme1]], [[mjx-rl-debugging-method]].
"""

from __future__ import annotations

import math


def pure_pursuit_cmd(pos, yaw, wpts, tgt_i, *, lookahead: float = 0.6, cruise_v: float = 0.8,
                     wz_max: float = 0.8, turn_gain: float = 1.6):
    """Path → body-velocity command. Returns ``(vx_cmd, wz_cmd, tgt_i)``: advance the target waypoint
    once inside ``lookahead``; steer toward it (w_z ∝ heading error) and slow forward speed when not
    facing it (so the body turns in place before driving). Commands stay within the policy's trained
    range (v_x∈[0,cruise], |w_z|≤wz_max)."""
    import numpy as np

    pos = np.asarray(pos, dtype=float)
    while tgt_i < len(wpts) - 1 and float(np.linalg.norm(np.asarray(wpts[tgt_i], float) - pos)) < lookahead:
        tgt_i += 1
    to = np.asarray(wpts[tgt_i], float) - pos
    desired = math.atan2(to[1], to[0])
    yaw_err = math.atan2(math.sin(desired - yaw), math.cos(desired - yaw))
    wz_cmd = float(np.clip(turn_gain * yaw_err, -wz_max, wz_max))
    vx_cmd = float(cruise_v * max(0.1, math.cos(yaw_err)))      # ~0 forward until roughly aligned
    return vx_cmd, wz_cmd, tgt_i


def follow_waypoints_legged(gene, policy, wpts, *, walls=None, max_steps: int = 6000,
                            goal_radius: float = 0.4, lookahead: float = 0.6, cruise_v: float = 0.8,
                            record_frames=None, frame_every: int = 25) -> dict:
    """Drive a legged ``gene`` along world-space ``wpts`` with a velocity-conditioned MorphPolicy.

    ``walls``: an optional ``maze.Maze`` to inject as obstacles (real maze); ``None`` follows the path on
    open ground (steerable-locomotion check). Returns reached/progress/distance like ``run_maze_episode``.
    """
    import mujoco
    import numpy as np

    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    from virturoid.services.morph_graph import encode_robot
    from virturoid.services.pick_place_controller import _capture_geom_frame

    if walls is not None:
        from virturoid.services.maze import build_maze_mjcf
        xml = build_maze_mjcf(gene, walls)
    else:
        xml = compile_gene_to_mjcf(gene, include_floor=True)
    model = mujoco.MjModel.from_xml_string(xml)
    model.opt.iterations = 20
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    graph = encode_robot(model)
    if graph.base_jid < 0 or graph.n_tokens == 0:
        return {"status": "not_legged", "reached": False, "waypoints_reached": 0, "waypoints_total": len(wpts)}
    bq = graph.base_qadr
    start = np.asarray(wpts[0], float)
    data.qpos[bq] = start[0]; data.qpos[bq + 1] = start[1]
    mujoco.mj_forward(model, data)

    def _yaw():
        q = data.qpos[bq + 3:bq + 7]
        return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))

    goal = np.asarray(wpts[-1], float)
    tgt_i, status, min_goal, step = 1, "timeout", 1e9, 0
    for step in range(max_steps):
        pos = np.asarray(data.qpos[bq:bq + 2], float)
        gd = float(np.linalg.norm(goal - pos))
        min_goal = min(min_goal, gd)
        if gd < goal_radius:
            status = "reached"; break
        vx_cmd, wz_cmd, tgt_i = pure_pursuit_cmd(pos, _yaw(), wpts, tgt_i,
                                                 lookahead=lookahead, cruise_v=cruise_v)
        obs = np.concatenate([graph.observe(model, data),
                              np.tile([vx_cmd, wz_cmd], (graph.n_tokens, 1))], axis=1)
        graph.apply(model, data, policy.act(obs), alpha=0.6)
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            status = "instability"; break
        if record_frames is not None and step % frame_every == 0:
            record_frames.append(_capture_geom_frame(data, model))
    return {"status": status, "reached": status == "reached", "steps": step + 1,
            "final_goal_distance_m": round(float(min_goal), 4),
            "waypoints_reached": tgt_i, "waypoints_total": len(wpts)}


def solve_maze_learned(gene, policy, *, n: int = 3, seed: int = 0, walls: bool = True,
                       max_steps: int = 6000, record_frames=None) -> dict:
    """Compose-to-solve, legged + learned: build a maze, plan A*, follow it with the velocity policy."""
    from virturoid.services.maze import astar, generate_maze, waypoints

    maze = generate_maze(n, seed=seed)
    path = astar(maze)
    if not path:
        return {"solved": False, "status": "no_path", "path_len": 0}
    wpts = waypoints(maze, path)
    out = follow_waypoints_legged(gene, policy, wpts, walls=(maze if walls else None),
                                  max_steps=max_steps, record_frames=record_frames)
    out.update({"solved": out["reached"], "path_len": len(path), "maze_n": n,
                "progress": round(out["waypoints_reached"] / max(1, out["waypoints_total"] - 1), 3)})
    return out
