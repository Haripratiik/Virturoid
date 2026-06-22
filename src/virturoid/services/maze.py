"""Maze task family (gap-closure plan W2): generate a maze, plan a global path, follow it.

Two-layer navigation (the ROS Nav2 pattern): a **global** A* planner over the maze's occupancy grid
finds the route through the walls; a **local** pure-pursuit follower drives the wheeled base along the
waypoints (reusing the rover wheel-grouping + sign auto-calibration from ``navigation_controller``). It
is classical and deterministic — no RL needed — and the legged variant (W3) follows the same path via
velocity commands. The composer builds the rover; the planner (intent_planner) routes "maze" → mobile.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field

WALL_H = 0.4          # wall height (m) — tall enough the rover can't climb it
CELL = 0.85           # world size of one grid cell (m); passages are one cell wide


@dataclass
class Maze:
    n: int                                  # n x n rooms
    grid: list = field(default_factory=list)   # (2n+1)x(2n+1) bool, True = wall
    start_rc: tuple = (1, 1)                 # grid coords of the start room
    goal_rc: tuple = (0, 0)                  # grid coords of the goal room (set in generate)
    cell: float = CELL                       # world size of one grid cell (m); SIZED TO THE ROVER so a
    #                                          one-cell passage is wide enough to TURN in (see solve_maze)

    @property
    def g(self) -> int:
        return 2 * self.n + 1

    def to_world(self, i: int, j: int) -> tuple:
        """Grid cell (row i, col j) -> world (x, y), with the START room at the origin."""
        si, sj = self.start_rc
        return ((j - sj) * self.cell, (si - i) * self.cell)


def generate_maze(n: int = 4, *, seed: int = 0, cell: float = CELL) -> Maze:
    """Perfect maze via recursive backtracker on an (2n+1)^2 wall grid (rooms at odd indices). ``cell`` is the
    world size of one passage — pass a value sized to the rover so it can turn the corners (see solve_maze)."""
    g = 2 * n + 1
    grid = [[True] * g for _ in range(g)]
    rng = random.Random(seed)

    def carve(r: int, c: int):
        grid[2 * r + 1][2 * c + 1] = False
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        rng.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and grid[2 * nr + 1][2 * nc + 1]:
                grid[2 * r + 1 + dr][2 * c + 1 + dc] = False   # knock the wall between rooms
                carve(nr, nc)

    carve(0, 0)
    return Maze(n=n, grid=grid, start_rc=(1, 1), goal_rc=(2 * n - 1, 2 * n - 1), cell=float(cell))


def astar(maze: Maze, start_rc: tuple | None = None, goal_rc: tuple | None = None) -> list:
    """Shortest grid path (4-connectivity, walls blocked) from start to goal room. List of (i, j)."""
    start = start_rc or maze.start_rc
    goal = goal_rc or maze.goal_rc
    grid, G = maze.grid, maze.g

    def h(a):
        return abs(a[0] - goal[0]) + abs(a[1] - goal[1])

    openq = [(h(start), 0, start)]
    came: dict = {start: None}
    cost = {start: 0}
    while openq:
        _, gc, cur = heapq.heappop(openq)
        if cur == goal:
            path = [cur]
            while came[path[-1]] is not None:
                path.append(came[path[-1]])
            return path[::-1]
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            ni, nj = cur[0] + dr, cur[1] + dc
            if 0 <= ni < G and 0 <= nj < G and not grid[ni][nj]:
                ng = gc + 1
                nxt = (ni, nj)
                if ng < cost.get(nxt, 1e9):
                    cost[nxt] = ng
                    came[nxt] = cur
                    heapq.heappush(openq, (ng + h(nxt), ng, nxt))
    return []


def waypoints(maze: Maze, path: list) -> list:
    """World-space waypoints for a grid path (room centres + the passage cells between them)."""
    return [maze.to_world(i, j) for (i, j) in path]


def wall_boxes(maze: Maze) -> list:
    """Static wall geoms as (x, y, half_x, half_y, half_z) for the MJCF worldbody."""
    h = maze.cell / 2.0
    out = []
    for i in range(maze.g):
        for j in range(maze.g):
            if maze.grid[i][j]:
                x, y = maze.to_world(i, j)
                out.append((x, y, h, h, WALL_H / 2.0))
    return out


def build_maze_mjcf(gene, maze: Maze) -> str:
    """Compile the robot + inject the maze walls + a goal marker into one MJCF world."""
    from virturoid.services.gene_compiler import compile_gene_to_mjcf

    xml = compile_gene_to_mjcf(gene)
    geoms = []
    for (x, y, hx, hy, hz) in wall_boxes(maze):
        geoms.append(f'    <geom type="box" pos="{x:.3f} {y:.3f} {hz:.3f}" '
                     f'size="{hx:.3f} {hy:.3f} {hz:.3f}" rgba="0.45 0.48 0.55 1"/>')
    gx, gy = maze.to_world(*maze.goal_rc)
    geoms.append(f'    <geom name="goal_marker" type="cylinder" pos="{gx:.3f} {gy:.3f} 0.02" '
                 f'size="0.18 0.02" rgba="0.18 0.83 0.75 0.6" contype="0" conaffinity="0"/>')
    return xml.replace("</worldbody>", "\n".join(geoms) + "\n  </worldbody>")


def run_maze_episode(model, wpts: list, *, horizon: int = 6000, lookahead: float = 0.55,
                     goal_radius: float = 0.32, record_frames=None, frame_every: int = 25) -> dict:
    """Pure-pursuit follow of a waypoint path with a differential-drive rover. Returns the outcome."""
    import mujoco
    import numpy as np

    from virturoid.services.pick_place_controller import _capture_geom_frame

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    jid = next((j for j in range(model.njnt) if int(model.jnt_type[j]) == 0), -1)
    if jid < 0:
        return {"status": "no_base", "steps": 0, "final_goal_distance_m": None}
    qadr = int(model.jnt_qposadr[jid])
    base_y = float(data.xpos[int(model.jnt_bodyid[jid])][1])
    left_act, right_act = [], []
    for u in range(model.nu):
        ajid = int(model.actuator_trnid[u, 0])
        if int(model.jnt_type[ajid]) != 3:
            continue
        wy = float(data.xpos[int(model.jnt_bodyid[ajid])][1])
        (left_act if wy >= base_y else right_act).append(u)
    if not left_act and not right_act:
        left_act, right_act = [0], [1] if model.nu > 1 else [0]

    def _yaw():
        q = data.qpos[qadr + 3: qadr + 7]
        return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))

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
    # start at the first waypoint (the maze start room is the world origin, but set it explicitly)
    start = np.array(wpts[0], dtype=float)
    data.qpos[qadr] = start[0]; data.qpos[qadr + 1] = start[1]
    mujoco.mj_forward(model, data)

    goal = np.array(wpts[-1], dtype=float)
    tgt_i = 1
    status, min_goal = "timeout", 1e9
    step = 0
    for step in range(horizon):
        pos = np.array(data.qpos[qadr:qadr + 2], dtype=float)
        gd = float(np.linalg.norm(goal - pos))
        min_goal = min(min_goal, gd)
        if gd < goal_radius:
            status = "reached"; break
        while tgt_i < len(wpts) - 1 and float(np.linalg.norm(np.array(wpts[tgt_i]) - pos)) < lookahead:
            tgt_i += 1
        tgt = np.array(wpts[tgt_i], dtype=float)
        to = tgt - pos
        yaw = _yaw()
        yaw_err = math.atan2(math.sin(math.atan2(to[1], to[0]) - yaw), math.cos(math.atan2(to[1], to[0]) - yaw))
        fr = float(model.actuator_forcerange[left_act[0] if left_act else 0, 1]) or 8.0
        # TURN-IN-PLACE when badly misaligned: in a one-cell corridor, creeping forward while turning scrapes a
        # wall and stalls (the maze-stall root cause). Pivot to within the heading band first, THEN drive — and
        # drive HARD (full force scaled by alignment) so the rover actually crosses the cell, not crawls.
        if abs(yaw_err) > 0.45:
            forward = 0.0
            turn = turn_sign * fr * (1.0 if yaw_err > 0 else -1.0)
        else:
            align = 1.0 - abs(yaw_err) / 0.45                     # 1 dead-ahead -> 0 at the band edge
            forward = fwd_sign * fr * (0.45 + 0.55 * align) * min(1.0, 2.2 * float(np.linalg.norm(to)))
            turn = turn_sign * fr * 0.7 * math.tanh(2.2 * yaw_err)
        left, right = forward - turn, forward + turn
        for u in left_act:
            data.ctrl[u] = float(np.clip(left, -model.actuator_forcerange[u, 1], model.actuator_forcerange[u, 1]))
        for u in right_act:
            data.ctrl[u] = float(np.clip(right, -model.actuator_forcerange[u, 1], model.actuator_forcerange[u, 1]))
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            status = "instability"; break
        if record_frames is not None and step % frame_every == 0:
            record_frames.append(_capture_geom_frame(data, model))
    return {"status": status, "steps": step + 1, "final_goal_distance_m": round(min_goal, 4),
            "waypoints_reached": tgt_i, "waypoints_total": len(wpts)}


def rover_footprint_m(gene) -> float:
    """The rover's planar footprint DIAGONAL (m) from its collision geoms — the width it sweeps when it
    rotates in place. A corridor must clear this for the rover to TURN a corner, not just drive straight."""
    import mujoco

    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    m = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    xs, ys = [], []
    for i in range(m.ngeom):
        if int(m.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_PLANE) or int(m.geom_contype[i]) == 0:
            continue
        c, rb = d.geom_xpos[i], float(m.geom_rbound[i])
        xs += [c[0] - rb, c[0] + rb]
        ys += [c[1] - rb, c[1] + rb]
    if not xs:
        return 0.5
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    return float((w * w + h * h) ** 0.5)


def cell_for_rover(gene, *, clearance: float = 1.7, floor: float = CELL) -> float:
    """Passage width sized to the rover: ``clearance`` × its turning DIAGONAL, so a one-cell passage is wide
    enough to rotate a corner in (the maze-stall root cause was a 0.73 m rover in a 0.85 m corridor). Floored
    at the default so a tiny rover still gets a sensible maze."""
    return round(max(floor, clearance * rover_footprint_m(gene)), 3)


def solve_maze(gene, *, n: int = 4, seed: int = 0, horizon: int | None = None, cell: float | None = None,
               record_frames=None) -> dict:
    """Compose-to-solve: build a maze SIZED TO THE ROVER, plan an A* path, and have ``gene`` follow it. The
    cell is widened to the rover's turning footprint so it can actually round the corners; the horizon and
    follower tolerances scale with the cell so a bigger maze still gets enough time. CPU MuJoCo."""
    import mujoco

    cell = cell_for_rover(gene) if cell is None else float(cell)
    maze = generate_maze(n, seed=seed, cell=cell)
    path = astar(maze)
    if not path:
        return {"solved": False, "status": "no_path", "path_len": 0}
    wpts = waypoints(maze, path)
    model = mujoco.MjModel.from_xml_string(build_maze_mjcf(gene, maze))
    model.opt.iterations = 20
    # scale the budget + follower tolerances with the (rover-sized) cell so corner turns have room + time.
    # Budget per waypoint covers a pivot + a full-cell crossing (the diff-drive turn-in-place is deliberate,
    # so each corner costs time); under-budgeting was timing out a rover that was still steadily progressing.
    horizon = int(max(8000, 950 * len(path) * (cell / CELL))) if horizon is None else int(horizon)
    out = run_maze_episode(model, wpts, horizon=horizon, lookahead=0.62 * cell,
                           goal_radius=0.40 * cell, record_frames=record_frames)
    out.update({"solved": out["status"] == "reached", "path_len": len(path),
                "maze_n": n, "optimal_cells": len(path), "cell_m": cell})
    return out


def maze_episode_view(gene, *, n: int = 3, seed: int = 0):
    """Solve a maze with ``gene`` while RECORDING per-frame geom poses, and return ``(view, xml)`` in the exact
    shape the desktop/web viewport replays (frames + outcome). This is the wire that lets a 'solve a maze' prompt
    show the rover actually navigating the maze in the app, instead of a generic score over a frozen viewport."""
    import mujoco

    cell = cell_for_rover(gene)
    maze = generate_maze(n, seed=seed, cell=cell)
    path = astar(maze)
    xml = build_maze_mjcf(gene, maze)
    if not path:
        return ({"frames": [], "scenes": [], "scene_index": 0,
                 "outcome": {"status": "no_path", "failure_label": "no_path", "placed_count": 0, "block_count": 1}},
                xml)
    wpts = waypoints(maze, path)
    model = mujoco.MjModel.from_xml_string(xml); model.opt.iterations = 20
    horizon = int(max(8000, 950 * len(path) * (cell / CELL)))
    frames: list = []
    out = run_maze_episode(model, wpts, horizon=horizon, lookahead=0.62 * cell,
                           goal_radius=0.40 * cell, record_frames=frames, frame_every=20)
    solved = out["status"] == "reached"
    view = {"frames": frames, "scenes": [], "scene_index": 0,
            "outcome": {"status": "success" if solved else out["status"],
                        "failure_label": None if solved else out["status"],
                        "placed_count": 1 if solved else 0, "block_count": 1,
                        "steps": out.get("steps"), "goal_distance_m": out.get("final_goal_distance_m")}}
    return view, xml
