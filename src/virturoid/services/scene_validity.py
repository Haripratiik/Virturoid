"""Physical-validity gate stack (scene-gen plan S4). A generated scene is only worth training on if it is
physically sane and solvable. Every SOTA generator gates its scenes; we implement the reusable ones with the
literature's exact thresholds, all FAIL-CLOSED:

- **unit sanity** — every object's dimensions are within the S2 category band; catches a 10 cm wall / 20 m box.
- **settle** — objects don't explode or drift when simulated (ClutterGen: near-zero velocity sustained); an
  interpenetrating spawn ejects and fails here.
- **navigability** (nav tasks) — an occupancy grid + A* from spawn to goal with the robot's clearance confirms a
  path exists; reports the geodesic:Euclidean ratio (Habitat's >=1.1 anti-triviality signal).
- **reachability** (manip tasks) — every manipulable object lies within the arm's reach (PhyScene R_reach).
- **solvability** (optional) — a scripted expert must actually succeed (MimicGen keep-if-success).

Pure-CPU (numpy + optional MuJoCo for the settle gate). ``validate_scene_physical`` runs the gates that apply to
the task and aggregates them fail-closed.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from virturoid.services.dimension_priors import check_dimensions, robot_scene_ratio_ok


# --------------------------------------------------------------------------------------------- unit sanity gate ---
def unit_sanity_gate(scene) -> dict:
    """Every object with a known category must have plausible dimensions; the robot:scene size ratio must be
    sane. Objects without size_xyz/category are skipped (nothing to check)."""
    flags = []
    for o in scene.objects:
        if o.size_xyz is not None and o.category:
            r = check_dimensions(o.category, o.size_xyz)
            flags += [f"{o.name}: {f}" for f in r["flags"] if not r["ok"]]
    if getattr(scene, "bounds", None):
        (xmn, ymn, _), (xmx, ymx, _) = scene.bounds
        scene_extent = max(xmx - xmn, ymx - ymn)
        rr = robot_scene_ratio_ok(0.4, scene_extent)         # ~0.4 m nominal robot; flags a robot-in-a-teacup unit bug
        if not rr["ok"] and rr["flag"]:
            flags.append(rr["flag"])
    return {"gate": "unit_sanity", "ok": not flags, "violations": flags}


# --------------------------------------------------------------------------------------------------- settle gate ---
def _scene_mjcf(scene) -> str:
    from virturoid.services.mujoco_exporter import _scene_objects_xml
    mats = "".join(f'<material name="{m}" rgba="0.6 0.6 0.62 1"/>' for m in
                   ("mat_gray", "mat_red", "mat_blue", "mat_metal", "mat_shell"))
    body = "\n".join(_scene_objects_xml(scene.objects))
    return (f'<mujoco><asset>{mats}</asset><option gravity="0 0 -9.81"/>'
            f'<worldbody><geom name="_ground" type="plane" size="50 50 0.1" pos="0 0 0"/>'
            f'{body}</worldbody></mujoco>')


def settle_gate(scene, *, steps: int = 120, tail: int = 25, vel_tol: float = 0.25, move_tol: float = 0.05) -> dict:
    """Simulate the scene and confirm the free bodies are STABLE — a scene whose objects interpenetrate at spawn
    ejects them (huge velocity/displacement) and fails here. The robust interpenetration signal is DRIFT (a body
    flying from its spawn); ``vel_tol`` is deliberately loose so normal settling jitter passes while an explosion
    (speeds of m/s) still trips it (ClutterGen uses a tighter bound over a longer settle; this is the quick gate)."""
    try:
        import mujoco
    except Exception:  # noqa: BLE001
        return {"gate": "settle", "ok": True, "violations": [], "skipped": "mujoco unavailable"}
    try:
        model = mujoco.MjModel.from_xml_string(_scene_mjcf(scene))
    except Exception as e:  # noqa: BLE001 - a scene that won't even compile is invalid
        return {"gate": "settle", "ok": False, "violations": [f"compile failed: {type(e).__name__}: {e}"]}
    data = mujoco.MjData(model)
    # record free-joint start positions
    starts = {}
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            adr = model.jnt_qposadr[j]
            starts[j] = np.array(data.qpos[adr:adr + 3])
    max_tail_speed = 0.0
    for t in range(steps):
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            return {"gate": "settle", "ok": False, "violations": ["simulation diverged (NaN) — invalid contacts"]}
        if t >= steps - tail:
            for j in starts:
                vadr = model.jnt_dofadr[j]
                max_tail_speed = max(max_tail_speed, float(np.linalg.norm(data.qvel[vadr:vadr + 3])))
    viol = []
    if max_tail_speed > vel_tol:
        viol.append(f"objects still moving at end (max speed {max_tail_speed:.3f} > {vel_tol} m/s)")
    max_xy_drift = 0.0
    for j, p0 in starts.items():
        adr = model.jnt_qposadr[j]
        # HORIZONTAL drift is the interpenetration signature — a body settling straight down onto a surface moves
        # in z (fine); a body EJECTED by an overlapping spawn shoots sideways (fails).
        max_xy_drift = max(max_xy_drift, float(np.linalg.norm(np.array(data.qpos[adr:adr + 2]) - p0[:2])))
    if max_xy_drift > move_tol:
        viol.append(f"a body shot {max_xy_drift:.3f} m sideways from spawn (>{move_tol}) — interpenetration/eject")
    return {"gate": "settle", "ok": not viol, "violations": viol,
            "max_tail_speed": round(max_tail_speed, 4), "max_xy_drift": round(max_xy_drift, 4)}


# ---------------------------------------------------------------------------------------------- navigability gate ---
def _occupancy(scene, robot_radius: float, cell: float):
    (xmn, ymn, _), (xmx, ymx, _) = scene.bounds
    nx = max(2, int(math.ceil((xmx - xmn) / cell)))
    ny = max(2, int(math.ceil((ymx - ymn) / cell)))
    grid = np.zeros((nx, ny), dtype=bool)                    # True = blocked
    inflate = robot_radius
    for o in scene.objects:
        if o.object_type not in ("wall", "obstacle"):
            continue
        hx, hy = (o.size_xyz[0] / 2, o.size_xyz[1] / 2) if o.size_xyz else (0.1, 0.1)
        cx, cy = o.pose_xyz_rpy[0], o.pose_xyz_rpy[1]
        # axis-aligned footprint inflated by robot radius (yaw~0 for our nav walls; conservative otherwise)
        ax0, ax1 = cx - hx - inflate, cx + hx + inflate
        ay0, ay1 = cy - hy - inflate, cy + hy + inflate
        i0, i1 = int((ax0 - xmn) / cell), int(math.ceil((ax1 - xmn) / cell))
        j0, j1 = int((ay0 - ymn) / cell), int(math.ceil((ay1 - ymn) / cell))
        grid[max(0, i0):max(0, i1), max(0, j0):max(0, j1)] = True
    return grid, (xmn, ymn), (nx, ny)


def _astar(grid, start, goal):
    nx, ny = grid.shape
    def h(a, b): return math.hypot(a[0] - b[0], a[1] - b[1])
    openq = [(h(start, goal), 0.0, start)]
    came, g = {}, {start: 0.0}
    while openq:
        _, gc, cur = heapq.heappop(openq)
        if cur == goal:
            n = 0
            while cur in came:
                cur = came[cur]; n += 1
            return gc, n
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nb = (cur[0] + dx, cur[1] + dy)
                if not (0 <= nb[0] < nx and 0 <= nb[1] < ny) or grid[nb]:
                    continue
                ng = gc + math.hypot(dx, dy)
                if ng < g.get(nb, 1e18):
                    g[nb] = ng; came[nb] = cur
                    heapq.heappush(openq, (ng + h(nb, goal), ng, nb))
    return None, 0


def navigability_gate(scene, *, robot_radius: float = 0.2, cell: float = 0.05, spawn_xy=None, goal_xy=None) -> dict:
    """Confirm a robot of ``robot_radius`` can actually reach the goal: build an inflated occupancy grid and run
    A* from spawn to goal. Fails if blocked (boxed in / corridor too narrow). Reports geodesic:Euclidean ratio
    (Habitat's >=1.1 anti-triviality signal — a near-1.0 straight shot is valid but trivially easy)."""
    if not getattr(scene, "bounds", None):
        return {"gate": "navigability", "ok": True, "violations": [], "skipped": "no bounds"}
    spawn = np.array(spawn_xy if spawn_xy is not None else scene.robot_spawn_xyz_rpy[:2], dtype=float)
    goal = None
    if goal_xy is not None:
        goal = np.array(goal_xy, dtype=float)
    else:
        for o in scene.objects:
            if o.object_type == "zone" and ("goal" in o.name or "target" in o.name):
                goal = np.array(o.pose_xyz_rpy[:2]); break
    if goal is None:
        return {"gate": "navigability", "ok": True, "violations": [], "skipped": "no goal in scene"}
    grid, (xmn, ymn), (nx, ny) = _occupancy(scene, robot_radius, cell)

    def to_cell(p):
        return (min(nx - 1, max(0, int((p[0] - xmn) / cell))), min(ny - 1, max(0, int((p[1] - ymn) / cell))))
    s, ggoal = to_cell(spawn), to_cell(goal)
    if grid[s] or grid[ggoal]:
        return {"gate": "navigability", "ok": False,
                "violations": ["spawn or goal is inside an obstacle (robot cannot stand there)"]}
    path_len, _ = _astar(grid, s, ggoal)
    if path_len is None:
        return {"gate": "navigability", "ok": False, "violations": ["no collision-free path from spawn to goal"]}
    geo = path_len * cell
    euc = float(np.linalg.norm(goal - spawn)) + 1e-9
    return {"gate": "navigability", "ok": True, "violations": [],
            "geodesic_m": round(geo, 3), "euclidean_m": round(euc, 3),
            "ratio": round(geo / euc, 3), "trivial": bool(geo / euc < 1.1)}


# --------------------------------------------------------------------------------------------- reachability gate ---
def reachability_gate(scene, *, base_xy=(0.0, 0.0), reach_m: float = 0.55) -> dict:
    """Every manipulable object (a free body) must lie within the arm's reach of ``base_xy`` (PhyScene R_reach).
    Fails if any is unreachable; reports the reachable fraction."""
    base = np.array(base_xy, dtype=float)
    manip = [o for o in scene.objects if o.object_type in ("cube", "box") and o.category != "obstacle"]
    if not manip:
        return {"gate": "reachability", "ok": True, "violations": [], "skipped": "no manipulable objects"}
    unreachable = [o.name for o in manip if np.linalg.norm(np.array(o.pose_xyz_rpy[:2]) - base) > reach_m]
    return {"gate": "reachability", "ok": not unreachable,
            "violations": [f"{n} beyond reach {reach_m} m" for n in unreachable],
            "r_reach": round(1.0 - len(unreachable) / len(manip), 3)}


# ------------------------------------------------------------------------------------------------ solvability gate ---
def solvability_gate(scene, expert_fn) -> dict:
    """Keep-if-success (MimicGen): a scripted expert must actually solve the scene. ``expert_fn(scene) -> bool``.
    Fail-closed on exception."""
    if expert_fn is None:
        return {"gate": "solvability", "ok": True, "violations": [], "skipped": "no expert provided"}
    try:
        ok = bool(expert_fn(scene))
    except Exception as e:  # noqa: BLE001
        return {"gate": "solvability", "ok": False, "violations": [f"expert errored: {type(e).__name__}: {e}"]}
    return {"gate": "solvability", "ok": ok, "violations": [] if ok else ["scripted expert could not solve it"]}


# ------------------------------------------------------------------------------------------------- the aggregator ---
_NAV_TYPES = {"wall", "floor"}


def validate_scene_physical(scene, *, robot_radius: float = 0.2, base_xy=(0.0, 0.0), reach_m: float = 0.55,
                            run_settle: bool = True, expert_fn=None) -> dict:
    """Run the gates that apply to this scene and aggregate FAIL-CLOSED. Navigability runs for scenes with
    walls/floor; reachability for scenes with manipulable objects; settle + unit-sanity always. Returns
    ``{"ok", "gates": [...], "violations": [...]}``."""
    gates = [unit_sanity_gate(scene)]
    if run_settle:
        gates.append(settle_gate(scene))
    is_nav = any(o.object_type in _NAV_TYPES for o in scene.objects)
    if is_nav:
        gates.append(navigability_gate(scene, robot_radius=robot_radius))
    else:
        gates.append(reachability_gate(scene, base_xy=base_xy, reach_m=reach_m))
    if expert_fn is not None:
        gates.append(solvability_gate(scene, expert_fn))
    viol = [f"[{g['gate']}] {v}" for g in gates for v in g.get("violations", [])]
    return {"ok": all(g["ok"] for g in gates), "gates": gates, "violations": viol}
