"""Task-matched evaluation for SPRAY/coating robots (startup plan §11): surface COVERAGE, not pick-place.

A spray robot's job is to sweep its nozzle over a target surface; scoring it on pick-place (as the
generic evaluator did) is meaningless. This drives the nozzle through a boustrophedon sweep of a panel's
grid in real physics and measures the fraction of grid cells the nozzle passed within spray range —
the natural metric for spraying/painting/coating, and the spray-class analogue of navigation distance
(mobile) and grasp success (manipulator). DOF-aware: high-DOF arms use task-space control, low-DOF the
joint-PD path.
"""

from __future__ import annotations


def evaluate_spray_coverage(gene, *, panel: dict | None = None, grid: int = 4, spray_radius: float = 0.08,
                            steps_per_point: int = 120, seed: int = 0) -> dict:
    """Sweep the nozzle over a panel and return ``{coverage, covered, total, panel}``. ``panel`` is a
    region dict ``{x:[a,b], y:[c,d], z:height}`` (default: a reachable tabletop panel). Needs MuJoCo."""
    import numpy as np
    import mujoco

    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    from virturoid.services.pick_place_controller import (
        plan_joint_targets, task_space_reach_ctrl, _actuated_joint_map,
        _actuator_force_clamps, _ee_site_id, _ee_position)

    panel = panel or {"x": [0.30, 0.46], "y": [-0.12, 0.12], "z": 0.18}
    xs = np.linspace(panel["x"][0], panel["x"][1], grid)
    ys = np.linspace(panel["y"][0], panel["y"][1], grid)
    # boustrophedon (zig-zag) order so the sweep is a continuous path, not teleporting across the panel
    pts = []
    for i, x in enumerate(xs):
        row = list(ys) if i % 2 == 0 else list(ys[::-1])
        pts += [(float(x), float(y), float(panel["z"])) for y in row]

    mj = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene))
    ee = _ee_site_id(mj)
    act = _actuated_joint_map(mj)
    clamps = _actuator_force_clamps(mj)
    n_rev = sum(1 for (_, _, jnt) in act if int(mj.jnt_type[jnt]) == 3)
    # Same reach-mode lesson as pick-place: firm joint-PD nails the panel cells for arms where it converges
    # (soft kp=8 + task-space under-reached, leaving ~37% of the panel uncoated); a redundant arm still needs
    # task-space. Probe THIS arm at the panel centre and use the mode that reaches.
    from virturoid.services.pick_place_controller import joint_pd_reaches_point
    kp, kd = 14.0, 2.0
    centre = (float(np.mean(panel["x"])), float(np.mean(panel["y"])), float(panel["z"]))
    use_task_space = (n_rev >= 5) and not joint_pd_reaches_point(mj, ee, act, clamps, kp, kd, centre)
    d = mujoco.MjData(mj); mujoco.mj_resetData(mj, d)

    covered = [False] * len(pts)
    for pt in pts:
        q = plan_joint_targets(mj, pt, seed=seed)[0]
        for _ in range(steps_per_point):
            if use_task_space:
                task_space_reach_ctrl(mj, d, ee, pt, act, clamps)
            else:
                for slot, (qadr, vadr, jnt) in enumerate(act):
                    tgt = q[slot] if slot < len(q) else 0.0
                    if mj.jnt_limited[jnt]:
                        lo, hi = mj.jnt_range[jnt]; tgt = min(max(tgt, lo), hi)
                    d.ctrl[slot] = float(np.clip(kp * (tgt - d.qpos[qadr]) - kd * d.qvel[vadr],
                                                 -clamps[slot], clamps[slot]))
            mujoco.mj_step(mj, d)
            if not np.all(np.isfinite(d.qpos)):
                break
            nozzle = _ee_position(d, ee)                  # spray as the nozzle sweeps: mark cells in range
            for j, p in enumerate(pts):
                if not covered[j] and float(np.linalg.norm(nozzle - np.asarray(p))) < spray_radius:
                    covered[j] = True

    n = len(pts)
    return {"coverage": round(sum(covered) / n, 3) if n else 0.0,
            "covered": sum(covered), "total": n, "panel": panel}
