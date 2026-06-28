"""§4.5 — a REAL non-prehensile PUSH skill (distinct from pick-place).

The audit found "push" routed to grasp-and-place — flatly wrong (a push doesn't grasp). This is a genuinely
different skill: a CLOSED gripper approaches BEHIND the object along the object->target axis and sweeps it across
the table to the target; success is measured by how far the object travels toward the target, not by a grasp.
Reuses the arm's top-down IK (grasp_pose) but with a low, lateral sweep and the jaw held shut as a solid pusher.
"""

from __future__ import annotations

import math


def push_attempt(gene, obj_xy, target_xy, *, ep_len: int = 320, kpj: float = 150.0, kdj: float = 18.0,
                 kpf: float = 300.0, kdf: float = 10.0, dq: float = 0.06, hover_z: float = 0.16,
                 push_z: float = 0.075, fclose: float = 0.0, back: float = 0.12, over: float = 0.10,
                 success_frac: float = 0.5, moved_min: float = 0.03) -> dict:
    """One scripted non-prehensile push of a box from ``obj_xy`` toward ``target_xy``. Returns
    ``{success, progress, moved_m, final_dist_to_target, start_dist, reason}`` — ``progress`` is the fraction of
    the start->target distance the object covered. Needs MuJoCo."""
    import mujoco
    import numpy as np

    from virturoid.schemas.scenes import SceneObject
    from virturoid.services.gene_compiler import compile_gene_with_scene
    from virturoid.services.grasp_pose import arm_revolute_joints, best_top_down_pose, solve_top_down_grasp_pose

    ox, oy = float(obj_xy[0]), float(obj_xy[1])
    tx, ty = float(target_xy[0]), float(target_xy[1])
    cube = SceneObject("box", "cube", (ox, oy, 0.05, 0, 0, 0), mass_kg=0.05, material="gray_block",
                       friction=1.0, scale=1.0)
    mj = mujoco.MjModel.from_xml_string(compile_gene_with_scene(gene, [cube]))
    mj.opt.iterations = 30
    mj.opt.ls_iterations = 12
    tcp = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    if tcp < 0:
        return {"success": False, "progress": 0.0, "moved_m": 0.0, "reason": "no_grasp_site"}
    ee = gene.end_effector()
    palm = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, ee.name)
    box_jid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, "free_box")
    bq = int(mj.jnt_qposadr[box_jid])

    nu = int(mj.nu)
    j_of = [int(mj.actuator_trnid[u, 0]) for u in range(nu)]
    arm_u = [u for u in range(nu) if int(mj.jnt_type[j_of[u]]) != int(mujoco.mjtJoint.mjJNT_SLIDE)]
    fin_u = [u for u in range(nu) if int(mj.jnt_type[j_of[u]]) == int(mujoco.mjtJoint.mjJNT_SLIDE)]
    act_dof = np.array([int(mj.jnt_dofadr[j_of[u]]) for u in range(nu)])
    arm_qadr = [int(mj.jnt_qposadr[j_of[u]]) for u in arm_u]
    fr_lim = mj.actuator_forcerange[:, 1]
    KP = np.array([kpj if u in arm_u else kpf for u in range(nu)])
    KD = np.array([kdj if u in arm_u else kdf for u in range(nu)])
    arm_joints = arm_revolute_joints(mj)

    d = mujoco.MjData(mj); mujoco.mj_resetData(mj, d); mujoco.mj_forward(mj, d)

    dx, dy = tx - ox, ty - oy
    L = math.hypot(dx, dy) or 1e-6
    ux, uy = dx / L, dy / L                                       # unit push direction (object -> target)
    pre = (ox - ux * back, oy - uy * back)                       # start BEHIND the object
    end = (ox + ux * over, oy + uy * over)                       # sweep a bit PAST the start toward the target
    start_dist = math.hypot(tx - ox, ty - oy)

    work = mujoco.MjData(mj); mujoco.mj_resetData(mj, work)
    above_map, _tilt, _r = best_top_down_pose(mj, work, site_id=tcp, palm_body_id=palm,
                                              target=(pre[0], pre[1], hover_z), arm_joints=arm_joints, iters=250)

    def ik(x, y, z, seed_map):
        return solve_top_down_grasp_pose(mj, work, site_id=tcp, palm_body_id=palm, target=(x, y, z),
                                         arm_joints=arm_joints, seed=[seed_map[qa] for qa, _ in arm_joints],
                                         iters=250)

    down_map = ik(pre[0], pre[1], push_z, above_map)
    push_map = ik(end[0], end[1], push_z, down_map)
    q_above = np.array([above_map[qa] for qa, _ in arm_joints])
    q_down = np.array([down_map[qa] for qa, _ in arm_joints])
    q_push = np.array([push_map[qa] for qa, _ in arm_joints])

    for k in range(len(arm_joints)):                             # pre-fold to the hover-behind pose
        d.qpos[arm_qadr[k]] = q_above[k]
    mujoco.mj_forward(mj, d)
    q_ref = q_above.copy()

    for t in range(ep_len):
        frac = t / ep_len
        q_tgt = q_above if frac < 0.18 else (q_down if frac < 0.38 else q_push)   # hover -> descend -> sweep
        q_ref = q_ref + np.clip(q_tgt - q_ref, -dq, dq)
        q_star = np.zeros(nu)
        for k, u in enumerate(arm_u):
            q_star[u] = q_ref[k]
        for u in fin_u:
            q_star[u] = fclose                                  # jaw held SHUT — a solid pusher (no grasp)
        q = d.qpos[act_dof]; qd = d.qvel[act_dof]
        ctrl = d.qfrc_bias[act_dof] + KP * (q_star - q) - KD * qd
        d.ctrl[:] = np.clip(ctrl, -fr_lim, fr_lim)
        mujoco.mj_step(mj, d)

    fox, foy = float(d.qpos[bq]), float(d.qpos[bq + 1])
    final_dist = math.hypot(tx - fox, ty - foy)
    moved = math.hypot(fox - ox, foy - oy)
    progress = max(0.0, (start_dist - final_dist) / max(start_dist, 1e-6))
    success = bool(progress >= success_frac and moved > moved_min)
    reason = None if success else ("did_not_move" if moved <= moved_min else "short_of_target")
    return {"success": success, "progress": round(progress, 3), "moved_m": round(moved, 4),
            "final_dist_to_target": round(final_dist, 4), "start_dist": round(start_dist, 4), "reason": reason}


def evaluate_push(gene, *, cases=None, seed: int = 0, **kw) -> dict:
    """Evaluate the push over a few (object, target) cases. Returns ``{success_rate, mean_progress, attempts}``."""
    if cases is None:
        cases = [((0.38, -0.06), (0.38, 0.10)), ((0.40, 0.05), (0.40, -0.10)),
                 ((0.36, 0.0), (0.46, 0.0))]
    attempts = [push_attempt(gene, o, t, **kw) for o, t in cases]
    n = len(attempts)
    return {"attempts": n, "success_rate": round(sum(a["success"] for a in attempts) / max(1, n), 3),
            "mean_progress": round(sum(a["progress"] for a in attempts) / max(1, n), 3),
            "cases": attempts}
