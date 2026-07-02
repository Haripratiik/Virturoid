"""Real grasp-lift evaluation: does a robot actually grasp and LIFT an object by friction (no pinning)?

The pick-place controller's success uses an idealized 'pin' grasp, so the benchmark's manipulator score
measures functional pick-place, not a real friction grip. This module measures the real thing: a
scripted top-down grasp — vertical-approach IK (``grasp_pose``) + phase joint-PD (settle → descend →
close → lift) — and reports the actual height the object comes off the table. Nothing is pinned; the
object stays only if the jaws hold it by friction.

It is the honest grasp metric AND the objective a grasp co-design / learned residual-RL skill optimize
(the residual-free *base* reward). It mirrors the contact model + phase structure of the M3 skill
rollout (``skill_evaluator._grasp_rollout``) with the learned residual set to zero — crucially the same
contact bitmasks (box ↔ table/fingers; arm collides with nothing) so the arm can fold down to the
object without its links knocking the table or the box.
"""

from __future__ import annotations

from pathlib import Path


def _set_grasp_contacts(mj, finger_bodies, box_body):
    """Match the M3 contact model: the box collides with everything, fingers with the box, the
    world/table with the box, and the ARM with nothing (so its links never knock the table or box)."""
    for g in range(mj.ngeom):
        b = int(mj.geom_bodyid[g])
        if b == box_body:
            mj.geom_contype[g], mj.geom_conaffinity[g] = 0b11, 0b11
        elif b in finger_bodies:
            mj.geom_contype[g], mj.geom_conaffinity[g] = 0b10, 0b10
        elif b == 0:                       # world / table
            mj.geom_contype[g], mj.geom_conaffinity[g] = 0b01, 0b01
        else:
            mj.geom_contype[g], mj.geom_conaffinity[g] = 0, 0


def grasp_lift_attempt(gene, gx: float, gy: float, *, ep_len: int = 260, kpj: float = 150.0,
                       kdj: float = 18.0, kpf: float = 300.0, kdf: float = 10.0, dq: float = 0.06,
                       phases=(0.18, 0.5, 0.62), fclose: float = 0.05, success_lift: float = 0.05,
                       hover_z: float = 0.16, lift_z: float = 0.22, aim_xy=None, friction: float = 1.0,
                       record_frames: bool = False, frame_every: int = 18, cam_px: int = 240) -> dict:
    """One scripted top-down grasp-and-lift attempt. The box sits at ``(gx, gy)``; the arm AIMS at ``aim_xy``
    (defaults to the box). Pass a separate ``aim_xy`` to grasp a perceived/predicted position while the box is
    really elsewhere — so a wrong perception genuinely misses. Returns
    ``{success, lifted, lifted_max, both_contact, tilt_deg, d_tcp, reason}`` — real friction grasp."""
    import mujoco
    import numpy as np

    from virturoid.schemas.scenes import SceneObject
    from virturoid.services.gene_compiler import compile_gene_with_scene
    from virturoid.services.grasp_pose import (
        arm_revolute_joints,
        best_top_down_pose,
        solve_top_down_grasp_pose,
    )

    cube = SceneObject("box", "cube", (gx, gy, 0.05, 0, 0, 0), mass_kg=0.03, material="gray_block",
                       friction=float(friction), scale=1.0)   # WS9: lower friction -> a harder, slippier grasp
    mj = mujoco.MjModel.from_xml_string(compile_gene_with_scene(gene, [cube]))
    mj.opt.iterations = 30
    mj.opt.ls_iterations = 12
    tcp = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    if tcp < 0:
        return {"success": False, "lifted": 0.0, "lifted_max": 0.0, "both_contact": 0,
                "tilt_deg": None, "d_tcp": None, "reason": "no_grasp_site"}
    ee = gene.end_effector()
    palm = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, ee.name)
    box_jid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, "free_box")
    bq = int(mj.jnt_qposadr[box_jid]); bv = int(mj.jnt_dofadr[box_jid])
    box_body = int(mj.jnt_bodyid[box_jid])
    box_geom = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, "box")

    nu = int(mj.nu)
    j_of = [int(mj.actuator_trnid[u, 0]) for u in range(nu)]
    arm_u = [u for u in range(nu) if int(mj.jnt_type[j_of[u]]) != int(mujoco.mjtJoint.mjJNT_SLIDE)]
    fin_u = [u for u in range(nu) if int(mj.jnt_type[j_of[u]]) == int(mujoco.mjtJoint.mjJNT_SLIDE)]
    finger_bodies = {int(mj.jnt_bodyid[j_of[u]]) for u in fin_u}
    finger_geoms = {g for g in range(mj.ngeom) if int(mj.geom_bodyid[g]) in finger_bodies}
    _set_grasp_contacts(mj, finger_bodies, box_body)

    act_dof = np.array([int(mj.jnt_dofadr[j_of[u]]) for u in range(nu)])
    arm_qadr = [int(mj.jnt_qposadr[j_of[u]]) for u in arm_u]
    fr_lim = mj.actuator_forcerange[:, 1]
    KP = np.array([kpj if u in arm_u else kpf for u in range(nu)])
    KD = np.array([kdj if u in arm_u else kdf for u in range(nu)])
    arm_joints = arm_revolute_joints(mj)

    d = mujoco.MjData(mj); mujoco.mj_resetData(mj, d); mujoco.mj_forward(mj, d)
    box_z = float(d.qpos[bq + 2])

    # vertical-approach IK reference poses. The hover pose is solved seed-robustly (several seeds, best
    # kept) so it works regardless of joint layout; the descend/lift poses CONTINUE from it (each seeded
    # from the previous) so the arm stays in one configuration branch.
    ax, ay = aim_xy if aim_xy is not None else (gx, gy)          # arm aims here; the box is at (gx, gy)
    work = mujoco.MjData(mj); mujoco.mj_resetData(mj, work)
    above_map, tilt, _res = best_top_down_pose(mj, work, site_id=tcp, palm_body_id=palm,
                                               target=(ax, ay, hover_z), arm_joints=arm_joints, iters=250)

    def ik(z, seed_map):
        return solve_top_down_grasp_pose(mj, work, site_id=tcp, palm_body_id=palm,
                                         target=(ax, ay, z), arm_joints=arm_joints,
                                         seed=[seed_map[qa] for qa, _ in arm_joints], iters=250)

    at_map = ik(box_z, above_map)
    lift_map = ik(lift_z, at_map)
    q_above = np.array([above_map[qa] for qa, _ in arm_joints])
    q_at = np.array([at_map[qa] for qa, _ in arm_joints])
    q_lift = np.array([lift_map[qa] for qa, _ in arm_joints])

    for k in range(len(arm_joints)):                              # pre-fold to the hover pose
        d.qpos[arm_qadr[k]] = q_above[k]
    mujoco.mj_forward(mj, d)
    box_z0 = float(d.qpos[bq + 2]); q_ref = q_above.copy()
    p_settle, p_grip, p_lift = phases

    frames: list = []
    renderer = cam = None
    if record_frames:                                            # side camera for a grasp-sequence montage
        mj.geom_rgba[box_geom] = [0.95, 0.1, 0.1, 1]; mj.geom_matid[box_geom] = -1
        cam = mujoco.MjvCamera(); cam.lookat[:] = [gx - 0.03, gy, 0.12]
        cam.distance, cam.azimuth, cam.elevation = 0.9, -120, -18
        renderer = mujoco.Renderer(mj, height=cam_px, width=cam_px)

    def finger_contacts():
        touched = set()
        for i in range(d.ncon):
            g1, g2 = int(d.contact[i].geom1), int(d.contact[i].geom2)
            if box_geom in (g1, g2):
                for g in (g1, g2):
                    if g in finger_geoms:
                        touched.add(int(mj.geom_bodyid[g]))
        return len(touched)

    max_both, lifted_max = 0, -1.0
    for t in range(ep_len):
        frac = t / ep_len
        q_tgt = q_lift if frac >= p_lift else (q_above if frac < p_settle else q_at)
        q_ref = q_ref + np.clip(q_tgt - q_ref, -dq, dq)
        f_tgt = 0.0 if frac < p_grip else fclose
        q_star = np.zeros(nu)
        for k, u in enumerate(arm_u):
            q_star[u] = q_ref[k]
        for u in fin_u:
            q_star[u] = f_tgt
        q = d.qpos[act_dof]; qd = d.qvel[act_dof]
        ctrl = d.qfrc_bias[act_dof] + KP * (q_star - q) - KD * qd
        d.ctrl[:] = np.clip(ctrl, -fr_lim, fr_lim)
        mujoco.mj_step(mj, d)
        if renderer is not None and t % frame_every == 0:
            renderer.update_scene(d, camera=cam); frames.append(renderer.render().copy())
        if frac >= p_grip:
            max_both = max(max_both, finger_contacts())
        lifted_max = max(lifted_max, float(d.qpos[bq + 2]) - box_z0)

    boxp = d.qpos[bq:bq + 3]
    d_tcp = float(np.linalg.norm(d.site_xpos[tcp] - boxp))
    lifted = float(d.qpos[bq + 2]) - box_z0
    success = bool(max_both >= 2 and lifted > success_lift and d_tcp < 0.08)
    if success:
        reason = None
    elif max_both < 2:
        reason = "no_grasp_contact"
    elif lifted <= success_lift:
        reason = "gripped_no_lift"
    elif d_tcp >= 0.08:
        reason = "lifted_then_dropped"
    else:
        reason = "low_lift"
    if renderer is not None:
        renderer.close()
    return {"success": success, "lifted": round(lifted, 4), "lifted_max": round(lifted_max, 4),
            "both_contact": int(max_both >= 2), "tilt_deg": round(float(tilt), 2),
            "d_tcp": round(d_tcp, 4), "reason": reason, "frames": frames}


def evaluate_grasp_lift(gene, *, positions=None, seed: int = 0, controller_params: dict | None = None,
                        friction: float = 1.0, **kw) -> dict:
    """Evaluate real grasp-and-lift over a few object positions. Returns ``{success_rate, mean_lift,
    max_lift, mean_tilt_deg, attempts}`` — a real friction-grasp metric usable as the honest grasp
    score and as the residual-free base reward for the learned grasp skill.

    WS9 residual dispatch: if ``controller_params`` carries a ``residual_params_path`` (a banked learned grasp
    residual MLP) that exists on disk, each attempt is run WITH that residual on top of the gravity-comp PD
    (``skill_evaluator._grasp_rollout``); otherwise the scripted grasp is the honest floor. ``friction`` sets
    the object's friction (the M5_arm_grasp_hard scene lowers it so the scripted grasp is brittle and a learned
    residual has room to win). Other ``controller_params`` (gains/phases) become the residual rollout's cfg."""
    if positions is None:
        positions = [(0.40, 0.0), (0.44, 0.06), (0.44, -0.06), (0.48, 0.0)]
    cp = controller_params or {}
    res_path = cp.get("residual_params_path")
    use_residual = bool(res_path) and Path(res_path).exists()
    if use_residual:                                              # LEARNED residual on top of the PD base
        from virturoid.services.skill_evaluator import _grasp_rollout, _load_policy
        layers = _load_policy(res_path)
        cfg = {k: v for k, v in cp.items() if k in ("kpj", "kdj", "kpf", "kdf", "alpha", "dq", "phases",
                                                    "ep_len", "fclose")}
        attempts = [_grasp_rollout(gene, cfg, layers, float(gx), float(gy), friction=float(friction))
                    for gx, gy in positions]
    else:                                                         # scripted grasp (the honest floor)
        attempts = [grasp_lift_attempt(gene, float(gx), float(gy), friction=float(friction), **kw)
                    for gx, gy in positions]
    n = max(1, len(attempts))
    succ = sum(a["success"] for a in attempts)
    tilts = [a["tilt_deg"] for a in attempts if a.get("tilt_deg") is not None]
    return {
        "success_rate": round(succ / n, 4),
        "mean_lift": round(sum(a["lifted"] for a in attempts) / n, 4),
        "max_lift": round(max(a["lifted"] for a in attempts), 4),
        "mean_tilt_deg": round(sum(tilts) / len(tilts), 2) if tilts else None,
        "residual": bool(use_residual),
        "attempts": attempts,
    }
