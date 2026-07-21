"""Scripted pick-and-place controller for the tabletop sorting task.

This closes the actual MVP demo loop: the arm reaches above a block, descends,
engages a grasp, lifts, transports to the matching bin, and releases. Arm motion
is real MuJoCo physics (PD-tracked joint targets found by a short cross-entropy
IK search); the grasp is an idealized engagement that pins the block to the
end-effector once the tip is close enough, and release hands the block back to
gravity so it falls into the bin.

Outcomes are task-grounded: success (blocks in matching bins), missed_grasp,
wrong_bin, dropped, or instability. These are the failure categories the product
clusters and turns into regression scenes.
"""

from __future__ import annotations

import hashlib

from virturoid.services.mujoco_runner import (
    _actuated_joint_map,
    _actuator_force_clamps,
    _ee_position,
    _ee_site_id,
    reach_rollout,
)

ENGAGE_RADIUS_M = 0.12
BIN_RADIUS_M = 0.10
HOVER_M = 0.20
GRASP_TIP_OFFSET_M = 0.02
_BLOCK_TO_BIN = {"red_block": "red_bin", "blue_block": "blue_bin"}


def plan_joint_targets(model, point, iterations: int | None = None, candidates: int | None = None,
                       seed: int = 0, site_name: str = "ee_site"):
    """Cross-entropy IK using forward kinematics (exact, no dynamic settling).

    Finds joint angles whose forward-kinematic ``site_name`` is nearest ``point``,
    respecting joint limits. The dynamic controller then PD-tracks these angles.
    ``site_name`` lets a gripper aim its grasp TCP (``grasp_site``, the finger-tip
    midpoint) at the object rather than the palm (``ee_site``) — the finger base
    overshoots the object by a finger-length, so targeting the palm mis-places the grasp.

    ``iterations``/``candidates`` default to a budget that SCALES with the joint count: a 3-DOF arm
    solves at 80 candidates but a 7-DOF arm needs more search or it under-reaches (residual 0.05 vs
    0.00) — the experiment showed control, not body, capped high-DOF arms. Pass explicit values to
    override (e.g. the coarse IK-grid precompute uses small budgets for speed).
    """
    import mujoco
    import numpy as np

    rng = np.random.default_rng(seed)
    data = mujoco.MjData(model)
    actuated = _actuated_joint_map(model)
    n_act = len(actuated)
    if iterations is None:
        iterations = min(20, 8 + n_act)        # 3-DOF: 11, 5: 13, 9: 17
    if candidates is None:
        candidates = min(260, 40 * max(2, n_act))   # 3-DOF: 120, 5: 200, 9: 260
    ee_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if ee_site < 0:
        ee_site = _ee_site_id(model)
    ranges = []
    for (_, _, jnt) in actuated:
        if model.jnt_limited[jnt]:
            ranges.append((float(model.jnt_range[jnt][0]), float(model.jnt_range[jnt][1])))
        else:
            ranges.append((-3.14, 3.14))

    n = len(actuated)
    mean = np.array([0.0] + [0.9] * (n - 1), dtype=float)[:n]
    std = np.full(n, 0.9)
    target = np.array(point, dtype=float)

    def fk_distance(angles):
        mujoco.mj_resetData(model, data)
        for slot, (qadr, _, _) in enumerate(actuated):
            lo, hi = ranges[slot]
            data.qpos[qadr] = min(max(float(angles[slot]), lo), hi)
        mujoco.mj_kinematics(model, data)
        return float(np.linalg.norm(_ee_position(data, ee_site) - target))

    best, best_dist = mean.copy(), fk_distance(mean)
    for _ in range(iterations):
        samples = rng.normal(mean, std, size=(candidates, n))
        scored = sorted(((fk_distance(c), c) for c in samples), key=lambda item: item[0])
        elites = np.array([c for _, c in scored[: max(2, candidates // 4)]])
        mean = elites.mean(axis=0)
        std = elites.std(axis=0) + 1e-3
        if scored[0][0] < best_dist:
            best_dist, best = scored[0][0], scored[0][1].copy()
    clamped = [min(max(float(best[i]), ranges[i][0]), ranges[i][1]) for i in range(n)]
    return clamped, best_dist


def task_space_reach_ctrl(model, data, ee_site, target, actuated, clamps,
                          kp: float = 80.0, kd: float = 12.0, kdj: float = 2.0, site_id=None):
    """Operational-space (Jacobian-transpose) reach: servo the end-effector toward ``target`` in task
    space, writing gravity-compensated, clamped torques to ``data.ctrl``. Unlike PD-tracking a fixed
    IK joint config, this drives the *end-effector* directly and resolves redundancy in the null space —
    so it controls HIGH-DOF arms (6-7+) that joint-PD cannot dynamically realize (CPU-verified: a 7-DOF
    arm reaches 0.04 m here vs 0.22 m under joint-PD). The right base for high-DOF arms.

    Position-only by design: it reliably gets the end-effector TO the object (enough for the pick-place
    pin-grasp). Controlling the gripper ORIENTATION too (so the fingers straddle the box for a real
    friction grasp) needs a properly-tuned 6-DOF operational-space solver or a learned policy — naive
    J-transpose and a quick DLS both failed to do it robustly (docs/morphology_experiment_findings.md §9);
    that is the documented next effort, kept OUT of this controller so it stays simple + working.
    """
    import mujoco
    import numpy as np

    sid = ee_site if site_id is None else site_id
    dofs = [v for (_, v, _) in actuated]
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, sid)
    Ja = jacp[:, dofs]                               # task Jacobian over the actuated dofs (3 x n_act)
    ee = _ee_position(data, sid)
    ee_vel = Ja @ data.qvel[dofs]
    wrench = kp * (np.asarray(target, float) - ee) - kd * ee_vel
    tau = Ja.T @ wrench                              # Jacobian-transpose: redundancy-safe task-space force
    for k, (qadr, vadr, jnt) in enumerate(actuated):
        cmd = data.qfrc_bias[vadr] + tau[k] - kdj * data.qvel[vadr]
        data.ctrl[k] = float(np.clip(cmd, -clamps[k], clamps[k]))


# Tuned for reliable grasp ENGAGEMENT (measured pick_place 0.38 -> 1.00 across 8 scenes): the old soft gains
# (kp 9) let joint-PD settle SHORT of the grasp pose, so the fingertip never reached the block during the
# engage window (-> missed_grasp). Firmer gains drive the arm fully onto the target; a slightly lower grasp_z
# and wider engage radius latch the pin grasp reliably.
DEFAULT_CONTROLLER_PARAMS = {
    "kp": 14.0,
    "kd": 2.0,
    "phase_steps": 300,
    "engage_radius": 0.15,
    "hover": HOVER_M,
    "grasp_z": 0.05,
    "place_z": 0.13,
}


def run_contact_pick_place_episode(gene, objects, assignments: dict, *, record_frames=None, frame_every: int = 12,
                                   kpj: float = 150.0, kdj: float = 18.0, kpf: float = 300.0, kdf: float = 10.0,
                                   dq: float = 0.06, fclose: float = 0.05, steps_per_block: int = 420,
                                   hover_z: float = 0.16, lift_z: float = 0.22, place_z: float = 0.12) -> dict:
    """REAL contact-grasp sort (NO _pin_block): for each assigned block, top-down grasp (close the finger
    actuators and hold the block by CONTACT FRICTION), lift, transport to its bin, lower, release. Drives the
    actual gripper; the block stays only by friction. Returns the same outcome shape as run_pick_place_episode
    plus ``grasp_model='contact'``. ``objects`` is the scene's SceneObject list; ``assignments`` maps block->bin.
    De-risked in build/_grasp_transport_derisk.py (4/4 place, grip held through transport)."""
    import mujoco
    import numpy as np

    from virturoid.services.gene_compiler import compile_gene_with_scene
    from virturoid.services.grasp_pose import arm_revolute_joints, best_top_down_pose, solve_top_down_grasp_pose

    obj_list = list(objects)
    pos = {o.name: tuple(o.pose_xyz_rpy[:3]) for o in obj_list}
    mj = mujoco.MjModel.from_xml_string(compile_gene_with_scene(gene, obj_list))
    mj.opt.iterations = 30; mj.opt.ls_iterations = 12
    tcp = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    ee = gene.end_effector()
    palm = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, ee.name) if ee else -1
    if tcp < 0 or palm < 0:
        return {"status": "failure", "failure_label": "no_grasp_site", "placed_count": 0,
                "block_count": len(assignments), "per_block": {}, "metrics": {"grasp_rate": 0.0}}
    nu = int(mj.nu); j_of = [int(mj.actuator_trnid[u, 0]) for u in range(nu)]
    arm_u = [u for u in range(nu) if int(mj.jnt_type[j_of[u]]) != int(mujoco.mjtJoint.mjJNT_SLIDE)]
    fin_u = [u for u in range(nu) if int(mj.jnt_type[j_of[u]]) == int(mujoco.mjtJoint.mjJNT_SLIDE)]
    finger_bodies = {int(mj.jnt_bodyid[j_of[u]]) for u in fin_u}
    finger_geoms = {g for g in range(mj.ngeom) if int(mj.geom_bodyid[g]) in finger_bodies}
    blocks = {}
    for name in assignments:
        jid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, f"free_{name}")
        if jid < 0 or assignments[name] not in pos:
            continue
        blocks[name] = {"qadr": int(mj.jnt_qposadr[jid]), "body": int(mj.jnt_bodyid[jid]),
                        "geom": mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, name)}
    # contact masks: the ARM collides with NOTHING (its links can't knock blocks/table); every manipulable BLOCK
    # collides with the table + the fingers; bins/decor are visual. (Generalizes grasp_eval._set_grasp_contacts.)
    block_bodies = {b["body"] for b in blocks.values()}
    for g in range(mj.ngeom):
        bd = int(mj.geom_bodyid[g])
        if bd in block_bodies:
            mj.geom_contype[g], mj.geom_conaffinity[g] = 0b11, 0b11
        elif bd in finger_bodies:
            mj.geom_contype[g], mj.geom_conaffinity[g] = 0b10, 0b10
        elif bd == 0:
            mj.geom_contype[g], mj.geom_conaffinity[g] = 0b01, 0b01
        else:
            mj.geom_contype[g], mj.geom_conaffinity[g] = 0, 0
    act_dof = np.array([int(mj.jnt_dofadr[j_of[u]]) for u in range(nu)])
    arm_qadr = [int(mj.jnt_qposadr[j_of[u]]) for u in arm_u]
    fr_lim = mj.actuator_forcerange[:, 1]
    KP = np.array([kpj if u in arm_u else kpf for u in range(nu)])
    KD = np.array([kdj if u in arm_u else kdf for u in range(nu)])
    arm_joints = arm_revolute_joints(mj)
    d = mujoco.MjData(mj); mujoco.mj_resetData(mj, d); mujoco.mj_forward(mj, d)
    work = mujoco.MjData(mj)
    stable = True
    frames = [] if record_frames is not None else None
    step = [0]

    def cap():
        if record_frames is not None and step[0] % frame_every == 0:
            record_frames.append(_capture_geom_frame(d, mj))
        step[0] += 1

    def finger_contacts(box_geom):
        touched = set()
        for i in range(int(d.ncon)):
            g1, g2 = int(d.contact[i].geom1), int(d.contact[i].geom2)
            if box_geom in (g1, g2):
                for g in (g1, g2):
                    if g in finger_geoms:
                        touched.add(int(mj.geom_bodyid[g]))
        return len(touched)

    def ik(x, y, z, seed_map):
        return solve_top_down_grasp_pose(mj, work, site_id=tcp, palm_body_id=palm, target=(x, y, z),
                                         arm_joints=arm_joints, seed=[seed_map[qa] for qa, _ in arm_joints], iters=250)

    q_ref = np.array([float(d.qpos[arm_qadr[k]]) for k in range(len(arm_joints))])
    per_block: dict = {}
    first = True
    p_settle, p_grip, p_lift, p_transport, p_lower, p_release = 0.12, 0.30, 0.40, 0.55, 0.82, 0.92
    for name, info in blocks.items():
        bx, by, _ = pos[name]
        binx, biny, _ = pos[assignments[name]]
        mujoco.mj_resetData(mj, work)
        above_map, _t, _r = best_top_down_pose(mj, work, site_id=tcp, palm_body_id=palm,
                                               target=(bx, by, hover_z), arm_joints=arm_joints, iters=250)
        bz = float(d.qpos[info["qadr"] + 2])
        Q = {"above": above_map}
        Q["at"] = ik(bx, by, bz, above_map)
        Q["lift"] = ik(bx, by, lift_z, Q["at"])
        Q["over"] = ik(binx, biny, lift_z, Q["lift"])
        Q["inb"] = ik(binx, biny, place_z, Q["over"])
        Qa = {k: np.array([m[qa] for qa, _ in arm_joints]) for k, m in Q.items()}
        # APPROACH: slew to THIS block's hover (fingers open) until the arm actually ARRIVES, so EVERY block starts
        # its grasp from the hover pose — not mid-slew from the previous bin. (The first-block-only pre-fold made
        # block 1 succeed but block 2+ missed: they never reached their hover before the timed descend fired.)
        for _ in range(180):
            q_ref = q_ref + np.clip(Qa["above"] - q_ref, -dq, dq)
            q_star = np.zeros(nu)
            for k, u in enumerate(arm_u):
                q_star[u] = q_ref[k]
            q = d.qpos[act_dof]; qd = d.qvel[act_dof]
            d.ctrl[:] = np.clip(d.qfrc_bias[act_dof] + KP * (q_star - q) - KD * qd, -fr_lim, fr_lim)
            mujoco.mj_step(mj, d)
            if not np.all(np.isfinite(d.qpos)):
                stable = False; break
            cap()
            arm_q = np.array([float(d.qpos[a]) for a in arm_qadr])
            if float(np.max(np.abs(q_ref - Qa["above"]))) < 0.02 and float(np.max(np.abs(arm_q - Qa["above"]))) < 0.06:
                break
        if not stable:
            break
        first = False
        max_both = 0
        for t in range(steps_per_block):
            frac = t / steps_per_block
            if frac < p_settle:
                q_tgt = Qa["above"]
            elif frac < p_lift:
                q_tgt = Qa["at"]
            elif frac < p_transport:
                q_tgt = Qa["lift"]
            elif frac < p_lower:
                q_tgt = Qa["over"]
            elif frac < p_release:
                q_tgt = Qa["inb"]
            else:
                q_tgt = Qa["over"]
            q_ref = q_ref + np.clip(q_tgt - q_ref, -dq, dq)
            f_tgt = fclose if (p_grip <= frac < p_release) else 0.0
            q_star = np.zeros(nu)
            for k, u in enumerate(arm_u):
                q_star[u] = q_ref[k]
            for u in fin_u:
                q_star[u] = f_tgt
            q = d.qpos[act_dof]; qd = d.qvel[act_dof]
            d.ctrl[:] = np.clip(d.qfrc_bias[act_dof] + KP * (q_star - q) - KD * qd, -fr_lim, fr_lim)
            mujoco.mj_step(mj, d)
            if not np.all(np.isfinite(d.qpos)):
                stable = False; break
            if p_grip <= frac < p_lower:
                max_both = max(max_both, finger_contacts(info["geom"]))
            cap()
        if not stable:
            break
        bp = d.qpos[info["qadr"]:info["qadr"] + 3]
        dist = float(np.hypot(float(bp[0]) - binx, float(bp[1]) - biny))
        grasped = max_both >= 2
        if dist < BIN_RADIUS_M and float(bp[2]) < 0.2:
            outcome = "placed"
        elif not grasped:
            outcome = "missed_grasp"
        else:
            outcome = "dropped"
        per_block[name] = {"grasped": bool(grasped), "outcome": outcome,
                           "final_xy": [round(float(bp[0]), 4), round(float(bp[1]), 4)], "dist_to_bin_m": round(dist, 4)}

    if not stable:
        status, failure_label = "failure", "instability"
    else:
        failure_label = next((b["outcome"] for b in per_block.values() if b["outcome"] != "placed"), None)
        status = "success" if (per_block and failure_label is None) else ("failure" if per_block else "failure")
    n = max(1, len(per_block))
    placed_count = sum(1 for b in per_block.values() if b.get("outcome") == "placed")
    grasped_count = sum(1 for b in per_block.values() if b.get("grasped"))
    return {"status": status, "failure_label": failure_label, "placed_count": placed_count,
            "block_count": len(per_block), "per_block": per_block, "grasp_model": "contact",
            "metrics": {"grasp_rate": round(grasped_count / n, 3), "place_rate": round(placed_count / n, 3),
                        "stable": stable}}


def run_pick_place_episode(model, scene: dict, params: dict | None = None, perception=None, target_policy=None, record_frames=None, frame_every: int = 12, **overrides) -> dict:
    """Run one full sorting episode in real physics and classify the outcome.

    ``scene["objects"]`` maps object name -> (x, y, z). ``params`` tunes the
    controller ("brain"). If ``perception`` is given, the controller targets
    *perceived* block poses (from a PerceptionAdapter) instead of privileged sim
    poses, and cannot act on blocks it fails to detect — perception in the loop.
    """
    import mujoco
    import numpy as np

    cfg = {**DEFAULT_CONTROLLER_PARAMS, **(params or {}), **overrides}
    kp = float(cfg["kp"])
    kd = float(cfg["kd"])
    phase_steps = int(cfg["phase_steps"])
    engage_radius = float(cfg["engage_radius"])
    hover = float(cfg["hover"])
    grasp_z = float(cfg["grasp_z"])
    place_z = float(cfg["place_z"])

    objects = scene["objects"]
    # Spec-driven object->target mapping (Phase 2): a task can assign any manipulable object to
    # any target zone. Defaults to the color-sort mapping so existing builds are unchanged.
    assignments = scene.get("assignments", _BLOCK_TO_BIN)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    actuated = _actuated_joint_map(model)
    clamps = _actuator_force_clamps(model)
    ee_site = _ee_site_id(model)

    addrs = {
        name: _free_addrs(model, name)
        for name, target_name in assignments.items()
        if name in objects and target_name in objects
    }

    # per-episode manipulation telemetry (plan §11.1/§32.3): EE path length + peak simultaneous contacts,
    # accumulated across all phases — feeds richer failure analysis + the future IL dataset.
    metrics_acc = {"ee_path": 0.0, "peak_contacts": 0, "prev_ee": None}
    last_targets = [0.0] * model.nu
    stable = True
    latched = {"block": None}
    step_counter = [0]
    # DOF-based base selection: a long redundant arm (>=5 revolute joints) can't be dynamically driven
    # to an IK joint config by joint-PD (the 7-DOF arm settled 0.22 m off — experiment §7); for those,
    # servo the end-effector in TASK space toward the phase's cartesian point instead. Low-DOF arms keep
    # the proven joint-PD path unchanged.
    n_revolute = sum(1 for (_, _, jnt) in actuated if int(model.jnt_type[jnt]) == 3)
    high_dof = n_revolute >= 5

    # Reach mode (per-arm). 'joint' drives PD toward the robust seed-IK joint targets (plan_joint_targets — the
    # same solver grasp_eval uses to hit 100%); 'task' servos the EE in task space (J-transpose/DLS). NEITHER is
    # universal: joint-PD nails a 6-DOF arm (1.00) but a redundant 7-DOF arm settles ~0.22 m off (-> 0.00),
    # while task-space reaches the 7-DOF arm but UNDER-reaches the 6-DOF one (0.38). So 'auto' (default) PROBES
    # this arm — can joint-PD actually settle the EE onto a representative grasp target? — and picks the mode
    # that works for THIS body. Measured: 6-DOF -> joint, 7-DOF -> task, both then 1.00.
    reach_mode = str(cfg.get("reach_mode", "auto")).lower()
    if reach_mode == "joint":
        use_task_space = False
    elif reach_mode == "task":
        use_task_space = high_dof
    else:
        use_task_space = high_dof and not _joint_pd_can_reach(
            model, ee_site, actuated, clamps, kp, kd, objects, addrs, grasp_z)

    def drive(targets, steps, grasp=None, engage=None, point=None):
        nonlocal stable, last_targets
        last_targets = targets
        engaged = False
        for _ in range(steps):
            if use_task_space and point is not None:
                task_space_reach_ctrl(model, data, ee_site, point, actuated, clamps)
            else:
                for slot, (qadr, vadr, jnt) in enumerate(actuated):
                    tgt = targets[slot] if slot < len(targets) else 0.0
                    if model.jnt_limited[jnt]:
                        lo, hi = model.jnt_range[jnt]
                        tgt = min(max(tgt, lo), hi)
                    # Gravity/Coriolis compensation: a heavy composed arm (real-BOM motor masses) needs up to
                    # ~11 Nm just to HOLD the shoulder against gravity at the lift pose — more than kp*err can
                    # supply, so without this the arm sags to table height with the block pinned to it (block
                    # never lifts -> "dropped"). qfrc_bias supplies exactly the hold torque so PD only closes the
                    # position error; ~0 for light arms (no-op). Mirrors task_space_reach_ctrl's gravcomp pattern.
                    torque = data.qfrc_bias[vadr] + kp * (tgt - data.qpos[qadr]) - kd * data.qvel[vadr]
                    c = clamps[slot]
                    data.ctrl[slot] = float(max(-c, min(c, torque)))
            mujoco.mj_step(model, data)
            if not np.all(np.isfinite(data.qpos)):
                stable = False
                return engaged
            ee = _ee_position(data, ee_site)
            if metrics_acc["prev_ee"] is not None:
                metrics_acc["ee_path"] += float(np.linalg.norm(ee - metrics_acc["prev_ee"]))
            metrics_acc["prev_ee"] = ee.copy()
            if int(data.ncon) > metrics_acc["peak_contacts"]:
                metrics_acc["peak_contacts"] = int(data.ncon)
            # Latch the grasp at the closest approach during the descent.
            if engage is not None and not engaged:
                if float(np.linalg.norm(ee - _block_pos(data, addrs[engage]))) < engage_radius:
                    engaged = True
            if grasp is not None or engaged:
                _pin_block(data, addrs[engage or grasp], ee)
            step_counter[0] += 1
            if record_frames is not None and step_counter[0] % frame_every == 0:
                record_frames.append(_capture_geom_frame(data, model))
        return engaged

    # Perception in the loop: target perceived poses; undetected blocks can't be acted on.
    detected = None
    if perception is not None:
        mujoco.mj_forward(model, data)  # compute geom/camera world poses before sensing
        seen = perception.perceive(model, data, list(addrs))
        detected = {b: tuple(seen[b]["pos"][:3]) for b in addrs if b in seen}

    per_block = {}
    for block in addrs:
        if detected is not None and block not in detected:
            per_block[block] = {"grasped": False, "detected": False}
            continue
        bx, by = (detected[block][0], detected[block][1]) if detected is not None else (objects[block][0], objects[block][1])
        bin_x, bin_y, _ = objects[assignments[block]]
        base_seed = _stable_seed(block)

        def plan(point, seed):
            # Learned policy (behavior cloning) predicts targets with no search;
            # otherwise fall back to the FK-IK planner.
            if target_policy is not None:
                return target_policy.predict(point)
            return plan_joint_targets(model, point, seed=seed)[0]

        p_above_block = (bx, by, 0.05 + hover)
        p_at_block = (bx, by, grasp_z)
        p_above_bin = (bin_x, bin_y, 0.05 + hover)
        p_in_bin = (bin_x, bin_y, place_z)
        above_block = plan(p_above_block, base_seed)
        at_block = plan(p_at_block, base_seed + 1)
        above_bin = plan(p_above_bin, base_seed + 2)
        in_bin = plan(p_in_bin, base_seed + 3)

        drive(above_block, phase_steps, point=p_above_block)
        grasped = drive(at_block, phase_steps, engage=block, point=p_at_block)
        if grasped and stable:
            drive(above_block, phase_steps, grasp=block, point=p_above_block)
            drive(above_bin, phase_steps, grasp=block, point=p_above_bin)
            # The PLACE phase is the one that decides success, yet it used to get HALF the steps of every
            # other phase. Task-space servoing (the redundant-arm path) converges by iterated J-transpose/DLS
            # steps, so a halved budget lands the block short: MEASURED a 7-DOF arm settling 7.7 cm from the
            # bin versus 1.4 cm for the 6-DOF joint-PD arm, which is the #185 sort regression (0.4 < 0.7).
            # Joint-PD arms already converge well inside the half budget, so their behaviour is unchanged.
            drive(in_bin, phase_steps if use_task_space else phase_steps // 2, grasp=block, point=p_in_bin)
            drive(above_bin, phase_steps // 2, point=p_above_bin)  # release over the bin
        per_block[block] = {"grasped": grasped}
        if not stable:
            break

    drive(last_targets, 200)  # settle

    failure_label = None
    for block, info in per_block.items():
        pos = _block_pos(data, addrs[block])
        match = objects[assignments[block]]
        # Any OTHER assigned target counts as a wrong placement (generalizes the sort's
        # red/blue swap to arbitrary object->target tasks).
        other_targets = [t for t in set(assignments.values()) if t != assignments[block] and t in objects]
        d_match = float(np.hypot(pos[0] - match[0], pos[1] - match[1]))
        info.update({"final_xy": [round(float(pos[0]), 4), round(float(pos[1]), 4)], "dist_to_bin_m": round(d_match, 4)})
        if d_match < BIN_RADIUS_M and pos[2] < 0.2:
            info["outcome"] = "placed"
        elif not info["grasped"]:
            info["outcome"] = "missed_grasp"
        elif any(float(np.hypot(pos[0] - objects[t][0], pos[1] - objects[t][1])) < BIN_RADIUS_M for t in other_targets):
            info["outcome"] = "wrong_bin"
        else:
            info["outcome"] = "dropped"
        if info["outcome"] != "placed" and failure_label is None:
            failure_label = info["outcome"]

    if not stable:
        status, failure_label = "failure", "instability"
    elif failure_label is None:
        status = "success"
    else:
        status = "failure"
    n_blocks = max(1, len(per_block))
    placed_count = sum(1 for b in per_block.values() if b.get("outcome") == "placed")
    grasped_count = sum(1 for b in per_block.values() if b.get("grasped"))
    # grasped but not placed = the object slipped/was mis-released (a real drop/mis-sort signal)
    drop_count = sum(1 for b in per_block.values() if b.get("grasped") and b.get("outcome") != "placed")
    dists = [b["dist_to_bin_m"] for b in per_block.values() if "dist_to_bin_m" in b]
    return {
        "status": status,
        "failure_label": failure_label,
        "placed_count": placed_count,
        "block_count": len(per_block),
        "per_block": per_block,
        # plan §11.1/§32.3 per-episode manipulation metrics (for failure clustering + the IL dataset)
        "metrics": {
            "grasp_rate": round(grasped_count / n_blocks, 3),
            "place_rate": round(placed_count / n_blocks, 3),
            "drop_count": drop_count,
            "mean_dist_to_bin_m": round(sum(dists) / len(dists), 4) if dists else None,
            "ee_path_len_m": round(metrics_acc["ee_path"], 4),
            "peak_contacts": metrics_acc["peak_contacts"],
            "stable": stable,
        },
    }


def _capture_geom_frame(data, model) -> list:
    """Per-geom world pose snapshot for the 3D viewer: [x, y, z, qw, qx, qy, qz]."""
    import numpy as np

    frame = []
    for gid in range(model.ngeom):
        pos = data.geom_xpos[gid]
        mat = np.array(data.geom_xmat[gid], dtype=float).reshape(3, 3)
        quat = _mat_to_quat(mat)
        frame.append([round(float(pos[0]), 4), round(float(pos[1]), 4), round(float(pos[2]), 4),
                      round(quat[0], 4), round(quat[1], 4), round(quat[2], 4), round(quat[3], 4)])
    return frame


def _mat_to_quat(m):
    import numpy as np

    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        return [0.25 / s, (m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s]
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        return [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    if m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        return [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
    return [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]


def _stable_seed(name: str) -> int:
    """Process-independent seed from a name (Python's str hash is randomized)."""
    return int.from_bytes(hashlib.md5(name.encode("utf-8")).digest()[:4], "big") % 100000


def _free_addrs(model, block_name: str) -> tuple[int, int]:
    import mujoco

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"free_{block_name}")
    if jid < 0:
        return (-1, -1)
    return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])


def _block_pos(data, addrs: tuple[int, int]):
    import numpy as np

    qadr = addrs[0]
    return np.array(data.qpos[qadr : qadr + 3], dtype=float)


def _pin_block(data, addrs: tuple[int, int], ee_pos) -> None:
    qadr, dadr = addrs
    if qadr < 0:
        return
    data.qpos[qadr] = ee_pos[0]
    data.qpos[qadr + 1] = ee_pos[1]
    data.qpos[qadr + 2] = ee_pos[2] - GRASP_TIP_OFFSET_M
    data.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    for i in range(6):
        data.qvel[dadr + i] = 0.0


def joint_pd_reaches_point(model, ee_site, actuated, clamps, kp, kd, point,
                           *, tol: float = 0.06, steps: int = 320) -> bool:
    """Probe whether joint-PD to the seed-IK target can actually settle the EE onto ``point`` for THIS arm —
    on a throwaway copy of the state. Returns True (-> joint-PD reaches it precisely) or False (-> the arm is
    redundant/under-damped enough that task-space servoing is more reliable). One short rollout; lets any
    task-space controller auto-pick the reach mode that works per body (a 6-DOF arm reaches; a redundant
    7-DOF one settles ~0.22 m off and needs task-space)."""
    import mujoco
    import numpy as np

    pt = np.asarray(point, dtype=float)
    tgt = plan_joint_targets(model, pt, seed=7)[0]
    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)
    for _ in range(steps):
        for slot, (qadr, vadr, jnt) in enumerate(actuated):
            t = tgt[slot] if slot < len(tgt) else 0.0
            if model.jnt_limited[jnt]:
                lo, hi = model.jnt_range[jnt]
                t = min(max(t, lo), hi)
            torque = kp * (t - d.qpos[qadr]) - kd * d.qvel[vadr]
            d.ctrl[slot] = float(max(-clamps[slot], min(clamps[slot], torque)))
        mujoco.mj_step(model, d)
        if not np.all(np.isfinite(d.qpos)):
            return False
    return float(np.linalg.norm(_ee_position(d, ee_site)[:2] - pt[:2])) < tol


def _joint_pd_can_reach(model, ee_site, actuated, clamps, kp, kd, objects, addrs, grasp_z,
                        *, tol: float = 0.06, steps: int = 320) -> bool:
    """Pick-place probe: can joint-PD settle the EE onto a representative grasp target (the first block)?"""
    block = next(iter(addrs), None)
    if block is None or block not in objects:
        return True
    bx, by = objects[block][0], objects[block][1]
    return joint_pd_reaches_point(model, ee_site, actuated, clamps, kp, kd, (bx, by, grasp_z),
                                  tol=tol, steps=steps)
