"""Top-down grasp pose: orientation-aware inverse kinematics for a vertical (palm-down) approach.

Position-only IK (``plan_joint_targets``) places a gripper's grasp TCP *at* the object but leaves the
palm at whatever orientation the arm's configuration happens to produce. The morphology experiment
(findings §12) traced composed grasp failures to exactly this: a yaw + 2-pitch arm reaches a tabletop
box with the palm tilted ~26 deg, so the jaws close at a slant and *sweep the object away* instead of
straddling it. A grasp needs the approach orientation decoupled from reach — which requires (a) a
**wrist** degree of freedom in the body (so the palm CAN be levelled) and (b) IK that solves for
position AND orientation together.

This module provides that solver: damped least-squares over the stacked translational + rotational
site Jacobian, driving the grasp TCP to ``target`` while rotating the palm's local +z axis to point
along ``down`` (straight down for a top-down tabletop grasp). On a wristed arm it converges to a
vertical approach (tilt ~0 deg, sub-mm position residual); on an underactuated arm it returns the best
compromise (and ``approach_tilt_deg`` reports the residual tilt, which the caller can treat as a
grasp-readiness signal). It is the shared approach primitive for both scripted grasp attempts and as
the reset / nominal pose for a learned grasp policy.
"""

from __future__ import annotations


def solve_top_down_grasp_pose(model, data, *, site_id: int, palm_body_id: int, target,
                              arm_joints, seed=None, iters: int = 300, step: float = 0.6,
                              damping: float = 0.01, down=(0.0, 0.0, -1.0),
                              orient_weight: float = 1.0):
    """Solve for arm joint angles placing ``site_id`` at ``target`` with the palm's +z along ``down``.

    ``arm_joints`` is a list of ``(qpos_adr, dof_col)`` for the revolute arm joints (e.g. from
    ``_actuated_joint_map`` filtered to hinge joints). Runs DLS IK in place on ``data`` (so callers
    can read the resulting kinematics) and returns ``{qpos_adr: angle}`` for the arm joints. ``seed``
    (optional, aligned with ``arm_joints``) sets the starting configuration — a roughly folded-down
    seed avoids the IK settling into an elbow-up branch.
    """
    import mujoco
    import numpy as np

    target = np.asarray(target, dtype=float)
    down = np.asarray(down, dtype=float)
    cols = [c for _, c in arm_joints]
    # Per-joint limits (jnt_range where limited) so the IK only returns poses the robot can actually
    # hold — without this the solver happily wraps a joint past its stop (e.g. a wrist to -3.87 rad on a
    # +/-3.14 joint), the sim then clamps it, and the gripper lands short of the object (findings §12).
    joints = [int(model.dof_jntid[c]) for c in cols]
    lo = [float(model.jnt_range[j, 0]) if model.jnt_limited[j] else -1e9 for j in joints]
    hi = [float(model.jnt_range[j, 1]) if model.jnt_limited[j] else 1e9 for j in joints]
    if seed is not None:
        for (qa, _), v in zip(arm_joints, seed):
            data.qpos[qa] = float(v)

    for _ in range(iters):
        mujoco.mj_forward(model, data)
        jp = np.zeros((3, model.nv))
        jr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, site_id)
        e_pos = target - data.site_xpos[site_id]
        z_axis = data.xmat[palm_body_id].reshape(3, 3)[:, 2]
        e_orient = np.cross(z_axis, down) * orient_weight     # rotate palm +z toward 'down'
        err = np.concatenate([e_pos, e_orient])
        jac = np.vstack([jp[:, cols], jr[:, cols]])
        dq = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(6), err)
        for i, ((qa, _), delta) in enumerate(zip(arm_joints, dq)):
            data.qpos[qa] = min(hi[i], max(lo[i], data.qpos[qa] + step * float(delta)))

    mujoco.mj_forward(model, data)
    return {qa: float(data.qpos[qa]) for qa, _ in arm_joints}


def _seed_candidates(n: int):
    """Generic starting configurations for the grasp IK — a few folded-arm patterns plus zeros — so the
    solver doesn't depend on a hand-crafted, layout-specific seed (a yaw-first seed fails on a purely
    planar pitch arm and vice-versa). The best is chosen by achieved tilt + position residual."""
    patterns = [[0.0] * n,
                [0.0, 0.9, -1.3, -0.5, 0.0, 0.0, 0.0],
                [0.0, 1.4, -1.0, -0.8, 0.0, 0.0, 0.0],
                [0.6, -1.0, -1.0, -0.5, 0.0, 0.0, 0.0],
                [0.0, 0.6, -1.2, 0.4, -0.5, 0.0, 0.0]]
    return [p[:n] + [0.0] * max(0, n - len(p)) for p in patterns]


def best_top_down_pose(model, data, *, site_id: int, palm_body_id: int, target, arm_joints,
                       seeds=None, iters: int = 250, down=(0.0, 0.0, -1.0)):
    """Solve the top-down grasp pose from several seeds and return the best ``(qmap, tilt_deg,
    residual)`` — the seed-robust entry point. ``qmap`` is ``{qpos_adr: angle}``; on return ``data`` is
    left at the best solution so the caller can read its kinematics or continue from it."""
    import numpy as np

    seeds = seeds if seeds is not None else _seed_candidates(len(arm_joints))
    best = None
    for seed in seeds:
        qmap = solve_top_down_grasp_pose(model, data, site_id=site_id, palm_body_id=palm_body_id,
                                         target=target, arm_joints=arm_joints, seed=seed, iters=iters,
                                         down=down)
        residual = float(np.linalg.norm(np.asarray(target, dtype=float) - data.site_xpos[site_id]))
        tilt = approach_tilt_deg(model, data, palm_body_id, down=down)
        score = residual + 0.01 * tilt           # position first, then orientation
        if best is None or score < best[0]:
            best = (score, dict(qmap), tilt, residual)
    # re-apply the best solution so 'data' reflects it on return
    for qa, val in best[1].items():
        data.qpos[qa] = val
    import mujoco
    mujoco.mj_forward(model, data)
    return best[1], best[2], best[3]


def approach_tilt_deg(model, data, palm_body_id: int, down=(0.0, 0.0, -1.0)) -> float:
    """Angle (degrees) between the palm's +z axis and ``down`` — 0 is a perfect top-down approach.

    A grasp-readiness signal: > ~10 deg means the jaws would close at a slant and risk sweeping the
    object (the underactuated-arm failure mode). ``mj_forward`` must have been called on ``data``.
    """
    import numpy as np

    z_axis = data.xmat[palm_body_id].reshape(3, 3)[:, 2]
    down = np.asarray(down, dtype=float)
    cos = abs(float(z_axis @ down) / (np.linalg.norm(z_axis) * np.linalg.norm(down) + 1e-12))
    return float(np.degrees(np.arccos(min(1.0, max(-1.0, cos)))))


def arm_revolute_joints(model):
    """List ``(qpos_adr, dof_col)`` for the actuated *revolute* joints (the arm joints, excluding
    prismatic gripper fingers) — the joint set the grasp IK is solved over."""
    import mujoco

    from virturoid.services.mujoco_runner import _actuated_joint_map

    out = []
    for (qadr, vadr, jnt) in _actuated_joint_map(model):
        if int(model.jnt_type[jnt]) == int(mujoco.mjtJoint.mjJNT_HINGE):
            out.append((qadr, vadr))
    return out
