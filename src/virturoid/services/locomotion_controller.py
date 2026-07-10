"""Locomotion controller + task-matched evaluation for LEGGED robots (startup plan §11/§32.1).

A scripted trot gait — the §32.1 baseline-first controller for a quadruped, the legged analogue of the
differential-drive baseline for wheeled bases. It settles the legs into a stance, then oscillates the
hips of diagonal leg pairs (with a knee lift on the forward swing) to walk forward, and measures the
distance travelled while staying upright. Generic: it finds the floating base, identifies hips (legs
attached directly to the torso) vs knees, and groups legs into diagonal pairs by their torso-frame
position — so it drives any composed quadruped, not a hard-coded one.
"""

from __future__ import annotations


def locomotion_status(forward_m: float, upright: bool, *, min_forward_m: float = 0.1) -> str:
    """Classify locomotion using signed forward progress.

    Planar distance is useful diagnostics, but cannot certify a walk: a robot
    that lurches backward half a metre has travelled far while failing the
    requested forward task.  Keep this small pure helper shared/testable so
    every caller uses the same anti-gaming boundary.
    """
    if not upright:
        return "fell"
    return "walked" if float(forward_m) > float(min_forward_m) else "stalled"


def run_locomotion_episode(model, horizon: int = 2400, settle: int = 400, freq: float = 0.05,
                           hip_amp: float = 0.4, knee_lift: float = 0.3, stance=(0.2, -0.4),
                           kp: float = 50.0, kd: float = 5.0, on_step=None) -> dict:
    """Trot a legged robot forward; return ``{distance_m, forward_m, upright, status}``. Needs MuJoCo.
    ``on_step(model, data, step)`` is called each gait step (e.g. to render a walking filmstrip)."""
    import math
    import numpy as np
    import mujoco

    from virturoid.services.pick_place_controller import _actuated_joint_map, _actuator_force_clamps

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)                                  # trot PD targets are tuned for qpos0; the
    mujoco.mj_forward(model, data)                                    # crawl gait (which adapts q_def) honors rest
    base_j = next((j for j in range(model.njnt) if int(model.jnt_type[j]) == 0), -1)
    if base_j < 0:
        return {"distance_m": 0.0, "forward_m": 0.0, "upright": False, "status": "no_floating_base"}
    qadr = int(model.jnt_qposadr[base_j])
    torso_body = int(model.jnt_bodyid[base_j])
    act = _actuated_joint_map(model)
    clamps = _actuator_force_clamps(model)

    # Classify each actuated revolute leg joint by AXIS + depth, so the gait handles real 3-DOF legs
    # (hip ABDUCTION/roll about body-x + hip PITCH + KNEE) as well as 2-DOF legs and any leg count:
    #   • roll/abduction (|axis_x| dominant) -> held at neutral (legs stay under the body),
    #   • first pitch joint down a leg        -> HIP (swings, phased by leg azimuth for a trot/tripod),
    #   • deeper pitch joint                  -> KNEE (lifts in phase with its hip).
    ROLL, HIP, KNEE = 0, 1, 2
    role = [None] * len(act)
    body_to_k = {int(model.jnt_bodyid[jnt]): k for k, (qa, va, jnt) in enumerate(act)
                 if int(model.jnt_type[jnt]) == 3}
    hips = []                                             # (k, azimuth) for each pitch hip
    for k, (qa, va, jnt) in enumerate(act):
        if int(model.jnt_type[jnt]) != 3:                 # only revolute leg joints
            continue
        ax = model.jnt_axis[jnt]
        if abs(ax[0]) > abs(ax[1]):                       # roll/abduction -> hold neutral
            role[k] = ROLL
            continue
        body = int(model.jnt_bodyid[jnt])
        pk = body_to_k.get(int(model.body_parentid[body]))
        if pk is not None and role[pk] in (HIP, KNEE):    # parent is a pitch joint -> this is the knee
            role[k] = KNEE
        else:                                             # first pitch in the leg -> hip
            role[k] = HIP
            bx = data.xpos[body][0] - data.xpos[torso_body][0]
            by = data.xpos[body][1] - data.xpos[torso_body][1]
            hips.append((k, math.atan2(by, bx)))
    hips.sort(key=lambda t: t[1])
    hip_phase = {k: math.pi * (rank % 2) for rank, (k, _) in enumerate(hips)}
    hip_body = {int(model.jnt_bodyid[act[k][2]]): k for k in hip_phase}

    def leg_phase_for(jnt):                               # walk up to the knee's hip to share its phase
        body = int(model.jnt_bodyid[jnt])
        while body > 0:
            if body in hip_body:
                return hip_phase[hip_body[body]]
            body = int(model.body_parentid[body])
        return 0.0

    stance_hip, stance_knee = stance

    def drive(step, gait):
        for k, (qa, va, jnt) in enumerate(act):
            r = role[k]
            if r == ROLL:
                tgt = 0.0                                 # hold abduction neutral
            elif not gait:
                tgt = stance_hip if r == HIP else stance_knee
            elif r == HIP:
                # PROPULSIVE trot: cos sweeps the hip MONOTONICALLY from +amp (forward) to -amp (back) over the
                # stance half (ph 0->pi), so a PLANTED foot drives the body forward. The old sin() form planted
                # the foot during a symmetric back-then-forward dip = zero net thrust (a shuffle-in-place, the
                # deep-analysis dog: forward -0.045 m). Diagonal legs are pi out of phase (hip_phase).
                tgt = stance_hip + hip_amp * math.cos(step * freq + hip_phase.get(k, 0.0))
            else:                                         # KNEE
                # Lift the foot during SWING only (sin<0, ph in (pi,2pi)) so it clears the ground on the forward
                # return and stays PLANTED through the whole backward propulsion sweep (sin>=0, ph in (0,pi)).
                tgt = stance_knee - knee_lift * max(0.0, -math.sin(step * freq + leg_phase_for(jnt)))
            data.ctrl[k] = float(np.clip(data.qfrc_bias[va] + kp * (tgt - data.qpos[qa]) - kd * data.qvel[va],
                                         -clamps[k], clamps[k]))
        mujoco.mj_step(model, data)

    for s in range(settle):
        drive(s, gait=False)
    p0 = np.array(data.qpos[qadr:qadr + 2])
    stand_z = float(data.qpos[qadr + 2]) or 1.0          # the body's OWN standing torso height (post-settle)
    min_tz = stand_z
    for s in range(horizon):
        drive(s, gait=True)
        min_tz = min(min_tz, float(data.qpos[qadr + 2]))
        if on_step is not None:
            on_step(model, data, s)
        if not np.all(np.isfinite(data.qpos)):
            return {"distance_m": 0.0, "forward_m": 0.0, "upright": False, "status": "instability"}
    p1 = np.array(data.qpos[qadr:qadr + 2])
    torso_z = float(data.qpos[qadr + 2])
    dist = float(np.linalg.norm(p1 - p0))
    # UN-GAMEABLE upright: the torso must END near, and never COLLAPSE far below, its OWN standing height — not
    # merely clear a fixed 0.12 m floor. A humanoid that stands 0.84 m and topples to 0.13 m cleared 0.12 m and was
    # falsely reported 'upright, walked' (a fall that drifts the base forward is not a walk). Relative to the body's
    # own height, so it still holds for a naturally-low quadruped/hexapod torso.
    upright = bool(torso_z > 0.5 * stand_z and min_tz > 0.4 * stand_z)
    # Honest status: an upright body that simply didn't travel far is STALLED (weak/mistuned gait), not fallen.
    # Conflating the two (the old label) hid the real failure mode the deep-analysis dog showed — upright but
    # shuffling in place — behind a 'fell' that implies a balance loss it never had.
    forward = float(p1[0] - p0[0])
    status = locomotion_status(forward, upright)
    return {"distance_m": round(dist, 3), "forward_m": round(forward, 3),
            "upright": upright, "status": status}
