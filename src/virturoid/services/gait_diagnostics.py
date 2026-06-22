"""Gait-QUALITY diagnostics — the metrics that distinguish WALKING from a forward shuffle/drift/lean.

Forward velocity alone is a gameable proxy (the locomotion-collapse lesson): a body can lean, slide, or
stiff-drag itself forward and score positive ``fwd_vel`` while never taking a step. This module reads a rollout
(base trajectory + per-foot heights over time) and computes process-level gait quality — foot-contact CADENCE,
left/right ALTERNATION, duty factor, swing CLEARANCE, base-height stability, uprightness, lateral drift — and a
single ``is_walking`` verdict + ``gait_quality`` score. This is the signal the AI-assisted trainer's LLM critic
reasons over to redesign the reward toward a real gait (Eureka-style), instead of cranking a gameable number.
"""

from __future__ import annotations


def diagnose_gait(qpos, foot_z, base_qadr: int, dt: float, *, foot_z0=None, stand_z: float | None = None,
                  contact_eps: float = 0.03) -> dict:
    """Compute gait-quality metrics from a recorded rollout.

    ``qpos`` (T, nq) base+joint positions per step; ``foot_z`` (T, n_feet) world heights of the foot geoms;
    ``base_qadr`` qpos address of the free-base joint (x,y,z, then wxyz quat); ``dt`` sim seconds/step;
    ``foot_z0`` (n_feet,) each foot's standing height (contact reference) — defaults to each foot's min height;
    ``stand_z`` the base STANDING height (the spawn/upright height) — the reference for fall/collapse detection;
    defaults to the initial base height. Returns metrics + ``is_walking`` + ``gait_quality`` (0..1) + a summary.

    A WALK must hold its standing height and keep moving the WHOLE time — a crouch-lunge that travels a bit then
    collapses and freezes is NOT walking (it games a height-vs-mean check + fakes cadence by flailing). So `fell`
    and `height_held` are referenced to the STANDING height (not the trajectory mean), and forward progress is
    also checked over the LAST third of the episode (sustained, not a one-time lunge)."""
    import numpy as np

    qpos = np.asarray(qpos, dtype=float)
    foot_z = np.asarray(foot_z, dtype=float)
    T = qpos.shape[0]
    nF = foot_z.shape[1] if foot_z.ndim == 2 else 0
    dur = max(T * dt, 1e-6)

    bx, by, bz = qpos[:, base_qadr], qpos[:, base_qadr + 1], qpos[:, base_qadr + 2]
    quat = qpos[:, base_qadr + 3:base_qadr + 7]                  # wxyz
    upright = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)    # world-up · body-up
    z_stand = float(stand_z) if stand_z is not None else float(bz[0])   # STANDING height = fall/collapse reference
    forward_m = float(bx[-1] - bx[0])
    mean_speed = forward_m / dur
    # SUSTAINED forward: progress over the LAST third — a lunge-then-freeze (x stuck) reads ~0 here, not a walk.
    k3 = max(1, (2 * T) // 3)
    late_speed = float(bx[-1] - bx[k3]) / max((T - k3) * dt, 1e-6)
    lateral_drift = float(np.max(np.abs(by - by[0]))) if T else 0.0
    bh_mean, bh_min = float(np.mean(bz)), float(np.min(bz))
    bh_cv = float(np.std(bz) / max(bh_mean, 1e-6))              # bobbing/collapse (high = unstable)
    up_mean = float(np.mean(upright))
    tall = bz > 0.7 * z_stand                                   # at near-standing height (not crouched/collapsed)
    height_held = float(np.mean(tall))                          # fraction of the episode spent at standing height
    # FELL/COLLAPSED is referenced to the STANDING height, NOT the trajectory mean (a slow sink drags the mean
    # down so a mean-relative check misses it — the bug your eyes caught): a real fall drops below ~0.6*z_stand.
    fell = bool(bh_min < 0.6 * z_stand or float(np.min(upright)) < 0.0 or height_held < 0.5)

    # --- foot-contact schedule: a foot is "in stance" when near its standing/ground height ---
    if nF >= 1:
        ref = np.asarray(foot_z0, float) if foot_z0 is not None else foot_z.min(axis=0)
        contact = foot_z <= (ref[None, :] + contact_eps)        # (T, nF) bool stance mask
        duty = contact.mean(axis=0)                             # fraction of time in stance per foot
        liftoffs = np.sum(contact[:-1] & ~contact[1:], axis=0)  # stance->swing transitions = steps
        n_steps = int(liftoffs.sum())
        cadence = n_steps / dur                                 # steps per second (0 = no stepping = drag)
        clearance = (foot_z - ref[None, :]).max(axis=0)         # peak swing height per foot
        max_clearance = float(np.max(clearance))
        mean_duty = float(np.mean(duty))
    else:
        duty = []; n_steps = 0; cadence = 0.0; max_clearance = 0.0; mean_duty = 1.0; contact = None

    # --- alternation: anti-phase contact between the two primary feet (1 = perfect L/R walk) ---
    alternation = 0.0
    if nF >= 2 and contact is not None:
        c0 = contact[:, 0].astype(float); c1 = contact[:, 1].astype(float)
        c0 -= c0.mean(); c1 -= c1.mean()
        denom = (np.linalg.norm(c0) * np.linalg.norm(c1)) or 1.0
        in_phase = float(np.dot(c0, c1) / denom)               # +1 in-phase (hop), -1 anti-phase (walk)
        alternation = float(max(0.0, min(1.0, (1.0 - in_phase) / 2.0)))

    # --- composite verdict: WALKING needs SUSTAINED forward progress, HELD standing height, real stepping,
    # clearance, alternation, and uprightness — all of them, the whole episode (no lunge-then-collapse). ---
    is_walking = bool(mean_speed > 0.1 and late_speed > 0.05 and height_held > 0.7
                      and cadence > 0.6 and max_clearance > 0.02
                      and up_mean > 0.6 and alternation > 0.3 and not fell)
    gait_quality = float(
        0.16 * min(mean_speed / 0.4, 1.0)            # makes forward progress
        + 0.16 * min(max(late_speed, 0.0) / 0.3, 1.0)  # KEEPS moving (not a one-time lunge)
        + 0.20 * height_held                         # HOLDS its standing height (anti-collapse)
        + 0.20 * min(cadence / 2.0, 1.0)             # actually steps (anti-shuffle)
        + 0.12 * min(max_clearance / 0.06, 1.0)      # lifts the feet
        + 0.16 * alternation                         # left/right gait, not a hop/drag
    )
    if fell:
        gait_quality = min(gait_quality, 0.2)         # a fall/collapse caps quality regardless of forward number

    diagnosis = _summarize(mean_speed, late_speed, height_held, cadence, max_clearance, alternation,
                           mean_duty, up_mean, fell, is_walking)
    return {
        "is_walking": is_walking, "gait_quality": round(gait_quality, 3),
        "forward_m": round(forward_m, 3), "mean_speed": round(mean_speed, 3),
        "late_speed": round(late_speed, 3), "height_held": round(height_held, 3),
        "cadence_steps_per_s": round(cadence, 3), "n_steps": n_steps,
        "alternation": round(alternation, 3), "mean_duty_factor": round(mean_duty, 3),
        "max_foot_clearance_m": round(max_clearance, 4), "lateral_drift_m": round(lateral_drift, 3),
        "base_height_mean": round(bh_mean, 3), "base_height_min": round(bh_min, 3), "stand_z": round(z_stand, 3),
        "base_height_cv": round(bh_cv, 3), "upright_mean": round(up_mean, 3), "fell": fell,
        "summary": diagnosis,
    }


def _summarize(speed, late_speed, height_held, cadence, clearance, alternation, duty, upright, fell, walking) -> str:
    """A short natural-language read of the gait — what the LLM critic (and the user) acts on."""
    if fell:
        if height_held < 0.5:
            return (f"COLLAPSED: crouches/sinks and holds standing height only {height_held*100:.0f}% of the time "
                    f"(travelled {speed:.2f} m/s early then froze at {late_speed:.2f} m/s). Not a gait.")
        return "FELL/collapsed — unstable; not a gait."
    if walking:
        return (f"WALKING: steps at {cadence:.1f}/s, feet alternate ({alternation:.2f}), "
                f"clearance {clearance*100:.0f}cm, sustained {late_speed:.2f} m/s upright.")
    bits = []
    if height_held < 0.7:
        bits.append(f"loses its stance (upright at standing height only {height_held*100:.0f}% of the time)")
    if late_speed < 0.05 and speed > 0.1:
        bits.append(f"lunges then STOPS (early {speed:.2f} m/s, late {late_speed:.2f} m/s) — not sustained")
    if cadence < 0.6:
        bits.append(f"NOT STEPPING (cadence {cadence:.2f}/s) — dragging/sliding feet")
    if clearance < 0.02:
        bits.append(f"feet barely leave ground (clearance {clearance*100:.1f}cm)")
    if duty > 0.85:
        bits.append(f"feet almost always in contact (duty {duty:.2f}) — shuffle, not stride")
    if alternation < 0.3 and cadence >= 0.6:
        bits.append(f"feet move together ({alternation:.2f}) — hopping, not L/R walk")
    if speed < 0.1:
        bits.append(f"barely moves forward ({speed:.2f} m/s)")
    if upright < 0.6:
        bits.append(f"leaning ({upright:.2f} upright)")
    return ("NOT WALKING: " + "; ".join(bits)) if bits else "borderline gait"
