"""Real-actuator sim model (breakthrough v5 B1). The single dominant sim-to-real actuator effect for the
servo/QDD tiers is NOT reflected inertia or friction — it is the TORQUE-SPEED CURVE: a real motor cannot deliver
peak torque at speed. Naive sims use a constant torque clamp, so policies learn to command peak torque at high
joint velocity (impossible on hardware) and shatter on transfer (Isaac Lab's `DCMotor`, ToddlerBot, legged-gym
all confirm training with REAL torque limits is what transfers).

Model (four-quadrant DC-motor saturation): available |torque| = tau_max up to the knee speed qd_tau_max, then
tapers LINEARLY to 0 at the no-load speed qd_max — but only in the DRIVING quadrants (torque and velocity same
sign, motor doing work). In the BRAKING quadrants (opposite sign) the full tau_max is available. All three
parameters come from catalog fields we already store: tau_max = peak_torque_nm, qd_max = max_speed_radps, and the
knee qd_tau_max defaults to the rated-torque speed (~0.5*qd_max unless the datasheet N-T graph says otherwise).

These functions are ``xp``-generic (numpy or jax.numpy) so the SAME clamp runs in the MJX trainer and the CPU
eval and the BOM certificate — if they diverged, the "executable-on-BOM" certificate would be fiction.
"""

from __future__ import annotations


def available_torque(qvel, tau_max, qd_tau_max, qd_max, *, xp):
    """Driving-side |torque| available at joint speed ``qvel`` (rad/s): tau_max, tapering to 0 between qd_tau_max
    and qd_max. Shapes broadcast; scalars or per-joint arrays both work."""
    speed = xp.abs(qvel)
    taper = xp.clip((qd_max - speed) / xp.maximum(qd_max - qd_tau_max, 1e-6), 0.0, 1.0)
    return tau_max * taper


def clamp_torque(tau_cmd, qvel, tau_max, qd_tau_max, qd_max, *, xp):
    """Four-quadrant torque clamp. DRIVING (``tau_cmd`` and ``qvel`` same sign): speed-limited by the envelope.
    BRAKING (opposite sign): full ``tau_max`` (a motor can always resist). Returns the realizable joint torque."""
    driving = available_torque(qvel, tau_max, qd_tau_max, qd_max, xp=xp)
    same_sign = (tau_cmd * qvel) >= 0.0
    lim = xp.where(same_sign, driving, tau_max)
    return xp.clip(tau_cmd, -lim, lim)


def knee_speed(qd_max: float, *, frac: float = 0.5) -> float:
    """Default knee (rated-torque) speed = a fraction of no-load speed when the datasheet N-T graph is unknown."""
    return max(1e-3, float(qd_max) * frac)
