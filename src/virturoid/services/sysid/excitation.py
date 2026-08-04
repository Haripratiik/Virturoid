"""The excitation experiment: the short, safe, information-rich thing an engineer actually runs on the robot.

This is the one artefact in the whole wedge that leaves our machine and touches somebody's hardware, so its
constraints are physical, not stylistic:

**It must be short.** Nobody runs a twenty-minute experiment on a robot they care about. The default budget is
120 seconds of motion for the whole machine; a 14-DOF quadruped comes out at ~50 s. When the budget cannot cover
every joint at the minimum useful length the plan says so in ``budget`` and does not quietly truncate, because a
shortened segment does not fail loudly -- it produces a confident number for a parameter the data never loaded,
which is exactly the failure ``identifiability`` exists to prevent.

**It must respect the hardware.** Every joint's amplitude is bounded by the gene's declared limits with a 15%
back-off at each end, and every joint's frequency is bounded by the motor the BOM actually sized for it
(``component_catalog.select_actuator``): a sinusoid of amplitude A at frequency f demands peak torque
``I*(2*pi*f)^2*A + b*(2*pi*f)*A`` on top of gravity, and peak speed ``2*pi*f*A``. Both are held under a fraction
of the datasheet's peak torque and no-load speed, and the plan names WHICH of the two bound each joint. When the
bound is too tight to leave a useful band the amplitude is reduced and the frequency re-solved; when even that
fails the joint is marked ``excitable: false`` with the binding constraint named rather than being commanded
anyway.

**It must be informative, and differently informative in different places.** Four segments per joint, each
carrying a parameter the others cannot:

  ``triangle``    slow, near-constant speed with repeated SIGN REVERSALS -> viscous damping, and the only thing
                  that makes Coulomb friction separable from a constant gravity offset (with no reversal the two
                  columns are collinear and neither is identifiable -- see ``identifiability``).
  ``chirp``       logarithmic sweep f_lo -> f_hi -> reflected inertia / armature, and the bandwidth at which the
                  sim and the hardware start to disagree.
  ``step_train``  square setpoint changes -> control-signal latency (rise delay) and breakaway stiction.
  ``hold``        a settle back at the start pose so the next joint begins from rest instead of inheriting the
                  last one's ringing.

Joints are excited ONE AT A TIME. Exciting them together would be shorter, and would also make each joint's
torque residual contain its neighbours' coupling terms -- buying a shorter experiment with an unidentifiable
result. The safety envelope reused here is the same ``select_actuator`` ladder the BOM and
``control_script_compiler`` size against, so the excitation cannot ask for a motor the robot does not have.

CAVEAT, stated because a script that moves someone's hardware should carry one: bounded by datasheet torque and
declared joint limits is NECESSARY, not SUFFICIENT. It does not know about cable routing, end-stop dampers,
thermal state, or a payload someone left bolted on. This is a plan to be reviewed by the robot's owner, not a
plan to be run unattended.
"""

from __future__ import annotations

# ---- safety envelope. Every one of these is a fraction of a DECLARED limit, never an absolute guess. ----
LIMIT_BACKOFF = 0.15          # shrink each joint's declared range by this at BOTH ends before using it
AMP_FRACTION = 0.30           # sinusoid amplitude as a fraction of the remaining half-range
AMP_CAP_RAD = 0.35            # ...and never more than ~20 deg of swing, whatever the range allows
TORQUE_FRACTION = 0.50        # peak commanded torque stays under half the motor's datasheet peak
SPEED_FRACTION = 0.35         # peak joint speed stays under this fraction of datasheet no-load speed
F_LO_HZ = 0.4                 # bottom of the sweep
F_HI_CAP_HZ = 5.0             # top of the sweep, before the torque/speed bounds cut it further
F_MIN_USEFUL_HZ = 0.8         # below this the chirp carries no inertia information -> shrink amplitude and retry
AMP_RETRY_SHRINK = 0.6
AMP_RETRY_MAX = 3

# ---- time budget ----
DEFAULT_BUDGET_S = 120.0      # the whole machine, not per joint
MIN_PER_JOINT_S = 2.5         # below this the four segments stop being separable
MAX_PER_JOINT_S = 6.0
SEGMENT_FRACTIONS = (("triangle", 0.34), ("chirp", 0.42), ("step_train", 0.18), ("hold", 0.06))
DEFAULT_CONTROL_HZ = 100.0


def _actuator_envelope(seg):
    """``(peak_torque_nm, no_load_speed_radps, part_name)`` for a segment, from the real motor the BOM sized.

    Selection is driven by ``torque_req_nm`` (the joint's fixed REQUIREMENT) rather than by
    ``actuator_torque_nm`` (the selected motor's peak), so re-selecting here does not ratchet up a rung the way
    ``bom_builder.py:469`` documents. The torque ceiling then takes the MORE CONSERVATIVE of the catalog peak and
    the gene's own figure: if the two disagree, the smaller one is the one that cannot break anything.
    """
    from virturoid.services.bom_builder import _DEFAULT_JOINT_TORQUE_NM
    from virturoid.services.component_catalog import select_actuator

    declared = getattr(seg, "actuator_torque_nm", None)
    req = getattr(seg, "torque_req_nm", None) or declared or _DEFAULT_JOINT_TORQUE_NM
    act = select_actuator(float(req))
    peak = float(act.peak_torque_nm)
    if declared:
        peak = min(peak, float(declared))
    return peak, float(act.max_speed_radps), act.name


def _max_frequency(inertia, damping, amp, gravity_nm, peak_nm, speed_radps):
    """Highest sinusoid frequency (Hz) whose peak torque and peak speed both stay inside the datasheet envelope.

    Torque budget: ``I*w^2*A + b*w*A + |tau_gravity| <= TORQUE_FRACTION * peak``. Solving the quadratic in
    ``w = 2*pi*f`` gives the torque ceiling; ``w*A <= SPEED_FRACTION * no_load`` gives the speed ceiling. Returns
    ``(f_hz, binding_constraint)`` -- and ``(0.0, 'gravity')`` when holding the joint up already spends the
    budget, which is a real finding about the robot, not an error.
    """
    import math

    headroom = TORQUE_FRACTION * peak_nm - abs(gravity_nm)
    if headroom <= 0.0:
        return 0.0, "gravity"
    if amp <= 1e-9:
        return 0.0, "amplitude"
    ia, ba = max(inertia, 1e-9) * amp, max(damping, 0.0) * amp
    w_torque = (-ba + math.sqrt(ba * ba + 4.0 * ia * headroom)) / (2.0 * ia)
    w_speed = SPEED_FRACTION * max(speed_radps, 1e-9) / amp
    f_torque, f_speed = w_torque / (2.0 * math.pi), w_speed / (2.0 * math.pi)
    f = min(F_HI_CAP_HZ, f_torque, f_speed)
    binding = "bandwidth_cap"
    if f_torque <= f_speed and f_torque < F_HI_CAP_HZ:
        binding = "peak_torque"
    elif f_speed < f_torque and f_speed < F_HI_CAP_HZ:
        binding = "no_load_speed"
    return max(f, 0.0), binding


def build_excitation(gene, *, budget_s: float = DEFAULT_BUDGET_S, control_hz: float = DEFAULT_CONTROL_HZ,
                     backoff: float = LIMIT_BACKOFF) -> dict:
    """Author the bench experiment for ``gene``. Pure measurement + arithmetic; no rollout, no LLM.

    The returned plan is JSON-serializable and is the contract between us and the engineer: it declares the
    setup, the PD gains, the per-joint signal, the safety envelope that produced it, and the log schema we need
    back. ``excitation_command_series`` turns it into the actual command array.
    """
    import numpy as np

    from virturoid.services.sysid.bench_rig import (
        bench_gains,
        bench_model,
        inverse_torque,
        joint_dof_map,
        rest_pose,
        safe_band,
        start_pose,
        torque_ceiling,
    )

    model, rig = bench_model(gene)
    kp, kd, inertia = bench_gains(model)
    dofs = joint_dof_map(model, gene)
    lo, hi = safe_band(model, gene, backoff=backoff)
    q0 = start_pose(model, gene, backoff=backoff)
    q_rest = rest_pose(model)
    ceil_model = torque_ceiling(model)

    # Gravity/bias torque at the start pose: mj_inverse with zero velocity and zero acceleration.
    zero = np.zeros((1, model.nv))
    gravity = inverse_torque(model, q0[None, :], zero, zero)[0]

    dt = float(model.opt.timestep)
    log_hz = 1.0 / dt
    ctrl_every = max(1, int(round(log_hz / float(control_hz))))
    control_hz_actual = log_hz / ctrl_every

    segs = [s for s in gene.actuated_joints() if s.name in dofs]
    n = len(segs)
    per_joint = MIN_PER_JOINT_S if n == 0 else min(MAX_PER_JOINT_S, max(MIN_PER_JOINT_S, budget_s / n))
    over_budget = bool(n and per_joint * n > budget_s + 1e-9)

    joints = []
    for seg in segs:
        adr = dofs[seg.name]
        peak_nm, speed_radps, part = _actuator_envelope(seg)
        band_lo, band_hi = float(lo[adr]), float(hi[adr])
        centre = float(np.clip(q0[adr], band_lo, band_hi))
        head = min(centre - band_lo, band_hi - centre)
        amp = min(AMP_FRACTION * head, AMP_CAP_RAD)
        f_hi, binding = _max_frequency(float(inertia[adr]), float(model.dof_damping[adr]), amp,
                                       float(gravity[adr]), peak_nm, speed_radps)
        shrinks = 0
        while f_hi < F_MIN_USEFUL_HZ and shrinks < AMP_RETRY_MAX and amp > 1e-4:
            amp *= AMP_RETRY_SHRINK
            shrinks += 1
            f_hi, binding = _max_frequency(float(inertia[adr]), float(model.dof_damping[adr]), amp,
                                           float(gravity[adr]), peak_nm, speed_radps)
        # A joint is EXCITABLE if it can be moved safely at all. A low frequency ceiling is a weaker, separate
        # verdict: the slow segments still load friction and damping, so declaring the whole joint unusable
        # would throw away two identifiable parameters to protect a third. This is reported as a FACT about the
        # chirp and nothing more -- an earlier version also predicted "expect armature unidentified" here, and
        # the measurement contradicted it (the step train carries acceleration whatever the chirp ceiling is,
        # and armature came back identified at t = 70 on a 0.64 Hz joint). Only the data rules.
        excitable = bool(f_hi > 0.0 and amp > 1e-4 and head > 1e-4)
        bandwidth_limited = bool(excitable and f_hi < F_MIN_USEFUL_HZ)
        joints.append({
            "joint": seg.name, "dof": int(adr), "excitable": excitable,
            "chirp_bandwidth_limited": bandwidth_limited,
            "q_start_rad": round(centre, 5),
            "rest_pose_rad": round(float(q_rest[adr]), 5),
            # The engineer physically has to put the joint here before starting, so a start pose that is NOT
            # the robot's rest stance has to be called out rather than buried in a number.
            "moved_off_rest_pose": bool(abs(centre - float(q_rest[adr])) > 1e-4),
            "amplitude_rad": round(float(amp), 5), "amplitude_deg": round(float(np.degrees(amp)), 3),
            "f_lo_hz": round(min(F_LO_HZ, f_hi / 2.0) if f_hi > 0 else 0.0, 4),
            "f_hi_hz": round(float(f_hi), 4),
            "binding_constraint": binding,
            "amplitude_shrinks": shrinks,
            "safe_band_rad": [round(band_lo, 5), round(band_hi, 5)],
            "declared_limits_rad": [seg.joint_lower, seg.joint_upper],
            "actuator_part": part,
            "datasheet_peak_torque_nm": round(peak_nm, 4),
            "datasheet_no_load_speed_radps": round(speed_radps, 3),
            "model_forcerange_nm": round(float(ceil_model[adr]), 4),
            "gravity_hold_nm": round(float(gravity[adr]), 4),
            "joint_inertia_kgm2": round(float(inertia[adr]), 6),
            "kp": round(float(kp[adr]), 4), "kd": round(float(kd[adr]), 4),
            "not_excitable_because": None if excitable else (
                f"no safe motion exists for this joint: at every amplitude down to {AMP_RETRY_MAX} shrinks, "
                f"holding it already spends {int(TORQUE_FRACTION*100)}% of the {part} peak torque "
                f"({peak_nm:.2f} N.m); binding constraint: {binding}"),
            "chirp_bandwidth_limited_because": None if not bandwidth_limited else (
                f"the safe frequency ceiling is {f_hi:.2f} Hz, under the {F_MIN_USEFUL_HZ} Hz target: {binding} "
                f"on the {part} ({peak_nm:.2f} N.m peak, {speed_radps:.1f} rad/s no-load) binds first. The "
                f"slow segments and the step train still run; whether that is enough is decided by the "
                f"identifiability report on the returned log, not predicted here"),
        })

    n_ex = sum(1 for j in joints if j["excitable"])
    duration = round(per_joint * n_ex, 3)
    return {
        "ok": True,
        "robot": getattr(gene, "id", ""),
        "setup": {
            "description": "Robot ON A STAND. Base rigidly clamped, limbs hanging free, nothing in contact.",
            "why": "isolates actuator dynamics + latency (the residual Hwangbo et al. attribute the gap to) "
                   "from contact, which this experiment does NOT measure",
            **rig,
        },
        "controller": {"law": "PD to a joint position setpoint, torque commanded",
                       "control_hz": round(control_hz_actual, 3),
                       "note": "these gains are part of the experiment -- the gap number is not reproducible "
                               "under different ones, and the latency search re-simulates this exact loop"},
        "budget": {"requested_s": float(budget_s), "per_joint_s": round(per_joint, 3),
                   "duration_s": duration, "n_joints": n, "n_excitable": n_ex,
                   "over_budget": over_budget,
                   "note": ("per-joint time is at the MIN_PER_JOINT floor; the four segments may not be "
                            "separable, expect identifiability to degrade" if per_joint <= MIN_PER_JOINT_S + 1e-9
                            else "")},
        "segments": [{"name": nm, "fraction": fr, "seconds": round(per_joint * fr, 3), "identifies": purpose}
                     for (nm, fr), purpose in zip(SEGMENT_FRACTIONS, (
                         "viscous damping; makes Coulomb friction separable from the gravity offset via velocity "
                         "sign reversals",
                         "reflected inertia / armature, and the frequency where sim and hardware diverge",
                         "control-signal latency (rise delay) and breakaway stiction",
                         "nothing -- a settle so the next joint starts from rest"))],
        "safety_envelope": {
            "limit_backoff_frac": backoff, "amplitude_frac_of_halfrange": AMP_FRACTION,
            "amplitude_cap_rad": AMP_CAP_RAD, "torque_frac_of_datasheet_peak": TORQUE_FRACTION,
            "speed_frac_of_no_load": SPEED_FRACTION,
            "reviewed_by_a_human": False,
            "caveat": "bounded by datasheet torque and declared joint limits is NECESSARY, not SUFFICIENT: it "
                      "knows nothing about cable routing, end-stop dampers, thermal state or a bolted-on "
                      "payload. Have the robot's owner review this before running it.",
        },
        "log_schema": {
            "log_hz": round(log_hz, 3),
            "required": ["t_s", "q_cmd_rad", "q_meas_rad"],
            "strongly_recommended": ["tau_meas_nm"],
            "optional": ["qd_meas_radps"],
            "note": "without tau_meas only the trajectory gap can be reported -- no parameter can be "
                    "attributed, because the residual that carries the attribution is a TORQUE residual",
        },
        "joints": joints,
        "excitation_order": [j["joint"] for j in joints if j["excitable"]],
    }


def excitation_command_series(gene, plan: dict):
    """Realize ``plan`` as the actual command array: ``(t, q_cmd)`` with one row per log sample.

    Joints not currently being excited are held at their start pose, so at any instant exactly one joint is
    moving -- which is what makes each joint's torque residual attributable to that joint.
    """
    import numpy as np

    from virturoid.services.sysid.bench_rig import bench_model, start_pose

    model, _ = bench_model(gene)
    dt = float(model.opt.timestep)
    log_hz = 1.0 / dt
    per_joint = float(plan["budget"]["per_joint_s"])
    rows_per_joint = int(round(per_joint * log_hz))
    active = [j for j in plan["joints"] if j["excitable"]]
    total = max(1, rows_per_joint * len(active))

    q0 = start_pose(model, gene)
    for j in plan["joints"]:
        q0[int(j["dof"])] = float(j["q_start_rad"])
    cmd = np.tile(q0, (total, 1))

    for i, j in enumerate(active):
        adr = int(j["dof"])
        a = i * rows_per_joint
        cmd[a:a + rows_per_joint, adr] = _joint_signal(rows_per_joint, log_hz, float(j["q_start_rad"]),
                                                       float(j["amplitude_rad"]), float(j["f_lo_hz"]),
                                                       float(j["f_hi_hz"]))
    return np.arange(total) * dt, cmd


def _joint_signal(rows, log_hz, centre, amp, f_lo, f_hi):
    """The four segments, concatenated, for one joint."""
    import numpy as np

    counts = [max(1, int(round(rows * fr))) for _, fr in SEGMENT_FRACTIONS]
    counts[-1] += rows - sum(counts)
    n_tri, n_chirp, n_step, n_hold = counts
    out = []

    # triangle: constant |speed|, repeated sign reversals
    t = np.arange(n_tri) / log_hz
    period = max(n_tri / log_hz / 2.0, 1e-6)
    out.append(centre + amp * (2.0 / np.pi) * np.arcsin(np.sin(2.0 * np.pi * t / period)))

    # logarithmic chirp f_lo -> f_hi
    t = np.arange(n_chirp) / log_hz
    dur = max(n_chirp / log_hz, 1e-6)
    lo = max(f_lo, 1e-3)
    ratio = max(f_hi / lo, 1.0001)
    phase = 2.0 * np.pi * lo * dur / np.log(ratio) * (ratio ** (t / dur) - 1.0)
    out.append(centre + amp * np.sin(phase))

    # step train: 6 alternating blocks at 60% amplitude
    step = np.zeros(n_step)
    blk = max(1, n_step // 6)
    for b in range(6):
        step[b * blk:(b + 1) * blk] = amp * 0.6 * (1.0 if b % 2 == 0 else -1.0)
    out.append(centre + step)

    out.append(np.full(n_hold, centre))
    return np.concatenate(out)[:rows]
