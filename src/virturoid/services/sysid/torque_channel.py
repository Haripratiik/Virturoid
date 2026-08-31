"""Motor CURRENT as a torque channel -- converted through a torque constant, and never silently.

Most robots do not have a joint torque sensor. They have a current sense resistor in the motor driver, and the
SDK reports amps. What the estimators here need is newton-metres, and the two differ by one number: the
torque constant ``kt``, in N.m per amp, referred to the JOINT output (so the gearbox ratio and its efficiency
are inside it). Refusing a current log because the field is spelled ``current`` rather than ``effort`` would
be refusing most of the market over a unit conversion.

So the conversion is done, and it is done OUT LOUD. Three things make that safe rather than sloppy:

**The constant is never invented.** It comes from one of three places, in this order, and the result says
which: the number the customer read off their motor's datasheet; the number IDENTIFIED from their own log (the
slope between the torque the controller commanded and the current the motor drew -- see
``identify_from_command``); or, last, a value DERIVED from the catalog actuator the BOM sized for that joint,
which is a stand-in part and is labelled as one.

**What a wrong constant does is measured, and it is different for the two answers.** A torque constant is a
pure SCALE on the channel. The actuation delay is a LAG, so it barely cares: MEASURED on the Menagerie Go2,
``_delay_from_command_response`` returns the injected 0 / 20 / 40 ms EXACTLY with the channel scaled anywhere
from 0.5x to 2.0x, and what changes is the reconstruction gate -- ``fraction_of_applied_torque_explained``
falls 1.00 -> 0.87 -> 0.77 at 1.0x / 1.15x / 1.3x and through the 0.5 floor at 0.5x and 2.0x, where the
estimator correctly refuses. The parameter FIT is a different story: it regresses the torque residual, so a
scale error on the channel goes straight into frictionloss, damping and armature. That asymmetry is reported
rather than averaged away.

**A magnitude-only channel is refused, not squared away.** Several drivers report |I| with the sign in a
separate direction bit. Multiplying that by kt gives a torque that never reverses, which reads as a robot
whose actuators only ever push one way -- a confident, physically impossible input. When a joint's commanded
torque changes sign and its logged current does not, that joint is refused with the reason.
"""

from __future__ import annotations

#: Mechanical efficiency assumed for the drivetrain when kt has to be DERIVED from a catalog part. A planetary
#: or cycloidal reducer of the kind these actuators use lands roughly here; a harmonic drive can be lower under
#: light load. It is an assumption, it is stated in the returned record, and it is one of the three reasons a
#: derived constant carries a wide band.
DEFAULT_DRIVETRAIN_EFFICIENCY = 0.85

#: How wrong the DERIVATION is once you have the right part. Not a confidence interval -- there is no sampling
#: distribution here -- but the span of two assumptions: the SI identity kt[N.m/A] = ke[V.s/rad], from which
#: real machines depart by ~5-15% (magnetic saturation, winding temperature, a no-load speed quoted at a
#: different bus voltage), and the drivetrain efficiency above.
DERIVED_KT_UNCERTAINTY_FRAC = 0.25

#: ...and the error that DOMINATES it, stated separately because it is not of the same kind and is not bounded.
#: ``select_actuator`` picks the smallest catalog part that meets the joint's REQUIRED torque with a safety
#: margin. That is the right answer for a bill of materials and it is a STAND-IN for a torque constant, because
#: the customer's robot has whatever motor the customer bought. MEASURED on the Menagerie Go2: the BOM sizes a
#: T-Motor AK10-9 (48 N.m peak, derived kt 1.299 N.m/A) for the hips and a T-Motor AK80-64 (120 N.m, kt 6.476)
#: for the calves, while Unitree actually ships a GO-M8010-6 on every joint (23.7 N.m, derived kt 0.519). That
#: is 2.5x out on the hip and 12.5x on the calf -- a factor, not a percentage.
#:
#: Which is why nothing here rests on it: pass ``torque_constant_nm_per_a`` and this whole paragraph is moot,
#: and when you cannot, ``identify_from_command`` re-derives the constant from the customer's own log and
#: ``measure_gap`` reports the ratio. MEASURED, at a constant 1.35x wrong the delay still comes back exactly at
#: 0 / 20 / 40 ms, the cross-check reports 1.35, and the tracking gate refuses the fit (1.27x against 1.50x) --
#: the delay survives a wrong constant and the parameters do not, and both say so.
DERIVED_KT_PART_SUBSTITUTION = (
    "the catalog part is the one the BOM SIZES for this joint's torque requirement, not the motor on the "
    "customer's robot, and that error is a FACTOR rather than a percentage. Measured on a Menagerie Go2, the "
    "sized part's constant is 2.5x the real motor's on the hips and 12.5x on the calves. Pass the datasheet "
    "value, or read 'identified_from_this_log' in the gap report, before believing a derived one."
)

#: Below this correlation the "slope between commanded torque and logged current" is not a torque constant, it
#: is a line through a cloud. Identification refuses under it rather than returning the slope.
MIN_IDENTIFY_R2 = 0.5


def _actuator_for(seg):
    """The catalog part the BOM would size for this joint -- the same selection ``excitation`` bounds against.

    Driven by ``torque_req_nm`` (the joint's fixed REQUIREMENT) rather than by ``actuator_torque_nm`` (the
    already-selected motor's peak), so reading a constant here does not ratchet the selection up a rung.
    """
    from virturoid.services.bom_builder import _DEFAULT_JOINT_TORQUE_NM
    from virturoid.services.component_catalog import select_actuator

    declared = getattr(seg, "actuator_torque_nm", None)
    req = getattr(seg, "torque_req_nm", None) or declared or _DEFAULT_JOINT_TORQUE_NM
    return select_actuator(float(req))


def derived_torque_constant(actuator, *, efficiency: float = DEFAULT_DRIVETRAIN_EFFICIENCY) -> dict:
    """``kt`` at the JOINT OUTPUT, in N.m/A, from a catalog actuator's headline specs.

    For an ideal machine the torque constant and the back-EMF constant are the same number in SI, so
    ``ke = V_nominal / omega_motor_noload`` and ``kt_motor = ke``. Referring that to the output multiplies by
    the gear ratio and the drivetrain efficiency, while ``omega_motor = omega_output * ratio`` divides by it --
    so the ratio CANCELS and what is left is ``kt_out = efficiency * V / omega_output_noload``. Convenient, and
    worth noticing, because it means the answer does not depend on the one catalog field (gear ratio) that
    varies most between a real part and a stand-in for it.
    """
    speed = max(float(getattr(actuator, "max_speed_radps", 0.0) or 0.0), 1e-6)
    volts = float(getattr(actuator, "voltage_v", 0.0) or 0.0)
    kt = float(efficiency) * volts / speed
    return {
        "nm_per_a": round(kt, 6),
        "source": "derived_from_catalog_actuator",
        "part": getattr(actuator, "name", ""),
        "basis": (f"kt = efficiency * V / no-load speed = {efficiency:g} * {volts:g} V / {speed:g} rad/s, "
                  f"using the SI identity kt[N.m/A] = ke[V.s/rad] for an ideal machine. The gear ratio cancels "
                  f"between referring kt to the output and referring the no-load speed to the motor"),
        "assumed_drivetrain_efficiency": float(efficiency),
        "uncertainty_frac": DERIVED_KT_UNCERTAINTY_FRAC,
        "uncertainty_note": (f"+/-{DERIVED_KT_UNCERTAINTY_FRAC:.0%} is the DERIVATION's own span given this "
                             f"part. It does not cover the part being the wrong one, which is the larger "
                             f"error and is not a percentage"),
        "implied_peak_current_a": round(float(getattr(actuator, "peak_torque_nm", 0.0)) / kt, 2) if kt else None,
        "warning": f"{getattr(actuator, 'name', 'this part')}: " + DERIVED_KT_PART_SUBSTITUTION,
    }


def torque_constants(gene, *, explicit=None, efficiency: float = DEFAULT_DRIVETRAIN_EFFICIENCY) -> dict:
    """``{joint: constant_record}`` for every actuated joint.

    ``explicit`` is what the customer read off their datasheet: one number for the whole robot, or a
    ``{joint: nm_per_a}`` mapping for a machine with more than one motor type. Anything not named there falls
    back to the catalog derivation, per joint, and says so in its own record.
    """
    if isinstance(explicit, (int, float)):
        explicit = {seg.name: float(explicit) for seg in gene.actuated_joints()}
    explicit = {str(k): float(v) for k, v in (explicit or {}).items() if v is not None}

    out = {}
    for seg in gene.actuated_joints():
        if seg.name in explicit:
            kt = explicit[seg.name]
            out[seg.name] = {"nm_per_a": kt, "source": "stated_by_the_customer",
                             "basis": "read off the motor's datasheet and passed in with the log",
                             "uncertainty_frac": 0.0}
            continue
        try:
            out[seg.name] = derived_torque_constant(_actuator_for(seg), efficiency=efficiency)
        except Exception as exc:  # noqa: BLE001 - a missing catalog entry must not take the whole report down
            out[seg.name] = {"nm_per_a": None, "source": "unavailable",
                             "basis": f"no catalog actuator could be sized for this joint: "
                                      f"{type(exc).__name__}: {exc}"}
    return out


def convert_current_to_torque(gene, log: dict, *, explicit=None,
                              efficiency: float = DEFAULT_DRIVETRAIN_EFFICIENCY) -> tuple:
    """``(log_with_tau_meas, record)``. The log is COPIED; ``i_meas`` is left on it beside the conversion.

    Refuses per joint rather than globally where it can: a robot with one magnitude-only channel is still worth
    measuring on its other eleven joints, and the estimators already report per joint.
    """
    import numpy as np

    if log.get("tau_meas") is not None:
        return log, {"converted": False, "why": "the log already carries measured torque; the current channel "
                                                "was not needed and was not used"}
    raw = log.get("i_meas")
    if raw is None:
        return log, {"converted": False, "why": "the log carries neither tau_meas nor a motor-current channel"}

    names = [str(s) for s in (log.get("joints") or [])]
    consts = torque_constants(gene, explicit=explicit, efficiency=efficiency)
    cur = np.asarray(raw, dtype=float)
    if cur.ndim != 2 or cur.shape[1] != len(names):
        return log, {"converted": False,
                     "why": f"the current channel is {cur.shape}, which does not line up with the log's "
                            f"{len(names)} joint columns"}

    q_cmd = np.asarray(log.get("q_cmd"), dtype=float)
    q_meas = np.asarray(log.get("q_meas"), dtype=float)
    tau = np.zeros_like(cur)
    per_joint, refused, used = {}, [], []
    for j, name in enumerate(names):
        if name not in consts:
            # In the log but not an actuated joint of this robot. Nothing to convert and nothing to refuse:
            # the estimators never read this column.
            per_joint[name] = {"used": False, "nm_per_a": None, "source": "not_applicable",
                               "basis": "this joint is in the log but not in the robot's actuated joint set"}
            continue
        rec = dict(consts[name])
        kt = rec.get("nm_per_a")
        col = cur[:, j]
        # A magnitude-only channel: the joint was driven both ways (its tracking error reversed sign) and the
        # current never did. kt * |I| is a torque that only ever pushes one way, which is not a modelling
        # approximation -- it is a different robot.
        err = (q_cmd[:, j] - q_meas[:, j]) if (q_cmd.ndim == 2 and q_meas.ndim == 2) else None
        driven_both_ways = bool(err is not None and err.size and err.max() > 1e-3 and err.min() < -1e-3)
        signed = bool(col.size and col.max() > 0.0 and col.min() < 0.0)
        if kt is None or not (kt > 0.0):
            rec["used"] = False
            rec["refused_because"] = "no torque constant is available for this joint"
            refused.append(name)
        elif driven_both_ways and not signed:
            rec["used"] = False
            rec["refused_because"] = (
                "this joint's commanded direction reversed during the experiment and its logged current never "
                "changed sign, so the channel is a MAGNITUDE. Multiplying it by a torque constant would "
                "describe an actuator that can only push one way. Log the signed (q-axis) current, or the "
                "direction bit alongside it")
            refused.append(name)
        else:
            tau[:, j] = col * float(kt)
            rec["used"] = True
            used.append(name)
        per_joint[name] = rec

    # ANY refusal refuses the whole channel, and that is not fastidiousness -- it is the only representation
    # available. The log is one dense array, so a refused joint would have to be written as SOMETHING, and a
    # zero torque column does not read as "unknown", it reads as "this actuator applied nothing", which is a
    # confident lie exactly where the estimator is about to regress. Found by measuring: on a magnitude-only
    # Go2 log the two joints the plan actually excites were refused (their commanded direction reversed and
    # the current did not) while the ten idle joints passed the sign gate for want of any motion -- so the
    # conversion "succeeded" on ten columns carrying nothing and zeroed the two carrying everything. A log that
    # falls through here is not a dead end: it is treated as position-only, and still yields the delay.
    if refused or not used:
        return log, {
            "converted": False, "per_joint": per_joint, "joints_refused": sorted(refused),
            "joints_that_would_have_converted": sorted(used),
            "why": (f"the current channel could not be converted on {sorted(refused) or 'any joint'}, and a "
                    f"partial torque channel cannot be represented -- the log is one array, so a refused "
                    f"joint would be written as a zero torque, which reads as 'this actuator applied nothing'. "
                    f"See per_joint for the reason on each"),
            "what_still_works": "this log is now read as POSITION-ONLY: the per-joint trajectory gap and the "
                                "actuation delay are still reported; no parameter may be named.",
        }

    out = dict(log)
    out["tau_meas"] = tau.tolist() if isinstance(raw, list) else tau
    srcs = sorted({(per_joint[n].get("source") or "") for n in used})
    return out, {
        "converted": True,
        "units": "N.m = A * kt, per joint, with kt referred to the JOINT OUTPUT",
        "joints_converted": sorted(used),
        "joints_refused": sorted(refused),
        "constant_sources": srcs,
        "per_joint": per_joint,
        "headline": (f"tau_meas was COMPUTED from the logged motor current for {len(used)} joint(s) using a "
                     f"torque constant ({', '.join(srcs)}); it was not measured with a torque sensor."),
        "what_it_costs": (
            "the delay is a LAG and is insensitive to this constant -- MEASURED on the Go2, the injected "
            "0/20/40 ms come back exactly with the channel scaled 0.5x to 2.0x, and the reconstruction gate is "
            "what refuses a grossly wrong one. The fitted PARAMETERS are not: they come from a torque "
            "residual, so an error in kt is an error of the same fraction in frictionloss, damping and "
            "armature. Read the intervals with that on top."),
    }


def identify_from_command(want, tau_used, *, kt_used):
    """``kt`` from the log itself: the slope between the torque the controller COMMANDED and the channel.

    ``want`` and ``tau_used`` are already lag-aligned 1-D arrays for one joint, in N.m -- ``tau_used`` being
    the current already multiplied by ``kt_used``. A current loop tracks its torque command closely, so the
    slope of the one against the other is the factor by which ``kt_used`` is wrong, and
    ``kt_true = slope * kt_used``.

    Stated plainly because it is an assumption and not a measurement of the motor: this identifies the constant
    that makes the ACTUATOR APPEAR TO HAVE APPLIED WHAT IT WAS ASKED FOR. Any systematic torque-tracking error
    -- thermal derating, a current loop that undershoots, an unmodelled gearbox loss -- is absorbed into it.
    That is why it is reported as a CROSS-CHECK on a stated constant and only substituted for a derived one.
    """
    import numpy as np

    x = np.asarray(tau_used, dtype=float)
    y = np.asarray(want, dtype=float)
    if x.size < 16 or float(np.std(x)) < 1e-9:
        return None
    xc, yc = x - x.mean(), y - y.mean()
    denom = float(np.dot(xc, xc))
    if denom <= 1e-12:
        return None
    slope = float(np.dot(xc, yc) / denom)
    resid = yc - slope * xc
    var = float(np.dot(yc, yc))
    r2 = 1.0 - float(np.dot(resid, resid)) / var if var > 1e-12 else 0.0
    if r2 < MIN_IDENTIFY_R2 or slope <= 0.0:
        return {"nm_per_a": None, "r2": round(r2, 4), "slope": round(slope, 6),
                "why": (f"the logged current explains only {max(r2, 0.0):.1%} of the commanded torque "
                        f"(need {MIN_IDENTIFY_R2:.0%}), so its slope is not a torque constant")}
    return {"nm_per_a": round(float(kt_used) * slope, 6), "r2": round(r2, 4),
            "slope_against_the_constant_used": round(slope, 4),
            "source": "identified_from_this_log",
            "assumes": "the actuator applied the torque it was commanded, so any torque-tracking error is "
                       "inside this number"}
