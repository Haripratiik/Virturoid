"""The sim2sim gate. WE OWN NO HARDWARE, so this is how every number in this package was validated.

A second MuJoCo model with KNOWN perturbations stands in for the robot: extra joint friction, extra damping,
extra reflected inertia, and a whole-tick delay between computing a torque and applying it. We run the
excitation on it, hand the resulting log to ``measure_gap`` as though it had come off a bench, and check that
what comes back is what went in.

That is a real test of a real thing, and it is worth being exact about which thing:
"""

from __future__ import annotations

WHAT_SIM2SIM_DOES_NOT_PROVE = (
    "This gate validates the PIPELINE and the ESTIMATOR: that the excitation is emitted inside the declared "
    "envelope, that a perturbation of known size is recovered at known accuracy, that the delay search finds "
    "the delay, and that parameters the excitation could not load are reported as unidentified rather than "
    "guessed. It does NOT validate the PHYSICS. Both the 'robot' and the 'sim' are MuJoCo, so every modelling "
    "error MuJoCo itself makes -- its friction constraint, its contact model, its rigid-link assumption, "
    "gearbox backlash and elasticity it does not represent at all -- cancels exactly and is invisible here. "
    "Those are precisely the errors a real log exists to expose. The recovery numbers below are therefore an "
    "UPPER BOUND on this tool's accuracy on real hardware, not an estimate of it. No claim about a physical "
    "robot may be made from this gate; the first real-hardware log requires a design partner and that is an "
    "explicit ask, not an implementation detail."
)

#: The default injection. Sized to be a realistic modelling error rather than a caricature: roughly a doubling
#: of a mid-size joint's frictionloss, ~60% of its damping, and a reflected inertia comparable to a small
#: gearbox's -- plus 2 control ticks (20 ms) of the actuator delay Hwangbo et al. name as the dominant term.
#:
#: That 20 ms used to be the exact point at which the wedge stopped working on a real body, and the fact that
#: it is the DEFAULT is why the failure was invisible: every published number was taken on a composed dog whose
#: bench loop tolerates delay (worst tracking RMS 0.0339 -> 0.0365 rad across 0-40 ms) while the Menagerie Go2's
#: hips ring (0.0298 -> 0.1848 rad). Both halves are now measured on the Go2 -- the delay is recovered exactly
#: at 0/20/40 ms and the tracking gate passes at 0 and 20 -- and the lesson stands: a default this package
#: validates against is the one place a body-specific number is least likely to be noticed.
#:
#: IT IS AN ABSOLUTE OFFSET, SO ITS SIZE IS A PROPERTY OF THE BODY, and the sentence above no longer describes
#: the Go2. When ``robot_import`` started carrying the customer's DECLARED drivetrain into the twin instead of
#: substituting a structural prior, this Go2's joints went from damping 0.8 / frictionloss 0.12 to the 2.0 /
#: 0.2 Unitree writes. The same +0.6 / +0.08 offset that was a +75% / +67% modelling error on the substituted
#: drivetrain is a +30% / +40% one on the declared one, and the ring it was sized against largely went with it:
#: prior-replay-vs-log RMS across 0-40 ms is now 0.0031 -> 0.0659 rad on the full plan. Every ``improvement_x``
#: this package quotes is a ratio of trajectory RMS before and after fitting, so a smaller RELATIVE error
#: leaves less to remove and the ratio falls -- at 40 ms it went 1.484x -> 1.121x, which reads as a regression
#: and is not one. Anyone re-sizing this dict should size it as a FRACTION of each joint's declared value, and
#: re-measure ``tests/test_sysid_delay_wedge.py`` when they do; it is the file that notices.
DEFAULT_PERTURBATION = {"frictionloss": 0.08, "damping": 0.6, "armature": 0.03}
DEFAULT_DELAY_TICKS = 2


def perturbed_model(model, perturbation: dict):
    """A deep COPY of ``model`` with the named ``dof_*`` fields shifted. Never mutates the original -- a
    perturbed model that leaked back into the shared compile cache would silently change every other rollout."""
    import copy

    import numpy as np

    hw = copy.deepcopy(model)
    for name, delta in (perturbation or {}).items():
        attr = f"dof_{name}"
        if not hasattr(hw, attr):
            raise ValueError(f"unknown model parameter {name!r} (expected one of frictionloss/damping/armature)")
        getattr(hw, attr)[:] = np.maximum(np.asarray(getattr(model, attr), dtype=float) + float(delta), 0.0)
    return hw


#: What a customer's log actually carries, and the three cases the estimators have to serve. Which one you get
#: is a property of their DRIVER, not of their robot, and it decides which answers are reachable:
#:
#:   "torque"         per-joint torque in N.m. ROS 2 JointState.effort / ros2_control's effort state interface
#:                    when the driver populates it; Franka's tau_J (real link-side sensors); ANYdrive's
#:                    spring-deflection torque. Everything works.
#:   "current"        per-joint motor current in amps, no torque field. The common case: a current sense
#:                    resistor is in every motor driver and a torque sensor is in almost none. Converted
#:                    through a torque constant by ``sysid.torque_channel``, out loud.
#:   "position_only"  q_cmd and q_meas (and usually velocity) and nothing else. The PARAMETERS are unreachable
#:                    -- their residual is a torque residual -- but the ACTUATION DELAY is not, and is
#:                    recovered by ``gap_report._delay_from_motion``.
LOG_CHANNELS = ("torque", "current", "position_only")


def synthetic_hardware_log(gene, *, perturbation: dict | None = None, delay_ticks: int = DEFAULT_DELAY_TICKS,
                           plan: dict | None = None, budget_s: float = 120.0,
                           hold_only: bool = False, link_scale: float = 1.0,
                           inertia_scale: float = 1.0, torque_scale: float = 1.0,
                           channel: str = "torque", torque_constant_error: float = 1.0) -> tuple:
    """Run the excitation on a perturbed model and return ``(plan, log)`` in the exact schema a real bench log
    would arrive in -- so ``measure_gap`` cannot tell the difference, which is the point of the harness.

    ``channel`` picks which of the three real log flavours to emit (see ``LOG_CHANNELS``). ``current`` divides
    the applied torque by each joint's torque constant and ships amps instead of newton-metres;
    ``torque_constant_error`` then multiplies the constant used to GENERATE it, so the customer's real motor
    differs from the catalog stand-in the reader will assume by exactly that factor -- which is the adversarial
    case for the conversion, and the reason the reader reports its constant instead of assuming one.

    ``hold_only`` replaces the excitation with a constant hold: the robot is perturbed exactly as before but
    never moves. It is the adversarial case for the EXPERIMENT -- a real gap exists, and NOTHING in the data
    can locate it. A tool that returns three confident numbers here is broken in the way that matters, so this
    is a first-class mode rather than a test fixture.

    ``link_scale`` is the adversarial case for the FIT, and it is a different failure. It multiplies every
    link's mass and rotational inertia, which is an error NO combination of frictionloss, damping and armature
    can express -- and which enters each joint's equation through the SAME ``qdd`` term armature does, so the
    estimator absorbs it into armature and reports intervals that exclude the true, unchanged value. Measured
    at 1.30: armature "identified" on 14/14 joints, 13/14 intervals excluding the truth, one joint at +66%,
    and ``trajectory.improvement_x`` at 0.993 -- the fit made tracking marginally WORSE while every number in
    it looked like a measurement. That last figure is what ``fit.application_gate`` rules on, and this mode is
    how it is tested. An excitation cannot fix this one; only a wider model or a refusal can.

    ``inertia_scale`` and ``torque_scale`` are the OTHER two misspecifications, and they exist because this
    harness could only build ``link_scale`` -- which is why ``fit.MIN_TRACKING_IMPROVEMENT_X``'s band looked
    empty for as long as it did. Both are errors the estimator cannot REPORT and can substantially ABSORB, and
    both take ``improvement_x`` well past 1.0 (measured, composed dog, 35 s, delay 0; see
    ``docs/calibration_wedge_under_delay.md`` section 13):

      * ``inertia_scale`` multiplies each link's rotational inertia with its MASS HELD EXACTLY RIGHT, so there
        is no gravity signature at all and the whole error is in the ``qdd`` term -- the one armature adds to.
        At 30 (a +60% error in each joint's own diag(M)) the fit scores **1.745x and CLEARS the 1.5x gate**,
        with armature "identified" on 14/14 joints and 14/14 intervals excluding the true unchanged value.
      * ``torque_scale`` multiplies the actuator gear, so the plant really receives ``torque_scale`` x the
        torque the log records: a wrong gear ratio, a wrong torque constant, a current-sense gain error, or an
        unmodelled gearbox efficiency. Scaling the torque by g is algebraically the same as dividing M, b and f
        by g -- every one of which the estimator CAN move -- so the absorption is exact except for the gravity
        term, and the fitted deltas come back at the predicted ``(1 - g) / g`` times each prior (measured at
        g = 0.5: damping +0.841 against a predicted +0.80). At 1.25 it scores **1.536x and CLEARS the gate**.

    ``link_scale`` moves mass and inertia TOGETHER because that is the realistic density/fill-fraction error;
    ``inertia_scale`` deliberately does not, because separating them is what shows the gravity term is the only
    reason the ``link_scale`` family stays pinned at ~1.0.
    """
    import numpy as np

    from virturoid.services.sysid.bench_rig import (
        bench_gains,
        bench_model,
        joint_dof_map,
        pd_replay,
        start_pose,
    )
    from virturoid.services.sysid.excitation import build_excitation, excitation_command_series

    plan = plan or build_excitation(gene, budget_s=budget_s)
    model, _ = bench_model(gene)
    kp, kd, _ = bench_gains(model)
    dofs = joint_dof_map(model, gene)
    hw = perturbed_model(model, perturbation if perturbation is not None else DEFAULT_PERTURBATION)
    if float(link_scale) != 1.0:
        # Body 0 is the world. Mass and inertia move TOGETHER because scaling only one is a physically
        # incoherent robot, and the point of this mode is a realistic modelling error (a density or fill
        # fraction that is wrong), not a caricature.
        hw.body_mass[1:] = np.asarray(hw.body_mass[1:], dtype=float) * float(link_scale)
        hw.body_inertia[1:] = np.asarray(hw.body_inertia[1:], dtype=float) * float(link_scale)
    if float(inertia_scale) != 1.0:
        # Rotational inertia ALONE, mass untouched -- the half of a link error that armature can express.
        hw.body_inertia[1:] = np.asarray(hw.body_inertia[1:], dtype=float) * float(inertia_scale)
    if float(torque_scale) != 1.0:
        # ``pd_replay`` logs the torque it COMMANDED (``data.ctrl``); MuJoCo applies ``gear * ctrl``. Scaling
        # the gear is therefore exactly the customer whose driver reports a torque computed with the wrong
        # constant -- the log says one number and the joint receives another.
        hw.actuator_gear[:, 0] = np.asarray(hw.actuator_gear[:, 0], dtype=float) * float(torque_scale)

    _, q_cmd = excitation_command_series(gene, plan)
    q0 = start_pose(model, gene)
    for j in plan["joints"]:
        q0[int(j["dof"])] = float(j["q_start_rad"])
    if hold_only:
        q_cmd = np.tile(q0, (np.asarray(q_cmd).shape[0], 1))
    ctrl_every = max(1, int(round((1.0 / float(model.opt.timestep)) / float(plan["controller"]["control_hz"]))))
    t, q, qd, tau = pd_replay(hw, q_cmd, kp=kp, kd=kd, q_start=q0,
                              ctrl_every=ctrl_every, delay_ticks=int(delay_ticks))

    names = list(dofs)
    cols = [dofs[n] for n in names]
    chan = str(channel or "torque").strip().lower()
    if chan not in LOG_CHANNELS:
        raise ValueError(f"unknown log channel {channel!r} (expected one of {', '.join(LOG_CHANNELS)})")
    log = {
        "joints": names,
        "control_hz": float(plan["controller"]["control_hz"]),
        "log_hz": 1.0 / float(model.opt.timestep),
        "t": t.tolist(),
        "q_cmd": np.asarray(q_cmd)[:, cols].tolist(),
        "q_meas": q[:, cols].tolist(),
        "qd_meas": qd[:, cols].tolist(),
        # MACHINE-READABLE, and required by the log schema. A prose provenance string can be read by a human
        # and by nothing else; Stage 2's actuator-fidelity rung turns on whether the log was measured on
        # hardware, and it must not have to infer that from an English sentence it could mis-parse.
        "measured_on": "sim2sim",
        "provenance": "SYNTHETIC - a second MuJoCo model standing in for hardware. NOT a measurement of any "
                      "physical robot."
                      + (f" Its links carry {link_scale:g}x our mass and inertia -- an error no fitted "
                         f"parameter can express." if float(link_scale) != 1.0 else "")
                      + (f" Its links carry {inertia_scale:g}x our ROTATIONAL INERTIA at our exact mass -- an "
                         f"error no fitted parameter can REPORT, and one armature can largely absorb."
                         if float(inertia_scale) != 1.0 else "")
                      + (f" Its joints receive {torque_scale:g}x the torque this log records -- a wrong gear "
                         f"ratio / torque constant / gearbox efficiency."
                         if float(torque_scale) != 1.0 else ""),
        "excitation": "hold_only (deliberately uninformative)" if hold_only else "full plan",
        "channel": chan,
    }
    if chan == "torque":
        log["tau_meas"] = tau[:, cols].tolist()
    elif chan == "current":
        from virturoid.services.sysid.torque_channel import torque_constants
        consts = torque_constants(gene)
        kt = np.array([float((consts.get(n) or {}).get("nm_per_a") or 0.0) * float(torque_constant_error)
                       for n in names])
        if not np.all(kt > 0.0):
            raise ValueError("no torque constant could be sized for every joint, so a current channel cannot "
                             "be synthesised for this robot")
        log["i_meas"] = (tau[:, cols] / kt).tolist()
        log["torque_constant_used_to_generate_nm_per_a"] = {n: round(float(k), 6) for n, k in zip(names, kt)}
        log["provenance"] += (
            f" Its torque channel is MOTOR CURRENT: tau / kt per joint, with kt "
            + ("as the catalog derivation" if float(torque_constant_error) == 1.0 else
               f"deliberately {torque_constant_error:g}x the catalog derivation, so the reader's constant is "
               f"wrong by that factor")
            + ".")
    else:
        log["provenance"] += (" It carries NO torque and NO current channel: position and velocity only, which "
                              "is what a driver that does not populate effort gives you.")
    return plan, log


def recovery_table(gene, *, perturbation: dict | None = None, delay_ticks: int = DEFAULT_DELAY_TICKS,
                   budget_s: float = 120.0, plan: dict | None = None) -> dict:
    """Inject a known perturbation, recover it, and report the error -- per parameter, per joint.

    The honest reading of this table is in ``WHAT_SIM2SIM_DOES_NOT_PROVE``. What it can legitimately establish:
    the estimator is unbiased in the direction it claims, its error is bounded, and the parameters it declares
    identified are the ones it actually got right.
    """
    import time

    import numpy as np

    from virturoid.services.sysid.gap_report import measure_gap

    t0 = time.perf_counter()
    inj = dict(DEFAULT_PERTURBATION if perturbation is None else perturbation)
    plan, log = synthetic_hardware_log(gene, perturbation=inj, delay_ticks=delay_ticks,
                                       plan=plan, budget_s=budget_s)
    t_excite = time.perf_counter() - t0
    gap = measure_gap(gene, log, plan=plan)
    if not gap.get("ok"):
        return {"ok": False, "gap": gap}

    rows = gap["attribution"]["per_joint"]
    per_param = {}
    for p, injected in inj.items():
        ests = [r["parameters"][p]["estimate"] for r in rows.values() if p in r["parameters"]]
        flags = [p in r["identified"] for r in rows.values()]
        if not ests:
            continue
        med = float(np.median(ests))
        per_param[p] = {
            "injected": round(float(injected), 6),
            "recovered_median": round(med, 6),
            "recovered_min": round(float(np.min(ests)), 6),
            "recovered_max": round(float(np.max(ests)), 6),
            "error_abs": round(med - float(injected), 6),
            "error_pct": round(100.0 * (med - float(injected)) / float(injected), 2) if injected else None,
            "n_joints_identified": int(sum(flags)),
            "n_joints": len(ests),
            "unit": gap["attribution"]["parameters"].get(p, ""),
        }

    ctrl_hz = float(plan["controller"]["control_hz"])
    lat = gap.get("latency") or {}
    # ``delay_ms`` is None when NO joint could even be scored -- a hold-only log, say. Reporting an error of
    # "None minus 20" as a crash would turn a legitimate refusal into a broken harness.
    injected_ms = round(delay_ticks * 1000.0 / ctrl_hz, 3)
    recovered = lat.get("delay_ms")
    return {
        "ok": True,
        "robot": getattr(gene, "id", ""),
        "provenance": "sim2sim",
        "what_this_does_not_prove": WHAT_SIM2SIM_DOES_NOT_PROVE,
        "excitation": {"duration_s": plan["budget"]["duration_s"],
                       "n_joints": plan["budget"]["n_joints"],
                       "wall_clock_s": round(t_excite, 3)},
        "measure_gap_wall_clock_s": gap.get("wall_clock_s"),
        "parameters": per_param,
        "delay": {"injected_ms": injected_ms,
                  "recovered_ms": recovered,
                  "identified": bool(lat.get("identified")),
                  "error_ms": None if recovered is None else round(float(recovered) - injected_ms, 3),
                  "source": lat.get("source", "uncorrected_model"),
                  "not_identified_because": lat.get("not_identified_because"),
                  "at_grid_edge": lat.get("at_grid_edge")},
        "gap": gap,
    }
