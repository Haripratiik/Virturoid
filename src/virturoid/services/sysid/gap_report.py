"""The gap number: how far is our simulator from THIS robot's own measured behaviour?

Deliberately not a score. A scalar "fidelity: 0.87" tells an engineer nothing they can act on; what they need is
the sentence "your simulator is 0.11 rad RMS off on the left knee at 3 Hz and lags your hardware by 18 ms", and
then which parameter would have to move to close it. So the output is a per-joint table in rad, ms and N.m, a
ranking that names the worst joints, and a per-parameter attribution -- and every one of those attributions is
run past ``identifiability`` before it is allowed to be stated.

Two independent measurements, because they answer different questions and fail differently:

**The forward replay** drives our sim with the same commands the hardware got and compares trajectories. This is
the number the engineer recognises, and it is an OBSERVABLE -- it makes no claim about mechanism.

**The inverse-dynamics residual** asks our model what torque it thinks the LOGGED motion required, and subtracts
that from the torque the hardware actually applied. What is left is, by construction, the physics our model is
missing. Regressing it on the model's own sensitivity ``d(tau)/d(parameter)`` gives a delta per parameter
already in physical units. This is the number that names a culprit.

Latency is identified separately and honestly, and it is identified OPEN-LOOP. The cross-correlation lag
between the two trajectories is reported as ``output_phase_lag_ms`` because that is what it is -- and it is a
LOWER BOUND on actuation delay, not an estimate of it: measured on a quadruped, injecting 40 ms of loop delay
moved the closed-loop output lag by only 18 ms, because feedback partially compensates delay. The parameter
itself comes from ``_delay_from_command_response``: the log carries the applied torque, we know the control law
because we authored it and shipped its gains in the plan, so the lag between the law evaluated on the logged
state and the torque the actuator actually applied IS the transport delay, with no dynamics model in the path
at all.

That last point is the whole design, and it was arrived at by measuring the alternative. Re-simulating the
closed loop across a grid of delays and taking the trajectory RMS minimum -- the obvious approach, and the one
this module shipped first -- does NOT work on a real robot. It works only when the model is already exactly
right, and it fails in a specific and instructive way: under-shooting the delay SATURATES. A replay with too
little delay simply does not ring, so its error against a ringing log is the log's ringing amplitude and stops
growing, while a replay with too much delay rings out of phase and scores WORSE than one that does not ring at
all. The objective is therefore a one-grid-point hole sitting on a monotone ramp, not a basin, and any residual
dynamics error fills the hole in and leaves the argmin at zero. MEASURED on the Menagerie Go2 at 40 ms injected
(``docs/calibration_wedge_under_delay.md``): the oracle-model sweep reads 0.1100 / 0.1099 / 0.1098 / 0.1186 /
**0.0000** / 0.1759 across 0-50 ms, and the prior-model sweep reads 0.1093 / **0.1061** / 0.1172 / ... -- a 3%
win for the wrong answer. Replacing RMS with a normalised cross-correlation was tried and has the same shape,
because incoherent ringing decorrelates just as fast as it de-superposes. The trajectory sweep is kept as
``_delay_search``, reported beside the others, and it is NOT allowed to claim a delay it merely won on.

A log with no torque channel at all is no longer handed to that sweep. ``_delay_from_motion`` reads the applied
torque out of the MOTION instead -- inverse dynamics evaluated POINTWISE on the logged trajectory rather than
forward-simulated, so a model error stays local, with reflected inertia / damping / dry friction / offset free
at every candidate lag. It recovers 0 / 20 / 40 ms exactly on the Go2 and it is what a position-only customer
gets; it has a plant in it, so ``_merge_latency`` ranks it below the torque channel and above the sweep, by how
much plant is in each. And a log carrying motor current is converted to torque by ``sysid.torque_channel``
before any of this, through a constant that is reported rather than assumed.

Everything here is CPU MuJoCo and non-iterative: four inverse-dynamics passes plus a small delay grid. It is a
measurement, not a fit -- fitting is Stage 2.
"""

from __future__ import annotations

#: Parameters attributed here, and how far to step each when measuring the model's sensitivity to it.
#: The step is relative to the joint's own current value with an absolute floor, so a joint whose nominal value
#: is zero still gets a probe instead of a zero column.
SENSITIVITY_PARAMS = {
    "frictionloss": {"rel": 0.25, "floor": 0.01, "unit": "N.m"},
    "damping": {"rel": 0.25, "floor": 0.05, "unit": "N.m.s/rad"},
    "armature": {"rel": 0.25, "floor": 0.005, "unit": "kg.m^2"},
}
DEFAULT_DELAY_MAX_TICKS = 8
#: Latency is only reported as identified when sweeping the delay explains at least this much of the trajectory
#: mismatch. Below it the mismatch is dominated by a DYNAMICS error, and delay cannot be separated from it by a
#: delay-only search -- measured directly: injecting 20 ms of delay together with a friction/damping/armature
#: error made delay=0 the best-fitting delay on the uncorrected model, i.e. the honest answer is "cannot tell".
DELAY_TRUST_FRACTION = 0.5

# ---- gates for the command/response (open-loop) delay estimate. All three MEASURED, see the table below. ----
#: The reconstructed control law must account for at least this fraction of the applied torque's own variation
#: at the winning lag. Below it we are not looking at the loop we think we are looking at -- a different control
#: law, a mis-stated gain, or a torque channel that is mostly noise.
DELAY_MIN_EXPLAINED = 0.5
#: ...and the winning lag must beat the next-best lag by at least this much. This is the gate that makes a ZERO
#: delay reportable: the argmin being the zero-delay entry says nothing on its own, but the zero-delay entry
#: beating every other entry by a margin does.
DELAY_MIN_MARGIN = 0.15
#: A joint whose applied torque barely varied over its window cannot resolve a lag at all -- every candidate
#: fits a constant equally well. Scaled by the joint's own ``forcerange`` so it means the same thing on a wrist
#: and on a hip. Without this gate the ten joints a two-joint plan never excites report a PERFECT match at every
#: lag (0 vs 0) and a spurious margin of 1.0.
DELAY_MIN_TORQUE_SWING_FRAC = 0.01

#: What the command/response estimate survives, MEASURED on the Menagerie Go2 (12 s, FL_hip + FL_thigh,
#: injections of 0/20/40 ms). Each row corrupts the log the way a real bench would and reports the recovered
#: delay; see ``docs/calibration_wedge_under_delay.md`` section 8.
#:
#:   clean sim2sim                            0 / 20 / 40 ms   exact
#:   14-bit encoder quantisation              0 / 20 / 40 ms   exact
#:   encoder noise, 1 mrad                    0 / 20 / 40 ms   exact
#:   velocity noise, 0.02 rad/s               0 / 20 / 40 ms   exact
#:   current noise, 2% of peak torque         0 / 20 / 40 ms   exact
#:   an unmodelled gravity feed-forward       0 / 20 / 40 ms   exact  (mean removal is what buys this)
#:   the gains actually run 10% off plan      0 / 20 / 40 ms   exact
#:   current channel low-passed at 100 Hz     0 / 20 / 40 ms   exact, margin 0.19
#:   current channel low-passed at 40 Hz     10 / 30 / 50 ms   ONE TICK HIGH -- see ``TORQUE_CHANNEL_CAVEAT``
#:   all of the above at once                 0 / 20 / 40 ms   exact at 20/40; at 0 ms the margin (0.09) falls
#:                                                             under the gate and it correctly refuses
TORQUE_CHANNEL_CAVEAT = (
    "this is the lag between the control law evaluated on the LOGGED state and the torque the log says was "
    "applied, so it includes any latency in the torque channel itself. If tau_meas is a filtered motor current, "
    "that filter's own group delay is inside this number: MEASURED sim2sim, a first-order low-pass at 100 Hz "
    "leaves the estimate exact, and one at 40 Hz biases it a full control tick HIGH. Log the least-filtered "
    "current you have, and read a one-tick disagreement between joints as a filtering difference before "
    "reading it as an actuator difference."
)

# ---- the POSITION-ONLY estimator. Same three gates; one extra dependency, stated. -------------------------
#: A log with neither measured torque nor motor current used to get no latency number at all -- the fallback
#: was the closed-loop trajectory sweep, which is biased toward zero and is not allowed to claim. It is
#: recoverable, and the reason the obvious attempts fail is a SAMPLING artefact rather than a property of the
#: experiment: the applied torque is a zero-order hold across ``ctrl_every`` physics steps, so a central
#: difference taken AT a control-tick boundary straddles two different applied torques and biases the answer a
#: full tick HIGH. MEASURED on the Go2 (2 joints, 12 s, prior model, injections 0/20/40 ms): a central
#: difference at the boundary gives 10 / 30 / 50 ms with margins of 0.03, a BACKWARD difference gives the same
#: +1 tick, and both a forward difference at the boundary and a central difference at the interval MIDPOINT
#: give 0 / 20 / 40 ms exactly with margins of 0.90-0.99. This estimator therefore integrates the equation of
#: motion over ONE hold interval, where the applied torque is constant by construction.
MOTION_MODEL_CAVEAT = (
    "this estimate reads the applied torque out of the MOTION, through our own inverse dynamics, so unlike the "
    "tau_meas estimate it DOES depend on the dynamics model -- the three parameters this package fits "
    "(frictionloss / damping / armature) are left free at every candidate lag, but an error outside that set "
    "is not absorbed. MEASURED on the Go2 with +30% link mass and inertia -- an error no fitted parameter can "
    "express -- the lag is still recovered exactly at 0 and 20 ms (margins 0.89 and 0.72) and at 40 ms the "
    "margin collapses to 0.00 and it REFUSES. It also needs the log sampled at or above the control rate: at "
    "500 / 250 / 100 Hz against a 100 Hz loop it is exact (margins 0.66-0.93), and at 50 Hz -- below the loop "
    "-- it is wrong by a tick and the margin falls to 0.03-0.19, so the gate refuses. Prefer a log with "
    "tau_meas or motor current; that estimate has no plant in it at all."
)


def _resample(t_src, values, t_dst):
    import numpy as np

    t_src = np.asarray(t_src, dtype=float)
    values = np.asarray(values, dtype=float)
    out = np.zeros((t_dst.size, values.shape[1]))
    for c in range(values.shape[1]):
        out[:, c] = np.interp(t_dst, t_src, values[:, c])
    return out


def _align_log(log: dict, model, dofs: dict):
    """Put a log onto the bench grid, in full DOF space, or say precisely what is missing.

    Every actuated joint must be present. A partial log cannot be used for inverse dynamics at all: the torque
    our model says joint 3 needed depends on where joints 1, 2 and 4 were, so silently zero-filling an unlogged
    joint would produce a confident residual attributable to nothing.
    """
    import numpy as np

    names = [str(s) for s in (log.get("joints") or [])]
    missing = [n for n in dofs if n not in names]
    if missing:
        return None, {"ok": False, "error": "log is missing actuated joints",
                      "missing_joints": sorted(missing), "logged_joints": names,
                      "why": "inverse dynamics on joint J depends on the whole configuration; a partial log "
                             "cannot attribute a residual to any joint"}
    dt = float(model.opt.timestep)
    t_src = np.asarray(log["t"], dtype=float)
    n_dst = int(round((t_src[-1] - t_src[0]) / dt)) + 1
    t_dst = t_src[0] + np.arange(n_dst) * dt
    resampled = not (t_src.size == n_dst and np.allclose(t_src, t_dst, atol=dt * 1e-3))

    nv = model.nv
    cols = {n: names.index(n) for n in dofs}
    out = {}
    for key, required in (("q_cmd", True), ("q_meas", True), ("qd_meas", False), ("tau_meas", False)):
        raw = log.get(key)
        if raw is None:
            if required:
                return None, {"ok": False, "error": f"log is missing required field {key!r}"}
            out[key] = None
            continue
        arr = np.asarray(raw, dtype=float)
        arr = _resample(t_src, arr, t_dst) if resampled else arr
        full = np.zeros((t_dst.size, nv))
        for name, adr in dofs.items():
            full[:, adr] = arr[:, cols[name]]
        out[key] = full
    out["t"] = t_dst
    return out, {"ok": True, "resampled": resampled, "log_hz": round(1.0 / dt, 3), "n_rows": int(t_dst.size)}


def _windows(plan, n_rows, dofs, log_hz):
    """``{joint: (start, stop)}`` -- each joint's own excitation window, so its residual is measured while it,
    and nothing else, was moving. Without a plan the whole log is used for every joint."""
    if not plan:
        return {name: (0, n_rows) for name in dofs}
    rows = int(round(float(plan["budget"]["per_joint_s"]) * log_hz))
    active = [j for j in plan["joints"] if j["excitable"]]
    out = {}
    for i, j in enumerate(active):
        a = min(i * rows, n_rows)
        out[j["joint"]] = (a, min(a + rows, n_rows))
    for name in dofs:
        out.setdefault(name, (0, n_rows))
    return out


def _phase_lag_ms(a, b, dt, max_lag):
    """Lag (ms) at which ``b`` best matches ``a``; positive means ``b`` trails."""
    import numpy as np

    best_l, best_e = 0, float("inf")
    n = a.size
    for lag in range(0, int(max_lag) + 1):
        if n - lag < 8:
            break
        e = float(np.mean((a[:n - lag] - b[lag:]) ** 2))
        if e < best_e:
            best_e, best_l = e, lag
    return best_l * dt * 1000.0


def _sensitivity_columns(model, q, qd, qacc):
    """``{param: (n, nv)}`` of ``d(qfrc_inverse)/d(param)`` by central difference on the model itself.

    Using the model's own derivative rather than a hand-written ``sign(qd)/qd/qdd`` basis matters: MuJoCo models
    frictionloss as a solver CONSTRAINT with a stick regime, not as ``f*sign(qd)``, and idealizing it recovered
    an injected friction perturbation ~35% low. The measured derivative recovers it correctly.
    """
    import copy

    import numpy as np

    from virturoid.services.sysid.bench_rig import inverse_torque

    cols = {}
    for name, spec in SENSITIVITY_PARAMS.items():
        p0 = np.asarray(getattr(model, f"dof_{name}"), dtype=float)
        delta = np.maximum(spec["rel"] * np.abs(p0), spec["floor"])
        up, dn = copy.deepcopy(model), copy.deepcopy(model)
        getattr(up, f"dof_{name}")[:] = p0 + delta
        getattr(dn, f"dof_{name}")[:] = np.maximum(p0 - delta, 0.0)
        step = np.maximum(np.asarray(getattr(up, f"dof_{name}")) - np.asarray(getattr(dn, f"dof_{name}")), 1e-12)
        cols[name] = (inverse_torque(up, q, qd, qacc) - inverse_torque(dn, q, qd, qacc)) / step
    return cols


def _delay_search(model, q_cmd, q_hw, *, kp, kd, q_start, ctrl_every, max_ticks, dt):
    """Sweep whole control ticks of actuation delay, re-simulate, and take the one that reproduces the log.

    Reported with a TRUST verdict, because this search is delay-only: a friction or inertia error also shifts
    the trajectory, and on an uncorrected model it can swamp the delay signature entirely. When sweeping delay
    fails to explain most of the mismatch, the honest output is that latency could not be separated -- not the
    argmin, which in that case is an artefact of whichever parameter error dominates.
    """
    import numpy as np

    from virturoid.services.sysid.bench_rig import pd_replay

    grid = []
    for d in range(0, int(max_ticks) + 1):
        _, q_try, _, _ = pd_replay(model, q_cmd, kp=kp, kd=kd, q_start=q_start,
                                   ctrl_every=ctrl_every, delay_ticks=d)
        grid.append({"delay_ticks": d, "delay_ms": round(d * ctrl_every * dt * 1000.0, 3),
                     "trajectory_rms_rad": round(float(np.sqrt(np.mean((q_try - q_hw) ** 2))), 8)})
    best = min(grid, key=lambda r: r["trajectory_rms_rad"])
    at_zero = grid[0]["trajectory_rms_rad"]
    explained = 1.0 - (best["trajectory_rms_rad"] / at_zero) if at_zero > 1e-12 else 1.0
    trusted = bool(explained >= DELAY_TRUST_FRACTION)
    return {
        "delay_ms": best["delay_ms"], "delay_ticks": best["delay_ticks"],
        "identified": trusted,
        "fraction_of_mismatch_explained_by_delay": round(float(explained), 4),
        "residual_after_best_delay_rad": best["trajectory_rms_rad"],
        "at_grid_edge": bool(best["delay_ticks"] == int(max_ticks)),
        "grid": grid,
        "not_identified_because": None if trusted else (
            f"sweeping delay explains only {explained:.1%} of the trajectory mismatch (need "
            f"{DELAY_TRUST_FRACTION:.0%}); it is dominated by a dynamics error, and a delay-only search cannot "
            f"separate the two. Close the parameter gap first, then re-run"),
    }


def _delay_from_command_response(model, aligned, dofs, plan, *, kp, kd, ctrl_every, max_ticks, dt) -> dict:
    """The actuation delay, measured OPEN-LOOP off the log itself. No dynamics model in the path.

    ``pd_replay`` -- and every real joint driver -- applies at tick ``k`` a torque that was COMPUTED at tick
    ``k - D``. The log carries the state the law was computed from (``q_cmd``, ``q_meas``, ``qd_meas``) and the
    torque that was actually applied (``tau_meas``), and we know the law because we authored it and shipped its
    gains inside the excitation plan. So ``D`` is the shift that best aligns ``clip(kp*(q_cmd - q) - kd*qd)``
    with ``tau_meas``, and nothing about the robot's mass, friction or inertia enters the question.

    That is the whole reason this exists. The closed-loop trajectory sweep (``_delay_search``) has to simulate
    the plant, so a wrong plant moves its answer; measured on the Go2, a 40 ms delay came back as 10 ms because
    the parameter error filled in the objective's one-grid-point minimum. This estimator's objective is a
    symmetric V with the truth at the bottom -- MEASURED at 40 ms injected, in N.m: 15.85 / 12.91 / 9.28 / 4.96
    / **0.00** / 4.96 / 9.28 -- and it does not move when the plant is wrong, because the plant is not in it.

    Three things are scored per joint and all three have to hold before a number is claimed:

      * the joint's applied torque actually VARIED (``DELAY_MIN_TORQUE_SWING_FRAC`` of its forcerange). A joint
        the plan never excited holds a constant torque that fits every candidate lag perfectly.
      * the reconstruction EXPLAINS that variation at the winning lag (``DELAY_MIN_EXPLAINED``). This is the
        gate that fires when the customer did not run the loop we think they ran.
      * the winning lag BEATS the next-best one (``DELAY_MIN_MARGIN``). This is the gate that lets a zero delay
        be a finding: ``_delay_search`` could never report one, because its trust metric was
        ``1 - best/at_zero`` and at zero delay those are the same entry.

    Means are removed from both sides before comparing, so a gravity feed-forward the customer's controller has
    and we do not model, or a torque-sensor bias, is not read as agreement -- it is a constant at every lag.

    Two assumptions worth stating because a real log can break them. The control ticks are taken to start at
    the log's first sample; if the customer's log starts part-way into a tick, the reconstruction is evaluated
    at a constant sub-tick offset from where the controller evaluated it, which inflates the residual at EVERY
    lag equally and so costs margin rather than moving the answer. And the lag measured is the whole
    controller-to-applied-torque path: a filtered current channel puts its own group delay inside this number,
    which is ``TORQUE_CHANNEL_CAVEAT`` and is the one bias here that is invisible from the inside.
    """
    import numpy as np

    from virturoid.services.sysid.bench_rig import central_derivative, torque_ceiling

    tau = aligned.get("tau_meas")
    if tau is None:
        return {"available": False, "why": "the log carries no tau_meas"}
    q_cmd, q = aligned["q_cmd"], aligned["q_meas"]
    qd = aligned["qd_meas"] if aligned.get("qd_meas") is not None else central_derivative(q, dt)
    ceil = torque_ceiling(model)
    want = np.clip(np.asarray(kp) * (q_cmd - q) - np.asarray(kd) * qd, -ceil, ceil)
    ce = max(1, int(ctrl_every))
    wins = _windows(plan, q.shape[0], dofs, 1.0 / dt)
    tick_ms = ce * dt * 1000.0
    # The swing floor is a fraction of the joint's own forcerange, so it means the same thing on a wrist and on
    # a hip. ``torque_ceiling`` returns a 1e6 SENTINEL for an actuator with no forcerange, and scaling a floor
    # off that would refuse every joint on the robot; there, fall back to the log's own torque scale.
    log_scale = max(float(np.max(np.std(tau, axis=0))) if tau.size else 0.0, 1e-6)

    per_joint, agreed = {}, []
    for name, adr in dofs.items():
        a, b = wins[name]
        first = a + ((-a) % ce)                      # the first control tick at or after the window start
        rows = np.arange(first, b, ce)
        if rows.size < 32:
            per_joint[name] = {"identified": False, "delay_ms": None, "delay_ticks": None,
                               "torque_swing_nm": 0.0,
                               "not_identified_because": "this joint's window holds fewer than 32 control "
                                                         "ticks, which is too few to resolve a shift"}
            continue
        w, g = want[rows, adr], tau[rows, adr]
        swing = float(np.std(g))
        scale = float(ceil[adr]) if float(ceil[adr]) < 1e6 else log_scale
        floor = max(DELAY_MIN_TORQUE_SWING_FRAC * scale, 1e-6)
        grid = []
        for d in range(0, int(max_ticks) + 1):
            n = w.size - d
            if n < 16:
                break
            x, y = w[:n], g[d:d + n]
            e = (x - x.mean()) - (y - y.mean())
            grid.append({"delay_ticks": d, "delay_ms": round(d * tick_ms, 3),
                         "command_residual_nm": round(float(np.sqrt(np.mean(e ** 2))), 8)})
        if len(grid) < 2:
            per_joint[name] = {"identified": False, "delay_ms": None, "delay_ticks": None,
                               "torque_swing_nm": round(swing, 6),
                               "not_identified_because": "this joint's window is too short to shift"}
            continue
        best = min(grid, key=lambda r: r["command_residual_nm"])
        nxt = min(r["command_residual_nm"] for r in grid if r["delay_ticks"] != best["delay_ticks"])
        explained = 1.0 - best["command_residual_nm"] / swing if swing > floor else 0.0
        margin = 1.0 - best["command_residual_nm"] / nxt if nxt > 1e-12 else 0.0
        why = None
        if swing <= floor:
            why = (f"this joint's applied torque varied by only {swing:.4g} N.m over its window (floor "
                   f"{floor:.4g} N.m): it was not driven hard enough for a one-tick shift to show")
        elif explained < DELAY_MIN_EXPLAINED:
            why = (f"the declared PD law reconstructs only {explained:.1%} of the applied torque at the best "
                   f"lag (need {DELAY_MIN_EXPLAINED:.0%}); the log was probably not produced by the controller "
                   f"the plan specifies, or tau_meas is dominated by noise")
        elif margin < DELAY_MIN_MARGIN:
            why = (f"the best lag beats the next-best by only {margin:.1%} (need {DELAY_MIN_MARGIN:.0%}); this "
                   f"excitation cannot separate {best['delay_ms']:g} ms from its neighbour")
        row = {"identified": why is None, "delay_ms": best["delay_ms"], "delay_ticks": best["delay_ticks"],
               "torque_swing_nm": round(swing, 6),
               "fraction_of_applied_torque_explained": round(float(explained), 4),
               "margin_over_next_best_tick": round(float(margin), 4),
               "at_grid_edge": bool(best["delay_ticks"] == grid[-1]["delay_ticks"]),
               "not_identified_because": why, "grid": grid}
        per_joint[name] = row
        if why is None:
            agreed.append(row)

    if not agreed:
        reasons = sorted({r.get("not_identified_because") for r in per_joint.values()
                          if r.get("not_identified_because")})
        # The argmin is still REPORTED, disclaimed, exactly as the trajectory sweep always did: an engineer
        # reading a refusal is entitled to see the number that was refused. Only joints whose torque actually
        # moved contribute -- a constant-torque joint's argmin is arbitrary and would poison the median.
        moved = sorted(r["delay_ticks"] for r in per_joint.values()
                       if r.get("delay_ticks") is not None and r.get("torque_swing_nm", 0.0) > 0.0
                       and "not driven hard enough" not in (r.get("not_identified_because") or ""))
        guess = int(moved[len(moved) // 2]) if moved else None
        return {
            "available": True, "identified": False,
            "delay_ms": None if guess is None else round(guess * tick_ms, 3), "delay_ticks": guess,
            "method": "command/response lag on the log (open loop)", "per_joint": per_joint,
            "joints_identified": 0, "joints_scored": len(per_joint),
            "at_grid_edge": bool(guess is not None and guess == int(max_ticks)),
            "reported_but_not_claimed": ("the argmin over the joints that moved, shown so the refusal can be "
                                         "read; it is NOT an estimate of the delay"),
            "not_identified_because": "no joint could resolve a lag: " + "; ".join(reasons[:3]),
        }

    ticks = sorted(r["delay_ticks"] for r in agreed)
    best_ticks = int(ticks[len(ticks) // 2])
    return {
        "available": True, "identified": True,
        "delay_ms": round(best_ticks * tick_ms, 3), "delay_ticks": best_ticks,
        "method": "command/response lag on the log (open loop): the shift that aligns the declared PD law "
                  "evaluated on the logged state with the logged applied torque. No plant model is involved, "
                  "so a parameter error cannot move it.",
        "per_joint": per_joint,
        "joints_identified": len(agreed), "joints_scored": len(per_joint),
        "joint_agreement": {"ticks_min": int(ticks[0]), "ticks_max": int(ticks[-1]),
                            "unanimous": bool(ticks[0] == ticks[-1]),
                            "note": ("every joint that could resolve a lag returned the same one"
                                     if ticks[0] == ticks[-1] else
                                     f"joints disagree across {ticks[0]}-{ticks[-1]} ticks; the MEDIAN is "
                                     f"reported. Read the per-joint table -- a one-tick spread is more often a "
                                     f"difference in torque-channel filtering than in the actuators")},
        "fraction_of_applied_torque_explained": round(
            float(min(r["fraction_of_applied_torque_explained"] for r in agreed)), 4),
        "margin_over_next_best_tick": round(float(min(r["margin_over_next_best_tick"] for r in agreed)), 4),
        "at_grid_edge": bool(any(r["at_grid_edge"] for r in agreed)),
        "not_identified_because": None,
        "caveat": TORQUE_CHANNEL_CAVEAT,
    }


def _delay_from_motion(model, aligned, dofs, plan, *, kp, kd, ctrl_every, max_ticks, dt,
                       native_log_hz=None) -> dict:
    """The actuation delay from a POSITION-ONLY log: no measured torque, no motor current, no re-simulation.

    The torque the actuator applied is not in the log, but its EFFECT is. Over one control tick the applied
    torque is a zero-order hold, so integrating the equation of motion across that interval gives

        tau_applied[j]  =  ID(q_j, qd_j, qacc_j) + a*qacc_j + b*qd_j + c*sign(qd_j) + d

    where ``qacc_j = (qd[(j+1)*ce] - qd[j*ce]) / (ce*dt)`` is the MEAN acceleration while that one torque was
    applied, ``ID`` is our model's inverse dynamics evaluated pointwise on the logged motion, and the four free
    coefficients are re-fitted AT EVERY CANDIDATE LAG so that reflected inertia, viscous damping, dry friction
    and a torque offset -- the exact set Stage 2 fits -- cannot bias the lag. The delay is then the shift that
    aligns that against the declared control law, exactly as ``_delay_from_command_response`` does.

    Two things distinguish this from ``_delay_search``, which is the estimator it replaces on these logs:

      * inverse dynamics is evaluated POINTWISE on the logged trajectory, so a model error stays local. The
        sweep FORWARD-SIMULATES, so a model error compounds along the rollout and fills in the objective's
        one-grid-point minimum -- that is the whole defect, and it is why the sweep is biased toward zero.
      * the parameters most likely to be wrong are free here rather than fixed at our prior. With them fixed,
        MEASURED, the same objective reads 0 / 70 / 20 ms for injections of 0 / 20 / 40 (margins 0.00-0.15);
        with them free it reads 0 / 20 / 40 exactly with margins of 0.91-0.93.

    The sampling is load-bearing and was the whole difficulty. Everything is evaluated INSIDE one hold
    interval: the acceleration as a difference across it, the configuration at its midpoint. A central
    difference taken at the interval BOUNDARY mixes the torque before the boundary with the torque after it and
    biases every answer one control tick high -- measured, on the oracle model as well as the prior, which is
    what showed it was ours and not the experiment's. See ``MOTION_MODEL_CAVEAT`` for what this still costs.
    """
    import numpy as np

    from virturoid.services.sysid.bench_rig import central_derivative, inverse_torque, torque_ceiling

    q, q_cmd = aligned["q_meas"], aligned["q_cmd"]
    if q is None or q_cmd is None or q.shape[0] < 4:
        return {"available": False, "why": "the log is too short to integrate one control interval"}
    qd = aligned["qd_meas"] if aligned.get("qd_meas") is not None else central_derivative(q, dt)
    ce = max(1, int(ctrl_every))
    n_rows = int(q.shape[0])
    if n_rows <= ce + 1:
        return {"available": False, "why": "the log holds fewer than two control ticks"}
    # The NATIVE sample rate, not the aligned one, and this gate is load-bearing. The delay lives on the
    # control-tick grid, so a log sampled slower than one tick cannot resolve one -- and the danger is not that
    # the estimate degrades, it is that ``_align_log`` INTERPOLATES the log back up to the physics rate first,
    # which smooths the very discontinuity the estimator reads and restores a confident margin on a wrong
    # answer. MEASURED on the Go2 at 20 ms injected: a raw 50 Hz log against a 100 Hz loop returned 30 ms with
    # a margin that cleared the gate. Refused here instead, by the one number the interpolation cannot fake.
    ctrl_hz = 1.0 / (ce * dt)
    if native_log_hz is not None and float(native_log_hz) < ctrl_hz * 0.99:
        return {"available": True, "identified": False, "delay_ms": None, "delay_ticks": None,
                "method": "motion reconstruction on the log (open loop)",
                "per_joint": {}, "joints_identified": 0, "joints_scored": 0,
                "not_identified_because": (
                    f"this log is sampled at {float(native_log_hz):.1f} Hz and the control loop runs at "
                    f"{ctrl_hz:.1f} Hz, so one control tick is shorter than the gap between samples and the "
                    f"delay cannot be resolved from the motion. Re-log at or above the control rate, or log "
                    f"tau_meas / motor current -- that estimate does not differentiate and survives this"),
                "caveat": MOTION_MODEL_CAVEAT}

    ceil = torque_ceiling(model)
    want_full = np.clip(np.asarray(kp) * (q_cmd - q) - np.asarray(kd) * qd, -ceil, ceil)
    tick0 = np.arange(0, n_rows - ce, ce)              # first row of each COMPLETE hold interval
    tick1 = tick0 + ce
    mid = tick0 + ce // 2
    qacc_i = (qd[tick1] - qd[tick0]) / (ce * dt)
    q_i, qd_i = q[mid], 0.5 * (qd[tick0] + qd[tick1])
    id_i = inverse_torque(model, q_i, qd_i, qacc_i)
    want_i = want_full[tick0]

    wins = _windows(plan, n_rows, dofs, 1.0 / dt)
    tick_ms = ce * dt * 1000.0
    log_scale = max(float(np.max(np.std(want_i, axis=0))) if want_i.size else 0.0, 1e-6)

    per_joint, agreed = {}, []
    for name, adr in dofs.items():
        a, b = wins[name]
        sel = np.nonzero((tick0 >= a) & (tick1 < b))[0]
        swing = float(np.std(want_i[sel, adr])) if sel.size else 0.0
        scale = float(ceil[adr]) if float(ceil[adr]) < 1e6 else log_scale
        floor = max(DELAY_MIN_TORQUE_SWING_FRAC * scale, 1e-6)
        if sel.size < 32:
            per_joint[name] = {"identified": False, "delay_ms": None, "delay_ticks": None,
                               "commanded_torque_swing_nm": round(swing, 6),
                               "not_identified_because": "this joint's window holds fewer than 32 complete "
                                                         "control intervals, which is too few to resolve a "
                                                         "shift"}
            continue
        grid = []
        for d in range(0, int(max_ticks) + 1):
            n = sel.size - d
            if n < 16:
                break
            src, dst = sel[:n], sel[d:d + n]
            y = want_i[src, adr] - id_i[dst, adr]
            X = np.column_stack([qacc_i[dst, adr], qd_i[dst, adr], np.sign(qd_i[dst, adr]), np.ones(n)])
            theta, *_ = np.linalg.lstsq(X, y, rcond=None)
            grid.append({"delay_ticks": d, "delay_ms": round(d * tick_ms, 3),
                         "motion_residual_nm": round(float(np.sqrt(np.mean((y - X @ theta) ** 2))), 8)})
        if len(grid) < 2:
            per_joint[name] = {"identified": False, "delay_ms": None, "delay_ticks": None,
                               "commanded_torque_swing_nm": round(swing, 6),
                               "not_identified_because": "this joint's window is too short to shift"}
            continue
        best = min(grid, key=lambda r: r["motion_residual_nm"])
        nxt = min(r["motion_residual_nm"] for r in grid if r["delay_ticks"] != best["delay_ticks"])
        explained = 1.0 - best["motion_residual_nm"] / swing if swing > floor else 0.0
        margin = 1.0 - best["motion_residual_nm"] / nxt if nxt > 1e-12 else 0.0
        why = None
        if swing <= floor:
            why = (f"this joint's commanded torque varied by only {swing:.4g} N.m over its window (floor "
                   f"{floor:.4g} N.m): it was not driven hard enough for a one-tick shift to show")
        elif explained < DELAY_MIN_EXPLAINED:
            why = (f"the motion accounts for only {explained:.1%} of the commanded torque at the best lag "
                   f"(need {DELAY_MIN_EXPLAINED:.0%}); either the log was not produced by the controller the "
                   f"plan specifies, or our dynamics model is too far off this joint to read the torque out "
                   f"of the motion. A log carrying tau_meas or motor current does not have this dependency")
        elif margin < DELAY_MIN_MARGIN:
            why = (f"the best lag beats the next-best by only {margin:.1%} (need {DELAY_MIN_MARGIN:.0%}); this "
                   f"log cannot separate {best['delay_ms']:g} ms from its neighbour. A log sampled below the "
                   f"control rate lands here, and so does a model error the free parameters cannot absorb")
        row = {"identified": why is None, "delay_ms": best["delay_ms"], "delay_ticks": best["delay_ticks"],
               "commanded_torque_swing_nm": round(swing, 6),
               "fraction_of_commanded_torque_explained": round(float(explained), 4),
               "margin_over_next_best_tick": round(float(margin), 4),
               "at_grid_edge": bool(best["delay_ticks"] == grid[-1]["delay_ticks"]),
               "not_identified_because": why, "grid": grid}
        per_joint[name] = row
        if why is None:
            agreed.append(row)

    method = ("motion reconstruction on the log (open loop): the shift that aligns the declared PD law with "
              "the torque our inverse dynamics says the logged motion required, integrated over one "
              "zero-order-hold interval, with reflected inertia / damping / dry friction / offset re-fitted at "
              "every candidate lag")
    if not agreed:
        reasons = sorted({r.get("not_identified_because") for r in per_joint.values()
                          if r.get("not_identified_because")})
        moved = sorted(r["delay_ticks"] for r in per_joint.values()
                       if r.get("delay_ticks") is not None
                       and "not driven hard enough" not in (r.get("not_identified_because") or ""))
        guess = int(moved[len(moved) // 2]) if moved else None
        return {
            "available": True, "identified": False,
            "delay_ms": None if guess is None else round(guess * tick_ms, 3), "delay_ticks": guess,
            "method": method, "per_joint": per_joint,
            "joints_identified": 0, "joints_scored": len(per_joint),
            "at_grid_edge": bool(guess is not None and guess == int(max_ticks)),
            "reported_but_not_claimed": ("the argmin over the joints that moved, shown so the refusal can be "
                                         "read; it is NOT an estimate of the delay"),
            "not_identified_because": "no joint could resolve a lag: " + "; ".join(reasons[:3]),
            "caveat": MOTION_MODEL_CAVEAT,
        }

    ticks = sorted(r["delay_ticks"] for r in agreed)
    best_ticks = int(ticks[len(ticks) // 2])
    return {
        "available": True, "identified": True,
        "delay_ms": round(best_ticks * tick_ms, 3), "delay_ticks": best_ticks,
        "method": method,
        "per_joint": per_joint,
        "joints_identified": len(agreed), "joints_scored": len(per_joint),
        "joint_agreement": {"ticks_min": int(ticks[0]), "ticks_max": int(ticks[-1]),
                            "unanimous": bool(ticks[0] == ticks[-1]),
                            "note": ("every joint that could resolve a lag returned the same one"
                                     if ticks[0] == ticks[-1] else
                                     f"joints disagree across {ticks[0]}-{ticks[-1]} ticks; the MEDIAN is "
                                     f"reported. Read the per-joint table")},
        "fraction_of_commanded_torque_explained": round(
            float(min(r["fraction_of_commanded_torque_explained"] for r in agreed)), 4),
        "margin_over_next_best_tick": round(float(min(r["margin_over_next_best_tick"] for r in agreed)), 4),
        "at_grid_edge": bool(any(r["at_grid_edge"] for r in agreed)),
        "not_identified_because": None,
        "caveat": MOTION_MODEL_CAVEAT,
    }


def _merge_latency(cmd: dict, traj: dict, lag_note: str, motion: dict | None = None) -> dict:
    """One latency verdict from up to three estimators, with the least model-dependent one in charge.

    Precedence is not a preference, it is what the measurements support, and it ranks by HOW MUCH PLANT is in
    the estimate:

      1. ``command_response`` -- the log's own applied torque against the declared law. No plant at all, so it
         is right whether or not the parameter fit is. Needs ``tau_meas`` (motor current converted through a
         torque constant counts; the argmin is scale-invariant, MEASURED exact from 0.5x to 2.0x on it).
      2. ``motion_reconstruction`` -- the applied torque read out of the motion by inverse dynamics evaluated
         POINTWISE, with the fitted parameters free. Some plant, bounded and stated in ``MOTION_MODEL_CAVEAT``.
         This is what a position-only log gets, and it recovers 0 / 20 / 40 ms exactly on the Go2.
      3. ``trajectory_sweep`` -- a forward re-simulation, so the plant is in it end to end and its errors
         COMPOUND. Its argmin is biased TOWARD ZERO whenever the plant is wrong, because under-shooting a delay
         saturates while over-shooting does not. Measured on the Go2 at 20 ms injected: the prior-model sweep
         puts 0 ms 49% ahead of the runner-up -- a confident, wrong answer -- so it is never allowed to promote
         itself on a margin. It keeps its original, stricter rule and is reported alongside.
    """
    out = {
        "delay_ms": traj.get("delay_ms"), "delay_ticks": traj.get("delay_ticks"),
        "identified": bool(traj.get("identified")),
        "source": "trajectory_sweep",
        "method": "closed-loop trajectory sweep (nothing better was available on this log)",
        "not_identified_because": traj.get("not_identified_because"),
        "at_grid_edge": traj.get("at_grid_edge"),
        "caveat": lag_note,
        "trajectory_sweep": traj,
    }
    if motion is not None and motion.get("available"):
        out["motion_reconstruction"] = motion
    out["command_response"] = cmd
    if cmd.get("available") and cmd.get("identified"):
        out.update({
            "delay_ms": cmd["delay_ms"], "delay_ticks": cmd["delay_ticks"], "identified": True,
            "source": "command_response", "method": cmd["method"], "not_identified_because": None,
            "at_grid_edge": cmd["at_grid_edge"],
            "fraction_of_applied_torque_explained": cmd["fraction_of_applied_torque_explained"],
            "margin_over_next_best_tick": cmd["margin_over_next_best_tick"],
            "joint_agreement": cmd["joint_agreement"],
            "caveat": lag_note + " " + TORQUE_CHANNEL_CAVEAT,
        })
        return out
    if motion is not None and motion.get("available") and motion.get("identified"):
        # No measured torque, or the torque channel could not resolve it. Inverse dynamics on the logged motion
        # is a WEAKER experiment than the torque channel and a strictly STRONGER one than the sweep, and it is
        # allowed to claim on its own margin because -- unlike the sweep -- its objective is not zero-biased.
        out.update({
            "delay_ms": motion["delay_ms"], "delay_ticks": motion["delay_ticks"], "identified": True,
            "source": "motion_reconstruction", "method": motion["method"], "not_identified_because": None,
            "at_grid_edge": motion["at_grid_edge"],
            "fraction_of_commanded_torque_explained": motion["fraction_of_commanded_torque_explained"],
            "margin_over_next_best_tick": motion["margin_over_next_best_tick"],
            "joint_agreement": motion["joint_agreement"],
            "caveat": lag_note + " " + MOTION_MODEL_CAVEAT,
        })
        return out
    # Nothing that is allowed to claim could resolve it. The refusal stands and says which estimator refused;
    # the argmin is still carried, disclaimed, the way the sweep's always was.
    ran = cmd if cmd.get("available") else (motion if (motion or {}).get("available") else None)
    if ran is None:
        return out
    src = "command_response" if cmd.get("available") else "motion_reconstruction"
    out.update({"delay_ms": ran.get("delay_ms"), "delay_ticks": ran.get("delay_ticks"), "identified": False,
                "source": src, "method": ran["method"],
                "at_grid_edge": ran.get("at_grid_edge"),
                "reported_but_not_claimed": ran.get("reported_but_not_claimed"),
                "not_identified_because": ran["not_identified_because"]})
    return out


def _excitation_stats(qd, qacc, a, b, adr):
    import numpy as np

    v = qd[a:b, adr]
    reversals = int(np.count_nonzero(np.diff(np.sign(v[np.abs(v) > 1e-4])) != 0)) if v.size else 0
    return {"velocity_sign_reversals": reversals,
            "speed_rms_radps": float(np.sqrt(np.mean(v ** 2))) if v.size else 0.0,
            "accel_rms_radps2": float(np.sqrt(np.mean(qacc[a:b, adr] ** 2))) if v.size else 0.0,
            "n_samples": int(b - a)}


def _attribute(model, aligned, plan, dofs, log_hz, noise_floor):
    """Per-joint residual regression + identifiability. Returns ``(rows, raw_estimates)``."""
    import numpy as np

    from virturoid.services.sysid.bench_rig import central_derivative, inverse_torque
    from virturoid.services.sysid.identifiability import identifiability_report

    dt = float(model.opt.timestep)
    q, qd_log, tau = aligned["q_meas"], aligned["qd_meas"], aligned["tau_meas"]
    qd = qd_log if qd_log is not None else central_derivative(q, dt)
    qacc = central_derivative(qd, dt)
    resid = tau - inverse_torque(model, q, qd, qacc)
    cols = _sensitivity_columns(model, q, qd, qacc)
    names = list(SENSITIVITY_PARAMS)
    wins = _windows(plan, q.shape[0], dofs, log_hz)

    rows, raw = {}, {}
    for name, adr in dofs.items():
        a, b = wins[name]
        if b - a < 16:
            continue
        y = resid[a:b, adr]
        X = np.column_stack([np.ones(b - a)] + [cols[p][a:b, adr] for p in names])
        rep = identifiability_report(name, X, y, names,
                                     excitation_stats=_excitation_stats(qd, qacc, a, b, adr),
                                     noise_floor=(noise_floor or {}).get(name))
        rows[name] = rep
        raw[name] = {p: rep["parameters"][p]["estimate"] for p in ("offset", *names)}
    return rows, raw


def _cross_check_torque_constant(model, aligned, dofs, plan, *, kp, kd, ctrl_every, dt, delay_ticks) -> dict:
    """Re-derive kt from the log and set it beside the one the conversion used.

    Only meaningful once the lag is known, which is why it runs after the delay: the current a motor drew at
    tick ``k`` answers the torque the controller commanded at tick ``k - D``, and comparing them at the wrong
    shift would report the correlation of two misaligned signals as a calibration.
    """
    import numpy as np

    from virturoid.services.sysid.bench_rig import central_derivative, torque_ceiling
    from virturoid.services.sysid.torque_channel import identify_from_command

    tau = aligned.get("tau_meas")
    if tau is None:
        return {}
    q_cmd, q = aligned["q_cmd"], aligned["q_meas"]
    qd = aligned["qd_meas"] if aligned.get("qd_meas") is not None else central_derivative(q, dt)
    ceil = torque_ceiling(model)
    want = np.clip(np.asarray(kp) * (q_cmd - q) - np.asarray(kd) * qd, -ceil, ceil)
    ce = max(1, int(ctrl_every))
    d = max(0, int(delay_ticks or 0))
    wins = _windows(plan, q.shape[0], dofs, 1.0 / dt)
    out = {}
    for name, adr in dofs.items():
        a, b = wins[name]
        first = a + ((-a) % ce)
        rows = np.arange(first, b, ce)
        n = rows.size - d
        if n < 16:
            continue
        out[name] = identify_from_command(want[rows[:n], adr], tau[rows[d:d + n], adr], kt_used=1.0)
    return {k: v for k, v in out.items() if v}


def measure_gap(gene, log: dict, *, plan: dict | None = None,
                delay_max_ticks: int = DEFAULT_DELAY_MAX_TICKS,
                measure_noise_floor: bool = True,
                torque_constant_nm_per_a=None) -> dict:
    """Measure how far our simulator is from ``log``, per joint, in units an engineer recognises.

    ``log`` carries ``joints``, ``t``, ``q_cmd``, ``q_meas`` and, to attribute anything, either ``tau_meas`` or
    a motor-current channel ``i_meas``; see ``build_excitation(...)['log_schema']``. A current channel is
    converted through a torque constant by ``torque_channel`` and the conversion is reported, never silent --
    ``torque_constant_nm_per_a`` is the datasheet value (one number, or ``{joint: value}``) and without it the
    constant is derived from the catalog actuator the BOM sized, with a wide stated band. ``plan`` is that same
    excitation plan, and supplying it is what lets each joint be scored over its OWN excitation window instead
    of over the whole run.
    """
    import time

    import numpy as np

    from virturoid.services.sysid.bench_rig import (
        bench_gains,
        bench_model,
        joint_dof_map,
        pd_replay,
        start_pose,
    )
    from virturoid.services.sysid.torque_channel import convert_current_to_torque

    t_wall = time.perf_counter()
    log, torque_channel = convert_current_to_torque(gene, log, explicit=torque_constant_nm_per_a)
    model, rig = bench_model(gene)
    kp, kd, _ = bench_gains(model)
    dofs = joint_dof_map(model, gene)
    dt = float(model.opt.timestep)
    log_hz = 1.0 / dt

    aligned, meta = _align_log(log, model, dofs)
    if aligned is None:
        return {**meta, "robot": getattr(gene, "id", "")}

    ctrl_hz = float(log.get("control_hz") or (plan or {}).get("controller", {}).get("control_hz") or 100.0)
    ctrl_every = max(1, int(round(log_hz / ctrl_hz)))
    # The rate the log ARRIVED at, before `_align_log` interpolated it onto the physics grid. Read off the raw
    # timestamps rather than trusted from a `log_hz` field, because a field can be stale and this number gates
    # a claim (see `_delay_from_motion`).
    t_raw = np.asarray(log.get("t") or [], dtype=float)
    native_hz = (float(t_raw.size - 1) / (t_raw[-1] - t_raw[0])
                 if t_raw.size > 1 and t_raw[-1] > t_raw[0] else None)
    q_cmd, q_hw = aligned["q_cmd"], aligned["q_meas"]
    # Start the replay where the LOG started, not where our own start-pose rule would have put the robot. On a
    # synthetic log the two are identical; on a real one they are not, and seeding from our own rule would
    # charge the engineer for a pose offset we invented and report it as a tracking error on every joint.
    q0 = start_pose(model, gene) if q_hw.shape[0] == 0 else q_hw[0].copy()
    _, q_sim, qd_sim, tau_sim = pd_replay(model, q_cmd, kp=kp, kd=kd, q_start=q0, ctrl_every=ctrl_every)
    lag_max = int(delay_max_ticks) * ctrl_every

    # ---- per-joint trajectory / torque gap -----------------------------------------------------------------
    wins = _windows(plan, q_hw.shape[0], dofs, log_hz)
    tau_hw = aligned["tau_meas"]
    joints = {}
    for name, adr in dofs.items():
        a, b = wins[name]
        if b - a < 8:
            continue
        e = q_sim[a:b, adr] - q_hw[a:b, adr]
        row = {
            "joint": name,
            "position_rms_rad": round(float(np.sqrt(np.mean(e ** 2))), 6),
            "position_rms_deg": round(float(np.degrees(np.sqrt(np.mean(e ** 2)))), 4),
            "position_p95_rad": round(float(np.percentile(np.abs(e), 95)), 6),
            "position_max_rad": round(float(np.abs(e).max()), 6),
            # None, not 0.0, when the log carries no velocity: a zero here would read as perfect agreement.
            "velocity_rms_radps": (round(float(np.sqrt(np.mean(
                (qd_sim[a:b, adr] - aligned["qd_meas"][a:b, adr]) ** 2))), 6)
                if aligned["qd_meas"] is not None else None),
            "output_phase_lag_ms": round(_phase_lag_ms(q_sim[a:b, adr], q_hw[a:b, adr], dt, lag_max), 3),
            "hardware_swing_rad": round(float(q_hw[a:b, adr].max() - q_hw[a:b, adr].min()), 6),
        }
        if tau_hw is not None:
            te = tau_sim[a:b, adr] - tau_hw[a:b, adr]
            row["torque_rms_nm"] = round(float(np.sqrt(np.mean(te ** 2))), 6)
            row["torque_bias_nm"] = round(float(np.mean(te)), 6)
            row["torque_max_nm"] = round(float(np.abs(te).max()), 6)
        joints[name] = row

    out = {
        "ok": True,
        "robot": getattr(gene, "id", ""),
        "setup": rig,
        "log": meta,
        "measured": "sim-vs-log gap. NOT a score: read the per-joint table.",
        "joints": joints,
        "worst_joints_by_position_rad": [r["joint"] for r in sorted(
            joints.values(), key=lambda r: -r["position_rms_rad"])[:5]],
        "wall_clock_s": None,
    }
    # Attached whether or not the conversion SUCCEEDED. A refusal is the more important message of the two:
    # it is the difference between "your parameters carry a kt error" and "your current channel is a
    # magnitude and nothing here read it".
    if torque_channel.get("converted") or torque_channel.get("per_joint"):
        out["torque_channel"] = torque_channel
    delay_kw = dict(kp=kp, kd=kd, q_start=q0, ctrl_every=ctrl_every,
                    max_ticks=int(delay_max_ticks), dt=dt)
    lag_note = ("per-joint 'output_phase_lag_ms' in the table above is the OBSERVABLE closed-loop output lag "
                "and is a LOWER BOUND on actuation delay -- feedback partially compensates delay (measured: "
                "40 ms injected showed as 18 ms of output lag). Use the identified figure as the parameter.")

    if tau_hw is None:
        # A position-only log. The PARAMETERS still cannot be attributed -- that residual is a torque residual
        # and there is nothing to regress -- but the DELAY can be, off the motion, and that is the term this
        # package's opening paragraph names as dominant. See ``_delay_from_motion``.
        motion = _delay_from_motion(model, aligned, dofs, plan, kp=kp, kd=kd, ctrl_every=ctrl_every,
                                    max_ticks=int(delay_max_ticks), dt=dt, native_log_hz=native_hz)
        out["latency"] = _merge_latency({"available": False, "why": "the log carries no tau_meas"},
                                        _delay_search(model, q_cmd, q_hw, **delay_kw), lag_note, motion)
        out["attribution"] = {
            "available": False,
            "why": "the log carries no tau_meas. The residual that attributes a gap to a parameter is a TORQUE "
                   "residual; without measured torque (or motor current) only the trajectory gap above and the "
                   "actuation delay can be reported, and no parameter may be named.",
            "what_is_still_available": "the actuation delay, from 'latency' above -- read out of the motion "
                                       "rather than out of a torque channel, with the model dependency that "
                                       "buys stated in its caveat.",
        }
        out["wall_clock_s"] = round(time.perf_counter() - t_wall, 3)
        return out

    # ---- the estimator's own floor, from a log with nothing wrong with it ---------------------------------
    floor = None
    if measure_noise_floor:
        _, q_c, qd_c, tau_c = pd_replay(model, q_cmd, kp=kp, kd=kd, q_start=q0, ctrl_every=ctrl_every)
        control = {"q_meas": q_c, "qd_meas": qd_c, "tau_meas": tau_c, "q_cmd": q_cmd, "t": aligned["t"]}
        _, raw0 = _attribute(model, control, plan, dofs, log_hz, None)
        floor = {j: {p: abs(v) for p, v in vals.items()} for j, vals in raw0.items()}

    rows, _ = _attribute(model, aligned, plan, dofs, log_hz, floor)
    out["attribution"] = {
        "available": True,
        "method": "residual of measured torque against our model's inverse dynamics, regressed on the model's "
                  "own d(tau)/d(parameter); coefficients are deltas TO ADD to our current values",
        "parameters": {p: s["unit"] for p, s in SENSITIVITY_PARAMS.items()},
        "noise_floor_source": ("replaying the same excitation on our own unperturbed model and running the "
                               "identical estimator; whatever it reports there is measurement error, and it is "
                               "dominated by differentiating logged velocity to get acceleration"
                               if floor else "not measured (measure_noise_floor=False)"),
        "per_joint": rows,
    }
    implicated = sorted({p for r in rows.values() for p in r["identified"] if p != "offset"})
    out["implicated_parameters"] = implicated
    out["joints_with_an_identified_gap"] = sorted(
        j for j, r in rows.items() if [p for p in r["identified"] if p != "offset"])

    # ---- latency: open-loop off the log, with the closed-loop sweep kept beside it -------------------------
    # The trajectory sweep is run twice -- on the model as it stands and on one carrying the identified deltas
    # -- because a delay-only sweep cannot see past a dynamics error, and the corrected pass is the only way
    # that estimator ever gets an answer. Neither pass is in charge: see ``_merge_latency``.
    traj = _delay_search(model, q_cmd, q_hw, **delay_kw)
    if implicated:
        corrected = _corrected_model(model, rows, dofs)
        after = _delay_search(corrected, q_cmd, q_hw, **delay_kw)
        traj["after_parameter_correction"] = {
            **after,
            "applied_deltas": implicated,
            "why": "the same sweep on a model carrying the identified parameter deltas; the uncorrected sweep "
                   "is dominated by the parameter error and this one is not",
        }
        if after["identified"] and not traj["identified"]:
            traj["identified"] = True
            traj["delay_ms"] = after["delay_ms"]
            traj["delay_ticks"] = after["delay_ticks"]
            traj["source"] = "after_parameter_correction"
    cmd = _delay_from_command_response(model, aligned, dofs, plan, kp=kp, kd=kd, ctrl_every=ctrl_every,
                                       max_ticks=int(delay_max_ticks), dt=dt)
    # The motion estimator is a fallback here, not a second opinion, and it runs ONLY when the torque channel
    # could not resolve the lag -- it costs one inverse-dynamics pass and it has a plant in it, so it must not
    # displace an answer that does not. It earns its place on exactly one case, and that case is common on a
    # CURRENT channel: a torque constant far enough out fails the reconstruction gate, and the customer would
    # otherwise get a refusal with nothing to act on. Measured on the Go2 with the constant 2.5x out -- the
    # real Go2 part-substitution factor -- the torque channel refuses at "the declared PD law reconstructs only
    # -5%" and this recovers the delay from the motion, which is what lets the cross-check below name the 2.5x.
    motion = None
    if not cmd.get("identified"):
        motion = _delay_from_motion(model, aligned, dofs, plan, kp=kp, kd=kd, ctrl_every=ctrl_every,
                                    max_ticks=int(delay_max_ticks), dt=dt, native_log_hz=native_hz)
    out["latency"] = _merge_latency(cmd, traj, lag_note, motion)

    # ---- and if the torque channel was a CURRENT channel, re-derive its constant from the log ---------------
    # Runs here rather than at the conversion because the answer depends on the lag: the current at tick k
    # answers the torque commanded at tick k - D. Reported as a cross-check, never silently substituted --
    # a slope of 1.3 against a customer's own datasheet number is a finding about their drivetrain, not a
    # licence for us to overwrite what they told us.
    if torque_channel.get("converted"):
        lat = out["latency"]
        checks = _cross_check_torque_constant(
            model, aligned, dofs, plan, kp=kp, kd=kd, ctrl_every=ctrl_every, dt=dt,
            delay_ticks=(lat.get("delay_ticks") or 0) if lat.get("identified") else 0)
        for name, chk in checks.items():
            rec = torque_channel.get("per_joint", {}).get(name)
            if not rec or not rec.get("used") or not chk.get("nm_per_a"):
                continue
            used = float(rec["nm_per_a"])
            implied = float(chk["nm_per_a"]) * used     # identify_from_command ran with kt_used=1.0 on tau
            rec["identified_from_this_log"] = {**chk, "nm_per_a": round(implied, 6)}
            rec["ratio_identified_over_used"] = round(implied / used, 4) if used else None
        torque_channel["cross_check"] = (
            "'identified_from_this_log' per joint is the constant that would make the logged current match the "
            "torque the controller commanded, measured at the identified lag. A ratio near 1 corroborates the "
            "constant used; a ratio far from 1 says the constant, the current loop's torque tracking, or the "
            "drivetrain efficiency is off by that factor -- and the fitted parameters carry that factor.")
        if not lat.get("identified"):
            torque_channel["cross_check_caveat"] = ("the actuation delay was NOT identified, so this "
                                                    "cross-check was taken at zero lag and is weakened by "
                                                    "however much delay the robot actually has")
    out["wall_clock_s"] = round(time.perf_counter() - t_wall, 3)
    return out


def _corrected_model(model, rows, dofs):
    """A copy of ``model`` with each joint's IDENTIFIED parameter deltas applied. Parameters that failed the
    identifiability gates are deliberately left alone -- applying an unidentified estimate would be exactly the
    over-claim this package exists to refuse."""
    import copy

    out = copy.deepcopy(model)
    for name, adr in dofs.items():
        rep = rows.get(name)
        if not rep:
            continue
        for p in SENSITIVITY_PARAMS:
            if p in rep["identified"]:
                arr = getattr(out, f"dof_{p}")
                arr[adr] = max(float(arr[adr]) + float(rep["parameters"][p]["estimate"]), 0.0)
    return out
