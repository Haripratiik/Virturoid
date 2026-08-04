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

Latency is identified separately and honestly. The cross-correlation lag between the two trajectories is
reported as ``output_phase_lag_ms`` because that is what it is -- and it is a LOWER BOUND on actuation delay,
not an estimate of it: measured on a quadruped, injecting 40 ms of loop delay moved the closed-loop output lag
by only 18 ms, because feedback partially compensates delay. The actual parameter comes from re-simulating the
closed loop across a grid of delays and taking the one that reproduces the log, which recovers the injected
value exactly (0/10/20/40/60 ms -> 0/10/20/40/60 ms).

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


def measure_gap(gene, log: dict, *, plan: dict | None = None,
                delay_max_ticks: int = DEFAULT_DELAY_MAX_TICKS,
                measure_noise_floor: bool = True) -> dict:
    """Measure how far our simulator is from ``log``, per joint, in units an engineer recognises.

    ``log`` carries ``joints``, ``t``, ``q_cmd``, ``q_meas`` and (to attribute anything) ``tau_meas``; see
    ``build_excitation(...)['log_schema']``. ``plan`` is that same excitation plan, and supplying it is what lets
    each joint be scored over its OWN excitation window instead of over the whole run.
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

    t_wall = time.perf_counter()
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
    delay_kw = dict(kp=kp, kd=kd, q_start=q0, ctrl_every=ctrl_every,
                    max_ticks=int(delay_max_ticks), dt=dt)
    lag_note = ("per-joint 'output_phase_lag_ms' in the table above is the OBSERVABLE closed-loop output lag "
                "and is a LOWER BOUND on actuation delay -- feedback partially compensates delay (measured: "
                "40 ms injected showed as 18 ms of output lag). Use the identified figure as the parameter.")

    if tau_hw is None:
        out["latency"] = {**_delay_search(model, q_cmd, q_hw, **delay_kw), "caveat": lag_note}
        out["attribution"] = {
            "available": False,
            "why": "the log carries no tau_meas. The residual that attributes a gap to a parameter is a TORQUE "
                   "residual; without measured torque (or motor current) only the trajectory gap above can be "
                   "reported, and no parameter may be named.",
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

    # ---- latency, twice: on the model as it stands, and on one carrying the identified parameter deltas ----
    # A delay-only sweep cannot see past a dynamics error (measured: a 20 ms delay injected alongside a
    # friction/damping/inertia error makes delay=0 the best fit). Applying the deltas we just measured removes
    # that competing explanation, and only then is the remaining trajectory mismatch actually about timing.
    out["latency"] = {**_delay_search(model, q_cmd, q_hw, **delay_kw), "caveat": lag_note}
    if implicated:
        corrected = _corrected_model(model, rows, dofs)
        after = _delay_search(corrected, q_cmd, q_hw, **delay_kw)
        out["latency"]["after_parameter_correction"] = {
            **after,
            "applied_deltas": implicated,
            "why": "the same sweep on a model carrying the identified parameter deltas; use this figure when "
                   "the uncorrected sweep reports not-identified",
        }
        if after["identified"] and not out["latency"]["identified"]:
            out["latency"]["identified"] = True
            out["latency"]["delay_ms"] = after["delay_ms"]
            out["latency"]["delay_ticks"] = after["delay_ticks"]
            out["latency"]["source"] = "after_parameter_correction"
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
