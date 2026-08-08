"""Stage 2: FIT the parameters Stage 1 measured -- as a POSTERIOR, never as a point estimate.

Stage 1 answers "how far is our sim from your robot, and which parameters could this experiment pin?".
It stops there deliberately. This module takes the next step and it takes it under one rule:

    **A parameter comes back as a value AND an interval, or it does not come back at all.**

That rule is not stylistic. A point estimate on an under-identified parameter is the exact failure this
codebase has spent its history removing: a flywheel that advertised clusters wider than a uniform draw, a gait
fitter that cached a rounded float and turned CREDIBLE WALK into FELL. A calibration number is worse than
either, because it is applied to the model and every downstream verdict inherits it silently.

Three things happen here that Stage 1 does not do.

**1. The fit is ITERATED, because one linearization is measurably biased.** Stage 1 regresses the torque
residual on ``d(tau)/d(parameter)`` evaluated at the CURRENT model. That derivative is only correct near the
current value, and MuJoCo's frictionloss is a solver constraint with a stick regime rather than
``f*sign(qd)`` -- so a perturbation that nearly doubles a joint's friction is read ~17% low (measured:
injected 0.080, recovered 0.0666). Re-linearizing about the corrected model and re-regressing removes most of
that: the same injection comes back at 0.0785 (-1.9%). Each iteration is guarded -- a joint whose torque
residual does not actually fall is frozen at its last accepted step, so a diverging joint cannot walk away.

**2. The interval is a MOVING-BLOCK BOOTSTRAP, not a textbook standard error.** These are 500 Hz time series
whose regression residuals are strongly autocorrelated; resampling rows independently would claim roughly a
hundred times more precision than the experiment contains. Blocks preserve the correlation. The interval is
then WIDENED by the estimator's own measured floor -- the estimate it produces on a log with nothing wrong
with it, ~0.017 N.m of phantom friction, dominated by differentiating logged velocity to get acceleration.
That floor is a BIAS, not noise, and widening symmetrically for it is the conservative reading: it does not
remove the bias, it stops the interval from confidently excluding the truth because of it.

**3. Coverage is MEASURED, not assumed.** ``coverage_table`` draws random perturbations, recovers them, and
counts how often the true value actually falls inside the reported interval. An interval whose coverage is
not measured is a decoration. Measured over 1260 (trial, joint, parameter) cells at a nominal 0.90: 1206
claims, coverage 0.9884, refusal rate 0.0429 -- conservative, and not conservative by refusing, since the
median interval is 9-26% of the value it brackets depending on the parameter. WE OWN NO HARDWARE, so this --
like everything in this package -- is sim2sim: see ``synthetic_hardware.WHAT_SIM2SIM_DOES_NOT_PROVE`` for the
precise statement of what that cannot prove.

**4. A fit that does not make the simulator TRACK BETTER is not applied.** ``application_gate`` rules on
``trajectory.improvement_x``, and it exists because the tool already measured that number and no consumer read
it. The failure it closes: a synthetic robot built with +30% link mass and inertia -- an error NO combination
of frictionloss/damping/armature can express, and one that enters the joint equation through the same ``qdd``
regressor as armature -- came back armature-"identified" on 14/14 joints with 13/14 intervals EXCLUDING the
true unchanged value, and all 15 misattributed numbers were written into the compiled model while
``improvement_x`` sat at 0.993 (the fit made tracking marginally *worse*). The parameters this fit holds fixed
are now stated in the engineer-facing output too, per parameter and per fit: see ``PARAMETER_ALSO_ABSORBS``
and ``parameters_not_fitted``.

What is reused from Stage 1, unchanged: the bench rig (``bench_model``/``bench_gains``/``pd_replay``/
``inverse_torque``/``central_derivative``), the log alignment and per-joint windowing (``_align_log``,
``_windows``), the model's own sensitivity columns (``_sensitivity_columns``), the excitation statistics
(``_excitation_stats``), and the whole identifiability gate stack (``identifiability_report``, whose only
change is an optional override so the gates rule on the ACCUMULATED estimate instead of the last step's).
"""

from __future__ import annotations

#: Iterations of Gauss-Newton re-linearization. 3 is where the sim2sim recovery error stops improving
#: (measured on the 14-DOF quadruped: 16.8% -> 2.4% -> 1.9% -> 1.9% on frictionloss).
DEFAULT_ITERATIONS = 3

#: Bootstrap resamples. 256 gives a stable 5th/95th percentile; the cost is a few hundred 3000x4 lstsq calls.
DEFAULT_BOOTSTRAP = 256

#: Nominal interval level. Reported alongside the MEASURED coverage, which is the only number that matters.
DEFAULT_LEVEL = 0.90

#: One iteration may not move a parameter further than this multiple of its own current magnitude, nor further
#: than this multiple of the sensitivity probe floor. A bound, not a prior: it exists so a joint whose
#: regression is degenerate cannot take a single enormous step, and it is reported when it binds.
TRUST_REL = 2.0
TRUST_FLOOR_MULT = 20.0

#: A fit is only allowed to reach the model if applying it makes the simulator track the log MEASURABLY better.
#: The number is ``trajectory.improvement_x`` -- position RMS before / after, on a quantity the fit did not
#: optimise -- and this is the threshold it has to clear.
#:
#: NOT a round number picked for looking reasonable. It comes from the measured separation between fits the
#: estimator CAN express and fits it cannot (sim2sim, composed quadruped, 14 joints, 35 s excitation). The
#: table was taken with this gate and the corrected floor gate BOTH absent, which is the right calibration
#: point: a gate has to be sized against what the unguarded estimator actually does, not against what it does
#: once something else is already catching the same cases.
#:
#:   hardware IS the sim (nothing to find)                 0.000     8/42 pairs "identified" from pure noise
#:   +30% link INERTIA only                                0.065
#:   armature +0.0008, under the per-joint floor           0.360
#:   +15% link mass and inertia                            0.978
#:   +30% link mass and inertia                            0.993    <- 13/14 armature intervals exclude truth
#:   +30% link MASS only                                   0.993
#:   +60% link mass and inertia                            0.999
#:   ------------------------------------------------------------ nothing was measured in this gap
#:   frictionloss/damping/armature +0.05/+0.20/+0.010      3.343
#:   the default injection + 20 ms of delay                3.536
#:   the default injection x0.25                           4.726
#:   the default injection x0.5                            5.269
#:   the default injection, no delay                      16.365
#:
#: Every misspecification measured lands at or below 0.999 -- i.e. it never improves tracking AT ALL, which is
#: the structural signature of an error no fitted parameter can express. Every correctly-specified fit lands at
#: or above 3.343. The threshold sits in that empty band, deliberately off-centre: the misspecified ceiling is
#: pinned at ~1.0 by construction and needs little margin, while the correct floor is a sample and could go
#: lower on a smaller perturbation, so the headroom is put on the side that must not refuse a good fit. In the
#: engineer's units it says: the position RMS has to fall by at least a THIRD before we write anything into
#: your model. A control that is 2% better is not evidence of calibration.
#:
#: RE-MEASURED with this gate and the corrected floor gate BOTH in place, same robot and 35 s excitation, so
#: the band can be checked against the code as shipped rather than only against the code it was sized on:
#:
#:   +30% link mass and inertia   0.995   refused; 8/42 pairs still "identified", 0/8 armature intervals
#:                                        covering the unchanged truth, 0 parameters reaching the model
#:   frictionloss/damping/armature +0.05/+0.20/+0.010   3.343   applied; 37/42 identified, 37/37 covering
#:   the default injection + 20 ms of delay             3.536   applied; 42/42 identified, 42/42 covering
#:
#: The floor gate moved the misspecified case from 14/14 armature cells to 8/14 and its improvement from 0.993
#: to 0.995 -- the six cells it now refuses were the ones nearest the floor, so the deltas that get replayed
#: change. It did not move the case across the band, which is the point: a resolution gate cannot detect a
#: misspecification, and only this one does.
MIN_TRACKING_IMPROVEMENT_X = 1.5

#: What a fitted number could ALSO be. Surfaced on every parameter cell and in the engineer brief, because the
#: mechanism is invisible in the output otherwise: the regressor holds the link's own rigid-body parameters
#: FIXED, so an error in them cannot be reported -- only absorbed.
PARAMETER_ALSO_ABSORBS = {
    "armature": "any error in LINK INERTIA or LINK MASS. Reflected inertia enters the joint equation through "
                "the same qdd term the link's own inertia does, so this experiment cannot separate them and "
                "the link side is held fixed at the compiler's value. MEASURED sim2sim: a robot built with "
                "+30% link mass and inertia and NO armature error came back armature-'identified' on 14/14 "
                "joints with 13/14 intervals excluding the true (unchanged) value, one of them +66%. A large "
                "armature move against an uncertain CAD mass is a candidate mass error, not a gearbox finding.",
    "damping": "any velocity-proportional loss the model does not carry -- gearbox viscous drag, seal drag, a "
               "motor back-EMF term. Under a narrow-band excitation it is also the parameter most often "
               "confused with armature; the VIF gate reports that when it happens.",
    "frictionloss": "any load-independent constant torque the model does not carry: a joint-zero offset, a "
                    "torque-sensor bias, or a mis-stated link mass / centre of mass acting through gravity. "
                    "The regression's intercept absorbs the part that does not change sign with velocity; "
                    "what does change sign lands here.",
}


def _trust_cap(param: str, current: float) -> float:
    from virturoid.services.sysid.gap_report import SENSITIVITY_PARAMS

    floor = float(SENSITIVITY_PARAMS[param]["floor"])
    return max(TRUST_REL * abs(float(current)), TRUST_FLOOR_MULT * floor)


def application_gate(trajectory: dict | None, n_identified: int, *,
                     threshold: float = MIN_TRACKING_IMPROVEMENT_X) -> dict:
    """May this fit be written into the model? Ruled on ``trajectory.improvement_x``, not on the fit's own
    objective.

    This exists because the tool already knew and nobody read it. ``improvement_x`` was computed here, copied
    into the calibration record, printed in the engineer brief -- and no consumer branched on it, so a fit that
    made tracking marginally WORSE (0.993x, from a hardware built with +30% link mass and inertia) wrote 15
    misattributed values into the compiled model and satisfied the L2 joint-coverage requirement on the way.

    Fail-closed in both directions that matter. A fit whose effect on tracking was never MEASURED is
    provisional too: "we did not check" is not "it helped".
    """
    base = {
        "gate": "tracking_improvement",
        "measured_on": "position RMS of our replay against the log, before vs after applying ONLY the "
                       "identified deltas -- a quantity the fit did not optimise (it minimised a TORQUE "
                       "residual)",
        "threshold_x": float(threshold),
        "why_this_threshold": (
            "measured, not chosen: across sim2sim cases every misspecification the estimator cannot express "
            "(+15/30/60% link mass and inertia, mass-only, inertia-only, a sub-floor injection, and hardware "
            "identical to the sim) improved tracking by at most 0.999x -- i.e. not at all -- while every "
            "correctly-specified perturbation improved it by at least 3.343x. The threshold sits in that empty "
            "band. In engineer units: the position RMS must fall by at least a third before anything is "
            "written to the model."),
    }
    x = (trajectory or {}).get("improvement_x")
    if int(n_identified) <= 0:
        return {**base, "improvement_x": x, "passed": None, "provisional": False,
                "verdict": "nothing was identified, so there is nothing to apply and nothing to gate"}
    if x is None:
        return {**base, "improvement_x": None, "passed": False, "provisional": True,
                "verdict": "the fit's effect on tracking was NOT MEASURED, so it cannot be shown to help. The "
                           "fit is PROVISIONAL and is withheld from the model. Re-fit with "
                           "measure_trajectory=True, or apply it deliberately with allow_provisional=True."}
    x = float(x)
    if x >= float(threshold):
        return {**base, "improvement_x": x, "passed": True, "provisional": False,
                "verdict": f"applying this fit tracks the log {x:g}x closer in position RMS "
                           f"(threshold {threshold:g}x)"}
    return {
        **base, "improvement_x": x, "passed": False, "provisional": True,
        "verdict": (f"applying this fit does NOT measurably improve tracking: {x:g}x against a threshold of "
                    f"{threshold:g}x"
                    + (" -- it makes it WORSE." if x < 1.0 else ".")
                    + " The usual cause is an error the estimator cannot express -- link mass, link inertia, "
                      "centre of mass, contact or gearbox elasticity -- being absorbed by the parameters it "
                      "can, so the numbers move and the simulator does not get closer. The fit is PROVISIONAL "
                      "and is withheld from the model; every value is still reported so it can be read, and "
                      # NAMES THE DOOR, not the Python function. This sentence is read by the customer, and
                      # until `sysid.tools` existed it pointed at a private call they could not make: there was
                      # no tool, no CLI verb and no route into this package at all. Telling somebody how to
                      # override a gate in a language they are not speaking is the same defect as the gate
                      # being undiscoverable -- it names an escape hatch that is not on their wire.
                      "fit_actuators {allow_provisional: true} will write it anyway."),
    }


def _model_with(model, deltas: dict, dofs: dict):
    """A COPY of ``model`` carrying ``deltas[joint][param]`` on each joint's DOF, floored at zero.

    Copy, never mutate: the sensitivity columns perturb ``dof_*`` in place and a leaked model would silently
    change every other rollout in the process (``bench_model`` documents the same trap).
    """
    import copy

    out = copy.deepcopy(model)
    for name, adr in dofs.items():
        for param, delta in (deltas.get(name) or {}).items():
            arr = getattr(out, f"dof_{param}", None)
            if arr is None:
                continue
            arr[adr] = max(float(arr[adr]) + float(delta), 0.0)
    return out


def _block_bootstrap(X, y, *, n_boot: int, block_len: int, seed: int):
    """Percentile spread of the regression coefficients under a MOVING-BLOCK resample.

    Blocks, not rows. The residuals of a 2 Hz sinusoid sampled at 500 Hz are nearly the same measurement
    consecutively; an i.i.d. row bootstrap would treat 3000 correlated samples as 3000 independent ones and
    hand back an interval a decimal order too narrow.
    """
    import numpy as np

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    L = int(max(2, min(block_len, n // 2)))
    n_blocks = int(np.ceil(n / L))
    rng = np.random.default_rng(seed)
    starts_max = max(1, n - L + 1)
    out = np.zeros((int(n_boot), p))
    offsets = np.arange(L)
    for b in range(int(n_boot)):
        starts = rng.integers(0, starts_max, size=n_blocks)
        idx = (starts[:, None] + offsets[None, :]).ravel()[:n]
        idx = np.minimum(idx, n - 1)
        theta, *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
        out[b] = theta
    return out


def _fit_one_pass(model, q, qd, qacc, tau, dofs, wins, deltas, param_names):
    """One Gauss-Newton pass. Returns ``{joint: {"step": {...}, "X": .., "y": .., "rms": ..}}``."""
    import numpy as np

    from virturoid.services.sysid.bench_rig import inverse_torque
    from virturoid.services.sysid.gap_report import _sensitivity_columns

    current = _model_with(model, deltas, dofs)
    resid = tau - inverse_torque(current, q, qd, qacc)
    cols = _sensitivity_columns(current, q, qd, qacc)

    out = {}
    for name, adr in dofs.items():
        a, b = wins[name]
        if b - a < 16:
            continue
        y = resid[a:b, adr]
        X = np.column_stack([np.ones(b - a)] + [cols[p][a:b, adr] for p in param_names])
        theta, *_ = np.linalg.lstsq(X, y, rcond=None)
        step = {}
        capped = []
        for i, p in enumerate(param_names, start=1):
            cap = _trust_cap(p, float(getattr(current, f"dof_{p}")[adr]))
            raw = float(theta[i])
            step[p] = float(np.clip(raw, -cap, cap))
            if abs(raw) > cap + 1e-12:
                capped.append(p)
        out[name] = {"step": step, "offset": float(theta[0]), "X": X, "y": y,
                     "rms": float(np.sqrt(np.mean(y ** 2))), "capped": capped}
    return out


def fit_parameters(gene, log: dict, *, plan: dict | None = None,
                   iterations: int = DEFAULT_ITERATIONS,
                   n_boot: int = DEFAULT_BOOTSTRAP,
                   level: float = DEFAULT_LEVEL,
                   seed: int = 0,
                   measure_noise_floor: bool = True,
                   measure_trajectory: bool = True,
                   measure_delay: bool = True,
                   delay_max_ticks: int = 8) -> dict:
    """Fit each joint's dissipative/inertial parameters to ``log`` and return a POSTERIOR per parameter.

    ``log`` is the schema ``build_excitation(...)['log_schema']`` describes and ``synthetic_hardware_log``
    produces. ``plan`` is the excitation plan, and supplying it is what scopes each joint's fit to its OWN
    excitation window instead of to the whole run.

    Every parameter comes back with ``estimate`` (a delta to add), ``value`` (the absolute fitted value),
    ``interval`` on both, and ``identified``. Only the identified ones may be applied to a model
    (``calibration.apply_calibration`` enforces that); the rest carry the reason and, when the experiment
    was the problem, the experiment that would fix it.
    """
    import time

    import numpy as np

    from virturoid.services.sysid.bench_rig import (
        bench_gains,
        bench_model,
        central_derivative,
        joint_dof_map,
        pd_replay,
        start_pose,
    )
    from virturoid.services.sysid.gap_report import (
        SENSITIVITY_PARAMS,
        _align_log,
        _attribute,
        _excitation_stats,
        _windows,
    )
    from virturoid.services.sysid.identifiability import _effective_n, _ols, identifiability_report

    t_wall = time.perf_counter()
    model, rig = bench_model(gene)
    kp, kd, _ = bench_gains(model)
    dofs = joint_dof_map(model, gene)
    dt = float(model.opt.timestep)
    log_hz = 1.0 / dt

    aligned, meta = _align_log(log, model, dofs)
    if aligned is None:
        return {**meta, "robot": getattr(gene, "id", ""), "stage": 2}
    if aligned.get("tau_meas") is None:
        return {"ok": False, "robot": getattr(gene, "id", ""), "stage": 2,
                "error": "the log carries no tau_meas",
                "why": "the quantity being fitted is a TORQUE residual. Without measured torque (or motor "
                       "current) there is nothing to regress, and a fit against the trajectory alone would "
                       "be an unidentifiable mixture of every parameter at once."}

    q, tau = aligned["q_meas"], aligned["tau_meas"]
    qd = aligned["qd_meas"] if aligned["qd_meas"] is not None else central_derivative(q, dt)
    qacc = central_derivative(qd, dt)
    wins = _windows(plan, q.shape[0], dofs, log_hz)
    param_names = list(SENSITIVITY_PARAMS)

    q0c = start_pose(model, gene) if q.shape[0] == 0 else q[0].copy()
    ctrl_hz = float(log.get("control_hz") or (plan or {}).get("controller", {}).get("control_hz") or 100.0)
    ctrl_every = max(1, int(round(log_hz / ctrl_hz)))

    def _estimator_bias(on_model):
        """Replay the same excitation on ``on_model`` and run the IDENTICAL estimator against the result.

        The log now has nothing wrong with it by construction, so every non-zero number that comes back is
        this estimator's own error at that model's operating point.
        """
        _, q_c, qd_c, tau_c = pd_replay(on_model, aligned["q_cmd"], kp=kp, kd=kd, q_start=q0c,
                                        ctrl_every=ctrl_every)
        control = {"q_meas": q_c, "qd_meas": qd_c, "tau_meas": tau_c,
                   "q_cmd": aligned["q_cmd"], "t": aligned["t"]}
        _, raw = _attribute(on_model, control, plan, dofs, log_hz, None)
        return {j: {p: abs(v) for p, v in vals.items()} for j, vals in raw.items()}

    # ---- the estimator's own floor, from a log with NOTHING wrong with it ---------------------------------
    # Same control Stage 1 runs, for the same reason: whatever the estimator reports on an unperturbed replay
    # is measurement error, and every estimate underneath it is indistinguishable from zero.
    floor = _estimator_bias(model) if measure_noise_floor else None

    # ---- iterated Gauss-Newton ---------------------------------------------------------------------------
    deltas: dict = {name: {p: 0.0 for p in param_names} for name in dofs}
    frozen: set = set()
    history: list = []
    first_rms: dict = {}
    last_pass = None
    for it in range(max(1, int(iterations))):
        result = _fit_one_pass(model, q, qd, qacc, tau, dofs, wins, deltas, param_names)
        rms = {name: r["rms"] for name, r in result.items()}
        if not first_rms:
            first_rms = dict(rms)
        if last_pass is not None:
            for name, r in result.items():
                # The residual this joint actually carries did not fall. Its previous step is kept (it was the
                # best one seen) and the joint stops moving: an accepted-then-worse step is how a diverging
                # joint would otherwise walk a physical parameter to nonsense while every other joint improves.
                if name not in frozen and r["rms"] > last_pass[name]["rms"] + 1e-12:
                    frozen.add(name)
                    for p in param_names:
                        deltas[name][p] -= last_pass[name]["step"][p]
        if all(name in frozen for name in result):
            history.append({"iteration": it, "residual_rms_nm": rms, "note": "all joints frozen; stopped"})
            break
        for name, r in result.items():
            if name in frozen:
                continue
            for p in param_names:
                cur = float(getattr(model, f"dof_{p}")[dofs[name]]) + deltas[name][p]
                deltas[name][p] = max(deltas[name][p] + r["step"][p], -cur)
        history.append({"iteration": it, "residual_rms_nm": {k: round(v, 6) for k, v in rms.items()},
                        "frozen_joints": sorted(frozen)})
        last_pass = result

    # One last linearization at the converged point: this is the design matrix whose covariance is the
    # covariance of the CONVERGED estimate (standard Gauss-Newton), and the one the bootstrap resamples.
    final = _fit_one_pass(model, q, qd, qacc, tau, dofs, wins, deltas, param_names)

    # ---- and the estimator's bias WHERE THE ANSWER IS, not where the search started ----------------------
    # MEASURED, and it is the reason this module's first interval was dishonest. The floor above is taken at
    # the PRIOR: it answers "how small an estimate is indistinguishable from zero?". It does NOT answer "how
    # wrong is this estimator at the value it just converged to?", and for reflected inertia those are very
    # different numbers -- the acceleration estimate that dominates the error is a differentiated velocity,
    # so its bias scales with the inertia term it multiplies. Widening by the prior-point floor alone gave a
    # MEASURED interval coverage of 0.336 on armature against a nominal 0.90 (10 trials, 140 claims): the
    # estimates were good (2.5% median error) and the error bars were fiction. Replaying the excitation on
    # the FITTED model and re-running the identical estimator measures the bias at the operating point, which
    # is the quantity the interval actually has to cover. This is a parametric-bootstrap bias estimate and it
    # costs one extra replay.
    bias_at_fit = _estimator_bias(_model_with(model, deltas, dofs)) if measure_noise_floor else None

    # ---- posterior per joint / parameter -----------------------------------------------------------------
    lo_q, hi_q = 0.5 * (1.0 - float(level)), 1.0 - 0.5 * (1.0 - float(level))
    joints: dict = {}
    for name, adr in dofs.items():
        r = final.get(name)
        if r is None:
            continue
        X, y = r["X"], r["y"]
        theta_hat, resid, sigma2, xtx_inv = _ols(X, y)
        n_rows = int(X.shape[0])
        n_eff = _effective_n(resid)
        inflate = max(n_rows / max(n_eff, 1.0), 1.0)
        se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2 * inflate, 0.0))

        block = int(max(16, n_rows // 20))
        boot = _block_bootstrap(X, y, n_boot=n_boot, block_len=block, seed=seed + adr)
        boot_med = np.median(boot, axis=0)
        boot_lo = np.quantile(boot, lo_q, axis=0) - boot_med
        boot_hi = np.quantile(boot, hi_q, axis=0) - boot_med

        base = {p: float(getattr(model, f"dof_{p}")[adr]) for p in param_names}
        fl = (floor or {}).get(name, {})
        bias = (bias_at_fit or {}).get(name, {})

        override = {}
        for i, p in enumerate(param_names, start=1):
            total = float(deltas[name][p])
            # The interval is: bootstrap spread of the converged estimate, widened by the LARGER of the two
            # measured estimator errors -- at the prior (how small is indistinguishable from zero) and at the
            # fitted point (how wrong is this estimator where the answer landed). Both are biases, not noise,
            # so widening does not remove them; it stops the interval from confidently excluding the truth
            # because of one. The identifiability GATE below still uses the prior-point floor only, because
            # that gate asks a question about zero and its Stage-1 semantics must not drift.
            widen = max(abs(float(fl.get(p, 0.0))), abs(float(bias.get(p, 0.0))))
            d_lo = total + float(boot_lo[i]) - widen
            d_hi = total + float(boot_hi[i]) + widen
            override[p] = {"estimate": total, "ci": [d_lo, d_hi], "se": float(se[i])}

        rep = identifiability_report(name, X, y, param_names,
                                     excitation_stats=_excitation_stats(qd, qacc, *wins[name], adr),
                                     noise_floor=fl or None, override=override)

        params = {}
        for p in param_names:
            src = rep["parameters"][p]
            total = float(deltas[name][p])
            d_lo, d_hi = override[p]["ci"]
            v_lo, v_hi = max(base[p] + d_lo, 0.0), max(base[p] + d_hi, 0.0)
            params[p] = {
                "prior": round(base[p], 6),
                "delta": round(total, 6),
                "value": round(max(base[p] + total, 0.0), 6),
                "delta_interval": [round(d_lo, 6), round(d_hi, 6)],
                "value_interval": [round(v_lo, 6), round(v_hi, 6)],
                "value_interval_clipped_at_zero": bool(base[p] + d_lo < 0.0),
                "interval_level": float(level),
                "interval_method": ("moving-block bootstrap of the converged regression, widened by the "
                                    "estimator's measured floor"),
                "estimator_bias_at_the_fitted_point": round(abs(float(bias.get(p, 0.0))), 6),
                "std_error": src["std_error"],
                "t_stat": src["t_stat"],
                "vif": src["vif"],
                "confounded_with": src["confounded_with"],
                "noise_floor": src["noise_floor"],
                # How far above its own per-joint floor this estimate is, and how far it had to be. Carried
                # through because the verdict alone hides the margin, and a cell sitting at 1.9x reads very
                # differently from one at 18x.
                "floor_margin_x": src["floor_margin_x"],
                "floor_margin_needed_x": src["floor_margin_needed_x"],
                "unit": src["unit"],
                "excitation_metric": src["excitation_metric"],
                "excitation_seen": src["excitation_seen"],
                "excitation_needed": src["excitation_needed"],
                "identified": src["identified"],
                "reasons_not_identified": src["reasons_not_identified"],
                "suggested_experiment": src["suggested_experiment"],
                "trust_region_bound_this_joint": p in (r.get("capped") or []),
                # What else this number could be. On the cell, not only in a docstring: an engineer reading
                # "armature identified, 90% CI" is entitled to know the link's own mass and inertia were held
                # fixed and that an error in them lands here.
                "also_absorbs": PARAMETER_ALSO_ABSORBS.get(p),
            }
        joints[name] = {
            "joint": name,
            "n_samples": rep["n_samples"], "n_effective": rep["n_effective"],
            "autocorrelation_inflation": rep["autocorrelation_inflation"],
            "residual_rms_nm_before": round(float(first_rms.get(name, final[name]["rms"])), 6),
            "residual_rms_nm_after": round(float(final[name]["rms"]), 6),
            "frozen_early": name in frozen,
            "identified": [p for p in param_names if params[p]["identified"]],
            "not_identified": [p for p in param_names if not params[p]["identified"]],
            "parameters": params,
        }

    identified_pairs = [(j, p) for j, row in joints.items() for p in row["identified"]]
    refused = [{"joint": j, "parameter": p,
                "reasons": row["parameters"][p]["reasons_not_identified"],
                "suggested_experiment": row["parameters"][p]["suggested_experiment"]}
               for j, row in joints.items() for p in row["not_identified"]]

    out = {
        "ok": True,
        "stage": 2,
        "robot": getattr(gene, "id", ""),
        "setup": rig,
        "log": meta,
        "log_provenance": str(log.get("provenance") or "unstated"),
        # Carried so a follow-up experiment can be authored at the SAME per-joint length on FEWER joints,
        # which is what makes it strictly shorter than the run that produced this fit.
        "experiment": {"per_joint_s": ((plan or {}).get("budget") or {}).get("per_joint_s"),
                       "duration_s": ((plan or {}).get("budget") or {}).get("duration_s"),
                       "n_excitable": ((plan or {}).get("budget") or {}).get("n_excitable")},
        "estimator": {
            "method": "iterated Gauss-Newton on the inverse-dynamics torque residual, re-linearizing the "
                      "model's own d(tau)/d(parameter) at each step",
            "iterations_requested": int(iterations), "iterations_run": len(history),
            "interval": f"moving-block bootstrap ({n_boot} resamples) at level {level}, widened by the "
                        f"estimator's measured floor",
            "noise_floor_source": ("replaying the same excitation on our own unperturbed model and running the "
                                   "identical estimator; whatever it reports there is measurement error, and "
                                   "it is dominated by differentiating logged velocity to get acceleration"
                                   if floor else "not measured (measure_noise_floor=False)"),
            "bias_at_fitted_point_source": ("the same control replayed on the FITTED model. The prior-point "
                                            "floor answers 'how small is indistinguishable from zero'; this "
                                            "answers 'how wrong is the estimator where the answer landed', "
                                            "and for reflected inertia the two differ by enough that using "
                                            "only the first gave a measured interval coverage of 0.336 "
                                            "against a nominal 0.90"
                                            if bias_at_fit else "not measured"),
            "history": history,
        },
        "parameters_fitted": param_names,
        "parameters_not_fitted": {
            "tau_max / qd_tau_max / qd_max (the datasheet torque-speed envelope)":
                "the excitation is deliberately bounded at a fraction of datasheet peak torque and no-load "
                "speed, so it never enters the saturation regime where the knee lives. A safe experiment "
                "cannot identify the motor's limit; that needs a bench run that approaches it.",
            "kp / kd (the controller gains)":
                "ours, declared in the excitation plan and re-simulated exactly. There is nothing to identify.",
            # The one that had to be said out loud. It is not that these are un-fittable; it is that they are
            # held FIXED and their error does not disappear -- it moves into a parameter that IS reported.
            "link mass / link inertia / centre of mass (the bodies' own rigid-body parameters)":
                "ASSUMED CORRECT, and held fixed at whatever the compiler emitted from your geometry and "
                "density. They are not columns in the regressor, so an error in them cannot be REPORTED -- it "
                "is ABSORBED, and it is absorbed mostly by ARMATURE, because reflected inertia enters the "
                "joint equation through the same qdd term link inertia does. MEASURED sim2sim: a robot built "
                "with +30% link mass and inertia and no armature error at all came back armature-'identified' "
                "on 14/14 joints, 13/14 intervals excluding the true (unchanged) value, one joint at +66% "
                "(0.0100 -> 0.01664, 90% CI [0.01317, 0.02082]). What catches that case is the "
                "tracking-improvement gate under 'application' -- the fit stops getting the simulator closer "
                "the moment the error stops being one it can express. This entry is what NAMES it.",
            "contact / gearbox elasticity / backlash":
                "not represented in the compiled model at all, so there is no parameter to move. The bench rig "
                "measures a joint on a stand rather than a foot on the ground precisely so contact is not in "
                "the residual; backlash and drivetrain compliance still are, and they land in damping and "
                "frictionloss.",
        },
        "assumed_correct": {
            "what": ["link mass", "link inertia", "centre of mass", "kinematic geometry", "gear ratio"],
            "consequence": "an error in any of these is not reported; it is absorbed by the three parameters "
                           "that ARE fitted, armature first. See parameters_not_fitted for the measured case "
                           "and 'application' for the gate that refuses the resulting fit.",
        },
        "joints": joints,
        "identified_pairs": len(identified_pairs),
        "candidate_pairs": len(joints) * len(param_names),
        "refused": refused,
        "wall_clock_s": None,
    }

    if measure_trajectory:
        out["trajectory"] = _trajectory_improvement(gene, model, aligned, deltas, dofs, kp, kd, log, plan,
                                                    joints)
    # THE CONSUMER ``improvement_x`` never had. Computed here so every caller -- apply_calibration,
    # l2_requirements, engineer_brief -- rules on the same field rather than each re-deriving it or, as before,
    # none of them reading it at all.
    out["application"] = application_gate(out.get("trajectory"), len(identified_pairs))
    if measure_delay:
        out["latency"] = _delay_on_the_fitted_model(gene, model, aligned, deltas, dofs, kp, kd, log, plan,
                                                    joints, int(delay_max_ticks))
    out["wall_clock_s"] = round(time.perf_counter() - t_wall, 3)
    return out


def _delay_on_the_fitted_model(gene, model, aligned, deltas, dofs, kp, kd, log, plan, joints,
                               max_ticks: int) -> dict:
    """Sweep the actuation delay on the FITTED model rather than the prior one.

    Stage 1 measured, and its tests guard, that a delay-only sweep cannot see past a dynamics error: a 20 ms
    delay injected alongside a friction/damping/inertia error makes delay=0 the best-fitting delay on the
    uncorrected model. Stage 2 IS that correction, so this is the sweep that is entitled to an answer -- and
    the answer still cannot be applied, which is recorded on the result rather than left for the reader to
    discover downstream.
    """
    from virturoid.services.sysid.bench_rig import start_pose
    from virturoid.services.sysid.gap_report import _delay_search

    dt = float(model.opt.timestep)
    log_hz = 1.0 / dt
    ctrl_hz = float(log.get("control_hz") or (plan or {}).get("controller", {}).get("control_hz") or 100.0)
    ctrl_every = max(1, int(round(log_hz / ctrl_hz)))
    q_cmd, q_hw = aligned["q_cmd"], aligned["q_meas"]
    q0 = start_pose(model, gene) if q_hw.shape[0] == 0 else q_hw[0].copy()
    applied = {name: {p: deltas[name][p] for p in (joints.get(name, {}).get("identified") or [])}
               for name in dofs}
    fitted = _model_with(model, applied, dofs)
    out = _delay_search(fitted, q_cmd, q_hw, kp=kp, kd=kd, q_start=q0, ctrl_every=ctrl_every,
                        max_ticks=int(max_ticks), dt=dt)
    out["searched_on"] = "the FITTED model (only identified deltas applied)"
    out["why_this_model"] = ("a delay-only sweep on the uncorrected model is dominated by the parameter "
                             "error; closing that first is what makes the remaining mismatch about timing")
    return out


def _trajectory_improvement(gene, model, aligned, deltas, dofs, kp, kd, log, plan, joints) -> dict:
    """Replay the log's own commands through the model BEFORE and AFTER the fit and report the position gap.

    This is the number the engineer recognises and the one that decides whether the fit was worth applying,
    and it is deliberately measured on an INDEPENDENT quantity: the fit minimised a torque residual, so a
    torque residual falling proves only that the optimiser worked. Only the identified deltas are applied --
    applying the refused ones here would flatter the number with parameters we are not allowed to ship.
    """
    import numpy as np

    from virturoid.services.sysid.bench_rig import pd_replay, start_pose

    dt = float(model.opt.timestep)
    log_hz = 1.0 / dt
    ctrl_hz = float(log.get("control_hz") or (plan or {}).get("controller", {}).get("control_hz") or 100.0)
    ctrl_every = max(1, int(round(log_hz / ctrl_hz)))
    q_cmd, q_hw = aligned["q_cmd"], aligned["q_meas"]
    q0 = start_pose(model, gene) if q_hw.shape[0] == 0 else q_hw[0].copy()

    applied = {name: {p: deltas[name][p] for p in (joints.get(name, {}).get("identified") or [])}
               for name in dofs}
    fitted = _model_with(model, applied, dofs)

    _, q_before, _, _ = pd_replay(model, q_cmd, kp=kp, kd=kd, q_start=q0, ctrl_every=ctrl_every)
    _, q_after, _, _ = pd_replay(fitted, q_cmd, kp=kp, kd=kd, q_start=q0, ctrl_every=ctrl_every)

    per_joint = {}
    for name, adr in dofs.items():
        b = float(np.sqrt(np.mean((q_before[:, adr] - q_hw[:, adr]) ** 2)))
        a = float(np.sqrt(np.mean((q_after[:, adr] - q_hw[:, adr]) ** 2)))
        per_joint[name] = {"before_rms_rad": round(b, 6), "after_rms_rad": round(a, 6),
                           "improvement_x": round(b / a, 3) if a > 1e-12 else None}
    before = float(np.sqrt(np.mean((q_before - q_hw) ** 2)))
    after = float(np.sqrt(np.mean((q_after - q_hw) ** 2)))
    return {
        "measured": "position RMS of our replay against the log, over all logged joints",
        "independent_of_the_objective": "the fit minimised a TORQUE residual; this is a POSITION gap, so it "
                                        "is not the quantity that was optimised",
        "before_rms_rad": round(before, 6),
        "after_rms_rad": round(after, 6),
        "improvement_x": round(before / after, 3) if after > 1e-12 else None,
        "before_rms_deg": round(float(np.degrees(before)), 4),
        "after_rms_deg": round(float(np.degrees(after)), 4),
        "per_joint": per_joint,
        "caveat": "the delay is NOT applied here (no actuation-delay model exists in the compiled model), so "
                  "any residual timing error is still in this number",
    }


# ---------------------------------------------------------------------------------------------------------
# The sim2sim gate for the INTERVAL. An error bar whose coverage was never measured is a decoration.
# ---------------------------------------------------------------------------------------------------------

def coverage_table(gene, *, trials: int = 8, seed: int = 0, level: float = DEFAULT_LEVEL,
                   budget_s: float = 120.0, iterations: int = DEFAULT_ITERATIONS,
                   n_boot: int = DEFAULT_BOOTSTRAP, delay_ticks: int = 0,
                   scale_range: tuple = (0.4, 2.0), plan: dict | None = None) -> dict:
    """Draw random perturbations, fit them, and count how often the TRUE value lands inside the interval.

    This is the only test that can tell an honest interval from a decorative one, and it is the reason the
    intervals here are bootstrap-and-floor rather than a textbook standard error: the textbook version was
    measured first and under-covered.

    Unidentified parameters are excluded from the coverage count -- they report no number, so there is
    nothing to cover -- but they ARE counted and reported, because a tool that reaches nominal coverage by
    refusing to answer is a different failure with the same number.
    """
    import time

    import numpy as np

    from virturoid.services.sysid.bench_rig import bench_model, joint_dof_map
    from virturoid.services.sysid.synthetic_hardware import (
        DEFAULT_PERTURBATION,
        WHAT_SIM2SIM_DOES_NOT_PROVE,
        synthetic_hardware_log,
    )

    t0 = time.perf_counter()
    model, _ = bench_model(gene)
    dofs = joint_dof_map(model, gene)
    rng = np.random.default_rng(seed)

    covered = missed = declined = 0
    per_param: dict = {p: {"covered": 0, "missed": 0, "declined": 0, "abs_pct_error": [], "rel_width": []}
                       for p in DEFAULT_PERTURBATION}
    misses: list = []
    for t in range(int(trials)):
        inj = {p: float(v) * float(rng.uniform(*scale_range)) for p, v in DEFAULT_PERTURBATION.items()}
        plan_t, log = synthetic_hardware_log(gene, perturbation=inj, delay_ticks=int(delay_ticks),
                                             plan=plan, budget_s=budget_s)
        fit = fit_parameters(gene, log, plan=plan_t, iterations=iterations, n_boot=n_boot, level=level,
                             seed=seed + t, measure_trajectory=False)
        if not fit.get("ok"):
            continue
        for name, row in fit["joints"].items():
            adr = dofs[name]
            for p, delta in inj.items():
                truth = float(getattr(model, f"dof_{p}")[adr]) + float(delta)
                cell = row["parameters"][p]
                if not cell["identified"]:
                    declined += 1
                    per_param[p]["declined"] += 1
                    continue
                lo, hi = cell["value_interval"]
                inside = bool(lo <= truth <= hi)
                covered += int(inside)
                missed += int(not inside)
                per_param[p]["covered" if inside else "missed"] += 1
                if truth > 0:
                    per_param[p]["abs_pct_error"].append(abs(cell["value"] - truth) / truth * 100.0)
                    per_param[p]["rel_width"].append((hi - lo) / truth * 100.0)
                if not inside:
                    misses.append({"trial": t, "joint": name, "parameter": p,
                                   "truth": round(truth, 6), "interval": cell["value_interval"],
                                   "estimate": cell["value"]})

    claimed = covered + missed
    summary = {}
    for p, d in per_param.items():
        c = d["covered"] + d["missed"]
        errs, widths = d["abs_pct_error"], d["rel_width"]
        summary[p] = {"claimed": c, "covered": d["covered"], "declined": d["declined"],
                      "coverage": round(d["covered"] / c, 4) if c else None,
                      "median_abs_pct_error": round(float(np.median(errs)), 3) if errs else None,
                      "p95_abs_pct_error": round(float(np.percentile(errs, 95)), 3) if errs else None,
                      # An interval can always reach nominal coverage by being wide, so its WIDTH is reported
                      # next to its coverage. Read them together or neither means anything.
                      "median_interval_width_pct_of_truth": (round(float(np.median(widths)), 3)
                                                             if widths else None)}
    return {
        "ok": True,
        "provenance": "sim2sim",
        "what_this_does_not_prove": WHAT_SIM2SIM_DOES_NOT_PROVE,
        "trials": int(trials), "nominal_level": float(level),
        "claims": claimed, "covered": covered, "missed": missed, "declined": declined,
        "coverage_rate": round(covered / claimed, 4) if claimed else None,
        "refusal_rate": round(declined / (claimed + declined), 4) if (claimed + declined) else None,
        "per_parameter": summary,
        "misses": misses[:20],
        "reading": "coverage_rate is the fraction of IDENTIFIED parameters whose true value fell inside the "
                   "reported interval. At or above nominal_level the interval is honest or conservative; "
                   "below it, the interval is too narrow and the estimates are over-confident. refusal_rate "
                   "is reported alongside because a tool can always reach 100% coverage by declining to "
                   "answer, and that is a different failure with the same number.",
        "wall_clock_s": round(time.perf_counter() - t0, 3),
    }
