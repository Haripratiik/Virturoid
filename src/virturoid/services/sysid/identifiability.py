"""Which parameters did this experiment actually pin, and which did it not?

A calibration tool that reports a confident number for a parameter its data could not constrain is the same
class of dishonesty as a flywheel advertising clusters wider than a uniform draw: the output looks like
knowledge, is consumed as knowledge, and is noise. So every estimate this package emits has to clear three
independent gates before it is allowed to be called identified, and the ones that fail have to say WHY in a
sentence the engineer can act on.

**Gate 1 -- was the parameter physically excited?** Checked first, and checked on the trajectory rather than on
the fit, so the reason comes back physical instead of statistical. Coulomb friction needs the velocity to change
SIGN: with one-directional motion its regressor column is a constant and is exactly collinear with the gravity
offset, so no amount of data pins it. Viscous damping needs sustained speed. Reflected inertia needs
acceleration. A joint that was commanded to hold still fails all three, and should be told so rather than
handed three numbers.

**Gate 2 -- is it separable from the others?** Variance inflation factor, the conventional diagnostic, at the
conventional threshold of 10. VIF answers the question that matters here exactly: "could a different parameter
have produced the same torque signature under this excitation?" When it fails we name the partner it is confused
with, because "damping and armature are indistinguishable in your data" is actionable and "VIF = 41" is not.

**Gate 3 -- is it bigger than what we can measure?** Two floors. The statistical one is the coefficient's own
standard error, computed with an EFFECTIVE sample size rather than the raw row count -- these are 500 Hz time
series whose residuals are strongly autocorrelated, and a naive standard error over 25,000 correlated rows would
claim about a hundred times more precision than the experiment contains. The empirical one is the estimator's
measured bias on a log with nothing wrong with it (``gap_report`` runs that control): it is dominated by
differentiating logged velocity to get acceleration, and it is PER JOINT -- 0.0023 to 0.0216 N.m of phantom
friction across one 14-joint quadruped, an order of magnitude of spread, so it is never collapsed to a single
constant.

The empirical floor is compared against a BIAS-CORRECTED estimate rather than against the raw one, and that is
a correction. The floor is not noise -- it is this estimator's own bias -- so the estimate carries it, and
checking the estimate straight against it lets a number clear its own bias. Measured: an armature injection of
+0.0008 N.m, under every joint's floor of 0.00158-0.00221, came back "identified" on 14/14 joints at a median
0.00275, which is that joint's floor plus its bias almost exactly. An estimate is therefore required to reach
``FLOOR_MARGIN`` (= 2) times its joint's floor, which is what "even after subtracting the whole bias, what is
left still clears the floor" works out to.

The thresholds below are conventional, not tuned to make our own numbers look good; they are stated as module
constants so a reader can see what "identified" was allowed to mean.
"""

from __future__ import annotations

T_STAT_MIN = 3.0        # |theta| must exceed 3 standard errors (~99.7% under normality)
VIF_MAX = 10.0          # conventional collinearity threshold (Kutner et al.)
CI_Z = 1.96             # 95% interval

#: How far above its own measurement floor an estimate has to be before the change is called RESOLVED. Two,
#: and the 2 is derived rather than chosen: the floor is the estimator's BIAS (what it returns on a log with
#: nothing wrong with it), of unknown sign, so a reported |estimate| is consistent with a truth as small as
#: |estimate| - floor, and that worst case has to clear the floor itself. See the gate below for the measured
#: check of the same number over 173 cells and six sim2sim regimes.
FLOOR_MARGIN = 2.0

#: What the trajectory must contain before a parameter can be identified at all, and what to add when it does
#: not. ``metric`` is measured by ``gap_report`` over that joint's own excitation window.
EXCITATION_REQUIREMENTS = {
    "frictionloss": {
        "metric": "velocity_sign_reversals", "minimum": 4,
        "unit": "N.m",
        "why": "Coulomb friction acts as tau = f*sign(qd). Without velocity sign reversals sign(qd) is a "
               "constant and is exactly collinear with the gravity/offset term, so the two cannot be told apart",
        "fix": "add a slow triangle-wave segment (~1 s) that reverses direction at least twice at low speed",
    },
    "damping": {
        "metric": "speed_rms_radps", "minimum": 0.05,
        "unit": "N.m.s/rad",
        "why": "viscous damping acts as tau = b*qd and is invisible at rest",
        "fix": "add sustained motion at moderate speed; a slow triangle or a low-frequency sinusoid works",
    },
    "armature": {
        "metric": "accel_rms_radps2", "minimum": 0.5,
        "unit": "kg.m^2",
        "why": "reflected inertia acts as tau = I*qdd and is invisible at constant velocity",
        "fix": "add a higher-frequency chirp segment or sharper steps; inertia only shows under acceleration",
    },
    "offset": {
        "metric": "n_samples", "minimum": 50,
        "unit": "N.m",
        "why": "a constant torque offset (gravity, mass, or a zero-point error) needs only samples",
        "fix": "log more of the joint's hold segment",
    },
}


def _ols(X, y):
    """Ordinary least squares with the pieces needed for an honest error bar."""
    import numpy as np

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ theta
    dof = max(n - p, 1)
    sigma2 = float(resid @ resid) / dof
    try:
        xtx_inv = np.linalg.pinv(X.T @ X)
    except np.linalg.LinAlgError:
        xtx_inv = np.zeros((p, p))
    return theta, resid, sigma2, xtx_inv


def _effective_n(resid):
    """Sample size deflated for autocorrelation (Bartlett): ``n*(1-rho)/(1+rho)`` at lag 1.

    Without this the standard errors are fiction. A 50-second log at 500 Hz is 25,000 rows but nothing like
    25,000 independent observations -- consecutive samples of a 2 Hz sinusoid are nearly the same measurement.
    """
    import numpy as np

    r = np.asarray(resid, dtype=float)
    n = r.size
    if n < 8:
        return float(max(n, 1))
    r = r - r.mean()
    denom = float(r @ r)
    if denom <= 1e-30:
        return float(n)
    rho = float(r[1:] @ r[:-1]) / denom
    rho = min(max(rho, -0.999), 0.999)
    return float(min(max(n * (1.0 - rho) / (1.0 + rho), 4.0), n))


def _vif(X):
    """Variance inflation factor per column, plus each column's worst partner.

    The intercept is excluded from the reported list but kept in every auxiliary regression, so a column that is
    collinear with a CONSTANT (Coulomb friction under one-directional motion, the exact failure this package is
    built to catch) shows up as inflated rather than as clean.
    """
    import numpy as np

    X = np.asarray(X, dtype=float)
    p = X.shape[1]
    vifs, partners = [], []
    for i in range(p):
        others = np.delete(X, i, axis=1)
        if others.shape[1] == 0:
            vifs.append(1.0); partners.append(None); continue
        col = X[:, i]
        var = float(((col - col.mean()) ** 2).sum())
        if var <= 1e-30:
            # A CONSTANT column (the intercept). Its variance inflation is undefined, not infinite -- reporting
            # inf here would fail the intercept's separability gate on every joint forever, which is a bug that
            # reads exactly like a finding. Real collinearity WITH the constant still shows up, on the other
            # column: friction under one-directional motion inflates the FRICTION column, which is the signal.
            vifs.append(None); partners.append(None); continue
        beta, *_ = np.linalg.lstsq(others, col, rcond=None)
        rss = float(((col - others @ beta) ** 2).sum())
        r2 = 1.0 - rss / var
        r2 = min(max(r2, 0.0), 1.0 - 1e-12)
        vifs.append(1.0 / (1.0 - r2))
        best, best_r = None, 0.0
        for k in range(p):
            if k == i:
                continue
            a, b = col - col.mean(), X[:, k] - X[:, k].mean()
            da, db = float(np.sqrt((a * a).sum())), float(np.sqrt((b * b).sum()))
            if da <= 1e-30 or db <= 1e-30:
                continue
            corr = abs(float(a @ b) / (da * db))
            if corr > best_r:
                best, best_r = k, corr
        partners.append((best, round(best_r, 4)) if best is not None else None)
    return vifs, partners


def identifiability_report(joint: str, X, y, param_names, *, excitation_stats: dict,
                           noise_floor: dict | None = None, override: dict | None = None) -> dict:
    """Fit one joint's residual and rule on every parameter in it.

    ``X`` columns must be ``[intercept] + [one sensitivity column per name in param_names]``, in that order --
    the columns are the MODEL'S OWN derivative of predicted torque with respect to each parameter, so a
    coefficient comes out already in the parameter's physical units and the collinearity we measure is real
    physical confusability rather than an artefact of some basis we invented.

    ``override`` (Stage 2) supplies ``{name: {"estimate", "ci", "se"}}`` for parameters whose value came from
    somewhere other than this single regression -- specifically an ITERATED fit, where ``X`` and ``y`` are the
    final linearization and the number being ruled on is the ACCUMULATED total. Without it, the gates would
    rule on the last Gauss-Newton step, which is near zero at convergence by construction: every parameter
    would fail "below statistical resolution" precisely because the fit had worked. The gates themselves --
    excitation, separability, resolution, floor -- are unchanged and still read off this ``X``, because they
    are properties of the EXPERIMENT and are what the override must not be allowed to soften.
    """
    import numpy as np

    names = ["offset", *list(param_names)]
    over = dict(override or {})
    theta, resid, sigma2, xtx_inv = _ols(X, y)
    n = int(np.asarray(X).shape[0])
    n_eff = _effective_n(resid)
    inflate = max(n / max(n_eff, 1.0), 1.0)
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2 * inflate, 0.0))
    vifs, partners = _vif(X)
    floor = dict(noise_floor or {})

    yv = float(((np.asarray(y) - np.mean(y)) ** 2).sum())
    r2 = 1.0 - float((resid @ resid)) / yv if yv > 1e-30 else 0.0

    params = {}
    for i, name in enumerate(names):
        req = EXCITATION_REQUIREMENTS.get(name, {})
        unit = req.get("unit", "")
        ov = over.get(name) or {}
        est = float(ov["estimate"]) if "estimate" in ov else float(theta[i])
        err = float(ov["se"]) if "se" in ov else float(se[i])
        ci = [float(ov["ci"][0]), float(ov["ci"][1])] if ov.get("ci") else [est - CI_Z * err, est + CI_Z * err]
        vif = None if vifs[i] is None else float(vifs[i])
        t_stat = abs(est) / err if err > 1e-30 else float("inf")
        fl = abs(float(floor.get(name, 0.0)))

        metric = req.get("metric")
        seen = float(excitation_stats.get(metric, 0.0)) if metric else float("inf")
        need = float(req.get("minimum", 0.0))
        excited = seen >= need

        reasons = []
        if not excited:
            reasons.append(f"not excited: {metric} = {seen:g}, need >= {need:g}. {req.get('why', '')}. "
                           f"FIX: {req.get('fix', '')}")
        if vif is not None and vif > VIF_MAX:
            pr = partners[i]
            with_name = names[pr[0]] if pr else "another term"
            corr = pr[1] if pr else float("nan")
            reasons.append(f"not separable: VIF = {vif:.1f} (> {VIF_MAX:g}); under this excitation it is "
                           f"confounded with '{with_name}' (|corr| = {corr}). Their torque signatures are the "
                           f"same shape here, so any split between them is arbitrary")
        if t_stat < T_STAT_MIN:
            reasons.append(f"below statistical resolution: |estimate| = {abs(est):.5g} {unit} is only "
                           f"{t_stat:.2f} standard errors (need >= {T_STAT_MIN:g}); the experiment cannot "
                           f"distinguish it from zero")
        # THE FLOOR GATE, and it compares a BIAS-CORRECTED estimate to the floor. Comparing the raw estimate
        # was the defect: the floor is not noise, it is this estimator's own BIAS -- what it reports on a log
        # with nothing wrong with it -- so the estimate carries it, and an estimate that carries its own floor
        # clears its own floor. Measured: an armature injection of +0.0008, under every joint's floor of
        # 0.00158-0.00221, came back "identified" on 14/14 joints at a median 0.00275, which is that joint's
        # floor plus its bias-at-the-fitted-point almost exactly. The intervals still covered, so this was a
        # weak gate rather than a false interval; it was also the mechanism by which a change SMALLER THAN THE
        # INSTRUMENT'S RESOLUTION got written into a customer's model.
        #
        # A bias of unknown sign and magnitude <= floor means a reported |estimate| is consistent with a truth
        # anywhere in [|est| - floor, |est| + floor]. The change is resolvable only if the SMALL end of that
        # range still clears the floor, which is |est| >= FLOOR_MARGIN * floor with FLOOR_MARGIN = 2. Derived,
        # then CHECKED against 173 (case, joint, parameter) cells from six sim2sim regimes -- default
        # injection, a quarter of it, an above-floor set, a sub-floor injection, hardware identical to the sim,
        # and a +30% link-mass/inertia misspecification. |est|/floor for cells whose TRUE change is above that
        # joint's floor: n=112, min 1.18, 5th pct 3.04. For cells whose true change is UNDER it: n=61, max
        # 3.02, and 53 of the 61 sit at ratio <= 1.1. At 2.0 the gate keeps 111 of 112 genuinely-resolvable
        # cells and rejects 53 of 61 unresolvable ones; the 8 it still admits are the misspecification, which
        # is not a resolution failure and is caught by fit.application_gate instead.
        #
        # The floor is PER JOINT throughout -- measured 0.0023 to 0.0216 for frictionloss across one 14-joint
        # quadruped, an order of magnitude of spread -- so a quiet joint is never held to a noisy joint's bar,
        # and the ratio below is against that joint's own number.
        margin_x = abs(est) / fl if fl > 0.0 else float("inf")
        if fl > 0.0 and margin_x < FLOOR_MARGIN:
            reasons.append(f"inside the measurement floor: |estimate| = {abs(est):.5g} {unit} is "
                           f"{margin_x:.2f}x this joint's floor of {fl:.5g} {unit} (need >= "
                           f"{FLOOR_MARGIN:g}x). That floor is what this estimator reports on a log with "
                           f"NOTHING wrong with it -- a BIAS, dominated by differentiating logged velocity to "
                           f"get acceleration -- so the estimate carries it and is consistent with a true "
                           f"change as small as {max(abs(est) - fl, 0.0):.5g}, which does not clear the floor "
                           f"itself. Comparing the estimate straight to the floor would let it clear its own "
                           f"bias. Log acceleration or motor current directly, or raise the log rate, to "
                           f"resolve a change this small")

        params[name] = {
            "estimate": round(est, 6), "unit": unit,
            "std_error": round(err, 6),
            "ci95": [round(ci[0], 6), round(ci[1], 6)],
            "t_stat": round(t_stat, 3) if np.isfinite(t_stat) else None,
            "vif": round(vif, 3) if (vif is not None and np.isfinite(vif)) else None,
            "confounded_with": ((names[partners[i][0]] if partners[i] else None)
                                if (vif is not None and vif > VIF_MAX) else None),
            "excitation_metric": metric, "excitation_seen": round(seen, 5) if np.isfinite(seen) else None,
            "excitation_needed": need,
            "noise_floor": round(fl, 6) if fl else 0.0,
            # What the floor gate actually reads, reported next to the verdict so the margin is visible rather
            # than only its sign. Anything under FLOOR_MARGIN is a change this instrument cannot resolve.
            "floor_margin_x": (round(margin_x, 3) if np.isfinite(margin_x) else None),
            "floor_margin_needed_x": FLOOR_MARGIN,
            "identified": not reasons,
            "reasons_not_identified": reasons,
            # Only offered when the experiment was the problem. A parameter that WAS excited and still failed
            # is confounded or under-resolved, and "add a slow triangle" would not fix either -- suggesting it
            # anyway would send the engineer back to the robot to re-run something that cannot help.
            "suggested_experiment": req.get("fix") if (reasons and not excited) else None,
        }

    ident = [k for k, v in params.items() if v["identified"]]
    return {
        "joint": joint,
        "n_samples": n, "n_effective": round(n_eff, 1),
        "autocorrelation_inflation": round(inflate, 2),
        "fit_r2": round(r2, 5),
        "residual_rms_nm": round(float(np.sqrt(np.mean(resid ** 2))), 6),
        "identified": ident,
        "not_identified": [k for k in params if k not in ident],
        "parameters": params,
    }
