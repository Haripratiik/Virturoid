"""The calibration wedge under ACTUATION DELAY -- the residual its own opening paragraph calls dominant.

`sysid/__init__.py` cites Hwangbo et al. (Science Robotics, ANYmal) for the claim that the dominant sim-to-real
residual is actuator dynamics AND control-signal delay. `synthetic_hardware.DEFAULT_DELAY_TICKS` injects 2
control ticks (20 ms) of it by default, and `sysid.tools.simulate_bench_log` defaults to `delay_ms=20.0`. So
the default path through the shipped surface runs the wedge against the term it names as the point of the
exercise.

That path used to fail. MEASURED on the Menagerie Unitree Go2 and recorded in
`docs/calibration_wedge_under_delay.md`: at the package's own default the delay search returned 10 ms for a
20 ms injection and 10 ms for a 40 ms one, and the tracking gate withheld the fit at 1.016x against a 1.5x
threshold -- with the metric's own CEILING at that delay also 1.016x, so no fit of any quality could have
passed. The refusal was correct and carried no information.

Both are closed, and this file is the contract for how. Four parts:

  * THE MECHANISM -- why the closed-loop delay sweep cannot work on a real log, measured rather than asserted.
    These are diagnosis, and they are what makes the fix the right fix instead of a lucky one.
  * THE FIX -- the delay is now measured OPEN-LOOP off the log (`_delay_from_command_response`), and the gate
    is scored with both replays at that delay. 0 / 20 / 40 ms all come back exact.
  * THE CONTROLS -- the gate must not have been opened on the way. A misspecification the estimator cannot
    express is still refused WITH a delay present, and a joint the experiment never moved still reports
    nothing.
  * WHAT IS STILL TRUE AND STILL COSTS US -- the one-tick bias a filtered torque channel introduces, and the
    fact that at 40 ms (double Hwangbo's figure) the gate still refuses on the merits.

Cost note: the Go2 at the package's 120 s default budget costs ~45 s per fit. The whole phenomenon survives
narrowing to two joints and a 12 s excitation, so that is what this file runs; the full-plan numbers are in the
doc and quoted in the assertions' messages where they differ.

WE OWN NO HARDWARE. Everything here is sim2sim -- both the "robot" and the "sim" are MuJoCo, so MuJoCo's own
modelling error cancels exactly and is invisible. See ``WHAT_SIM2SIM_DOES_NOT_PROVE``. The open-loop estimator
recovers the delay EXACTLY here, and it does so because the reconstruction is algebraically the expression the
rig used; the tests that matter for a real bench are the dirty-log ones below, not the clean one.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("VIRTUROID_DISABLE_GAIT_HINTS", "1")

_MUJOCO = importlib.util.find_spec("mujoco") is not None
_GO2 = Path(os.path.expanduser("~/.cache/robot_descriptions/mujoco_menagerie/unitree_go2/go2.xml"))

pytestmark = [
    pytest.mark.skipif(not _MUJOCO, reason="system identification needs MuJoCo"),
    # A REAL robot, deliberately. The defect this file records was invisible on the composed dog every other
    # number in this package was measured on, and that is part of the finding -- so a fixture would erase it.
    pytest.mark.skipif(not _GO2.exists(), reason=f"needs the MuJoCo Menagerie Go2 at {_GO2}"),
    # This file never reads a walk verdict or a gait parameter; it drives the bench rig, which welds the base.
    pytest.mark.no_gait_fit,
]

#: Two joints and 12 s instead of twelve joints and 72 s. One hip (where the Go2's bench gain is floored at
#: 7.6x its inertia-scaled value, so the loop is delay-sensitive) and one thigh (where it is not).
ONLY_JOINTS = ["FL_hip", "FL_thigh"]
BUDGET_S = 12.0

#: One control tick at the plan's 100 Hz. The package's own default injection is 2 ticks.
MS_PER_TICK = 10.0
DEFAULT_DELAY_MS = 20.0


@pytest.fixture(scope="module")
def go2():
    from virturoid.services.robot_import import import_robot
    return import_robot(str(_GO2), robot_id="go2_delay_wedge")["gene"]


@pytest.fixture(scope="module")
def plan(go2):
    from virturoid.services.sysid import build_excitation
    return build_excitation(go2, budget_s=BUDGET_S, only_joints=ONLY_JOINTS)


@pytest.fixture(scope="module")
def rig(go2, plan):
    """``(model, kp, kd, dofs, ctrl_every, q_cmd, q0)`` -- the bench setup, built once."""
    from virturoid.services.sysid.bench_rig import (
        bench_gains,
        bench_model,
        joint_dof_map,
        start_pose,
    )
    from virturoid.services.sysid.excitation import excitation_command_series

    model, _ = bench_model(go2)
    kp, kd, _ = bench_gains(model)
    dofs = joint_dof_map(model, go2)
    dt = float(model.opt.timestep)
    ctrl_every = max(1, int(round((1.0 / dt) / float(plan["controller"]["control_hz"]))))
    _, q_cmd = excitation_command_series(go2, plan)
    q0 = start_pose(model, go2)
    for j in plan["joints"]:
        q0[int(j["dof"])] = float(j["q_start_rad"])
    return {"model": model, "kp": kp, "kd": kd, "dofs": dofs, "dt": dt,
            "ctrl_every": ctrl_every, "q_cmd": q_cmd, "q0": q0}


def _hardware(rig, delay_ticks, perturbation=None):
    """The 'robot': our model + the default perturbation + ``delay_ticks`` of transport delay."""
    from virturoid.services.sysid.bench_rig import pd_replay
    from virturoid.services.sysid.synthetic_hardware import DEFAULT_PERTURBATION, perturbed_model

    hw = perturbed_model(rig["model"], DEFAULT_PERTURBATION if perturbation is None else perturbation)
    t, q, qd, tau = pd_replay(hw, rig["q_cmd"], kp=rig["kp"], kd=rig["kd"], q_start=rig["q0"],
                              ctrl_every=rig["ctrl_every"], delay_ticks=int(delay_ticks))
    return hw, {"t": t, "q": q, "qd": qd, "tau": tau}


def _aligned(rig, log):
    """The ``_align_log`` shape ``_delay_from_command_response`` consumes, straight off a replay."""
    return {"q_cmd": rig["q_cmd"], "q_meas": log["q"], "qd_meas": log["qd"], "tau_meas": log["tau"],
            "t": log["t"]}


def _cmd_response(rig, plan, aligned, model=None, max_ticks=8):
    from virturoid.services.sysid.gap_report import _delay_from_command_response

    return _delay_from_command_response(model or rig["model"], aligned, rig["dofs"], plan,
                                        kp=rig["kp"], kd=rig["kd"], ctrl_every=rig["ctrl_every"],
                                        max_ticks=max_ticks, dt=rig["dt"])


def _gate_ceiling(rig, delay_ticks):
    """The largest ``improvement_x`` the OLD, delay-blind metric could EVER return at this delay.

    The gate divided RMS(prior replay vs log) by RMS(fitted replay vs log) with both replays at delay 0.
    Substitute the TRUE hardware model for the fitted one and the ratio is maximal by construction: no fit can
    do better than the parameters that generated the log. Three PD replays, no fit.
    """
    import numpy as np

    from virturoid.services.sysid.bench_rig import pd_replay

    hw, log = _hardware(rig, delay_ticks)
    q_log = log["q"]
    kw = dict(kp=rig["kp"], kd=rig["kd"], q_start=rig["q0"], ctrl_every=rig["ctrl_every"], delay_ticks=0)
    _, q_prior, _, _ = pd_replay(rig["model"], rig["q_cmd"], **kw)
    _, q_true, _, _ = pd_replay(hw, rig["q_cmd"], **kw)
    before = float(np.sqrt(np.mean((q_prior - q_log) ** 2)))
    after = float(np.sqrt(np.mean((q_true - q_log) ** 2)))
    return float("inf") if after <= 1e-12 else before / after


@pytest.fixture(scope="module")
def fit_at(go2, plan):
    """``fit_at(delay_ticks, **synthetic_hardware_log kwargs)`` -- a real fit, memoised per configuration."""
    from virturoid.services.sysid import fit_parameters
    from virturoid.services.sysid.synthetic_hardware import DEFAULT_PERTURBATION, synthetic_hardware_log

    cache: dict = {}

    def _get(delay_ticks: int, **kw):
        key = (int(delay_ticks), tuple(sorted((k, str(v)) for k, v in kw.items())))
        if key not in cache:
            kw.setdefault("perturbation", DEFAULT_PERTURBATION)
            _, log = synthetic_hardware_log(go2, delay_ticks=int(delay_ticks), plan=plan, **kw)
            cache[key] = fit_parameters(go2, log, plan=plan, n_boot=32)
        return cache[key]

    return _get


# =========================================================================================================
# THE MECHANISM. Why a closed-loop delay sweep cannot do this job, measured. These are the diagnosis, and
# they are why the fix is an OPEN-LOOP estimator rather than a wider grid or a finer one.
# =========================================================================================================

@pytest.mark.parametrize("delay_ticks", [0, 2, 4])
def test_the_delay_is_recoverable_from_this_excitation_given_the_right_model(rig, delay_ticks):
    """Hand the closed-loop sweep a model that already carries the true perturbation and it returns the
    injected delay EXACTLY.

    This is the control that separates "our search is broken" from "this experiment cannot see delay" -- and
    they need opposite fixes, so guessing was not allowed. It passes at 0, 20 and 40 ms with
    ``fraction_of_mismatch_explained_by_delay == 1.0``: the excitation contains everything needed to pin the
    delay, the range is wide enough, and the grid lands on the truth. Every failure this file records was
    therefore OURS, which is what made it worth closing.
    """
    from virturoid.services.sysid.gap_report import _delay_search

    hw, log = _hardware(rig, delay_ticks)
    q_log = log["q"]
    out = _delay_search(hw, rig["q_cmd"], q_log, kp=rig["kp"], kd=rig["kd"], q_start=q_log[0].copy(),
                        ctrl_every=rig["ctrl_every"], max_ticks=8, dt=rig["dt"])
    assert out["delay_ms"] == pytest.approx(delay_ticks * MS_PER_TICK), (
        f"the ORACLE model missed an injected {delay_ticks * MS_PER_TICK} ms: got {out['delay_ms']} ms")
    assert out["at_grid_edge"] is False


def test_the_delay_objective_is_a_hole_not_a_basin(rig):
    """WHY the closed-loop sweep is fragile, measured rather than asserted.

    On the true model the objective drops to EXACTLY zero at the injected tick and its immediate neighbours are
    ~0.016 and ~0.069 rad. The global minimum is one grid point wide and exists only because an exact model
    plus an exact delay reproduces the log exactly. There is no basin to fall into, so any residual dynamics
    error fills the hole and leaves a plateau whose argmin is decided in the fourth significant figure.

    The shape has a cause worth naming: under-shooting a delay SATURATES. A replay with too little delay does
    not ring, so its error against a ringing log is just the log's ringing amplitude and stops growing; a
    replay with too much delay rings out of phase and scores WORSE than one that does not ring at all. Sub-tick
    interpolation cannot fix that, and neither can a correlation objective -- both were measured.
    """
    from virturoid.services.sysid.gap_report import _delay_search

    ticks = int(DEFAULT_DELAY_MS / MS_PER_TICK)
    hw, log = _hardware(rig, ticks)
    q_log = log["q"]
    grid = {r["delay_ticks"]: r["trajectory_rms_rad"] for r in _delay_search(
        hw, rig["q_cmd"], q_log, kp=rig["kp"], kd=rig["kd"], q_start=q_log[0].copy(),
        ctrl_every=rig["ctrl_every"], max_ticks=4, dt=rig["dt"])["grid"]}

    assert grid[ticks] < 1e-9, f"the truth did not reproduce the log exactly: {grid[ticks]}"
    # ...and one tick either side is enormous by comparison. A basin would be gradual.
    assert grid[ticks - 1] > 1e-3, grid
    assert grid[ticks + 1] > 1e-3, grid
    # The two WRONG neighbours are within a whisker of each other, which is the plateau the search lands on
    # once the hole is filled in.
    assert abs(grid[0] - grid[1]) / max(grid[0], 1e-12) < 0.05, (
        f"0 and 10 ms are meant to be near-indistinguishable off the truth: {grid[0]} vs {grid[1]}")


def test_the_trajectory_sweep_is_biased_toward_zero_and_may_not_promote_itself_on_a_margin(rig):
    """The reason ``_merge_latency`` never lets the closed-loop sweep claim a delay on a MARGIN.

    A margin gate is what makes the open-loop estimate able to report a zero delay, and it is the obvious thing
    to reuse here. It must not be: because under-shooting saturates, the sweep on the PRIOR model puts the
    zero-delay entry comfortably ahead of every alternative on a log that carries 20 ms. That is a confident,
    wrong answer, and it is strictly worse than the refusal it would replace.
    """
    from virturoid.services.sysid.gap_report import _delay_search

    ticks = int(DEFAULT_DELAY_MS / MS_PER_TICK)
    _hw, log = _hardware(rig, ticks)
    q_log = log["q"]
    out = _delay_search(rig["model"], rig["q_cmd"], q_log, kp=rig["kp"], kd=rig["kd"],
                        q_start=q_log[0].copy(), ctrl_every=rig["ctrl_every"], max_ticks=4, dt=rig["dt"])
    grid = {r["delay_ticks"]: r["trajectory_rms_rad"] for r in out["grid"]}
    assert out["delay_ticks"] != ticks, (
        "this test is no longer adversarial: the prior-model sweep now lands on the truth")
    best = min(grid.values())
    runner_up = min(v for k, v in grid.items() if v != best)
    assert 1.0 - best / runner_up > 0.20, (
        f"the WRONG answer should look convincing on a margin -- that is the trap. best {best}, "
        f"runner-up {runner_up}, grid {grid}")


# =========================================================================================================
# THE FIX. The delay comes off the LOG, open loop, and the gate is scored at it.
# =========================================================================================================

@pytest.mark.parametrize("delay_ticks", [0, 2, 4])
def test_the_delay_is_recovered_on_a_real_robot(fit_at, delay_ticks):
    """THE HEADLINE, and the cell the engineer's sweep found empty. 0 / 20 / 40 ms, exactly, on an imported
    customer robot, through the shipped ``fit_parameters``.

    Tolerance is EXACT and that is the right bar, not a strict one: the delay lives on a control-tick grid, so
    a neighbouring tick is a different answer rather than a noisy one. The full 12-joint 120 s plan returns the
    same three values (`docs/calibration_wedge_under_delay.md` section 9).
    """
    lat = fit_at(delay_ticks)["latency"]
    assert lat["identified"] is True, lat.get("not_identified_because")
    assert lat["delay_ms"] == pytest.approx(delay_ticks * MS_PER_TICK), (
        f"injected {delay_ticks * MS_PER_TICK} ms, recovered {lat['delay_ms']} ms")
    assert lat["source"] == "command_response"
    assert lat["at_grid_edge"] is False


def test_finding_no_delay_is_a_finding_not_a_refusal(fit_at):
    """A zero delay is now REPORTED, which the old estimator could not do on any log that was not a perfect
    sim2sim.

    Its trust metric was ``explained = 1 - best_rms / rms_at_zero_delay``: when the true delay is zero the
    argmin IS the zero-delay entry, so ``explained`` was identically 0 and ``identified`` was always False. The
    Go2 at 0 ms injected returned the right answer and reported *"sweeping delay explains only 0.0% of the
    trajectory mismatch (need 50%)"*. "Your robot has no meaningful actuation delay" is a useful finding and
    that function could not say it. A margin over the next-best lag can.
    """
    lat = fit_at(0)["latency"]
    assert lat["delay_ms"] == 0.0
    assert lat["identified"] is True, lat.get("not_identified_because")
    assert lat["margin_over_next_best_tick"] >= 0.15


def test_the_open_loop_estimate_does_not_depend_on_how_good_the_model_is(rig, plan):
    """THE property that makes this estimator the right one, isolated.

    The closed-loop sweep has to simulate the plant, so a wrong plant moves its answer -- that is the whole
    defect. This one aligns two signals that are both IN THE LOG: the declared control law evaluated on the
    logged state, and the torque the log says was applied. Hand it the prior model and the true model and it
    must return the same delay, because neither one enters the calculation beyond the forcerange clamp.
    """
    ticks = int(DEFAULT_DELAY_MS / MS_PER_TICK)
    hw, log = _hardware(rig, ticks)
    aligned = _aligned(rig, log)
    on_prior = _cmd_response(rig, plan, aligned, model=rig["model"])
    on_oracle = _cmd_response(rig, plan, aligned, model=hw)
    assert on_prior["identified"] and on_oracle["identified"]
    assert on_prior["delay_ms"] == on_oracle["delay_ms"] == DEFAULT_DELAY_MS


def test_a_fit_at_the_packages_own_default_delay_is_applied(fit_at):
    """The engineer's headline cell, flipped. At the package's own default 20 ms the gate now PASSES.

    The threshold is untouched at 1.5x; what moved is the quantity it rules on. Full plan: 1.683x.
    """
    from virturoid.services.sysid.fit import MIN_TRACKING_IMPROVEMENT_X

    app = fit_at(2)["application"]
    assert app["passed"] is True, app["verdict"]
    assert app["provisional"] is False
    assert app["improvement_x"] >= MIN_TRACKING_IMPROVEMENT_X


def test_the_gate_is_scored_at_the_identified_delay_and_that_is_what_moved_it(fit_at):
    """The mechanism of the flip above, pinned so it cannot be mistaken for a threshold change.

    Both replays now run at the identified delay, so the timing error cancels in the ratio the way the ratio
    always assumed it did. The SAME fit scored at the old point (both replays at zero delay) is still under the
    threshold -- 1.016x on the full plan -- and that number is carried alongside rather than discarded.
    """
    from virturoid.services.sysid.fit import MIN_TRACKING_IMPROVEMENT_X

    traj = fit_at(2)["trajectory"]
    assert traj["scored_at_delay_ms"] == pytest.approx(DEFAULT_DELAY_MS)
    assert traj["delay_identified"] is True
    assert traj["improvement_x"] >= MIN_TRACKING_IMPROVEMENT_X
    assert traj["improvement_x_at_zero_delay"] < MIN_TRACKING_IMPROVEMENT_X, (
        "this test is no longer adversarial: the OLD scoring point now clears the gate too, so the change "
        "being tested is not what moved the verdict")


def test_the_old_scoring_points_ceiling_is_still_under_the_threshold(rig):
    """WHY the scoring point had to move, and it is not a tuning argument.

    Scored with both replays at delay 0, ``improvement_x`` has a CEILING that depends only on the delay: no fit
    can beat the parameters that generated the log. Measured here, and on the full plan 4.964x at 10 ms /
    1.016x at 20 ms / 1.000x at 40 ms. Below the threshold, a refusal is not evidence about the fit.
    """
    from virturoid.services.sysid.fit import MIN_TRACKING_IMPROVEMENT_X

    at_10, at_20, at_40 = _gate_ceiling(rig, 1), _gate_ceiling(rig, 2), _gate_ceiling(rig, 4)
    assert at_10 >= MIN_TRACKING_IMPROVEMENT_X, f"10 ms should be inside reach; got {at_10:.3f}x"
    assert at_20 < MIN_TRACKING_IMPROVEMENT_X, (
        f"this test is no longer adversarial: the delay-blind ceiling at the package's DEFAULT 20 ms is "
        f"{at_20:.3f}x, which now clears the {MIN_TRACKING_IMPROVEMENT_X}x threshold")
    assert at_40 < 1.05, f"the ceiling at 40 ms should be ~1.0 (no headroom at all); got {at_40:.3f}x"
    assert at_20 > at_40, "the ceiling should fall as the delay grows"


# =========================================================================================================
# THE CONTROLS. A gate that stopped refusing the things it exists to refuse would be a worse defect than the
# one this file closed.
# =========================================================================================================

def test_a_misspecification_is_still_refused_when_the_robot_also_has_delay(fit_at):
    """THE control on the scoring change. ``application_gate`` was sized against a robot built with +30% link
    mass and inertia -- an error NO combination of frictionloss/damping/armature can express, and one the
    estimator absorbs into armature. Scoring at the identified delay must not let it through.

    MEASURED on the full 12-joint plan at the new scoring point: 1.001x with no delay, 0.996x at 20 ms. The
    empty band the 1.5x threshold sits in is narrower than it was -- see the constant's docstring -- and it is
    still empty.
    """
    fit = fit_at(2, perturbation={}, link_scale=1.30)
    app = fit["application"]
    assert app["passed"] is False, (
        f"a +30% link mass/inertia error passed the tracking gate at {app['improvement_x']}x -- the gate that "
        f"exists for exactly this case has been opened")
    assert app["provisional"] is True


def test_nothing_reaches_the_model_when_the_gate_refuses(go2, fit_at):
    """The refusal has to be load-bearing, not decorative: a withheld fit must change nothing."""
    from virturoid.services.sysid import apply_calibration, calibration_of

    fit = fit_at(2, perturbation={}, link_scale=1.30)
    rec = calibration_of(apply_calibration(go2, fit)) or {}
    assert not (rec.get("joints") or {}), (
        "a provisional fit wrote parameters into the model; only allow_provisional=True may do that")


def test_a_joint_the_experiment_never_moved_reports_no_latency(rig, plan):
    """The gate that stops the estimator inventing a delay out of a constant.

    This plan excites two of the Go2's twelve joints; the other ten hold a constant torque that fits EVERY
    candidate lag perfectly. Without the torque-swing floor they came back with a residual of 0 at the best lag
    and a spurious margin of 1.0 -- twelve unanimous joints, ten of which had measured nothing.
    """
    _hw, log = _hardware(rig, 2)
    out = _cmd_response(rig, plan, _aligned(rig, log))
    identified = {n for n, r in out["per_joint"].items() if r["identified"]}
    assert identified == set(ONLY_JOINTS), f"expected only {ONLY_JOINTS} to resolve a lag; got {identified}"
    for name, row in out["per_joint"].items():
        if name in ONLY_JOINTS:
            continue
        assert "not driven hard enough" in (row["not_identified_because"] or ""), row


def test_a_log_the_declared_control_law_does_not_explain_is_refused(rig, plan):
    """A customer who ran a different loop than the plan specifies must not get a latency number.

    Replacing the applied torque with noise leaves an alignment problem with no signal in it. The reconstruction
    gate is what catches that; the margin gate alone would eventually pick a lucky lag.
    """
    import numpy as np

    _hw, log = _hardware(rig, 2)
    rng = np.random.default_rng(3)
    aligned = _aligned(rig, log)
    aligned["tau_meas"] = rng.normal(0.0, float(np.std(log["tau"])) + 1e-9, log["tau"].shape)
    out = _cmd_response(rig, plan, aligned)
    assert out["identified"] is False, out
    assert out["not_identified_because"]


# =========================================================================================================
# WHAT IS STILL TRUE AND STILL COSTS US. Pinned as facts, not wishes -- each is a real limit of the closure.
# =========================================================================================================

def test_the_estimate_survives_a_realistically_dirty_log(rig, plan):
    """Sim2sim recovers the delay exactly because the reconstruction is algebraically the expression the rig
    used, so the clean number proves nothing about a bench. This is the test that does.

    Encoder noise at 1 mrad, velocity noise at 0.02 rad/s and current noise at 2% of the peak commanded torque,
    all at once, on top of a gravity feed-forward our reconstruction does not model. The delay still comes back
    exact -- the feed-forward because both sides are mean-removed, the noise because a one-tick shift moves the
    residual by far more than the noise does.
    """
    import numpy as np

    rng = np.random.default_rng(5)
    ticks = int(DEFAULT_DELAY_MS / MS_PER_TICK)
    _hw, log = _hardware(rig, ticks)
    peak = float(np.abs(log["tau"]).max()) or 1.0
    aligned = _aligned(rig, log)
    aligned["q_meas"] = log["q"] + rng.normal(0.0, 1e-3, log["q"].shape)
    aligned["qd_meas"] = log["qd"] + rng.normal(0.0, 0.02, log["qd"].shape)
    aligned["tau_meas"] = (log["tau"] + rng.normal(0.0, 0.02 * peak, log["tau"].shape)
                           + np.tile(log["tau"][0], (log["tau"].shape[0], 1)))
    out = _cmd_response(rig, plan, aligned)
    assert out["identified"] is True, out.get("not_identified_because")
    assert out["delay_ms"] == pytest.approx(DEFAULT_DELAY_MS)


def test_a_heavily_filtered_torque_channel_biases_the_estimate_one_tick_high(rig, plan):
    """A REAL, KNOWN, UNDETECTABLE-FROM-INSIDE bias, recorded so nobody rediscovers it as a surprise.

    The number identified is the lag between the control law and the torque the log says was applied, so any
    latency in the torque channel itself is inside it. MEASURED: a first-order low-pass at 100 Hz leaves the
    estimate exact; one at 40 Hz moves it a full control tick. This is ``TORQUE_CHANNEL_CAVEAT``, and it is the
    honest limit of the closure -- the estimator cannot tell a slow actuator from a slow current sensor.
    """
    import numpy as np

    from virturoid.services.sysid.gap_report import TORQUE_CHANNEL_CAVEAT

    def _lp(x, dt, fc):
        a = dt / (dt + 1.0 / (2.0 * np.pi * fc))
        out, acc = np.zeros_like(x), x[0].copy()
        for k in range(x.shape[0]):
            acc = acc + a * (x[k] - acc)
            out[k] = acc
        return out

    ticks = int(DEFAULT_DELAY_MS / MS_PER_TICK)
    _hw, log = _hardware(rig, ticks)
    for fc, expected_ms in ((100.0, DEFAULT_DELAY_MS), (40.0, DEFAULT_DELAY_MS + MS_PER_TICK)):
        aligned = _aligned(rig, log)
        aligned["tau_meas"] = _lp(log["tau"], rig["dt"], fc)
        out = _cmd_response(rig, plan, aligned)
        assert out["identified"] is True, (fc, out.get("not_identified_because"))
        assert out["delay_ms"] == pytest.approx(expected_ms), (
            f"a {fc:g} Hz low-pass on the torque channel was measured to give {expected_ms} ms for a "
            f"{DEFAULT_DELAY_MS} ms injection; it now gives {out['delay_ms']} ms. If the 40 Hz row stopped "
            f"biasing, the caveat in TORQUE_CHANNEL_CAVEAT is now false and must be rewritten")
    assert "40 Hz biases it a full control tick" in TORQUE_CHANNEL_CAVEAT


def test_at_twice_the_realistic_delay_the_metric_regains_range_but_the_verdict_is_marginal(fit_at):
    """40 ms is double the figure Hwangbo et al. name and double this package's own default, and the closure
    is only partial there. Recorded as a limit, not as a win.

    MEASURED, same fit, two scoring points: at the OLD one the score is pinned at 1.000x -- the metric's own
    ceiling, so the refusal carried no information about the fit at all. At the new one it is 1.484x on the
    full 12-joint plan and 1.502x on this narrow one, straddling the 1.5x threshold. So the metric got most of
    its dynamic range back (0.0% -> ~48%) and the VERDICT at 40 ms is decided by which plan you ran, which is
    not a property anyone should rely on. The cause is physical: the Go2's hips ring hard enough at 40 ms that
    trajectory RMS stops being a sensitive function of the parameters. The delay itself is still recovered
    exactly, and the fit still recovers all three parameters to within 13.3% on 32/36 pairs.
    """
    traj = fit_at(4)["trajectory"]
    assert traj["scored_at_delay_ms"] == pytest.approx(40.0)
    assert traj["improvement_x_at_zero_delay"] < 1.05, (
        "the old scoring point should still be pinned at ~1.0 here -- that is the ceiling effect")
    assert traj["improvement_x"] > 1.3, (
        f"only {traj['improvement_x']}x at 40 ms; measured 1.484x (full plan) and 1.502x (this plan)")


# =========================================================================================================
# THE POSITION-ONLY LOG. The coverage limit section 10 recorded as open -- "the estimate requires tau_meas,
# and a log with position only cannot do this" -- and it is closed. Most of a delay-wedge customer base does
# not have a joint torque sensor; a good part of it does not populate ROS 2's `effort` either.
# =========================================================================================================

@pytest.fixture(scope="module")
def gap_at(go2, plan):
    """``gap_at(delay_ticks, **synthetic_hardware_log kwargs)`` -- measure_gap, memoised per configuration."""
    from virturoid.services.sysid import measure_gap
    from virturoid.services.sysid.synthetic_hardware import DEFAULT_PERTURBATION, synthetic_hardware_log

    cache: dict = {}

    def _get(delay_ticks: int, **kw):
        key = (int(delay_ticks), tuple(sorted((k, str(v)) for k, v in kw.items())))
        if key not in cache:
            kw.setdefault("perturbation", DEFAULT_PERTURBATION)
            _, log = synthetic_hardware_log(go2, delay_ticks=int(delay_ticks), plan=plan, **kw)
            cache[key] = (log, measure_gap(go2, log, plan=plan, measure_noise_floor=False))
        return cache[key]

    return _get


@pytest.mark.parametrize("delay_ticks", [0, 2, 4])
def test_the_delay_is_recovered_from_a_log_with_no_torque_channel_at_all(gap_at, delay_ticks):
    """THE LIFT. 0 / 20 / 40 ms, exactly, from ``q_cmd`` and ``q_meas`` -- no torque, no motor current.

    The applied torque is not in the log but its EFFECT is, so ``_delay_from_motion`` reads it back out of the
    motion by inverse dynamics and aligns that against the declared control law. The three parameters Stage 2
    fits are left free at every candidate lag, so the estimate does not lean on our priors for them.

    Why this was not obviously possible: every naive attempt reads ONE CONTROL TICK HIGH, on the oracle model
    as well as the prior, which looks exactly like a property of the experiment. It is not -- see the test
    below. Tolerance is EXACT, for the same reason it is exact on the torque channel: the delay lives on the
    control-tick grid, so a neighbouring tick is a different answer rather than a noisy one.
    """
    _log, gap = gap_at(delay_ticks, channel="position_only")
    lat = gap["latency"]
    assert lat["identified"] is True, lat.get("not_identified_because")
    assert lat["source"] == "motion_reconstruction"
    assert lat["delay_ms"] == pytest.approx(delay_ticks * MS_PER_TICK), (
        f"injected {delay_ticks * MS_PER_TICK} ms, recovered {lat['delay_ms']} ms")
    assert lat["margin_over_next_best_tick"] >= 0.15


def test_the_position_only_estimate_is_a_SAMPLING_problem_not_an_identifiability_one(rig, plan):
    """WHY it took a fix rather than an implementation, measured on the ORACLE model so it cannot be blamed on
    the prior.

    The applied torque is a zero-order hold across ``ctrl_every`` physics steps. A central difference of the
    logged velocity taken AT a tick boundary averages the acceleration under the OLD torque with the
    acceleration under the NEW one, so the reconstruction it feeds looks one tick late -- and the estimator
    faithfully reports a delay one tick too large. Sampled INSIDE the interval, the same objective is exact.

    This is the whole difference between "a position-only log cannot see the delay" and "we were reading it at
    the wrong instant", and it is the reason the boundary variant below is measured here rather than argued.
    """
    import numpy as np

    from virturoid.services.sysid.bench_rig import inverse_torque, torque_ceiling
    from virturoid.services.sysid.gap_report import _delay_from_motion, _windows

    ticks = int(DEFAULT_DELAY_MS / MS_PER_TICK)
    hw, log = _hardware(rig, ticks)
    ce, dt = rig["ctrl_every"], rig["dt"]

    def _boundary_variant(model):
        """The same estimator with the acceleration taken across the tick BOUNDARY. Nothing else differs."""
        q, qd = log["q"], log["qd"]
        qacc = np.zeros_like(qd)
        qacc[1:-1] = (qd[2:] - qd[:-2]) / (2.0 * dt)
        idt = inverse_torque(model, q, qd, qacc)
        ceil = torque_ceiling(model)
        want = np.clip(rig["kp"] * (rig["q_cmd"] - q) - rig["kd"] * qd, -ceil, ceil)
        wins = _windows(plan, q.shape[0], rig["dofs"], 1.0 / dt)
        out = {}
        for name in ONLY_JOINTS:
            adr = rig["dofs"][name]
            a, b = wins[name]
            rows = np.arange(a + ((-a) % ce), b, ce)
            grid = []
            for d in range(0, 9):
                n = rows.size - d
                src, dst = rows[:n], rows[d:d + n]
                y = want[src, adr] - idt[dst, adr]
                X = np.column_stack([qacc[dst, adr], qd[dst, adr], np.sign(qd[dst, adr]), np.ones(n)])
                th, *_ = np.linalg.lstsq(X, y, rcond=None)
                grid.append((d, float(np.sqrt(np.mean((y - X @ th) ** 2)))))
            out[name] = min(grid, key=lambda g: g[1])[0] * ce * dt * 1000.0
        return out

    on_prior = _boundary_variant(rig["model"])
    on_oracle = _boundary_variant(hw)
    for name in ONLY_JOINTS:
        assert on_prior[name] == pytest.approx(DEFAULT_DELAY_MS + MS_PER_TICK), (
            f"the boundary-sampled variant is meant to read exactly one tick HIGH on {name}; got "
            f"{on_prior[name]} ms for a {DEFAULT_DELAY_MS} ms injection. If this stopped biasing, the "
            f"mechanism recorded in gap_report.MOTION_MODEL_CAVEAT is no longer the one that was fixed")
        assert on_oracle[name] == pytest.approx(DEFAULT_DELAY_MS + MS_PER_TICK), (
            f"...and it must bias on the ORACLE model too ({name}: {on_oracle[name]} ms) -- that is what "
            f"shows the bias is the sampling and not the prior")

    shipped = _delay_from_motion(rig["model"], _aligned(rig, log), rig["dofs"], plan, kp=rig["kp"],
                                 kd=rig["kd"], ctrl_every=ce, max_ticks=8, dt=dt)
    assert shipped["identified"] is True, shipped.get("not_identified_because")
    assert shipped["delay_ms"] == pytest.approx(DEFAULT_DELAY_MS), (
        "sampled inside the hold interval the SAME objective must be exact")


def test_a_log_sampled_below_the_control_rate_is_refused_rather_than_guessed(go2, plan, gap_at):
    """The trap that makes this gate load-bearing, and it is not the obvious one.

    One control tick is 10 ms; a 50 Hz log has 20 ms between samples, so the delay is simply not in the data.
    The danger is not that the estimate degrades -- it is that ``_align_log`` INTERPOLATES the log back up to
    the physics rate before the estimator sees it, which smooths the discontinuity the estimator reads and
    hands back a confident margin on a wrong answer. MEASURED at 20 ms injected before the gate existed: 30 ms,
    ``identified: true``. The refusal reads the NATIVE timestamps, which interpolation cannot fake.
    """
    from virturoid.services.sysid import measure_gap

    log, _ = gap_at(2, channel="position_only")
    keep = 10                                          # 500 Hz -> 50 Hz, half the 100 Hz control rate
    idx = list(range(0, len(log["t"]), keep))
    slow = {**log, "t": [log["t"][i] for i in idx],
            "q_cmd": [log["q_cmd"][i] for i in idx], "q_meas": [log["q_meas"][i] for i in idx],
            "qd_meas": [log["qd_meas"][i] for i in idx]}
    lat = measure_gap(go2, slow, plan=plan, measure_noise_floor=False)["latency"]
    assert lat["identified"] is False, (
        f"a 50 Hz log against a 100 Hz loop claimed {lat['delay_ms']} ms; one control tick is not in this data")
    assert "sampled at" in (lat.get("not_identified_because") or ""), lat


def test_an_error_the_free_parameters_cannot_express_costs_the_position_only_margin(gap_at):
    """The control on the new estimator, and the price of having a plant in it at all.

    ``_delay_from_command_response`` has no plant, so a +30% link mass and inertia error cannot move it. This
    one reads the torque out of the motion, so it can -- and the question is whether it DEGRADES GRACEFULLY.
    MEASURED: still exact at 0 ms (margin 0.887) and 20 ms (0.287), and at 40 ms the reconstruction collapses
    and it REFUSES. Degrading into a refusal is the required behaviour; degrading into a confident wrong tick
    is what the trajectory sweep does and why it may never claim.
    """
    _log, gap = gap_at(4, channel="position_only", perturbation={}, link_scale=1.30)
    lat = gap["latency"]
    assert lat["identified"] is False, (
        f"a +30% link mass/inertia error at 40 ms was CLAIMED at {lat['delay_ms']} ms on a position-only log; "
        f"the free frictionloss/damping/armature columns cannot absorb that error and must not appear to")
    _log0, gap0 = gap_at(0, channel="position_only", perturbation={}, link_scale=1.30)
    assert gap0["latency"]["delay_ms"] == 0.0 and gap0["latency"]["identified"] is True, (
        "...and it should still be right where the misspecification is not yet fatal: 0 ms, identified")


def test_a_position_only_log_still_cannot_fit_a_parameter_and_says_what_it_can(go2, plan, gap_at):
    """The limit that did NOT move, stated so the lift is not read as more than it is.

    The delay is now reachable without a torque channel. The PARAMETERS are not, and no estimator can change
    that: the quantity being regressed is a torque residual and there is no torque. What must not happen is a
    refusal that reads as "this log is useless" when it just yielded the residual Hwangbo et al. call dominant.
    """
    from virturoid.services.sysid import fit_parameters

    log, gap = gap_at(2, channel="position_only")
    fit = fit_parameters(go2, log, plan=plan, n_boot=16)
    assert fit["ok"] is False
    assert "tau_meas" in fit["error"]
    assert "actuation delay" in fit["what_is_still_available"]
    assert gap["attribution"]["available"] is False
    assert gap["latency"]["identified"] is True, "the delay is the thing this log CAN still give"
