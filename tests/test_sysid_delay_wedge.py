"""The calibration wedge under ACTUATION DELAY -- the residual its own opening paragraph calls dominant.

`sysid/__init__.py` cites Hwangbo et al. (Science Robotics, ANYmal) for the claim that the dominant sim-to-real
residual is actuator dynamics AND control-signal delay. `synthetic_hardware.DEFAULT_DELAY_TICKS` injects 2
control ticks (20 ms) of it by default, and `sysid.tools.simulate_bench_log` defaults to `delay_ms=20.0`. So
the default path through the shipped surface runs the wedge against the term it names as the point of the
exercise.

MEASURED on the Menagerie Unitree Go2 (`docs/calibration_wedge_under_delay.md`): at that default the
tracking-improvement gate withholds the fit, and the delay search returns 10 ms for a 20 ms injection and
10 ms for a 40 ms one. On the composed dog every published number was taken on, the same experiment passes.
The wedge is not broken -- it refuses, correctly, and it refuses ALWAYS once the delay is real, because the
number it rules on has no dynamic range left.

This file is the contract for that. It is written in three parts:

  * FACTS -- the current behaviour, locked so it cannot drift unnoticed. These pass.
  * THE CONTROL -- proof that the delay IS identifiable from this excitation, so a failure to recover it is
    ours and not the experiment's. This passes, and it is what makes the xfails legitimate rather than wishes.
  * THE SPEC -- what we want and do not have, as `xfail(strict=True)`. Strict on purpose: the day someone
    makes one of these true, the suite fails until they delete the marker, which is the only way a
    known-wrong behaviour gets un-forgotten.

Cost note: the Go2 at the package's 120 s default budget costs ~2 minutes per fit. The whole phenomenon
survives narrowing to two joints and a 12 s excitation (MEASURED: fit 11 s, oracle delay sweep 2 s, ceiling
1.029x at 20 ms against 1.016x on the full plan), so that is what this file runs.

WE OWN NO HARDWARE. Everything here is sim2sim -- both the "robot" and the "sim" are MuJoCo, so MuJoCo's own
modelling error cancels exactly and is invisible. See ``WHAT_SIM2SIM_DOES_NOT_PROVE``. The point of this file
is that even INSIDE that gate, where the wedge is at its most favourable, it stops working at the delay it
names as dominant.
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
    # A REAL robot, deliberately. The defect this file records is invisible on the composed dog every other
    # number in this package was measured on, and that is the finding -- so a fixture would erase it.
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


def _hardware(rig, delay_ticks):
    """The 'robot': our model + the default perturbation + ``delay_ticks`` of transport delay."""
    from virturoid.services.sysid.bench_rig import pd_replay
    from virturoid.services.sysid.synthetic_hardware import DEFAULT_PERTURBATION, perturbed_model

    hw = perturbed_model(rig["model"], DEFAULT_PERTURBATION)
    _, q, _, _ = pd_replay(hw, rig["q_cmd"], kp=rig["kp"], kd=rig["kd"], q_start=rig["q0"],
                           ctrl_every=rig["ctrl_every"], delay_ticks=int(delay_ticks))
    return hw, q


def _gate_ceiling(rig, delay_ticks):
    """The largest ``improvement_x`` the shipped metric can EVER return at this delay.

    ``fit._trajectory_improvement`` divides RMS(prior replay vs log) by RMS(fitted replay vs log) with both
    replays at delay 0. Substitute the TRUE hardware model for the fitted one and the ratio is maximal by
    construction: no fit can do better than the parameters that generated the log. Three PD replays, no fit.
    """
    import numpy as np

    from virturoid.services.sysid.bench_rig import pd_replay

    hw, q_log = _hardware(rig, delay_ticks)
    kw = dict(kp=rig["kp"], kd=rig["kd"], q_start=rig["q0"], ctrl_every=rig["ctrl_every"], delay_ticks=0)
    _, q_prior, _, _ = pd_replay(rig["model"], rig["q_cmd"], **kw)
    _, q_true, _, _ = pd_replay(hw, rig["q_cmd"], **kw)
    before = float(np.sqrt(np.mean((q_prior - q_log) ** 2)))
    after = float(np.sqrt(np.mean((q_true - q_log) ** 2)))
    return float("inf") if after <= 1e-12 else before / after


@pytest.fixture(scope="module")
def fit_at_default_delay(go2, plan):
    """One real ``fit_parameters`` against a log carrying the package's OWN default 20 ms of delay."""
    from virturoid.services.sysid import fit_parameters
    from virturoid.services.sysid.synthetic_hardware import DEFAULT_PERTURBATION, synthetic_hardware_log

    _, log = synthetic_hardware_log(go2, perturbation=DEFAULT_PERTURBATION,
                                    delay_ticks=int(DEFAULT_DELAY_MS / MS_PER_TICK), plan=plan)
    return fit_parameters(go2, log, plan=plan, n_boot=32)


# =========================================================================================================
# THE CONTROL. Everything below turns on this: is the delay recoverable from this experiment AT ALL?
# =========================================================================================================

@pytest.mark.parametrize("delay_ticks", [0, 2, 4])
def test_the_delay_is_recoverable_from_this_excitation_given_the_right_model(rig, delay_ticks):
    """Hand ``_delay_search`` a model that already carries the true perturbation and it must return the
    injected delay EXACTLY.

    This is the control that separates "our search is broken" from "this experiment cannot see delay" -- and
    they need opposite fixes, so guessing is not allowed. It passes at 0, 20 and 40 ms with
    ``fraction_of_mismatch_explained_by_delay == 1.0``: the excitation contains everything needed to pin the
    delay, the range is wide enough, and the grid lands on the truth. Every failure recorded in this file is
    therefore OURS.
    """
    from virturoid.services.sysid.gap_report import _delay_search

    hw, q_log = _hardware(rig, delay_ticks)
    out = _delay_search(hw, rig["q_cmd"], q_log, kp=rig["kp"], kd=rig["kd"], q_start=q_log[0].copy(),
                        ctrl_every=rig["ctrl_every"], max_ticks=8, dt=rig["dt"])
    assert out["delay_ms"] == pytest.approx(delay_ticks * MS_PER_TICK), (
        f"the ORACLE model missed an injected {delay_ticks * MS_PER_TICK} ms: got {out['delay_ms']} ms. "
        f"If this ever fails, the delay is NOT identifiable from this excitation and the xfails below are "
        f"describing a different problem than the one they claim")
    assert out["at_grid_edge"] is False


def test_the_delay_objective_is_a_hole_not_a_basin(rig):
    """WHY the search is fragile, measured rather than asserted.

    On the true model the objective drops to EXACTLY zero at the injected tick and its immediate neighbours
    are ~0.024 and ~0.070 rad. The global minimum is one grid point wide and exists only because an exact
    model plus an exact delay reproduces the log exactly. There is no basin to fall into, so any residual
    dynamics error fills the hole and leaves a flat plateau whose argmin is decided in the fourth significant
    figure. That is the mechanism behind every xfail below.
    """
    from virturoid.services.sysid.gap_report import _delay_search

    ticks = int(DEFAULT_DELAY_MS / MS_PER_TICK)
    hw, q_log = _hardware(rig, ticks)
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


# =========================================================================================================
# FACTS. The behaviour as it stands, pinned so it cannot move without someone noticing.
# =========================================================================================================

def test_the_gates_own_ceiling_falls_under_its_threshold_once_the_delay_is_real(rig):
    """``application_gate`` rules on a RATIO whose numerator and denominator both carry the timing error, and
    no fitted parameter can touch it. So the ratio has a CEILING that depends only on the delay.

    Measured on this robot: reachable at 10 ms, unreachable at 20 ms and above -- and 20 ms is the package's
    own default and `simulate_bench_log`'s. A gate that cannot pass is not evidence about the fit.
    """
    from virturoid.services.sysid.fit import MIN_TRACKING_IMPROVEMENT_X

    at_10 = _gate_ceiling(rig, 1)
    at_20 = _gate_ceiling(rig, 2)
    at_40 = _gate_ceiling(rig, 4)

    assert at_10 >= MIN_TRACKING_IMPROVEMENT_X, (
        f"10 ms should still be inside the gate's reach; got {at_10:.3f}x")
    assert at_20 < MIN_TRACKING_IMPROVEMENT_X, (
        f"this test is no longer adversarial: the ceiling at the package's DEFAULT 20 ms is {at_20:.3f}x, "
        f"which now clears the {MIN_TRACKING_IMPROVEMENT_X}x threshold")
    assert at_40 < 1.05, f"the ceiling at 40 ms should be ~1.0 (no headroom at all); got {at_40:.3f}x"
    assert at_20 > at_40, "the ceiling should fall as the delay grows"


def test_a_fit_at_the_packages_own_default_delay_is_withheld(fit_at_default_delay):
    """The engineer's headline cell, locked. At 20 ms the gate refuses and the fit is provisional.

    This is recorded as a FACT rather than a wish because withholding is the right ACTION -- nothing false
    reaches the model. What is wrong is that the refusal carries no information (see the ceiling test above),
    not that it happened.
    """
    from virturoid.services.sysid.fit import MIN_TRACKING_IMPROVEMENT_X

    app = fit_at_default_delay["application"]
    assert app["passed"] is False, app["verdict"]
    assert app["provisional"] is True
    assert app["improvement_x"] < MIN_TRACKING_IMPROVEMENT_X


def test_the_delay_is_reported_but_never_claimed_as_identified(fit_at_default_delay):
    """The one thing that keeps this honest: the wrong number is not asserted.

    The search returns 10 ms for a 20 ms injection, and marks it ``identified: False`` with a numeric reason.
    An engineer reading the output is told the latency could not be separated -- so the failure is a silent
    gap in COVERAGE, not a lie. If this ever flips to True while the value is still wrong, that is a much
    worse defect than the one this file records.
    """
    lat = fit_at_default_delay["latency"]
    assert lat["identified"] is False, (
        f"the delay search claimed {lat['delay_ms']} ms as identified against an injected "
        f"{DEFAULT_DELAY_MS} ms")
    assert lat["not_identified_because"], "a refusal must say why"
    assert lat["fraction_of_mismatch_explained_by_delay"] < 0.5


def test_nothing_reaches_the_model_when_the_gate_refuses(go2, fit_at_default_delay):
    """The refusal has to be load-bearing, not decorative: a withheld fit must change nothing."""
    from virturoid.services.sysid import apply_calibration, calibration_of

    rec = calibration_of(apply_calibration(go2, fit_at_default_delay)) or {}
    assert not (rec.get("joints") or {}), (
        "a provisional fit wrote parameters into the model; only allow_provisional=True may do that")


# =========================================================================================================
# THE SPEC. What we want, as strict xfails. Deleting a marker is the only way to close one.
# =========================================================================================================

@pytest.mark.xfail(strict=True, reason=(
    "MEASURED: returns 10.0 ms for a 20.0 ms injection on the Go2 (and 10.0 ms for 40.0 ms). The delay IS "
    "identifiable -- the oracle control above recovers it exactly -- but the search is only well-conditioned "
    "on an exactly-correct model, and the parameter correction it relies on is 4.7x smaller than the delay "
    "it is meant to reveal. See docs/calibration_wedge_under_delay.md section 3"))
def test_SPEC_the_delay_search_recovers_the_delay_on_a_real_robot(fit_at_default_delay):
    """WANT: the sweep on the fitted model returns the injected delay, on an imported customer robot.

    `gap_report.py` claims '0/10/20/40/60 ms -> 0/10/20/40/60 ms'. True on the composed dog. Not true here.
    """
    assert fit_at_default_delay["latency"]["delay_ms"] == pytest.approx(DEFAULT_DELAY_MS)


@pytest.mark.xfail(strict=True, reason=(
    "MEASURED: improvement_x is 1.016 against a threshold of 1.5 -- and the metric's own CEILING at this "
    "delay is 1.016, so no fit can pass. At 40 ms the fit recovers all three parameters to within 13.3% and "
    "still scores 1.000. The gate reads a quantity that has stopped varying with the fit. See "
    "docs/calibration_wedge_under_delay.md section 4"))
def test_SPEC_a_correctly_specified_fit_survives_the_gate_when_the_robot_has_delay(fit_at_default_delay):
    """WANT: a perturbation made ENTIRELY of the three fitted regressor columns is applied, even though the
    robot also has actuation delay -- because real robots do, and that is the stated premise of the package.

    Closing this needs the gate to stop being delay-blind: either score both replays at the identified delay
    (MEASURED 1.683x at 20 ms, 1.484x at 40 ms -- the first passes), or report the ceiling alongside the ratio
    so a refusal that could not have gone the other way says so.
    """
    assert fit_at_default_delay["application"]["passed"] is True, (
        fit_at_default_delay["application"]["verdict"])


@pytest.mark.xfail(strict=True, reason=(
    "MEASURED: _delay_search reports explained = 1 - best_rms/rms_at_zero_delay. When the true delay is zero "
    "the argmin IS the zero-delay entry, so explained is identically 0 and 'identified' can never be True -- "
    "on any model that does not reproduce the log to 1e-12, i.e. on every log that is not a perfect sim2sim. "
    "Observed on the Go2 at 0 ms injected: the search returned the right answer (0.0 ms) and reported "
    "'sweeping delay explains only 0.0% of the trajectory mismatch (need 50%)'. See "
    "docs/calibration_wedge_under_delay.md section 6"))
def test_SPEC_finding_no_delay_is_a_finding_not_a_refusal(rig):
    """WANT: landing on 0 ms, correctly, is a reportable finding rather than a refusal.

    Run against our PRIOR model -- the realistic case, where a parameter gap exists and the zero-delay replay
    is close but not exact. That is the branch the escape hatch below does not cover.
    """
    from virturoid.services.sysid.gap_report import _delay_search

    _hw, q_log = _hardware(rig, 0)
    out = _delay_search(rig["model"], rig["q_cmd"], q_log, kp=rig["kp"], kd=rig["kd"],
                        q_start=q_log[0].copy(), ctrl_every=rig["ctrl_every"], max_ticks=4, dt=rig["dt"])
    assert out["delay_ms"] == 0.0, out
    assert out["identified"] is True, out["not_identified_because"]


def test_the_zero_delay_verdict_is_structurally_unreachable_except_on_a_perfect_model(rig):
    """The mechanism behind the xfail above, pinned -- and the escape hatch that hides it.

    ``explained = 1 - best/at_zero if at_zero > 1e-12 else 1.0``. Two regimes:

      * ORACLE model, zero delay: the replay reproduces the log exactly, ``at_zero`` is 0, the guard fires and
        explained is 1.0. This is the ONLY way a zero delay is ever reported as identified, and it can only
        happen when the 'hardware' is our own model -- never on a customer's log.
      * PRIOR model, zero delay: ``at_zero`` is the parameter gap, the argmin is still entry 0, so
        ``explained`` is exactly 0.0 and the correct answer is reported as a refusal.

    So the package's own sim2sim gate is the one configuration in which this defect is invisible.
    """
    from virturoid.services.sysid.gap_report import _delay_search

    hw, q_log = _hardware(rig, 0)
    kw = dict(kp=rig["kp"], kd=rig["kd"], q_start=q_log[0].copy(), ctrl_every=rig["ctrl_every"],
              max_ticks=4, dt=rig["dt"])

    oracle = _delay_search(hw, rig["q_cmd"], q_log, **kw)
    assert oracle["delay_ms"] == 0.0
    assert oracle["fraction_of_mismatch_explained_by_delay"] == 1.0
    assert oracle["identified"] is True, "the exact-model escape hatch stopped firing"

    prior = _delay_search(rig["model"], rig["q_cmd"], q_log, **kw)
    assert prior["delay_ms"] == 0.0, "the prior-model sweep should still land on the right answer"
    assert prior["fraction_of_mismatch_explained_by_delay"] == 0.0, (
        "explained must be exactly 0 when the argmin is the zero-delay entry")
    assert prior["identified"] is False, "this test is no longer adversarial -- the defect is fixed"
