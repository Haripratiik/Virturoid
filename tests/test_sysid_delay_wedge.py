"""The calibration wedge under ACTUATION DELAY -- the residual its own opening paragraph calls dominant.

`sysid/__init__.py` cites Hwangbo et al. (Science Robotics, ANYmal) for the claim that the dominant sim-to-real
residual is actuator dynamics AND control-signal delay. `synthetic_hardware.DEFAULT_DELAY_TICKS` injects 2
control ticks (20 ms) of it by default, and `sysid.tools.simulate_bench_log` defaults to `delay_ms=20.0`. So
the default path through the shipped surface runs the wedge against the term it names as the point of the
exercise.

That path used to fail. MEASURED on the Menagerie Unitree Go2 and recorded in
`docs/calibration_wedge_under_delay.md`: at the package's own default the delay search returned 10 ms for a
20 ms injection and 10 ms for a 40 ms one, and the tracking gate withheld the fit at 1.016x against a 1.5x
threshold -- with the score of the TRUE parameters at that delay also 1.016x, i.e. the metric had no range left
to reward a fit with. The refusal was correct and carried no information. (That reference was recorded as a
"ceiling" and is not one; the third retraction below is about exactly that word.)

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

THE DRIVETRAIN UNDER ALL OF IT MOVED, and six of these tests said so out loud instead of going quiet -- which
is the only reason this paragraph exists. ``robot_import`` now carries the customer's DECLARED joint damping /
armature / frictionloss into the twin instead of substituting ``gene_compiler._joint_dynamics_prior``'s
structural guess. On this Go2 that is damping 0.8 -> 2.0 and frictionloss 0.12 -> 0.2 (armature was already
0.01): every constant below used to be measured on a robot carrying 40% of Unitree's declared damping. A body
with 2.5x the damping does not ring, and THE RING WAS THE PHENOMENON. Prior-replay-vs-log RMS across 0-40 ms
was 0.0061 -> 0.1093 rad and is now 0.0039 -> 0.0652 (full 12-joint plan: 0.0031 -> 0.0659). Two claims are
RETRACTED as a result, each at its own test:

  * "at the package's default 20 ms the delay-blind metric's CEILING is 1.016x, under the 1.5x threshold, so
    no fit of any quality could pass" -- on the declared drivetrain that reference is 1.745x here and 1.791x on
    the full plan. The claim first becomes true at 30 ms (1.009x), not at 20.
  * "the same fit scored at the OLD point is still under the threshold" -- it is 1.503x here and 1.858x on the
    full plan, i.e. it clears. What the scoring change moves at 20 ms is no longer the VERDICT; it is the
    quantity, by 10x here and 15.7x on the full plan.

A THIRD claim is RETRACTED, and this one is about the yardstick rather than the operating point. The number
above was called a CEILING -- "substitute the true hardware model for the fitted one and the ratio is maximal
by construction, no fit can do better than the parameters that generated the log". **That is false for a
delay-blind metric.** The argument needs the replay family to contain the data-generating process, and at
delay 0 against a delayed log it does not, so the truth is not the residual's minimiser: a twin whose reflected
inertia is too high MIMICS transport lag and scores better. MEASURED over a 60-point sweep of the estimator's
own three parameters at 20 ms, the best score is 2.166x (armature +0.08) against the truth's 1.745x -- 24%
above the supposed ceiling, and 2.277x vs 1.791x (+27%) on the full 12-joint plan. The full-plan fit exceeded
the reference too, 1.858x vs 1.791x, and that +3.8% had been
written down here as a windowing artefact; instrumenting both computations on one Go2 log REFUTED that
(identical start state, commands, log and row range 0..36000 -- ``_windows`` never touches
``_trajectory_improvement``). The helper is now ``_true_parameter_score`` / ``_best_delay_blind_score``, the
one-sided bound is now a BRACKET, and the conclusion that mattered survives on the envelope: at 30 ms nothing
in the family reaches 1.5x (best 1.036x here, 1.012x full plan), at 40 ms nothing reaches 1.01x (1.003x both).

What did NOT move: the delay still comes back EXACTLY at 0 / 20 / 40 ms through every path (torque channel,
position-only, dirty log, filtered channel), the closed-loop sweep is still wrong at every delay, and a
misspecification the estimator cannot express is still refused. The estimator is untouched; its operating
point is not. Numbers quoted from ``docs/calibration_wedge_under_delay.md`` below predate this and are marked
where they are now stale.

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

#: The drivetrain every constant in this file USED to be measured at, written as the delta off the one the
#: Go2's own file declares: damping 2.0 - 1.2 = 0.8, frictionloss 0.2 - 0.08 = 0.12, armature untouched at
#: 0.01. That is exactly what ``gene_compiler._joint_dynamics_prior`` substituted before it learned to read
#: the source model, so it is a real operating point this product used to ship rather than a strawman -- and
#: it is roughly where a Panda-class arm sits (Menagerie's Panda declares damping 1.0 and no dry friction at
#: all). One test below needs a body that RINGS to demonstrate a trap the Go2 no longer exhibits, and this is
#: the honest way to get one: a named, previously-shipped drivetrain, not an invented robot.
SUBSTITUTED_DRIVETRAIN = {"damping": -1.2, "frictionloss": -0.08}


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


#: The parameter family the estimator can actually express, as absolute offsets off the compiled twin, spanning
#: "no error at all" to well past the injected truth (frictionloss +0.08 / damping +0.6 / armature +0.03).
#: Sweeping it is what replaced an ARGUMENT with a MEASUREMENT: see ``_best_delay_blind_score``.
_FAMILY_GRID = {"frictionloss": (0.0, 0.08, 0.3), "damping": (0.0, 0.6, 2.0, 6.0),
                "armature": (0.0, 0.03, 0.08, 0.2, 0.5)}

#: The two replays every delay-blind score shares, so a 60-point sweep costs 60 replays rather than 180.
#:
#: CACHED ON THE RIG, NOT IN A MODULE-LEVEL DICT, and that is the whole point. This was
#: ``_DELAY_BLIND_BASE[delay_ticks]`` -- keyed on the delay ALONE, while everything it stores (the hardware log
#: and the prior's RMS against it) is a function of the RIG: its model, its drivetrain, its command trajectory,
#: its gains. Correct while exactly one rig existed; a second rig asking for a delay the first had already
#: cached would have been handed the FIRST rig's log and silently scored against it -- no exception, no
#: mismatch, just plausible wrong numbers. Section 12 of the doc contemplates measuring an envelope on the
#: SUBSTITUTED drivetrain, which is precisely that second rig.
#:
#: Keying the dict harder (on ``id(rig["model"])``, say) would work until someone forgot a field. Hanging the
#: cache off the rig removes the key, so there is nothing left to get wrong -- the same chokepoint move as
#: putting the memory-bank destination policy in ``MemoryDB.__init__`` instead of at each call site.


def _delay_blind_score(rig, delay_ticks, offsets):
    """``improvement_x`` as the OLD, delay-blind gate computed it, for a twin carrying ``offsets``.

    RMS(prior replay vs log) / RMS(offset replay vs log), with BOTH replays at delay 0 against a log that
    carries ``delay_ticks`` of it. Three PD replays, no fit.

    THE NAME MATTERS AND THE PREVIOUS ONE WAS WRONG. This helper used to be ``_gate_ceiling``, and it was
    called with the true perturbation on the reasoning that "no fit can do better than the parameters that
    generated the log, so the ratio is maximal by construction". That reasoning does not hold for a
    delay-blind metric and the claim is RETRACTED -- measured, see ``_best_delay_blind_score``. The
    data-generating parameters minimise the residual only when the replay family CONTAINS the data-generating
    process, and at delay 0 against a delayed log it does not: the timing error is unmodelled, so a twin whose
    reflected inertia is too high partially MIMICS transport lag and scores better than the truth.
    """
    import numpy as np

    from virturoid.services.sysid.bench_rig import pd_replay
    from virturoid.services.sysid.synthetic_hardware import perturbed_model

    d = int(delay_ticks)
    kw = dict(kp=rig["kp"], kd=rig["kd"], q_start=rig["q0"], ctrl_every=rig["ctrl_every"], delay_ticks=0)
    base = rig.setdefault("_delay_blind_base", {})
    if d not in base:
        _, log = _hardware(rig, d)
        _, q_prior, _, _ = pd_replay(rig["model"], rig["q_cmd"], **kw)
        before = float(np.sqrt(np.mean((q_prior - log["q"]) ** 2)))
        base[d] = (log["q"], before)
    q_log, before = base[d]
    _, q_after, _, _ = pd_replay(perturbed_model(rig["model"], dict(offsets)), rig["q_cmd"], **kw)
    after = float(np.sqrt(np.mean((q_after - q_log) ** 2)))
    return float("inf") if after <= 1e-12 else before / after


def _true_parameter_score(rig, delay_ticks):
    """The delay-blind score of the parameters that GENERATED the log. A REFERENCE POINT, not a bound."""
    from virturoid.services.sysid.synthetic_hardware import DEFAULT_PERTURBATION

    return _delay_blind_score(rig, delay_ticks, DEFAULT_PERTURBATION)


def _best_delay_blind_score(rig, delay_ticks):
    """``(score, offsets)`` at the MAX over ``_FAMILY_GRID`` -- the measured envelope of the delay-blind metric.

    This is the honest replacement for the retracted "ceiling". It is still not a proof of a bound (it is a
    grid maximum over one family, and every joint moves together), but it is 60 measured models instead of an
    argument about one, and it is the family the estimator is drawn from -- which is what the downstream claim
    ("at this delay no fit could clear the gate") actually needs.
    """
    best = (0.0, {})
    for f in _FAMILY_GRID["frictionloss"]:
        for dm in _FAMILY_GRID["damping"]:
            for a in _FAMILY_GRID["armature"]:
                off = {"frictionloss": f, "damping": dm, "armature": a}
                s = _delay_blind_score(rig, delay_ticks, off)
                if s > best[0]:
                    best = (s, off)
    return best


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

    On the true model the objective drops to EXACTLY zero at the injected tick and every neighbour is orders of
    magnitude worse. The global minimum is one grid point wide and exists only because an exact model plus an
    exact delay reproduces the log exactly. There is no basin to fall into, so any residual dynamics error
    fills the hole and leaves a plateau whose argmin is decided in the noise.

    The shape has a cause worth naming: under-shooting a delay SATURATES. A replay with too little delay does
    not ring, so its error against a ringing log is just the log's ringing amplitude and stops growing; a
    replay with too much delay rings out of phase and scores WORSE than one that does not ring at all. Sub-tick
    interpolation cannot fix that, and neither can a correlation objective -- both were measured.

    RE-MEASURED on the drivetrain the customer's file declares (module docstring). The assertion that used to
    close this test was that the two WRONG under-shoot entries at 20 ms agree within 5%; on a body that barely
    rings they are 0.00277 and 0.00207 rad, 25% apart -- because the plateau's absolute height collapsed 5.7x
    while the residual trend across it did not. That 5% was a property of the RING, not of the saturation, and
    it was the wrong proxy for a claim about the search. The saturation is measured directly instead, and it is
    STRONGER than it was: one tick short costs 10.4x less than one tick long (it was 4.4x), and the plateau,
    read where it is fully developed, is flat to 0.16% across three grid points (it was 0.22%).
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
    # SATURATION, as the ASYMMETRY it actually is rather than as a tie between two points. One tick short is
    # far cheaper than one tick long, and two ticks short is still cheaper than one tick long. A basin would
    # be roughly symmetric about the minimum; this is a cliff on one side and a shelf on the other.
    assert grid[ticks + 1] / grid[ticks - 1] > 3.0, (
        f"under-shooting is meant to SATURATE and over-shooting to ring out of phase, so +1 tick must cost far "
        f"more than -1 tick: {grid[ticks + 1]} vs {grid[ticks - 1]}. If they have evened out, the objective "
        f"has a basin and the diagnosis in this docstring is wrong. grid {grid}")
    assert grid[0] < 0.5 * grid[ticks + 1], (
        f"under-shooting by TWO ticks must still be cheaper than over-shooting by one -- that is the shelf: "
        f"{grid[0]} vs {grid[ticks + 1]}, grid {grid}")

    # THE PLATEAU, read at 40 ms where three under-shoot entries exist and saturation is fully developed. Two
    # points can only ever show a tie; three show that the search has nothing to descend. This is the half of
    # "no basin" the 20 ms grid is too narrow to carry, and it holds on both drivetrains (0.16% now, 0.22%
    # under the substituted one), which is why it is the assertion that replaced the 5% tie.
    _hw4, log4 = _hardware(rig, 4)
    far = {r["delay_ticks"]: r["trajectory_rms_rad"] for r in _delay_search(
        _hw4, rig["q_cmd"], log4["q"], kp=rig["kp"], kd=rig["kd"], q_start=log4["q"][0].copy(),
        ctrl_every=rig["ctrl_every"], max_ticks=6, dt=rig["dt"])["grid"]}
    assert far[4] < 1e-9, f"the 40 ms hole is not exact either: {far[4]}"
    under = [far[0], far[1], far[2]]
    assert (max(under) - min(under)) / max(under) < 0.01, (
        f"the under-shoot side is meant to be a FLAT plateau once saturation is fully developed -- three grid "
        f"points the search cannot separate. Got {under} (spread "
        f"{(max(under) - min(under)) / max(under):.2%}), which is a gradient, i.e. a basin")


def _prior_sweep(rig, delay_ticks, base_drivetrain=None):
    """``(argmin_ticks, margin_over_next_best, grid)`` for the closed-loop sweep run on the PRIOR model.

    ``base_drivetrain`` shifts the prior model's ``dof_*`` before anything else happens, so the SAME
    experiment can be run on a differently-damped version of the same robot; the injected perturbation and
    delay are then applied on top of that, exactly as they are on the shipped one.
    """
    from virturoid.services.sysid.bench_rig import pd_replay
    from virturoid.services.sysid.gap_report import _delay_search
    from virturoid.services.sysid.synthetic_hardware import DEFAULT_PERTURBATION, perturbed_model

    prior = rig["model"] if base_drivetrain is None else perturbed_model(rig["model"], base_drivetrain)
    hw = perturbed_model(prior, DEFAULT_PERTURBATION)
    _, q_log, _, _ = pd_replay(hw, rig["q_cmd"], kp=rig["kp"], kd=rig["kd"], q_start=rig["q0"],
                               ctrl_every=rig["ctrl_every"], delay_ticks=int(delay_ticks))
    out = _delay_search(prior, rig["q_cmd"], q_log, kp=rig["kp"], kd=rig["kd"], q_start=q_log[0].copy(),
                        ctrl_every=rig["ctrl_every"], max_ticks=max(4, int(delay_ticks) + 2), dt=rig["dt"])
    grid = {r["delay_ticks"]: r["trajectory_rms_rad"] for r in out["grid"]}
    best = min(grid.values())
    runner_up = min(v for v in grid.values() if v != best)
    return out["delay_ticks"], 1.0 - best / runner_up, grid


def test_the_trajectory_sweep_is_biased_toward_zero_and_may_not_promote_itself_on_a_margin(rig):
    """The reason ``_merge_latency`` never lets the closed-loop sweep claim a delay on a MARGIN.

    A margin gate is what makes the open-loop estimate able to report a zero delay, and it is the obvious thing
    to reuse here. It must not be, and this test used to say why with a single number: on a log carrying 20 ms
    the sweep put the zero-delay entry 48.9% ahead of every alternative -- a confident, wrong answer, strictly
    worse than the refusal it would replace.

    RE-MEASURED on the declared drivetrain, THAT NUMBER IS 0.75%. The trap did not close; it moved. The sweep
    is still wrong at every delay (truth 1/2/3/4 ticks, argmin 0/0/1/1) but its margin is now 0.08-4.3%, so a
    margin gate would refuse rather than lie -- on THIS body. Run the identical experiment on the drivetrain
    this product substituted until the import path learned to read the source model
    (``SUBSTITUTED_DRIVETRAIN``) and the 48.9% comes straight back: 79.4% at 10 ms, 47.6% at 20 ms, for answers
    that are just as wrong.

    So the finding is stronger than the one it replaces, and it is the one that actually justifies the design:
    THE SWEEP'S MARGIN IS NOT EVIDENCE. The same wrong answer carries 0.08% or 79% depending on how hard the
    body rings -- a property of the customer's robot that we do not control and cannot bound in advance. A gate
    keyed on it is a gate keyed on someone else's damping.
    """
    # (1) BIASED TOWARD ZERO -- at every delay now, not at one. This half is untouched by the drivetrain.
    declared = {tk: _prior_sweep(rig, tk) for tk in (1, 2, 3, 4)}
    for tk, (got, _margin, grid) in declared.items():
        assert got != tk, (
            f"this test is no longer adversarial: the prior-model sweep lands on the truth at "
            f"{tk * MS_PER_TICK:g} ms, grid {grid}")
        assert got < tk, (
            f"the bias is meant to be toward ZERO; at {tk * MS_PER_TICK:g} ms the sweep OVER-shot to "
            f"{got * MS_PER_TICK:g} ms, grid {grid}")

    # (2) ...and on this body the margin it offers is worthless in the safe direction.
    worst = max(margin for _got, margin, _grid in declared.values())
    assert worst < 0.10, (
        f"the Go2 as its author declares it is no longer supposed to show the confident-wrong-answer trap; "
        f"the largest margin on a wrong answer across 10-40 ms is now {worst:.1%}. If it is back above 10%, "
        f"the RE-MEASURED note in this docstring is stale and the 48.9% story is live again on this body")

    # (3) THE TRAP ITSELF, demonstrated on a body that rings -- the one this product shipped until the import
    #     path learned to read the customer's declared drivetrain. A margin gate here promotes a wrong answer.
    for tk, floor in ((1, 0.60), (2, 0.40)):
        got, margin, grid = _prior_sweep(rig, tk, base_drivetrain=SUBSTITUTED_DRIVETRAIN)
        assert got != tk, (
            f"this test is no longer adversarial: on the SUBSTITUTED drivetrain the sweep now lands on the "
            f"truth at {tk * MS_PER_TICK:g} ms too, so there is no body left on which the trap is real and "
            f"`_merge_latency`'s refusal to promote the sweep needs a different justification. grid {grid}")
        assert margin > floor, (
            f"the WRONG answer must still look convincing on a LOW-DAMPING body -- that is the trap a margin "
            f"gate would fall into. At {tk * MS_PER_TICK:g} ms it beats the next-best by only {margin:.1%} "
            f"(measured 79.4% at 10 ms and 47.6% at 20 ms), grid {grid}")


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

    The threshold is untouched at 1.5x; what moved is the quantity it rules on. RE-MEASURED on the declared
    drivetrain: 15.538x here, 29.168x on the full plan (it was 1.789x / 1.683x under the substituted one).
    """
    from virturoid.services.sysid.fit import MIN_TRACKING_IMPROVEMENT_X

    app = fit_at(2)["application"]
    assert app["passed"] is True, app["verdict"]
    assert app["provisional"] is False
    assert app["improvement_x"] >= MIN_TRACKING_IMPROVEMENT_X


def test_the_gate_is_scored_at_the_identified_delay_and_that_is_what_moved_it(rig, fit_at):
    """The mechanism of the flip above, pinned so it cannot be mistaken for a threshold change.

    Both replays now run at the identified delay, so the timing error cancels in the ratio the way the ratio
    always assumed it did.

    RETRACTED, and this is the retraction: this test used to close by asserting that the SAME fit scored at the
    old point is *still under the threshold* -- 1.016x on the full plan. On the drivetrain the Go2's own file
    declares it is 1.503x here and 1.858x on the full plan, i.e. IT CLEARS. At 20 ms the scoring change is no
    longer what flips the verdict, because on a body that does not ring the delay-blind gate at 20 ms was
    survivable all along. The verdict claim survives only from 30 ms on; see the range test below.

    A SECOND RETRACTION, 2026-08-12, and it is about the yardstick rather than the fit. This test used to say
    the old point was "PINNED against a CEILING that depends only on the delay -- substitute the true hardware
    for the fitted model and no fit can do better". **There is no such ceiling.** The by-construction argument
    needs the replay family to CONTAIN the data-generating process, and at delay 0 against a delayed log it does
    not: the timing error is unmodelled, so the true parameters are not the minimiser of the position residual.
    MEASURED -- the delay-blind residual falls monotonically as reflected inertia is raised past the injected
    truth, because a too-heavy rotor makes the replay sluggish and thereby MIMICS transport lag. At 20 ms on
    this plan the injected truth scores 1.745x while ``armature +0.08`` (2.7x the injected 0.03) scores 2.166x,
    24% ABOVE the supposed ceiling; on the full 12-joint plan it is 2.277x against 1.791x, +27%. The real fit
    exceeds the reference too on the full plan: 1.858x against 1.791x, +3.8%, because the fit's own armature
    lands 17-21% high (0.0350-0.0362 against 0.030 injected).
    That +3.8% was previously written off as a windowing artefact -- ``_gate_ceiling`` replaying from the bench
    start pose over the whole log while the fit scored inside the excitation windows. **That explanation was a
    hypothesis and it is REFUTED**: instrumented on the same Go2 log, the two computations agree bit-for-bit on
    the start state (``q_meas[0] == start_pose``, maxabs 0.0), on the command series, on the log itself and on
    the row range (0..36000 for both -- ``_windows`` is used by the TORQUE fit, never by
    ``_trajectory_improvement``). The only difference was the model in the denominator, and the truth is not the
    best model there.

    So the helper is renamed (``_true_parameter_score``, a REFERENCE POINT), and what this test asserts is a
    BRACKET rather than a one-sided bound -- which is strictly stronger and says the thing that is actually
    true: the fit's delay-blind score cannot get FAR from the truth's delay-blind score in EITHER direction,
    and that is what makes the old number a measurement of the delay rather than of the fit. Measured
    0.861x the reference here, 1.038x on the full plan.

    What the scoring change is worth: an ORDER OF MAGNITUDE on the same fit -- 10.3x here, 15.7x on the full
    plan.
    """
    from virturoid.services.sysid.fit import MIN_TRACKING_IMPROVEMENT_X

    traj = fit_at(2)["trajectory"]
    ref = _true_parameter_score(rig, 2)
    frac = traj["improvement_x_at_zero_delay"] / ref
    assert traj["scored_at_delay_ms"] == pytest.approx(DEFAULT_DELAY_MS)
    assert traj["delay_identified"] is True
    assert traj["improvement_x"] >= MIN_TRACKING_IMPROVEMENT_X
    assert 0.80 <= frac <= 1.20, (
        f"the OLD scoring point is meant to sit CLOSE to the delay-blind score of the true parameters, on "
        f"either side -- that is what makes it a measurement of the delay rather than of the fit. It reads "
        f"{traj['improvement_x_at_zero_delay']}x against a reference of {ref:.3f}x, i.e. {frac:.1%} of it "
        f"(measured 86.1% on this plan, 103.8% on the full one). Under 80% and the old point has room to "
        f"discriminate a fit after all, so this docstring's argument is wrong; over 120% and the estimator is "
        f"buying delay-blind score by MIMICKING the delay far harder than the fit measured here does, which is "
        f"a finding about the estimator and has to be written down rather than absorbed")
    assert traj["improvement_x"] > 5.0 * traj["improvement_x_at_zero_delay"], (
        f"this test is no longer adversarial: scoring at the identified delay is worth only "
        f"{traj['improvement_x'] / traj['improvement_x_at_zero_delay']:.2f}x on the same fit "
        f"({traj['improvement_x']}x vs {traj['improvement_x_at_zero_delay']}x), so the scoring point is not "
        f"what moves the quantity and the change being tested has stopped mattering")


def test_the_old_scoring_points_range_collapses_with_delay_and_the_whole_family_falls_under_the_threshold(rig):
    """WHY the scoring point had to move, and it is not a tuning argument.

    Scored with both replays at delay 0, ``improvement_x`` loses its dynamic range as the delay grows: the log
    carries a timing error no fitted parameter can remove, it is in BOTH terms of the ratio, and it dominates
    them both. Where nothing the estimator can express clears the threshold, a refusal is not evidence about
    the fit -- it is arithmetic.

    RE-MEASURED TWICE, and both re-measurements are recorded because each moved a claim.

    2026-08-12 (drivetrain): it was 4.964x / 1.016x / 1.000x at 10 / 20 / 40 ms on the full plan, so the
    package's own DEFAULT injection sat in the dead zone -- which is what made the original finding sharp. On
    the drivetrain the Go2 actually declares, the true parameters score 3.350x at 10 ms, 1.745x at 20 ms (full
    plan 1.791x), 1.009x at 30 and 0.996x at 40. The dead zone starts at 30 ms, not at 20, and "at the
    package's default no fit could pass" is RETRACTED.

    2026-08-12 (the yardstick): the sentence this test used to open with -- "``improvement_x`` has a CEILING
    that depends only on the delay: no fit can beat the parameters that generated the log" -- is RETRACTED as
    well. It is not a bound and it never was; see ``_delay_blind_score``. Sweeping the estimator's own three
    parameters over a 60-point grid at each delay, the MAXIMUM delay-blind score is **2.166x at 20 ms (24%
    above the true parameters' 1.745x, at armature +0.08), 1.036x at 30 ms and 1.003x at 40 ms** -- and on the
    full 12-joint plan **3.823x / 2.277x / 1.012x / 1.003x at 10 / 20 / 30 / 40 ms** against references of
    3.403x / 1.791x / 1.005x / 1.001x. So:

      * the conclusion SURVIVES and is now measured over 60 models instead of argued from one -- at 30 ms and
        beyond NOTHING in the family the estimator draws from reaches the 1.5x threshold, which is what the
        "a refusal here is arithmetic" claim actually needs;
      * the yardstick was mislabelled, and at 20 ms it can be beaten by a quarter -- which is why the fit
        exceeding it by 3.8% on the full plan was never a defect in the fit.

    Cost: 3 x 60 delay-blind replays (~50 s here) on top of the reference sweep. That is the price of turning
    "by construction" into a measurement, and this file already paid it once by believing the argument.
    """
    from virturoid.services.sysid.fit import MIN_TRACKING_IMPROVEMENT_X

    at_10, at_20 = _true_parameter_score(rig, 1), _true_parameter_score(rig, 2)
    at_30, at_40 = _true_parameter_score(rig, 3), _true_parameter_score(rig, 4)
    assert at_10 >= MIN_TRACKING_IMPROVEMENT_X, f"10 ms should be inside reach; got {at_10:.3f}x"
    assert at_20 >= MIN_TRACKING_IMPROVEMENT_X, (
        f"the RETRACTION in this docstring is itself stale: the delay-blind score of the TRUE parameters at the "
        f"package's DEFAULT 20 ms is back under the {MIN_TRACKING_IMPROVEMENT_X}x threshold at {at_20:.3f}x, so "
        f"the original 'no fit could pass at 20 ms' finding is live again and the retraction must be withdrawn")
    assert at_40 < 1.05, f"the truth's own score at 40 ms should be ~1.0 (no range at all); got {at_40:.3f}x"
    assert at_10 > at_20 > at_30 > at_40, (
        f"the reference must fall monotonically as the delay grows: {at_10:.3f} / {at_20:.3f} / {at_30:.3f} / "
        f"{at_40:.3f}")

    best_20, off_20 = _best_delay_blind_score(rig, 2)
    best_30, off_30 = _best_delay_blind_score(rig, 3)
    best_40, off_40 = _best_delay_blind_score(rig, 4)

    # THE RETRACTION, asserted so it cannot be quietly reinstated: the true parameters are NOT the maximum.
    assert best_20 > at_20 * 1.05, (
        f"the retracted 'ceiling' claim is back in force: the best model in the estimator's own family scores "
        f"{best_20:.3f}x at 20 ms against the true parameters' {at_20:.3f}x ({best_20 / at_20:.3f}x), so on "
        f"this body the truth IS effectively maximal and the by-construction argument this test retracted "
        f"(measured 2.166x vs 1.745x, at {off_20}) would have to be reinstated -- deliberately, in writing")
    # THE ADVERSARIAL CLAIM, at the delays where it is true, over the whole family rather than one point.
    assert best_30 < MIN_TRACKING_IMPROVEMENT_X, (
        f"this test is no longer adversarial: at 30 ms the delay-blind metric can still reach {best_30:.3f}x "
        f"(at {off_30}), at or above the {MIN_TRACKING_IMPROVEMENT_X}x threshold. If some model the estimator "
        f"can express clears the gate at every delay, a delay-blind refusal is never uninformative and the "
        f"scoring point did not need to move at all")
    assert best_40 < 1.10, (
        f"at 40 ms the delay-blind metric should have no range left for ANY model in the family; the best is "
        f"{best_40:.3f}x at {off_40} (measured 1.003x)")
    assert best_20 > best_30 > best_40, (
        f"the envelope must fall with delay too: {best_20:.3f} / {best_30:.3f} / {best_40:.3f}")


# =========================================================================================================
# THE CONTROLS. A gate that stopped refusing the things it exists to refuse would be a worse defect than the
# one this file closed.
# =========================================================================================================

def test_a_misspecification_is_still_refused_when_the_robot_also_has_delay(fit_at):
    """THE control on the scoring change. ``application_gate`` was sized against a robot built with +30% link
    mass and inertia -- an error NO combination of frictionloss/damping/armature can express, and one the
    estimator absorbs into armature. Scoring at the identified delay must not let it through.

    MEASURED on the full 12-joint plan at the new scoring point: 1.001x with no delay, 0.996x at 20 ms. Those
    two predate the drivetrain fix; RE-MEASURED on this narrow plan at 20 ms the same misspecification now
    scores 1.078x (it was 0.996x). The band the 1.5x threshold sits in is narrower than it was -- see the
    constant's docstring.

    "...AND IT IS STILL EMPTY" USED TO CLOSE THAT SENTENCE, AND IT IS RETRACTED (2026-08-12, same day, a
    different probe): the band is not empty and this test is not the general control it reads as. What it
    controls is the LINK MASS AND INERTIA family, whose ratio is pinned near 1.0 by the GRAVITY term rather
    than by anything structural -- measured x0.4 ... x3.0 on the composed dog, 0.939 ... 1.000, never above 1.
    Take gravity out of the error and it goes straight through the gate: link rotational inertia at CORRECT
    mass reaches 1.753x and a 20% torque-scale error reaches 1.507x, both applied, both with intervals
    excluding the truth. Contract for that: ``tests/test_sysid_stage2.py``, and
    ``docs/calibration_wedge_under_delay.md`` section 13. This test still asserts exactly what it always
    measured; only the sentence claiming it generalised is withdrawn.
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


def test_at_twice_the_realistic_delay_the_metric_regains_range_but_the_gate_still_refuses(fit_at):
    """40 ms is double the figure Hwangbo et al. name and double this package's own default, and the closure
    is only partial there. Recorded as a limit, not as a win.

    Same fit, two scoring points: at the OLD one the score is pinned at ~1.000x -- and so is every other model
    in the estimator's family (best 1.003x, measured), so the refusal carries no information about the fit at
    all. At the new one it is 1.148x here and 1.121x on the full 12-joint plan. So the metric gets SOME of its dynamic range back and the gate then refuses ON THE
    MERITS, which is the honest outcome and the reason this section exists.

    RE-MEASURED, and the verdict got WORSE rather than better. It was 1.484x (full plan) / 1.502x (narrow),
    straddling the 1.5x threshold, and the note here used to be that the verdict at 40 ms depended on which
    plan you ran. It no longer straddles: it refuses either way. The cause is the harness, not the estimator,
    and it is worth naming because it looks like a regression and is not. ``DEFAULT_PERTURBATION`` is an
    ABSOLUTE offset (+0.6 N.m.s/rad of damping, +0.08 N.m of dry friction). Against the drivetrain we used to
    substitute (damping 0.8) that is a +75% modelling error; against the one Unitree declares (2.0) the same
    offset is +30%. ``improvement_x`` is a ratio of trajectory RMS before and after fitting, so a smaller
    relative error leaves less to remove and the ratio falls. The fit is measuring a smaller mistake and
    reporting a smaller improvement, which is correct.

    The delay itself is still recovered exactly at 40 ms, which is the part that matters for the wedge.
    """
    from virturoid.services.sysid.fit import MIN_TRACKING_IMPROVEMENT_X

    fit = fit_at(4)
    traj = fit["trajectory"]
    assert traj["scored_at_delay_ms"] == pytest.approx(40.0)
    assert fit["latency"]["delay_ms"] == pytest.approx(40.0)
    assert fit["latency"]["identified"] is True
    assert traj["improvement_x_at_zero_delay"] < 1.05, (
        "the old scoring point should still be pinned at ~1.0 here -- that is the ceiling effect")
    assert traj["improvement_x"] > 1.05, (
        f"the new scoring point must still recover SOME dynamic range at 40 ms -- that is the half of the "
        f"closure that works. Got {traj['improvement_x']}x against {traj['improvement_x_at_zero_delay']}x at "
        f"the old point; measured 1.148x (this plan) and 1.121x (full plan)")
    # A LIMIT, asserted as a limit. If this starts passing, the section heading above is wrong.
    assert traj["improvement_x"] < MIN_TRACKING_IMPROVEMENT_X, (
        f"this test is no longer adversarial: 40 ms now CLEARS the {MIN_TRACKING_IMPROVEMENT_X}x gate at "
        f"{traj['improvement_x']}x, so the limit this test records has closed and the docstring must be "
        f"rewritten to claim it rather than to concede it")
    assert fit["application"]["passed"] is False, (
        f"...and the refusal must actually reach the verdict: {fit['application'].get('verdict')}")


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

    ``_delay_from_command_response`` has no plant, so a link mass and inertia error cannot move it. This one
    reads the torque out of the motion, so it can -- and the question is whether it DEGRADES GRACEFULLY.

    THE REQUIRED BEHAVIOUR IS RESTATED, and this is the most important note in the file. The test used to
    assert that at 40 ms with +30% links the estimator REFUSES, on the reasoning that the free
    frictionloss/damping/armature columns cannot express a mass error and must not appear to. It now identifies
    -- AT 40.0 ms, WHICH IS THE INJECTED VALUE. Going back to the old drivetrain shows the refusal was never
    the property that was claimed: under it the argmin was ALSO 40.0 ms and the gate refused a CORRECT answer,
    because the reconstruction residual had gone negative (-43% to -446% of the commanded torque explained).
    So "it refuses" recorded a conservative gate, not an estimator that cannot be fooled -- and asserting it
    would have been asserting that a right answer stays suppressed.

    What must be true is narrower and is the thing the original sentence was reaching for: A MISSPECIFICATION
    MAY COST THE MARGIN AND MAY COST THE ANSWER, BUT IT MAY NEVER BUY A CONFIDENT WRONG TICK. Both halves are
    measured here.

      * THE COST is real and monotone, which is what the test's name claims -- and the contrast is the point.
        At 20 ms the margin over the next-best tick is 0.930 under the DEFAULT perturbation, an error the free
        columns CAN express (frictionloss + damping + armature), which costs it essentially nothing. Swap that
        for one they cannot and it falls: +15% links 0.362 -> +30% 0.242 -> +50% 0.178, against a
        ``DELAY_MIN_MARGIN`` floor of 0.15. The third rung is nearly on the floor.
      * NEVER CONFIDENTLY WRONG: swept over link scales 1.15 / 1.30 / 1.50 / 2.00 / 3.00 crossed with 0 / 20 /
        40 ms, all 15 configurations return the injected tick. Five of those are asserted here; the sweep is in
        the report, not in the suite, because it is 15 fits' worth of wall clock.

    Why it stopped collapsing at 40 ms rather than because anything in the estimator changed: with damping 2.0
    and dry friction 0.2 the joint's torque budget is dominated by terms the free columns DO express, so the
    inertial term carrying the +30% error is a smaller share of it and the reconstruction survives. At damping
    0.8 it was not, and it did not.
    """
    from virturoid.services.sysid.gap_report import DELAY_MIN_MARGIN

    def _lat(ticks, scale=None):
        kw = {} if scale is None else {"perturbation": {}, "link_scale": scale}
        return gap_at(ticks, channel="position_only", **kw)[1]["latency"]

    # (1) THE COST. Each rung of misspecification spends margin, monotonically, toward the refusal floor.
    ladder = [_lat(2)["margin_over_next_best_tick"]] + [
        _lat(2, s)["margin_over_next_best_tick"] for s in (1.15, 1.30)]
    assert all(a > b for a, b in zip(ladder, ladder[1:])), (
        f"a link mass/inertia error the free columns cannot express is meant to COST the margin at every rung: "
        f"clean / +15% / +30% came back {ladder}")
    assert ladder[-1] < 0.35 * ladder[0], (
        f"this test is no longer adversarial: a +30% link mass/inertia error costs only "
        f"{1.0 - ladder[-1] / ladder[0]:.1%} of the clean margin ({ladder}), so the error is being absorbed "
        f"somewhere it should not be rather than showing up as lost confidence")
    assert ladder[-1] > DELAY_MIN_MARGIN, (
        f"...and +30% is meant to still clear the {DELAY_MIN_MARGIN} floor; got {ladder[-1]}")

    # (2) NEVER CONFIDENTLY WRONG. Refusing is allowed at every rung. Claiming a tick that was not injected is
    #     not, and that is the only failure mode a customer cannot detect from the report.
    for scale, ticks in ((1.15, 2), (1.30, 2), (1.50, 2), (1.30, 0), (1.30, 4)):
        lat = _lat(ticks, scale)
        assert (not lat["identified"]) or lat["delay_ms"] == pytest.approx(ticks * MS_PER_TICK), (
            f"a +{(scale - 1.0) * 100:.0f}% link mass/inertia error at {ticks * MS_PER_TICK:g} ms was CLAIMED "
            f"at {lat['delay_ms']} ms (margin {lat.get('margin_over_next_best_tick')}) on a position-only log; "
            f"the free frictionloss/damping/armature columns cannot absorb that error and must not appear to")

    # (3) ...and where the misspecification is not yet fatal it is still right AND still confident.
    lat0 = _lat(0, 1.30)
    assert lat0["delay_ms"] == 0.0 and lat0["identified"] is True, (
        "the estimator should still be right where the misspecification is not yet fatal: 0 ms, identified")


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
