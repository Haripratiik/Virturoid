"""MOTOR CURRENT as the torque channel -- the log most robots can actually produce.

`docs/calibration_wedge_under_delay.md` section 10 listed "the estimate requires `tau_meas`" as an open
coverage limit. Surveying what the common stacks expose says that limit binds on the majority of real robots,
not on a corner: ROS 2's `sensor_msgs/JointState` has an `effort` field and `ros2_control`'s
`joint_state_broadcaster` only publishes it for joints whose hardware exposes an effort interface, which on a
robot with no joint torque sensor it does not. What those robots DO have is a current sense in every motor
driver -- Unitree's `MotorState.tau_est` (itself current-derived), Dynamixel's `Present Current`, ODrive and
moteus `Iq`, ANYdrive's spring deflection being the exception rather than the rule.

Current times the torque constant IS torque, and the constant is on the datasheet the BOM already carries. So
this file is the contract for doing that conversion HONESTLY, which means four things, one per section below:

  * it happens, it is reported, and a `measured_on` style provenance rides with it -- the conversion is never
    silent and current is never treated as newton-metres;
  * a WRONG constant costs the two answers differently, and the difference is measured: the delay is a LAG and
    survives it, the fitted parameters are a MAGNITUDE and do not;
  * the log itself can re-derive the constant, so a customer with no datasheet still gets told the factor;
  * a magnitude-only current channel is REFUSED, because kt*|I| is an actuator that can only push one way.

Everything here is sim2sim on the Menagerie Go2 -- see `synthetic_hardware.WHAT_SIM2SIM_DOES_NOT_PROVE`. Two
joints and 12 s, the same narrow plan `test_sysid_delay_wedge.py` uses and for the same cost reason.
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
    pytest.mark.skipif(not _GO2.exists(), reason=f"needs the MuJoCo Menagerie Go2 at {_GO2}"),
    pytest.mark.no_gait_fit,
]

ONLY_JOINTS = ["FL_hip", "FL_thigh"]
BUDGET_S = 12.0
MS_PER_TICK = 10.0

#: The same 12 s experiment on the FL thigh and calf. It leaves out the hip, whose bench gain is floored far above
#: its inertia-scaled value and whose delayed loop is the one that rings, and on the twin carrying the Go2's own
#: link inertials that is the difference between a wrong torque constant being refused and being applied at the
#: package's default delay. See ``test_the_fit_is_refused_when_the_torque_constant_is_wrong``.
NO_HIP_JOINTS = ["FL_thigh", "FL_calf"]

#: How far the constant is out in the control. 1.35x has been this file's case from the start.
KT_ERROR = 1.35


@pytest.fixture(scope="module")
def go2():
    from virturoid.services.robot_import import import_robot
    return import_robot(str(_GO2), robot_id="go2_torque_channel")["gene"]


@pytest.fixture(scope="module")
def plan(go2):
    from virturoid.services.sysid import build_excitation
    return build_excitation(go2, budget_s=BUDGET_S, only_joints=ONLY_JOINTS)


@pytest.fixture(scope="module")
def gap_at(go2, plan):
    """``gap_at(delay_ticks, **kw)`` -> ``(log, gap)``, memoised per configuration."""
    from virturoid.services.sysid import measure_gap
    from virturoid.services.sysid.synthetic_hardware import synthetic_hardware_log

    cache: dict = {}

    def _get(delay_ticks: int, **kw):
        key = (int(delay_ticks), tuple(sorted((k, str(v)) for k, v in kw.items())))
        if key not in cache:
            _, log = synthetic_hardware_log(go2, delay_ticks=int(delay_ticks), plan=plan, **kw)
            cache[key] = (log, measure_gap(go2, log, plan=plan, measure_noise_floor=False))
        return cache[key]

    return _get


@pytest.fixture(scope="module")
def kt_fit(go2, plan):
    """``kt_fit(delay_ticks, error, joints=ONLY_JOINTS)`` -> ``fit_parameters`` on a motor-current log whose
    torque constant is ``error`` x the one the reader derives, on top of the default injection. Memoised, and at
    n_boot=16 as this file has always fitted. ``error`` 1.0 is the same log with the constant right."""
    from virturoid.services.sysid import build_excitation, fit_parameters
    from virturoid.services.sysid.synthetic_hardware import synthetic_hardware_log

    plans = {tuple(ONLY_JOINTS): plan}
    cache: dict = {}

    def _get(delay_ticks: int, error: float, joints=tuple(ONLY_JOINTS)):
        joints = tuple(joints)
        key = (int(delay_ticks), float(error), joints)
        if key not in cache:
            if joints not in plans:
                plans[joints] = build_excitation(go2, budget_s=BUDGET_S, only_joints=list(joints))
            _, log = synthetic_hardware_log(go2, delay_ticks=int(delay_ticks), plan=plans[joints],
                                            channel="current", torque_constant_error=float(error))
            cache[key] = fit_parameters(go2, log, plan=plans[joints], n_boot=16)
        return cache[key]

    return _get


def _used(record):
    """The first joint whose current channel was actually converted."""
    return next(r for n, r in (record.get("per_joint") or {}).items()
                if r.get("used") and n in ONLY_JOINTS)


# =========================================================================================================
# 1. THE CONVERSION HAPPENS AND IT IS REPORTED.
# =========================================================================================================

def test_a_current_only_log_reaches_every_answer_a_torque_log_does(gap_at):
    """A log with amps and no torque field gets the delay AND the parameter attribution."""
    _log, gap = gap_at(2, channel="current")
    assert gap["latency"]["identified"] is True
    assert gap["latency"]["delay_ms"] == pytest.approx(2 * MS_PER_TICK)
    assert gap["attribution"]["available"] is True, (
        "a converted current channel is a torque channel; the attribution must be reachable")


def test_the_conversion_is_reported_never_silent(gap_at):
    """Current is not torque, and nothing downstream may read as though it were.

    The record names the constant, where it came from, its uncertainty, and the arithmetic. Without this a
    customer reads a frictionloss in N.m that is only as good as a number nobody showed them.
    """
    _log, gap = gap_at(2, channel="current")
    rec = gap["torque_channel"]
    assert rec["converted"] is True
    assert "COMPUTED from the logged motor current" in rec["headline"]
    assert "not measured with a torque sensor" in rec["headline"]
    row = _used(rec)
    assert row["nm_per_a"] > 0.0
    assert row["source"] == "derived_from_catalog_actuator"
    assert "kt = efficiency * V / no-load speed" in row["basis"]
    assert row["uncertainty_frac"] > 0.0


def test_a_stated_datasheet_constant_wins_over_the_derived_one(go2, plan, gap_at):
    """The customer's own number is used when they have it, and is labelled as theirs.

    It matters that this is the FIRST source rather than a fallback: the derived one is sized off the catalog
    part the BOM picks for the joint's torque requirement, and on this very robot that part is a T-Motor
    AK10-9 where Unitree actually ships a GO-M8010-6 -- a factor, not a percentage.
    """
    from virturoid.services.sysid import measure_gap

    log, _ = gap_at(2, channel="current")
    gap = measure_gap(go2, log, plan=plan, measure_noise_floor=False, torque_constant_nm_per_a=0.75)
    row = _used(gap["torque_channel"])
    assert row["nm_per_a"] == pytest.approx(0.75)
    assert row["source"] == "stated_by_the_customer"


# =========================================================================================================
# 2. A WRONG CONSTANT COSTS THE DELAY AND THE PARAMETERS DIFFERENTLY, AND BOTH SAY SO.
# =========================================================================================================

@pytest.mark.parametrize("delay_ticks", [0, 2, 4])
def test_the_delay_survives_a_wrong_torque_constant_because_it_is_a_lag(go2, plan, gap_at, delay_ticks):
    """THE asymmetry this whole file rests on, measured.

    A torque constant is a pure SCALE. ``_delay_from_command_response`` finds the SHIFT that aligns two
    signals, and a shift does not move when one of them is rescaled -- so the injected 0 / 20 / 40 ms come
    back exactly with the channel 1.35x out. What a wrong constant does spend is the reconstruction gate's
    margin: MEASURED on the full scale sweep, ``fraction_of_applied_torque_explained`` runs 1.00 / 0.87 / 0.77
    at 1.0x / 1.15x / 1.3x and falls through the 0.5 floor by 2.0x, where the estimate is refused rather than
    reported. This is why current is worth accepting at all.
    """
    _log, gap = gap_at(delay_ticks, channel="current", torque_constant_error=1.35)
    lat = gap["latency"]
    assert lat["identified"] is True, lat.get("not_identified_because")
    assert lat["delay_ms"] == pytest.approx(delay_ticks * MS_PER_TICK), (
        f"a 1.35x torque-constant error moved the delay: injected {delay_ticks * MS_PER_TICK} ms, got "
        f"{lat['delay_ms']} ms. The lag is not supposed to care about the scale of the channel")


#: Where the gate still refuses a fit taken through a 1.35x-wrong torque constant, measured 2026-10-10 on the twin
#: that carries the Go2's own link inertials: ``(delay_ticks, joints)``. In every row the constant-RIGHT fit on the
#: same cell clears the tracking gate, so each refusal is the gate telling the two apart, not a metric with no
#: range left.
STILL_REFUSED = [
    (0, tuple(ONLY_JOINTS)),       # wrong 1.024x, refused; right 2.652x
    (4, tuple(ONLY_JOINTS)),       # wrong 1.239x, refused; right 2.499x (itself refused by a rival; see the
                                   # 30 / 40 / 50 ms test in tests/test_sysid_delay_wedge.py)
    (2, tuple(NO_HIP_JOINTS)),     # THE CONTROL at the package's default delay: wrong 1.010x; right 3.185x
]


@pytest.mark.parametrize("delay_ticks, joints", STILL_REFUSED,
                         ids=[f"{t * 10}ms-{'+'.join(j)}" for t, j in STILL_REFUSED])
def test_the_fit_is_refused_when_the_torque_constant_is_wrong(kt_fit, delay_ticks, joints):
    """...and the PARAMETERS are a magnitude, so they do not survive it, and the gate catches that WHERE IT
    STILL DOES.

    The tracking gate is untouched at 1.5x. MEASURED at a 1.35x constant error on this plan: 1.272 / 1.276 /
    1.126 at 0 / 20 / 40 ms, all refused, against 1.903 / 1.789 / 1.502 with the constant right. So the same
    log yields a usable latency and an unusable fit, which is the honest split.

    RE-MEASURED 2026-10-10 on the twin that carries the Go2's own link inertials, and this test FAILED at 20 ms:
    the 1.35x-wrong fit scored 4.974x and was APPLIED. It is not weakened into a pass. It is split the way
    ``tests/test_sysid_delay_wedge.py`` splits its claims, into what still holds (here) and what does not (the
    next test, which pins it as a limit and says why). On the hip and thigh the refusal still holds at 0 ms
    (1.024x, against 2.652x with the constant right) and at 40 ms (1.239x, against 2.499x), and both cells stay.
    At the package's default 20 ms it holds only where the experiment leaves the HIP alone: on the FL thigh and
    calf the wrong constant reads 1.010x and is refused while the right one reads 3.185x and is applied, and
    that is the control this test now guards at the default delay (the FL thigh alone: 1.014x, refused). With
    the hip excited the same error is applied: 4.974x with the thigh, 5.711x alone.

    Two more things the re-measurement found, so this is not read as "only 1.35x is a problem". At 20 ms on the
    hip and thigh the gate passes every constant from 0.7x to 1.35x that was tried (4.011x / 6.132x / 5.975x /
    4.974x at 0.7 / 0.8 / 1.2 / 1.35), while one 1.5x out or worse, or 0.5x or less, is refused for a different
    reason: the channel can no longer reproduce the declared control law at any lag, no delay is claimed, the
    replay falls back to 0 ms and the fit reads 0.38-0.86x, a refusal about the LOG, which is what its verdict
    says. And a wrong constant on a log with nothing else wrong is refused at 0 / 20 / 40 ms at 0.000x: our
    prior already reproduces that motion exactly, and the fit's negative dampings (-0.67 and -0.55 N.m.s/rad at
    0 ms, intervals excluding zero) can only make it worse.

    Each cell asserts that the wrong constant is refused by the TRACKING gate, at the identified delay, under
    the threshold, and that the right constant on the same cell clears that gate.
    """
    from virturoid.services.sysid.fit import MIN_TRACKING_IMPROVEMENT_X

    ms = delay_ticks * MS_PER_TICK
    wrong, right = kt_fit(delay_ticks, KT_ERROR, joints), kt_fit(delay_ticks, 1.0, joints)
    where = f"{'+'.join(joints)} at {ms:g} ms"
    assert wrong["ok"] is True
    assert wrong["application"]["passed"] is False, (
        f"a fit taken through a {KT_ERROR}x-wrong torque constant scored {wrong['trajectory']['improvement_x']}x "
        f"on {where} and PASSED the {MIN_TRACKING_IMPROVEMENT_X}x gate; every parameter in it carries that "
        f"factor. Measured 1.024x / 1.239x / 1.010x on 2026-10-10 for the three cells this test keeps")
    # Refused by the TRACKING gate, on the merits, at the identified delay: not by a rival, and not because the
    # channel was too wrong to find the delay and the score fell back to 0 ms.
    assert wrong["application"]["refused_by"] is None, wrong["application"].get("verdict")
    assert wrong["latency"]["identified"] is True, wrong["latency"].get("not_identified_because")
    assert wrong["latency"]["delay_ms"] == pytest.approx(ms)
    assert wrong["trajectory"]["improvement_x"] < MIN_TRACKING_IMPROVEMENT_X
    # ...and the constant-RIGHT fit on the same cell clears the tracking gate, so the refusal above is the gate
    # discriminating rather than a metric that cannot pass anything here.
    assert right["trajectory"]["improvement_x"] >= MIN_TRACKING_IMPROVEMENT_X, (
        f"the constant-right fit on {where} reads {right['trajectory']['improvement_x']}x, under the gate, so "
        f"this cell no longer shows the gate telling a wrong constant from a right one (measured 2.652x / "
        f"2.499x / 3.185x on 2026-10-10)")
    assert wrong["torque_channel"]["converted"] is True, "and the fit must carry the conversion that caused it"


def test_at_the_default_delay_a_wrong_constant_is_APPLIED_once_the_hip_is_excited(kt_fit):
    """THE MISS, pinned as a limit rather than absorbed. Measured 2026-10-10 on the twin that carries the Go2's
    own link inertials, at the package's default 20 ms, on this file's plan (FL hip and thigh), through the
    shipped ``fit_parameters``: a fit taken through a 1.35x-wrong constant on top of the default injection is
    APPLIED.

      * The tracking gate passes it at 4.974x. The constant-right fit on the same log reads 13.646x, so the ratio
        still ORDERS the two, by 2.7x; the gate's fixed 1.5x no longer separates them, and nothing here retunes
        it.
      * Neither rival reaches the 1.0x it needs to claim it. The global-scale one replays our prior with every
        gear at 0.836x and explains 0.727x; the link-inertia one, at 6.1x the tensors, explains 0.975x and gets
        4.85x closer to the log than the prior does against a 7.3x threshold. A real dissipation change sits
        under the constant error and dilutes the coherence a torque scale would show, which is the hole
        ``application.what_this_gate_does_not_catch`` already discloses from the composed dog (a 1.35x
        constant on a genuine perturbation applied at 1.689x, gear rival 0.869).
      * What reaches the model is two ARMATURE deltas, +0.0172 and +0.0194, whose intervals [0.0131, 0.0214]
        and [0.0147, 0.0249] both EXCLUDE the injected +0.030: a confident wrong answer, the failure this gate
        exists to stop. The damping and dry-friction cells, which carry the factor most visibly, were refused
        by the identifiability gate, so what ships is the quiet part of the error.

    It is the inertials that opened it: the same gene with its ``source_inertials`` table removed reads 1.053x
    here and is refused (16.283x with the constant right). And it is the HIP: excite the FL thigh and calf
    instead and the same error is refused at 1.010x (the control in the previous test), excite the FL hip
    alone and it is applied at 5.711x. It does not depend on the bootstrap: n_boot 16 / 32 and seeds 0 / 1 / 2
    all read 4.974x and identify the same two cells.

    Asserted so that it fails in BOTH directions: if this cell is refused again (rewrite the docstring to claim
    the closure only after re-checking the selections and the 0.7x-1.35x range it was measured over), or if the
    constant-right fit stops clearing (then the refusals in the previous test mean less than they say).
    """
    from virturoid.services.sysid.fit import MIN_TRACKING_IMPROVEMENT_X
    from virturoid.services.sysid.synthetic_hardware import DEFAULT_PERTURBATION

    wrong, right = kt_fit(2, KT_ERROR), kt_fit(2, 1.0)
    assert wrong["ok"] is True and wrong["torque_channel"]["converted"] is True
    assert wrong["latency"]["identified"] is True and wrong["latency"]["delay_ms"] == pytest.approx(20.0)
    # The constant-right fit still clears, so this cell still has a correct case for the gate to pass.
    assert right["application"]["passed"] is True, right["application"].get("verdict")
    assert right["trajectory"]["improvement_x"] >= MIN_TRACKING_IMPROVEMENT_X
    # THE MISS. If this starts failing because the fit is refused, the limit closed: see the docstring.
    assert wrong["application"]["passed"] is True, (
        f"this test is no longer adversarial: a fit taken through a {KT_ERROR}x-wrong torque constant at 20 ms on "
        f"the hip and thigh is REFUSED again ({wrong['application'].get('verdict')}). On 2026-10-10 it was "
        f"applied at 4.974x with neither rival firing. Before claiming the closure, re-check it across joint "
        f"selections and constants from 0.7x to 1.35x the way the miss was measured")
    assert wrong["trajectory"]["improvement_x"] >= MIN_TRACKING_IMPROVEMENT_X
    assert wrong["trajectory"]["improvement_x"] < right["trajectory"]["improvement_x"], (
        f"the wrong constant now tracks at least as well as the right one ({wrong['trajectory']['improvement_x']}x "
        f"vs {right['trajectory']['improvement_x']}x; measured 4.974x vs 13.646x), so the ratio no longer even "
        f"orders them")
    for check in ("global_scale", "link_inertia"):
        rival = (wrong.get(check) or {}).get("rival") or {}
        assert rival, f"the {check} rival did not run on a fit that is being applied: {wrong.get(check)}"
        assert rival["explains_x"] < 1.0, (
            f"the {check} rival now explains this fit at {rival['explains_x']}x and claims it (measured 0.727x for "
            f"the gear rival and 0.975x for the inertia rival on 2026-10-10); if that is what refuses it, the "
            f"miss closed through a rival and the docstring must say which")
    # ...and what it writes into the model is confidently wrong.
    applied = {(j, p): row["parameters"][p]["delta_interval"]
               for j, row in wrong["joints"].items() for p in row["identified"]}
    assert applied, "the miss applies nothing at all, so it is no longer the case this test records"
    covering = [cell for cell, (lo, hi) in applied.items() if lo <= DEFAULT_PERTURBATION[cell[1]] <= hi]
    assert not covering, (
        f"some applied intervals now cover the injected value ({covering} of {sorted(applied)}); on 2026-10-10 "
        f"both armature intervals excluded +0.030, which is what makes this a confident wrong answer")


# =========================================================================================================
# 3. THE LOG CAN RE-DERIVE THE CONSTANT, SO A CUSTOMER WITHOUT A DATASHEET IS STILL TOLD THE FACTOR.
# =========================================================================================================

@pytest.mark.parametrize("error", [0.4, 1.35, 2.5])
def test_the_log_names_the_factor_the_constant_is_wrong_by(gap_at, error):
    """The cross-check: the slope between the torque the controller COMMANDED and the channel it got back.

    0.4 and 2.5 are not arbitrary -- 2.5 is the MEASURED ratio between the catalog part the BOM sizes for the
    Go2's hip (T-Motor AK10-9) and the motor Unitree ships (GO-M8010-6), i.e. exactly how wrong a derived
    constant is on this robot. A customer with no datasheet gets handed that factor rather than a silent
    error, and can multiply their fitted parameters by it or re-run with the right number.
    """
    _log, gap = gap_at(2, channel="current", torque_constant_error=error)
    row = _used(gap["torque_channel"])
    assert row["ratio_identified_over_used"] == pytest.approx(error, rel=0.05), (
        f"the log was generated with a constant {error}x the one the reader derived; the cross-check should "
        f"recover that factor, and reported {row['ratio_identified_over_used']}")
    assert row["identified_from_this_log"]["r2"] > 0.9


@pytest.mark.parametrize("error", [0.4, 2.5])
def test_a_grossly_wrong_constant_still_yields_the_delay_via_the_motion(gap_at, error):
    """The composition that turns a dead end into a diagnosis, and the reason the motion estimator runs here.

    At 2.5x out the reconstruction gate refuses the torque channel outright ("the declared PD law reconstructs
    only -5% of the applied torque"), and before this the customer got a refusal with nothing to act on. The
    motion estimator needs no torque channel at all, so it supplies the lag -- which is what makes the
    cross-check above readable, because that slope has to be taken at the right shift.
    """
    _log, gap = gap_at(2, channel="current", torque_constant_error=error)
    lat = gap["latency"]
    assert lat["identified"] is True, lat.get("not_identified_because")
    assert lat["source"] == "motion_reconstruction", (
        "the torque channel is 2.5x out and must NOT be the source of a claimed delay here")
    assert lat["delay_ms"] == pytest.approx(2 * MS_PER_TICK)
    assert _used(gap["torque_channel"])["ratio_identified_over_used"] == pytest.approx(error, rel=0.05)


# =========================================================================================================
# 4. THE REFUSALS.
# =========================================================================================================

def test_a_magnitude_only_current_channel_is_refused_entirely(go2, plan, gap_at):
    """kt * |I| is an actuator that can only push one way, and several drivers report exactly that.

    The subtlety that made this a whole-channel refusal rather than a per-joint one, found by measuring: the
    sign gate can only fire on a joint that was DRIVEN BOTH WAYS, so on this two-joint plan the ten idle
    joints passed it for want of any motion while the two that carry every bit of information were refused.
    Converting the ten and zeroing the two "succeeded" and destroyed the log -- a zero torque column does not
    read as unknown, it reads as "this actuator applied nothing". So any refusal refuses the channel, and the
    log degrades to position-only, which still yields the delay.
    """
    import numpy as np

    from virturoid.services.sysid import measure_gap

    log, _ = gap_at(2, channel="current")
    magnitude = {**log, "i_meas": np.abs(np.asarray(log["i_meas"])).tolist()}
    gap = measure_gap(go2, magnitude, plan=plan, measure_noise_floor=False)

    rec = gap["torque_channel"]
    assert rec["converted"] is False
    assert set(rec["joints_refused"]) >= set(ONLY_JOINTS), rec["joints_refused"]
    assert "MAGNITUDE" in rec["per_joint"][ONLY_JOINTS[0]]["refused_because"]
    assert gap["attribution"]["available"] is False, (
        "a refused channel must not leave a zeroed torque array behind for the regressor to fit")
    assert gap["latency"]["identified"] is True and gap["latency"]["delay_ms"] == pytest.approx(20.0), (
        "...and the log is still worth something: position-only still gives the actuation delay")


def test_the_gear_ratio_cancels_out_of_the_derivation(go2):
    """A property of the derivation worth pinning, because it is why it is usable at all.

    ``kt_out = efficiency * V / omega_out`` -- the gear ratio divides out between referring the constant to
    the output and referring the no-load speed to the motor. Gear ratio is the catalog field that varies most
    between a real part and a stand-in for it, so a formula that does not use it is meaningfully more robust
    than one that does. Two catalog parts with the same voltage and output speed and very different ratios
    must give the same constant.
    """
    import dataclasses

    from virturoid.services.component_catalog import ACTUATORS
    from virturoid.services.sysid.torque_channel import derived_torque_constant

    base = ACTUATORS[0]
    a = dataclasses.replace(base, voltage_v=48.0, max_speed_radps=20.0, gear_ratio=6.0)
    b = dataclasses.replace(base, voltage_v=48.0, max_speed_radps=20.0, gear_ratio=100.0)
    assert derived_torque_constant(a)["nm_per_a"] == pytest.approx(derived_torque_constant(b)["nm_per_a"])


def test_a_log_with_neither_channel_says_so_rather_than_converting_nothing(go2, plan, gap_at):
    """The no-op path: a position-only log must not produce a conversion record that reads like one."""
    from virturoid.services.sysid.torque_channel import convert_current_to_torque

    log, _ = gap_at(2, channel="position_only")
    out, rec = convert_current_to_torque(go2, log)
    assert out is log
    assert rec["converted"] is False
    assert "neither tau_meas nor a motor-current channel" in rec["why"]
