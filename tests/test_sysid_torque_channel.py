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


def test_the_fit_is_refused_when_the_torque_constant_is_wrong(go2, plan, gap_at):
    """...and the PARAMETERS are a magnitude, so they do not survive it -- and the gate catches that.

    The tracking gate is untouched at 1.5x. MEASURED at a 1.35x constant error on this plan: 1.272 / 1.276 /
    1.126 at 0 / 20 / 40 ms, all refused, against 1.903 / 1.789 / 1.502 with the constant right. So the same
    log yields a usable latency and an unusable fit, which is the honest split.
    """
    from virturoid.services.sysid import fit_parameters
    from virturoid.services.sysid.fit import MIN_TRACKING_IMPROVEMENT_X

    log, _gap = gap_at(2, channel="current", torque_constant_error=1.35)
    fit = fit_parameters(go2, log, plan=plan, n_boot=16)
    assert fit["ok"] is True
    assert fit["application"]["passed"] is False, (
        f"a fit taken through a 1.35x-wrong torque constant scored {fit['trajectory']['improvement_x']}x and "
        f"PASSED the {MIN_TRACKING_IMPROVEMENT_X}x gate; every parameter in it carries that factor")
    assert fit["torque_channel"]["converted"] is True, "and the fit must carry the conversion that caused it"


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
