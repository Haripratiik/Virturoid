"""The tool issued a confident DRIVETRAIN verdict on a log that is a record of a divergent integration.

Every other gate in ``sysid.fit`` decides WHICH of the customer's parts is wrong. This file is about the
question that has to come first, and about the day it was found not to be asked.

WHAT WAS MEASURED (2026-08-13, composed 8-legged spider, 24 joints, 35 s excitation, delay 0, n_boot 64,
bit-identical at 96, through the shipped ``fit_parameters``):

    armature-only +0.009   improvement_x 1.575   torque rival explains_x 1.023   REFUSED as global_scale
    armature-only +0.050   improvement_x 2.057   torque rival explains_x 1.218   REFUSED as global_scale

Both are CORRECT calibrations, and both were refused with a verdict telling the customer to go and check their
gear ratio and torque constant. That alone is a false refusal by the check next door. But the number that
matters is not the rival's:

    max|q_meas| = 298.9 rad   against a commanded envelope of 1.634 rad   -- a ratio of 182.9

Forty-seven revolutions of a joint that was asked for a quarter turn, with MuJoCo emitting "Nan, Inf or huge
value in QACC" while the log was being generated, and EVERY injection on that body doing the same -- including
the control with nothing wrong with it at all (1675). The plant diverged. The log is a record of that, not a
measurement of a robot, and no parameter verdict taken from it can mean anything.

So the defect is not that the torque rival lost. It is that the tool ACCEPTED the log and named a part.

=========================================================================================================
AND THEN THE GUARD'S OWN BOUND TURNED OUT TO BE SIZED ON A SURVEY TAKEN AT DELAY 0 (2026-08-13, same day).
=========================================================================================================

``LOG_EXCURSION_RATIO_MAX`` shipped at 5.80, the geometric midpoint of "sane 0.874-1.405 over 56 logs on 7
bodies" and "degenerate 23.974-11244.9 on the spider". **Every one of those 56 sane logs was generated with
``delay_ticks=0``** -- while ``synthetic_hardware.DEFAULT_DELAY_TICKS`` is 2, ``tools.simulate_bench_log``
defaults to ``delay_ms=20``, and ``tests/test_sysid_delay_wedge.py`` validates at 0/20/40 ms *because
actuation delay is the dominant sim-to-real term*. The guard was sized with its most dangerous axis pinned at
that axis's zero.

RE-SURVEYED across 19 composed bodies x 8 injections x 11 delays (0 ... 640 ms), 1672 logs:

    max excursion ratio      0ms    20ms    40ms   100ms   200ms   300ms   640ms
      composed millipede    1.483   5.671   6.268   6.587   6.331   6.444   6.237
      composed centipede    1.078   5.322   5.692   6.139   6.513   6.029   6.017
      composed rover        0.872   0.892   0.912   4.151   9.748  18.922  50.076
      composed 6-axis arm   0.985   1.079   1.371   2.919   6.644   8.861   9.144
      SPIDER, minimum      23.974  47.023  28.023  16.118 411.462 633.899 695.398

At 5.80 that is a REAL FALSE REFUSAL AT 40 ms -- five composed-millipede logs, one of them the control with
NOTHING WRONG WITH IT (5.859) -- inside the band this package validates. And the escape hatch ("those had
really diverged") is closed: re-running ``bench_rig.pd_replay``'s loop with the MjData kept shows **0 of 1368
non-spider logs emit a single MuJoCo instability warning at any delay to 400 ms**.

THE RE-CENTRED BOUND, by the rule that sized ``LINK_INERTIA_RIVAL_IMPROVEMENT_X``:

    SANE, 0-200 ms, n=1064      0.554 ... 9.748    ceiling = rover / nothing wrong @ 200 ms
    DEGENERATE, 0-200 ms, n=56 16.118 ... 51005    floor   = spider / armature +0.050 @ 100 ms
    sqrt(9.748 x 16.118) = 12.535  ->  LOG_EXCURSION_RATIO_MAX = 12.5

and it is SCOPED: past 200 ms the populations overlap (rover 18.922 at 300 ms, above the spider's 16.118), so
no threshold on this ratio can be right there and ``LOG_EXCURSION_VALID_DELAY_MS`` ships that limit.

These tests are slow (real fits on a real composed body). They are the acceptance evidence, so they run the
shipped thing rather than a stub.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("VIRTUROID_DISABLE_GAIT_HINTS", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="system identification needs MuJoCo")

BUDGET_S = 35.0

#: Bodies whose logs are SANE, and the injections measured on each. ``inertia_scale`` is in here deliberately:
#: it is a real misspecification that the link-inertia check must still catch, so if this guard ever fired on
#: it the fix would have bought honesty with a hole.
_SANE_BODIES = ["a four legged robot dog", "a small robot cat", "a humanoid robot"]
_SANE_CASES = [
    ("nothing wrong", {"perturbation": {}}),
    ("default injection", {"perturbation": {"frictionloss": 0.08, "damping": 0.6, "armature": 0.03}}),
    ("armature +0.009", {"perturbation": {"armature": 0.009}}),
    ("inertia_scale x40", {"perturbation": {}, "inertia_scale": 40.0}),
    ("inertia_scale x100", {"perturbation": {}, "inertia_scale": 100.0}),
]

#: THE BODY THAT FOUND THE DELAY DEFECT, and the delays it is checked at. The composed millipede's excursion
#: ratio goes 1.483 -> 6.268 between 0 and 40 ms, so at the retired 5.80 line it was FALSE-REFUSED at the very
#: delay ``test_sysid_delay_wedge.py`` validates. 20 ms is the package's own default injection.
_DELAY_BODY = "a millipede robot"
_DELAY_MS = [0, 20, 40, 100, 200]


def _log(gene, plan, delay_ticks=0, **kw):
    from virturoid.services.sysid.synthetic_hardware import synthetic_hardware_log
    _, lg = synthetic_hardware_log(gene, delay_ticks=delay_ticks, plan=plan, **kw)
    return lg


def _plausibility_of(gene, lg, plan=None, with_model=True):
    """Run the guard the way ``fit_parameters`` does -- on the ALIGNED log, in DOF space, WITH THE MODEL.

    The model is what carries ``jnt_limited``, and passing it is the whole of the unbounded-joint repair.
    ``with_model=False`` reproduces the pre-repair call, which is also the call a customer's bare arrays make.
    """
    from virturoid.services.sysid.bench_rig import bench_model, joint_dof_map
    from virturoid.services.sysid.fit import log_plausibility
    from virturoid.services.sysid.gap_report import _align_log

    model, _ = bench_model(gene)
    dofs = joint_dof_map(model, gene)
    aligned, meta = _align_log(lg, model, dofs)
    assert aligned is not None, meta
    return log_plausibility(aligned, dofs, model if with_model else None)


@pytest.fixture(scope="module")
def spider():
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.sysid import build_excitation, fit_parameters
    g = compose_robot("an eight legged spider robot", llm=None)
    p = build_excitation(g, budget_s=BUDGET_S)
    out = {}
    for label, arm in (("armature +0.009", 0.009), ("armature +0.050", 0.05)):
        lg = _log(g, p, perturbation={"armature": arm})
        out[label] = fit_parameters(g, lg, plan=p, n_boot=64, measure_delay=False)
    return out


def test_a_divergent_log_is_refused_as_a_log_and_not_as_the_customers_drivetrain(spider):
    """THE ACCEPTANCE TEST, and the assertion that matters is the NEGATIVE one.

    It is not enough that these fits are refused -- they already were, as ``global_scale``. What has to be true
    is that the refusal no longer names a part of the customer's robot, because this experiment cannot tell a
    wrong gear ratio from an unstable integrator and the previous verdict said it could.
    """
    for label, fit in spider.items():
        app = fit["application"]
        assert app["refused_by"] == "implausible_log", (
            f"{label}: refused_by is {app['refused_by']!r}. Measured before the guard: 'global_scale' on both "
            f"of these CORRECT armature-only calibrations, at torque rival explains_x 1.023 and 1.218. "
            f"log_plausibility: {fit.get('log_plausibility')}")
        low = app["verdict"].lower()
        for named in ("gear ratio", "torque constant", "inertia tensor", "your cad"):
            assert named not in low, (
                f"{label}: the refusal still names the customer's hardware ({named!r}) on a log that cannot "
                f"support any verdict. That is the defect, restated: {app['verdict']}")
        assert "log" in low and "cannot be ruled on" in low, app["verdict"]


def test_no_rival_is_simulated_on_a_log_that_cannot_be_ruled_on(spider):
    """A rival's number IS a confident answer. Spending one on a divergent log manufactures the evidence the
    verdict above refuses to give -- and it costs ~10 replays to do it."""
    for label, fit in spider.items():
        assert not (fit["global_scale"] or {}).get("rival"), (
            f"{label}: a torque rival was still simulated on an implausible log: "
            f"{fit['global_scale'].get('rival')}")
        assert not (fit.get("link_inertia") or {}).get("rival"), (
            f"{label}: an inertia rival was still simulated on an implausible log")
        assert (fit["global_scale"] or {}).get("suspected") is False, label


def test_the_excursion_ratio_reproduces_on_the_body_that_found_this(spider):
    """The measurement itself, pinned. If the composed spider ever becomes a stable body this test should be
    re-pointed at whatever else diverges -- not deleted, because then the guard has nothing behind it."""
    lp = spider["armature +0.009"]["log_plausibility"]
    assert lp["finite"] is True, "this log's failure is EXCURSION, not NaN -- see the docstring"
    assert lp["excursion_ratio"] > 100.0, (
        f"measured 182.941 (298.9 rad against a 1.634 rad commanded envelope); now {lp['excursion_ratio']}")
    assert lp["commanded_envelope_rad"] == pytest.approx(1.634, abs=0.05), lp
    assert lp["plausible"] is False, lp
    # The number that decides how far the OTHER spider case is from the line: it reads 23.974 at delay 0, and
    # 23.974 is the SMALLEST value the spider takes anywhere at or below 40 ms.
    lp2 = spider["armature +0.050"]["log_plausibility"]
    assert lp2["excursion_ratio"] > 10.0, f"measured 23.974; now {lp2['excursion_ratio']}"


@pytest.mark.parametrize("prompt", _SANE_BODIES)
def test_a_sane_log_is_never_refused_including_the_misspecifications_that_must_still_be_caught(prompt):
    """THE OTHER HALF. A guard that refuses real logs is worse than the defect, and a guard that refuses the
    ``inertia_scale`` family would silently delete the third check's whole catch population."""
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.sysid import build_excitation
    from virturoid.services.sysid.fit import LOG_EXCURSION_RATIO_MAX

    g = compose_robot(prompt, llm=None)
    p = build_excitation(g, budget_s=BUDGET_S)
    seen = {}
    for label, kw in _SANE_CASES:
        lp = _plausibility_of(g, _log(g, p, **kw), p)
        seen[label] = lp["excursion_ratio"]
        assert lp["plausible"] is True, (
            f"{prompt}/{label}: a SANE log was refused as implausible ({lp['excursion_ratio']}x against "
            f"{LOG_EXCURSION_RATIO_MAX}x). Do not raise the threshold to fix this -- re-measure BOTH "
            f"populations across BOTH bodies AND delays, because one of them has moved. {lp}")
    assert max(seen.values()) < 2.0, (
        f"{prompt}: this body's delay-0 population has drifted; it sat at 0.90-1.41 when the bound was "
        f"re-sized: {seen}")


# ---------------------------------------------------------------------------------------------------------
# THE DELAY AXIS -- the one the first survey held at zero, and the reason the bound moved.
# ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("delay_ms", _DELAY_MS)
def test_the_guard_does_not_refuse_a_sane_log_AT_THE_DELAY_THIS_PACKAGE_INJECTS(delay_ms):
    """THE REGRESSION TEST FOR THE DEFECT ITSELF, and it fails at the retired threshold.

    ``LOG_EXCURSION_RATIO_MAX`` was sized on a sane survey taken at ``delay_ticks=0`` while the harness's own
    default is 2 ticks and the wedge validates at 40 ms. On the composed millipede the ratio goes 1.483 (0 ms)
    -> 5.671 (20 ms) -> 6.268 (40 ms), so at the old 5.80 line FIVE of this body's eight injections were
    false-refused at 40 ms -- including the one with nothing wrong with it.

    Every case here is a CORRECT model or a misspecification the OTHER checks own. None of them is a divergent
    plant: re-running the replay with MuJoCo's warning counters read shows zero instability warnings on this
    body at any of these delays.
    """
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.sysid import build_excitation
    from virturoid.services.sysid.fit import LOG_EXCURSION_RATIO_MAX, LOG_EXCURSION_VALID_DELAY_MS

    assert delay_ms <= LOG_EXCURSION_VALID_DELAY_MS, "this test must stay inside the bound's declared scope"
    g = compose_robot(_DELAY_BODY, llm=None)
    p = build_excitation(g, budget_s=BUDGET_S)
    ticks = int(round(delay_ms * float(p["controller"]["control_hz"]) / 1000.0))
    seen = {}
    for label, kw in _SANE_CASES:
        lp = _plausibility_of(g, _log(g, p, delay_ticks=ticks, **kw), p)
        seen[label] = lp["excursion_ratio"]
        assert lp["plausible"] is True, (
            f"millipede/{label} at {delay_ms} ms: a SANE log was refused ({lp['excursion_ratio']}x against "
            f"{LOG_EXCURSION_RATIO_MAX}x). This is the exact defect the 5.80 threshold had. The fix is NOT to "
            f"raise the line -- it is to re-take both populations across delay and re-centre. {lp}")
    if delay_ms >= 40:
        # ...and the measurement that makes the point: at 40 ms this body is ABOVE the retired 5.80 line.
        assert max(seen.values()) > 5.8, (
            f"the composed millipede at {delay_ms} ms no longer exceeds the RETIRED 5.80 threshold "
            f"({seen}). That threshold's false refusal is the reason this file's bound moved; if this body "
            f"stopped reproducing it, re-point the test at whatever body now does rather than deleting it")


#: THE EDGES THE SHIPPED LINE IS SIZED ON, re-measured 2026-08-13 after BOTH of the previous pair turned out
#: to be wrong -- and they are kept here as module constants so a future re-measurement changes one place.
#:
#:   SANE ceiling      6.721  composed MILLIPEDE / frictionloss +0.030 @ 60 ms.
#:                            n = 2864 readings, 18 WHEEL-FREE composed bodies x 8 injection families x the
#:                            0-200 ms delay grid in 10 ms steps (millipede and centipede on 20 ms steps,
#:                            for cost).
#:   DEGENERATE floor 14.648  composed SPIDER / armature +0.050 @ 110 ms, n = 168 on the same grid.
#:
#: WHAT MOVED AND WHY, because both edges of the PREVIOUS pair (9.748 / 16.118) were artefacts:
#:   * 9.748 was a composed WHEELED ROVER. Its four wheels carry jnt_limited=False, so their position
#:     integrates and the ratio has no ceiling on them at all. The rover is not a high sane reading; it is a
#:     body this statistic cannot be formed on, and it is now reported NOT MEASURABLE.
#:   * 16.118 was the minimum over an 11-point delay grid that skipped 50/80/110 ms. On the 10 ms grid -- the
#:     finest a 100 Hz controller can represent -- the spider reads 14.897 / 14.834 / 14.648 there.
_SANE_CEILING = 6.721
_DEGENERATE_FLOOR = 14.648


def test_the_bound_sits_between_the_two_RE_MEASURED_populations_with_the_margins_it_actually_has():
    """No invented constants, and the margins are asserted AS MEASURED rather than as a symmetry that may not
    hold. Both edges come from the same wheel-free 8-injection sweep across the 0-200 ms delay grid.

    This test used to hardcode 9.748 / 16.118 and assert the line was EQUIDISTANT from them. Both numbers were
    wrong (see the module constants above) and the equidistance was an accident of the rule that picked the
    line, not a property worth pinning: the rule is "geometric midpoint of the measured gap", and what a reader
    needs from this test is how much room the line actually has on each side.
    """
    import numpy as np

    from virturoid.services.sysid.fit import (
        LOG_EXCURSION_RATIO_MAX,
        LOG_EXCURSION_VALID_DELAY_MS,
        log_plausibility,
    )

    midpoint = float(np.sqrt(_SANE_CEILING * _DEGENERATE_FLOOR))
    assert LOG_EXCURSION_RATIO_MAX == pytest.approx(midpoint, abs=0.05), (
        f"LOG_EXCURSION_RATIO_MAX = {LOG_EXCURSION_RATIO_MAX} is no longer the geometric midpoint of the "
        f"measured wheel-free sane ceiling ({_SANE_CEILING}, composed millipede / frictionloss +0.030 @ 60 ms) "
        f"and degenerate floor ({_DEGENERATE_FLOOR}, composed spider / armature +0.050 @ 110 ms), which is "
        f"sqrt({_SANE_CEILING} x {_DEGENERATE_FLOOR}) = {midpoint:.3f}. If you moved one edge, re-measure the "
        f"OTHER before moving this line -- both of the previous pair were wrong at once.")
    # THE MARGINS, ASSERTED AS MEASURED. Not equidistance: that held for the retired pair by construction and
    # is not what protects anyone. What protects them is that the line is strictly inside the gap with room.
    above = LOG_EXCURSION_RATIO_MAX / _SANE_CEILING
    below = _DEGENERATE_FLOOR / LOG_EXCURSION_RATIO_MAX
    assert above == pytest.approx(1.473, abs=0.02), f"margin above the sane ceiling is now {above:.3f}x"
    assert below == pytest.approx(1.480, abs=0.02), f"margin below the degenerate floor is now {below:.3f}x"
    assert _SANE_CEILING < LOG_EXCURSION_RATIO_MAX < _DEGENERATE_FLOOR
    assert LOG_EXCURSION_VALID_DELAY_MS == 200.0

    lp = log_plausibility({"q_meas": np.zeros((32, 2)), "q_cmd": np.ones((32, 2)) * 0.5})
    text = lp["sampled_range_on_sane_logs"].lower()
    for token in ("0.584", "6.721", "14.648", "geometric midpoint", "composed bodies", "delay"):
        assert token in text, f"the shipped bound stopped stating {token!r}: {lp}"
    # THE RETIRED FIGURES HAVE TO STAY NAMED AT THE POINT OF USE, not quietly deleted -- this package's
    # standing rule about superseded numbers. There are now two generations of them.
    assert "0.874" in text and "retired" in text, (
        f"the disclosure no longer marks the delay-0 survey as retired: {lp['sampled_range_on_sane_logs']}")
    assert "9.748" in text and "16.118" in text, (
        f"the disclosure no longer names the RETIRED 12.5 pair. 9.748 was a wheeled rover and 16.118 was a "
        f"coarse-grid artefact; a reader who meets either elsewhere must be able to find that here: "
        f"{lp['sampled_range_on_sane_logs']}")
    # AND THE SCOPE OF THE RANGE, which is the thing this field twice got wrong by omission.
    assert "eight more families" in text or "eight more" in text, (
        f"the disclosure states a range without saying which injection families it covers: {text}")
    for token in ("22.181", "18.778", "does not separate"):
        assert token in text, (
            f"the disclosure stopped naming the family this statistic CANNOT separate (dissipation removed): "
            f"{token!r} missing")
    assert lp["excursion_ratio_threshold_x"] == LOG_EXCURSION_RATIO_MAX
    assert lp["valid_delay_window_ms"] == LOG_EXCURSION_VALID_DELAY_MS
    assert "nothing is measured" in lp["outside_the_valid_delay_window"].lower()


def test_the_disclosure_names_the_unbounded_joint_blind_spot():
    """A guard that divides a peak POSITION by a commanded envelope assumes every joint's position is bounded.
    That assumption is invisible in the output unless it is written down."""
    import numpy as np

    from virturoid.services.sysid.fit import application_gate, log_plausibility

    lp = log_plausibility({"q_meas": np.zeros((16, 2)), "q_cmd": np.ones((16, 2)) * 0.5})
    txt = application_gate({"improvement_x": 3.0}, 4, log_plausibility=lp)["what_this_gate_does_not_catch"]
    low = txt.lower()
    for token in ("wheel", "continuous-rotation", "integrates without bound", "bench_model"):
        assert token in low, f"the shipped disclosure does not name the unbounded-joint blind spot: {token!r}"


# ---------------------------------------------------------------------------------------------------------
# THE UNBOUNDED JOINT -- a FALSE REFUSAL that deleted a real drivetrain finding, and the repair.
# ---------------------------------------------------------------------------------------------------------

_ROVER = "a six wheeled rover"
#: The regression, measured through the shipped ``fit_parameters`` before the repair. A composed six-wheeled
#: rover carrying a genuine 25% torque-scale error and NOTHING else wrong:
#:
#:     0 ms    ratio  0.812   plausible True    refused_by 'global_scale'    implied g 1.2974 (truth 1.25)
#:   200 ms    ratio 12.598   plausible FALSE   refused_by 'implausible_log' -- THE FINDING IS DELETED
#:
#: 200 ms is INSIDE ``LOG_EXCURSION_VALID_DELAY_MS``, and the rover's log carries zero MuJoCo instability
#: warnings. The ratio crossed only because a wheel's position integrates: 0.28 rad of travel at 0 ms and
#: 4.41 rad at 200 ms against a commanded envelope of 0.35 rad.
_ROVER_DELAYS_MS = [0, 40, 100, 200]


def _rover_fit(gene, plan, delay_ms):
    from virturoid.services.sysid import fit_parameters
    ticks = int(round(delay_ms * float(plan["controller"]["control_hz"]) / 1000.0))
    lg = _log(gene, plan, delay_ticks=ticks, perturbation={}, torque_scale=1.25)
    return fit_parameters(gene, lg, plan=plan, n_boot=64, measure_delay=True)


@pytest.fixture(scope="module")
def rover():
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.sysid import build_excitation
    g = compose_robot(_ROVER, llm=None)
    return g, build_excitation(g, budget_s=BUDGET_S)


def test_a_wheeled_body_is_NOT_MEASURABLE_rather_than_refused(rover):
    """THE REGRESSION TEST. The guard must not delete a real finding on a body it cannot form its statistic on.

    Every joint on this body is a continuous-rotation hinge, so the excursion ratio has no bound and there is
    no channel to rule on. That is NOT MEASURABLE -- neither a pass nor a refusal -- and the fit must go on to
    be judged by the checks that CAN rule on it.
    """
    gene, plan = rover
    for delay_ms in _ROVER_DELAYS_MS:
        fit = _rover_fit(gene, plan, delay_ms)
        lp = fit["log_plausibility"]
        assert lp["measurable"] is False, (
            f"{delay_ms} ms: every joint on this body is unbounded, so the ratio cannot be formed: {lp}")
        assert lp["plausible"] is None, (
            f"{delay_ms} ms: NOT MEASURABLE must be neither a pass nor a refusal, and `plausible` is the field "
            f"every consumer branches on. It is {lp['plausible']!r}: {lp}")
        assert lp["excursion_ratio"] is None, lp
        assert fit["application"]["refused_by"] != "implausible_log", (
            f"{delay_ms} ms: the log guard still refuses this rover. MEASURED before the repair: ratio 12.598 "
            f"at 200 ms, refused_by 'implausible_log', deleting a torque_scale finding whose implied g was "
            f"1.2974 against a truth of 1.25. {lp}")


def test_the_excluded_channels_are_NAMED_and_counted_not_silently_dropped(rover):
    """A customer must be able to see WHICH channels this guard did not look at. Dropping them quietly would
    trade one dishonesty for another."""
    gene, plan = rover
    lp = _rover_fit(gene, plan, 200)["log_plausibility"]
    assert lp["unbounded_joints_checked"] is True, lp
    assert lp["unbounded_joint_count"] == len(lp["unbounded_joints"]) > 0, lp
    assert all(n.startswith("wheel") for n in lp["unbounded_joints"]), (
        f"the excluded channels are supposed to be this rover's wheels: {lp['unbounded_joints']}")
    assert "jnt_limited" in (lp.get("not_measurable_because") or ""), lp
    for name in lp["unbounded_joints"]:
        assert name in lp["not_measurable_because"], f"{name} is excluded but not named: {lp}"


def test_the_drivetrain_finding_survives_at_the_delays_where_the_fit_itself_does(rover):
    """The other half: the repair must not have bought silence. Where this fit still clears the TRACKING gate,
    the torque-scale verdict has to come back with the right number.

    MEASURED across 0/40/100/200 ms on the composed six-wheeled rover at torque_scale=1.25:

        0 ms    improvement_x 7.565  refused_by 'global_scale'  implied g 1.2974  rival g 1.2488
       40 ms    improvement_x 3.349  refused_by 'global_scale'  implied g 1.2277  rival g 1.2277
      100 ms    improvement_x 0.835  refused_by None            -- the TRACKING gate refuses, correctly
      200 ms    improvement_x 1.000  refused_by None            -- likewise

    At 100 and 200 ms the actuation delay is not recovered on this body (the search returns 80 ms against a
    true 200 ms), so the replay carries a timing error no parameter can remove and the fit genuinely does not
    improve tracking. That is a TRUE refusal by a different gate and it is not this guard's to fix; what
    matters here is that it is no longer dressed up as a broken log.
    """
    gene, plan = rover
    fit0 = _rover_fit(gene, plan, 0)
    assert fit0["application"]["refused_by"] == "global_scale", fit0["application"]["verdict"]
    assert fit0["global_scale"]["implied_torque_scale_g"] == pytest.approx(1.25, rel=0.1), (
        f"the drivetrain finding is back but its number moved; measured 1.2974 against a truth of 1.25: "
        f"{fit0['global_scale']}")
    assert fit0["global_scale"]["rival"]["torque_scale_g"] == pytest.approx(1.25, rel=0.1), fit0["global_scale"]
    # ...and at every delay in the window the drivetrain SIGNATURE is still reported even where the tracking
    # gate refuses first, because deleting it is what the regression did.
    for delay_ms in _ROVER_DELAYS_MS:
        gs = _rover_fit(gene, plan, delay_ms)["global_scale"]
        assert gs.get("implied_torque_scale_g"), f"{delay_ms} ms: no implied torque scale is reported: {gs}"


def test_a_PARTLY_unbounded_body_measures_the_bounded_channels_and_names_the_rest():
    """The case the composed corpus cannot produce -- a robot with SOME continuous joints -- built from a two
    joint model so the exclusion logic is pinned rather than inferred.

    The wheel spins to 20 rad while the elbow tracks a 0.2 rad command perfectly. Measured over every channel
    that is 100x; measured over the channels this statistic is defined on it is 1.0, and the wheel is named.
    """
    import mujoco
    import numpy as np

    from virturoid.services.sysid.fit import log_plausibility

    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody><body name="hub">
      <joint name="wheel" type="hinge" axis="0 1 0"/>
      <geom type="box" size=".05 .05 .05"/>
      <body name="link" pos="0 0 .1">
        <joint name="elbow" type="hinge" axis="0 1 0" limited="true" range="-1 1"/>
        <geom type="capsule" fromto="0 0 0 0 0 .1" size=".02"/>
      </body>
    </body></worldbody></mujoco>""")
    dofs = {"wheel": 0, "elbow": 1}
    n = 800
    t = np.arange(n) * 0.002
    q = np.zeros((n, 2))
    q[:, 0] = 20.0 * t / t[-1]
    q[:, 1] = 0.2 * np.sin(2 * np.pi * 0.5 * t)
    cmd = np.zeros((n, 2))
    cmd[:, 1] = q[:, 1]

    lp = log_plausibility({"q_meas": q, "q_cmd": cmd}, dofs, model)
    assert lp["measurable"] is True and lp["plausible"] is True, lp
    assert lp["excursion_ratio"] == pytest.approx(1.0, abs=0.05), (
        f"the ratio was formed over the wheel as well as the elbow: {lp}")
    assert lp["worst_joint"] == "elbow", lp
    assert lp["unbounded_joints"] == ["wheel"] and lp["unbounded_joint_count"] == 1, lp
    assert "wheel" in lp["excluded_because_unbounded"], lp
    # ...and the same arrays WITHOUT the model, which is what a customer's bare log is: nothing can be
    # excluded, so the blind spot is still open and the output says which of the two it is.
    blind = log_plausibility({"q_meas": q, "q_cmd": cmd}, dofs)
    assert blind["unbounded_joints_checked"] is False, blind
    assert blind["plausible"] is False and blind["excursion_ratio"] > 50.0, blind
    assert "no compiled model" in blind["excluded_because_unbounded"], blind


def test_a_spinning_wheel_is_a_SANE_log_that_this_guard_refuses():
    """THE BLIND SPOT, DEMONSTRATED rather than asserted -- and it is a FALSE REFUSAL, not a miss.

    A drive wheel under a velocity loop has no position setpoint and no joint limit: its logged angle
    integrates without bound while the commanded envelope stays small. Nothing is wrong with the robot, the
    plant or the log, and the ratio is unbounded by construction.

    It cannot be produced by this package's own sim2sim gate -- ``bench_rig.bench_model`` welds the base and
    removes the floor, so nothing rolls and every joint is driven to a bounded position setpoint -- so it is
    built from arrays, which is exactly the form a customer's log arrives in.
    """
    import numpy as np

    from virturoid.services.sysid.fit import log_plausibility

    n = 2000
    t = np.arange(n) * 0.002
    q = np.zeros((n, 2))
    q[:, 0] = 5.0 * t                                 # the wheel: a steady 5 rad/s, 4 s of it
    q[:, 1] = 0.2 * np.sin(2 * np.pi * 0.5 * t)       # an ordinary joint, tracking perfectly
    cmd = np.zeros((n, 2))
    cmd[:, 1] = q[:, 1]
    lp = log_plausibility({"q_meas": q, "q_cmd": cmd}, {"wheel_0": 0, "arm_0": 1})
    assert lp["finite"] is True
    assert lp["plausible"] is False and lp["excursion_ratio"] > 50.0, (
        f"the blind spot has closed, which would be good news -- but it means this test and the disclosure "
        f"beside it are now wrong and both need re-writing: {lp}")
    assert lp["worst_joint"] == "wheel_0", lp
    # ...and the reason it is a blind spot rather than a bug: the arm alone is entirely ordinary.
    solo = log_plausibility({"q_meas": q[:, 1:], "q_cmd": cmd[:, 1:]})
    assert solo["plausible"] is True and solo["excursion_ratio"] == pytest.approx(1.0, abs=0.05), solo


def test_a_hold_only_log_is_not_mistaken_for_a_divergent_one():
    """The degenerate denominator, on a body where the floor does NOT bind -- so this pins the shape of the
    statistic, not the constant. Measured: 0.4175 rad of commanded hold on the dog, and a held joint's peak IS
    its command, so the ratio is 1.000 with or without the floor."""
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.sysid import build_excitation

    g = compose_robot("a four legged robot dog", llm=None)
    p = build_excitation(g, budget_s=BUDGET_S)
    lp = _plausibility_of(g, _log(g, p, perturbation={"armature": 0.03}, hold_only=True), p)
    assert lp["plausible"] is True, lp
    assert lp["excursion_ratio"] == pytest.approx(1.0, abs=0.05), (
        f"measured exactly 1.000 on the dog, cat and humanoid at delay 0; now {lp['excursion_ratio']}. A "
        f"hold-only log is the deliberately uninformative EXPERIMENT, and it must be refused for being "
        f"uninformative, not for looking divergent: {lp}")
    assert lp["commanded_envelope_rad"] > 0.1, (
        f"the dog's hold-only envelope is 0.4175 rad, 4.2x ABOVE LOG_ENVELOPE_FLOOR_RAD, so the floor does "
        f"NOT bind here -- the docstring that said it did was wrong and this assertion is what keeps the "
        f"corrected story true: {lp}")


def test_the_envelope_floor_is_load_bearing_on_a_HOLD_AT_THE_ZERO_POSE():
    """``LOG_ENVELOPE_FLOOR_RAD``'s real job, and the test that fails without it.

    Its docstring used to claim the floor is why hold-only "reads exactly 1.000 on the dog, the cat and the
    humanoid". MEASURED, those three hold envelopes are 0.4175 / 0.4175 / 0.6175 rad -- four to six times
    ABOVE the floor, so it never binds there and they read 1.000 with the floor at 1e-9. It binds on exactly
    two of the nineteen composed bodies surveyed, the OCTOPUS and the WHEELED ROVER, whose start pose IS the
    zero pose and whose hold-only commanded envelope is therefore EXACTLY 0.000 rad.

    That is the case reproduced here, from arrays: a stationary robot commanded to hold zero, with a
    millimetre of real sensor motion. Without the floor the denominator is zero and a perfectly ordinary
    stationary experiment is refused as a divergent integration.
    """
    import numpy as np

    from virturoid.services.sysid.fit import LOG_ENVELOPE_FLOOR_RAD, log_plausibility

    n = 512
    q = np.full((n, 3), 1e-3)
    cmd = np.zeros((n, 3))
    lp = log_plausibility({"q_meas": q, "q_cmd": cmd})
    assert lp["commanded_envelope_rad"] == 0.0, lp
    assert lp["plausible"] is True, (
        f"a HOLD AT THE ZERO POSE was refused as an implausible log. This is what LOG_ENVELOPE_FLOOR_RAD "
        f"exists for and it is measurable on real composed bodies -- the octopus and the wheeled rover both "
        f"hold at exactly 0.000 rad: {lp}")
    assert lp["excursion_ratio"] == pytest.approx(1e-3 / LOG_ENVELOPE_FLOOR_RAD, rel=1e-6), lp
    # WITHOUT the floor this same log divides by zero. Asserted directly so the constant cannot be deleted as
    # decorative -- which is precisely what its false docstring made it look like. (Re-computed here with the
    # floor set to nothing, which is the edit a reader tempted to delete the constant would actually make.)
    with np.errstate(divide="ignore", invalid="ignore"):
        unfloored = np.float64(1e-3) / np.float64(max(lp["commanded_envelope_rad"], 0.0))
    assert not np.isfinite(unfloored), (
        "the floor is no longer load-bearing on this input; re-derive it before removing it")

    # AND THE COST, measured on the 6-axis arm: its hold envelope is 0.01069 rad, BELOW the floor, so the
    # floor binds and the guard reads 0.1069 where the true ratio is 1.000 -- 9.4x more permissive.
    arm = log_plausibility({"q_meas": np.full((n, 2), 0.01069), "q_cmd": np.full((n, 2), 0.01069)})
    assert arm["excursion_ratio"] == pytest.approx(0.1069, rel=0.01), (
        f"the composed 6-axis arm's hold envelope is 0.01069 rad and the floor makes this guard 9.4x more "
        f"permissive there. That is the direction a false-refusal guard should err in, but it is a real loss "
        f"of sensitivity and the docstring says so: {arm}")


def test_a_non_finite_log_is_refused_without_naming_anything():
    """The other half of the guard. A NaN anywhere means the plant left the reals; there is nothing to rule on
    and nothing to name."""
    import numpy as np

    from virturoid.services.sysid.fit import log_plausibility

    n = 64
    q = np.zeros((n, 3))
    q[10, 1] = np.nan
    lp = log_plausibility({"q_meas": q, "q_cmd": np.zeros((n, 3)), "qd_meas": None, "tau_meas": None})
    assert lp["finite"] is False and lp["plausible"] is False, lp
    assert "non-finite" in lp["reading"], lp
