"""Stage 2 of the calibration wedge: the fit, the interval, the application, and the L2 verdict.

Stage 1 measures the gap and says which parameters the experiment pinned. Stage 2 fits them and applies the
result, so an engineer's simulator actually moves toward their machine. The tests below are weighted the same
way Stage 1's were -- toward what the tool REFUSES to do, because recovering a perturbation is the easy half:

  * an interval whose COVERAGE is measured, not asserted (an error bar nobody checked is a decoration);
  * a parameter the data could not constrain coming back as "not identified", never as a confident number,
    with the follow-up experiment offered ONLY when a different experiment could actually fix it;
  * a calibration that a customer can see and undo, byte-for-byte;
  * an actuator-fidelity rung that is EARNED rather than declared -- and, on this build, correctly refused.

WE OWN NO HARDWARE. Every gate here is sim2sim: a second MuJoCo model with known perturbations stands in for
the robot. That validates the estimator and the plumbing and cannot validate the physics, because MuJoCo's own
modelling error cancels when both sides are MuJoCo. See ``WHAT_SIM2SIM_DOES_NOT_PROVE``.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("VIRTUROID_DISABLE_GAIT_HINTS", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="system identification needs MuJoCo")

INJECTED = {"frictionloss": 0.08, "damping": 0.6, "armature": 0.03}
DELAY_TICKS = 2
# The whole suite runs on a 35 s excitation rather than the 120 s default: the estimator is identical, the
# per-joint segments stay above their separability floor, and the fit costs a third as much wall clock.
BUDGET_S = 35.0


@pytest.fixture(scope="module")
def dog():
    from virturoid.services.morphology_composer import compose_robot
    return compose_robot("a four legged robot dog", llm=None)


@pytest.fixture(scope="module")
def plan(dog):
    from virturoid.services.sysid import build_excitation
    return build_excitation(dog, budget_s=BUDGET_S)


@pytest.fixture(scope="module")
def log(dog, plan):
    from virturoid.services.sysid.synthetic_hardware import synthetic_hardware_log
    _, lg = synthetic_hardware_log(dog, perturbation=INJECTED, delay_ticks=DELAY_TICKS, plan=plan)
    return lg


@pytest.fixture(scope="module")
def fit(dog, plan, log):
    from virturoid.services.sysid import fit_parameters
    return fit_parameters(dog, log, plan=plan, n_boot=96)


@pytest.fixture(scope="module")
def calibrated(dog, fit, log, plan):
    from virturoid.services.sysid import apply_calibration
    return apply_calibration(dog, fit, log=log, plan=plan)


# --------------------------------------------------------------------------------------------------------
# The fit: a POSTERIOR, never a point estimate.
# --------------------------------------------------------------------------------------------------------

def test_every_parameter_comes_back_with_an_interval_not_just_a_number(fit):
    """THE rule of this module. A point estimate on a physical parameter is consumed as knowledge and applied
    to the model; if the experiment could not resolve it, the interval is the only thing that says so."""
    assert fit["ok"]
    assert fit["joints"], "no joint was fitted at all"
    for name, row in fit["joints"].items():
        for p, cell in row["parameters"].items():
            lo, hi = cell["value_interval"]
            assert hi > lo, f"{name}/{p} reported a degenerate interval"
            assert lo <= cell["value"] <= hi, f"{name}/{p} estimate {cell['value']} outside its own interval"
            assert cell["interval_level"] == pytest.approx(0.90)
            assert "bootstrap" in cell["interval_method"]
            dlo, dhi = cell["delta_interval"]
            assert dlo <= cell["delta"] <= dhi


def test_the_injected_perturbation_comes_back_and_the_iteration_beats_one_linearization(dog, plan, log):
    """Recovery, and the reason Stage 2 iterates at all. One linearization is biased -- MuJoCo models
    frictionloss as a solver constraint with a stick regime, and the model's sensitivity is taken about the
    CURRENT value, so a perturbation that doubles a joint's friction is read low. Re-linearizing about the
    corrected model must measurably close that, or the extra passes are only costing wall clock."""
    import numpy as np

    from virturoid.services.sysid import fit_parameters

    def err(iterations):
        f = fit_parameters(dog, log, plan=plan, iterations=iterations, n_boot=32,
                           measure_trajectory=False, measure_delay=False)
        out = {}
        for p, injected in INJECTED.items():
            med = float(np.median([r["parameters"][p]["delta"] for r in f["joints"].values()]))
            out[p] = abs(med - injected) / injected
        return out

    one, three = err(1), err(3)
    assert three["frictionloss"] < one["frictionloss"], (
        f"iterating did not reduce the frictionloss bias: 1 pass {one['frictionloss']:.1%}, "
        f"3 passes {three['frictionloss']:.1%}")
    tol = {"frictionloss": 0.20, "damping": 0.10, "armature": 0.20}
    for p, e in three.items():
        assert e <= tol[p], f"{p} recovered {e:.1%} off (tolerance {tol[p]:.0%})"


def test_the_reported_interval_actually_contains_the_truth_at_the_rate_it_claims(dog, plan):
    """An error bar nobody measured is a decoration, and this one was measurably wrong the first time it was
    written: widening only by the estimator's floor AT THE PRIOR gave armature a coverage of 0.336 against a
    nominal 0.90 over 140 claims -- the estimates were good (2.5% median error) and the intervals were
    fiction. Widening by the estimator's bias AT THE FITTED POINT fixed it. Coverage at or above nominal is
    the pass condition; below it the tool is over-confident.

    The interval WIDTH is asserted alongside, because 100% coverage is trivially reachable by reporting a
    useless interval, and the refusal rate is asserted because it is trivially reachable by declining."""
    from virturoid.services.sysid import coverage_table

    cov = coverage_table(dog, trials=3, seed=11, budget_s=BUDGET_S, n_boot=64, iterations=2,
                         delay_ticks=DELAY_TICKS, plan=plan)
    assert cov["claims"] >= 20, f"too few claims to say anything: {cov['claims']}"
    assert cov["coverage_rate"] >= cov["nominal_level"], (
        f"intervals under-cover: {cov['coverage_rate']} of truths inside a "
        f"{cov['nominal_level']} interval. Misses: {cov['misses'][:3]}")
    assert cov["refusal_rate"] < 0.5, "coverage was bought by declining to answer"
    for p, row in cov["per_parameter"].items():
        assert row["median_interval_width_pct_of_truth"] < 100.0, (
            f"{p}'s interval is wider than the value it brackets ("
            f"{row['median_interval_width_pct_of_truth']}% of truth) -- covering by being useless")


def test_the_fit_moves_the_trajectory_and_that_is_measured_on_something_it_did_not_optimise(fit):
    """The fit minimises a TORQUE residual. A torque residual falling proves the optimiser ran. The number
    that decides whether the fit was worth applying is the POSITION gap against the log, which was not the
    objective -- and only the identified deltas are allowed into it."""
    t = fit["trajectory"]
    assert t["after_rms_rad"] < t["before_rms_rad"], "the fitted model is no closer to the log"
    assert t["improvement_x"] > 2.0, f"only {t['improvement_x']}x closer"
    assert "POSITION gap" in t["independent_of_the_objective"]


def test_the_delay_is_searched_on_the_fitted_model_and_recovered_exactly(fit, plan):
    """Stage 1 measured that a delay-only sweep cannot see past a dynamics error. Stage 2 IS that correction,
    so this is the sweep entitled to an answer -- and the answer must land on the injected tick, because the
    delay lives on a grid and anything else is a different answer, not a noisy one."""
    lat = fit["latency"]
    assert lat["identified"] is True, lat
    assert lat["delay_ms"] == pytest.approx(DELAY_TICKS * 1000.0 / plan["controller"]["control_hz"])
    assert lat["at_grid_edge"] is False
    assert "FITTED model" in lat["searched_on"]


def test_a_log_without_measured_torque_is_refused_rather_than_fitted_from_the_trajectory(dog, plan, log):
    """The quantity being fitted is a torque residual. Fitting the trajectory instead would produce numbers
    in the right units from an unidentifiable mixture of every parameter at once."""
    from virturoid.services.sysid import fit_parameters

    partial = dict(log)
    partial.pop("tau_meas")
    out = fit_parameters(dog, partial, plan=plan)
    assert out["ok"] is False
    assert "tau_meas" in out["error"]
    assert "joints" not in out


# --------------------------------------------------------------------------------------------------------
# What the tool REFUSES to fit.
# --------------------------------------------------------------------------------------------------------

def test_an_uninformative_experiment_fits_nothing_and_offers_the_experiment_that_would(dog, plan):
    """THE test, carried over from Stage 1 and made stricter. The robot is perturbed exactly as in the
    recovery gate -- a real gap exists -- but it is commanded to hold still, so nothing in the data can
    locate it. Stage 2 must not only decline every parameter; it must decline to APPLY anything, and it must
    hand back a real follow-up experiment instead of three confident numbers."""
    from virturoid.services.sysid import apply_calibration, calibration_report, fit_parameters
    from virturoid.services.sysid.synthetic_hardware import synthetic_hardware_log

    _, held = synthetic_hardware_log(dog, perturbation=INJECTED, delay_ticks=0, plan=plan, hold_only=True)
    out = fit_parameters(dog, held, plan=plan, n_boot=32, measure_delay=False)
    assert out["ok"]
    assert out["identified_pairs"] == 0, (
        f"named {out['identified_pairs']} parameters from a robot that never moved")
    assert out["refused"], "declined everything without recording a single reason"
    for row in out["refused"]:
        assert row["reasons"], f"{row['joint']}/{row['parameter']} declined with no reason"
        assert any("not excited" in r for r in row["reasons"]), "gave a statistical reason, not a physical one"

    cal = apply_calibration(dog, out, log=held, plan=plan)
    rep = calibration_report(cal)
    assert rep["calibrated"] is True, "the refusal itself should be recorded on the robot"
    assert rep["n_changes"] == 0, "a hold-only log moved a parameter"
    assert rep["n_refused"] > 0


def test_a_hold_only_log_leaves_the_compiled_model_byte_identical(dog, plan):
    """The refusal has to reach the SIMULATOR, not just the report. A calibration that identified nothing and
    still perturbed the model would be the same over-claim wearing a different hat."""
    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    from virturoid.services.sysid import apply_calibration, fit_parameters
    from virturoid.services.sysid.synthetic_hardware import synthetic_hardware_log

    _, held = synthetic_hardware_log(dog, perturbation=INJECTED, delay_ticks=0, plan=plan, hold_only=True)
    out = fit_parameters(dog, held, plan=plan, n_boot=32, measure_trajectory=False, measure_delay=False)
    cal = apply_calibration(dog, out, log=held, plan=plan)
    assert compile_gene_to_mjcf(cal, include_floor=False) == compile_gene_to_mjcf(dog, include_floor=False)


def test_a_follow_up_experiment_is_offered_only_when_a_different_experiment_could_help(dog, plan):
    """A parameter that WAS excited and still failed is confounded with another or is under the measurement
    floor; re-running the robot fixes neither. Sending an engineer back to their hardware for nothing is the
    most expensive output this tool has, so ``follow_up_experiment`` must return None in that case and a real
    narrowed plan in the other."""
    from virturoid.services.sysid import follow_up_experiment, fit_parameters
    from virturoid.services.sysid.synthetic_hardware import synthetic_hardware_log

    _, held = synthetic_hardware_log(dog, perturbation=INJECTED, delay_ticks=0, plan=plan, hold_only=True)
    starved = fit_parameters(dog, held, plan=plan, n_boot=32, measure_trajectory=False, measure_delay=False)
    nxt = follow_up_experiment(dog, starved)
    assert nxt is not None, "an unexcited robot was offered no way forward"
    assert nxt["duration_s"] > 0 and nxt["joints"]
    assert nxt["plan"]["ok"] and nxt["plan"]["scope"]["only_joints"] == sorted(nxt["joints"])
    # A REAL plan, not advice: it must be shorter than the full experiment and its commands must be realizable.
    assert nxt["duration_s"] <= plan["budget"]["duration_s"] + 1e-6

    everything_pinned = {"ok": True, "joints": {
        "j": {"joint": "j", "identified": ["damping"], "not_identified": ["armature"],
              "parameters": {"armature": {"reasons_not_identified": ["not separable: VIF = 41"],
                                          "suggested_experiment": None, "excitation_metric": "accel"}}}}}
    assert follow_up_experiment(dog, everything_pinned) is None, (
        "offered an experiment for a parameter that was excited and confounded -- re-running cannot fix that")


def test_the_follow_up_plan_commands_the_segment_mix_it_advertises(dog, plan):
    """A narrowed re-run declares its own segment fractions. If ``excitation_command_series`` kept reading the
    module constant, the plan would advertise a triangle-heavy sweep and command the default one -- and the
    engineer would run the wrong experiment on their robot and get the same refusal back."""
    import numpy as np

    from virturoid.services.sysid import build_excitation, excitation_command_series
    from virturoid.services.sysid.bench_rig import bench_model, joint_dof_map

    fractions = (("triangle", 0.60), ("chirp", 0.16), ("step_train", 0.18), ("hold", 0.06))
    model, _ = bench_model(dog)
    dofs = joint_dof_map(model, dog)
    target = sorted(dofs)[0]
    narrow = build_excitation(dog, budget_s=8.0, only_joints=[target], segment_fractions=fractions)
    assert [j["joint"] for j in narrow["joints"]] == [target]
    assert narrow["segment_fractions"][0] == ["triangle", 0.60]

    _, cmd = excitation_command_series(dog, narrow)
    col = cmd[:, dofs[target]]
    rows = cmd.shape[0]
    tri = col[:int(rows * 0.60)]
    # The triangle is a constant-|speed| ramp: its second difference is ~0 except at the turning points.
    curvature = float(np.mean(np.abs(np.diff(tri, 2))))
    chirp = col[int(rows * 0.60):int(rows * 0.76)]
    assert curvature < float(np.mean(np.abs(np.diff(chirp, 2)))), (
        "the first 60% of the command is not the triangle the plan says it is")
    for other in sorted(dofs)[1:]:
        assert float(np.ptp(cmd[:, dofs[other]])) == pytest.approx(0.0, abs=1e-9), (
            f"{other} moved in a plan scoped to {target}")


# --------------------------------------------------------------------------------------------------------
# The adversarial case the identifiability gates CANNOT catch: an error that is not in the model at all.
#
# Every gate in ``identifiability`` asks a question about the EXPERIMENT -- was this parameter excited, is it
# separable, is it above the floor. None of them asks whether the parameter is the right parameter. A hardware
# built with +30% link mass and inertia passes all three on armature, because link inertia and reflected
# inertia enter the joint equation through the same acceleration term, and comes back with confident numbers
# and intervals that exclude the true, unchanged value. What catches it is the one number measured on a
# quantity the fit did not optimise: does applying it make the simulator TRACK BETTER.
#
# THAT LAST SENTENCE IS TRUE OF THIS CASE AND WAS GENERALISED TOO FAR; the retraction is the section after
# next, and it is measured. The gate catches a misspecification whose signature is GRAVITY. It does not catch
# one the fitted parameters can mimic, and two of those clear it.
# --------------------------------------------------------------------------------------------------------

LINK_SCALE = 1.30


@pytest.fixture(scope="module")
def misspecified(dog, plan):
    """A fit against hardware whose LINKS are 30% heavier -- an error no fitted parameter can express."""
    from virturoid.services.sysid import fit_parameters
    from virturoid.services.sysid.synthetic_hardware import synthetic_hardware_log

    _, lg = synthetic_hardware_log(dog, perturbation={}, delay_ticks=0, plan=plan, link_scale=LINK_SCALE)
    return lg, fit_parameters(dog, lg, plan=plan, n_boot=64, measure_delay=False)


def test_a_fit_that_does_not_improve_tracking_is_refused_rather_than_silently_applied(dog, plan, misspecified):
    """THE gap this section exists for, and it was a live defect: ``improvement_x`` was computed, copied into
    the calibration record and printed in the brief, and NO consumer branched on it. A fit at 0.993x -- which
    had made tracking marginally worse -- wrote 15 misattributed values into the compiled model and satisfied
    the L2 joint-coverage requirement on the way through.

    The parameters here are 'identified' in every sense the experiment can check. They are still wrong, and
    the reported intervals exclude the truth. Only the trajectory says so."""
    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    from virturoid.services.sysid import MIN_TRACKING_IMPROVEMENT_X, apply_calibration, calibration_report
    from virturoid.services.sysid.bench_rig import bench_model, joint_dof_map

    log, fit = misspecified
    app = fit["application"]
    assert app["improvement_x"] < MIN_TRACKING_IMPROVEMENT_X, (
        f"the misspecified fit improved tracking {app['improvement_x']}x -- this test is no longer adversarial")
    assert app["passed"] is False and app["provisional"] is True

    # It really did produce confident, WRONG numbers: the gates upstream cannot see this.
    model, _ = bench_model(dog)
    dofs = joint_dof_map(model, dog)
    claimed = [(n, r["parameters"]["armature"]) for n, r in fit["joints"].items()
               if r["parameters"]["armature"]["identified"]]
    assert claimed, "the fixture stopped reproducing the failure; nothing was claimed at all"
    excluded = [n for n, c in claimed
                if not (c["value_interval"][0] <= float(model.dof_armature[dofs[n]]) <= c["value_interval"][1])]
    assert excluded, ("every interval covered the unchanged truth, so this fit is not the adversarial case "
                      "any more and the assertions below prove nothing")

    # ...and none of it reaches the simulator.
    cal = apply_calibration(dog, fit, log=log, plan=plan)
    assert compile_gene_to_mjcf(cal, include_floor=False) == compile_gene_to_mjcf(dog, include_floor=False), (
        "a fit that does not improve tracking was written into the compiled model")
    rep = calibration_report(cal)
    assert rep["calibrated"] is True, "the refusal must be recorded on the robot, not just in a log line"
    assert rep["provisional"] is True and rep["applied_to_model"] is False
    assert rep["n_changes"] == 0
    assert rep["n_withheld"] == len(claimed), "the withheld numbers were not reported"
    assert any("EVERY FITTED PARAMETER" in s for s in rep["not_applied"])


def test_the_l2_rung_cannot_be_earned_by_a_fit_that_left_the_simulator_no_closer(dog, plan, misspecified):
    """``l2_requirements`` had its 80%-of-joints requirement SATISFIED by that fit -- 14/14 joints carried an
    'identified, applied' parameter. Coverage now counts APPLIED parameters, of which a provisional fit has
    none, and the tracking requirement is named explicitly so the reason is legible instead of surfacing as a
    mysterious 0/14."""
    from virturoid.services.sysid import apply_calibration, l2_requirements

    log, fit = misspecified
    cal = apply_calibration(dog, fit, log=log, plan=plan)
    l2 = l2_requirements(cal)
    reqs = {r["requirement"]: r for r in l2["requirements"]}
    track = next(r for k, r in reqs.items() if k.startswith("applying the fit measurably improves"))
    assert track["met"] is False
    assert str(fit["application"]["improvement_x"]) in track["evidence"]
    cover = next(r for k, r in reqs.items() if k.startswith("at least 80%"))
    assert cover["met"] is False, "joint coverage was satisfied by parameters that never reached the model"
    assert "APPLIED to the model" in cover["evidence"]
    assert l2["earned"] is False


def test_a_withheld_fit_is_still_readable_and_takes_a_deliberate_opt_in_to_apply(dog, plan, misspecified):
    """Withheld is not discarded. A fit that vanished would be indistinguishable from one never run, and the
    engineer needs to see what it wanted to write in order to recognise it as a mass error. Applying it anyway
    stays possible and has to be asked for at the call site, in a named argument.

    And the OVERRIDE must not erase the thing it overrode. ``provisional`` is a fact about the fit, not about
    what we chose to do with it, so a forced fit stays provisional and gains ``applied_over_the_gate``."""
    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    from virturoid.services.sysid import apply_calibration, calibration_of, calibration_report

    log, fit = misspecified
    held = apply_calibration(dog, fit, log=log, plan=plan)
    rec = calibration_of(held)
    assert rec["joints"] == {} and rec["joints_withheld"], "the numbers were dropped instead of withheld"
    assert rec["application_gate"]["improvement_x"] == fit["application"]["improvement_x"]
    assert rec["provisional"] is True and rec["applied_over_the_gate"] is False
    for row in calibration_report(held)["withheld_changes"]:
        assert {"joint", "parameter", "would_have_moved_from", "to", "interval"} <= set(row)

    forced = apply_calibration(dog, fit, log=log, plan=plan, allow_provisional=True)
    assert compile_gene_to_mjcf(forced, include_floor=False) != compile_gene_to_mjcf(dog, include_floor=False)
    rep = calibration_report(forced)
    assert rep["n_changes"] > 0
    assert calibration_of(forced)["provisional"] is True, (
        "the override cleared the flag that says there was something to override")
    assert rep["applied_over_the_gate"] is True and rep["withheld"] is False
    assert "hypothesis, not a measurement" in rep["applied_over_the_gate_note"]


def test_the_brief_does_not_read_as_an_improvement_when_there_was_none(dog, plan, misspecified):
    """0.993x closer reads as closer. The headline sentence is what an engineer quotes to their team, so it
    has to CHANGE rather than gain a footnote."""
    from virturoid.services.sysid import apply_calibration, engineer_brief

    log, fit = misspecified
    cal = apply_calibration(dog, fit, log=log, plan=plan)
    brief = engineer_brief(cal, fit, log=log)
    assert brief["applied"] is False
    assert "NOT applied" in brief["lines"][0]
    assert "closer" not in brief["lines"][0].lower(), brief["lines"][0]
    assert any("link mass" in ln for ln in brief["lines"]), (
        "the brief does not name the mechanism that produced the wrong numbers")
    assert brief["actuator_ladder"]["earned"] is False


def test_the_brief_reports_what_reached_the_model_not_what_the_gate_wanted(dog, plan, misspecified):
    """"Applied" is a fact about the GENE, and the brief was reading it off the FIT. On a robot the caller had
    deliberately calibrated with ``allow_provisional=True``, the headline said "This fit was NOT applied" about
    a model that already carried every one of those numbers -- an untrue sentence about the customer's
    simulator, which is the class of claim this gate exists to stop rather than to create.

    Nor may it flip to the flattering branch: the override is applied, so it is not withheld, and the "0.993x
    closer" sentence sits one branch away."""
    from virturoid.services.sysid import apply_calibration, engineer_brief

    log, fit = misspecified
    forced = apply_calibration(dog, fit, log=log, plan=plan, allow_provisional=True)
    brief = engineer_brief(forced, fit, log=log)
    assert brief["applied"] is True and brief["applied_over_the_gate"] is True
    head = brief["lines"][0]
    assert "FAILED the tracking gate" in head and "applied ANYWAY" in head, head
    assert "NOT applied" not in brief["sentence"], "the brief denies a change the model actually carries"
    assert "closer" not in brief["sentence"].lower(), brief["sentence"]
    # A way back out, NAMED AS SOMETHING THE READER CAN CALL. This asserted `revert_calibration` -- the Python
    # function -- until `sysid.tools` gave the package a door; a customer driving this over MCP could not call
    # that, so the escape hatch offered for a hypothesis sitting in their model was not on their wire.
    assert "calibration_status" in head and "revert" in head, head
    # And the certificate says the same thing in its own words, because that is the line that gets quoted.
    bench = brief["actuator_ladder"]["bench_identified"]
    assert bench["applied_over_the_gate"] is True
    assert "not a bench identification" in bench["note"], bench["note"]
    assert brief["actuator_ladder"]["earned"] is False


def test_a_correctly_specified_fit_is_not_caught_by_the_same_gate(fit):
    """The gate must not be reachable by refusing everything. The default injection is an error the estimator
    CAN express, and it has to sail through -- with the separation between the two cases wide, not marginal."""
    from virturoid.services.sysid import MIN_TRACKING_IMPROVEMENT_X

    app = fit["application"]
    assert app["passed"] is True, app["verdict"]
    assert app["improvement_x"] >= 2 * MIN_TRACKING_IMPROVEMENT_X, (
        f"the correctly-specified fit clears the gate by only {app['improvement_x']}x against a threshold of "
        f"{MIN_TRACKING_IMPROVEMENT_X}x -- too close to call the threshold measured rather than tuned")


# --------------------------------------------------------------------------------------------------------
# ...AND THE TWO MISSPECIFICATIONS THAT CLEAR THE SAME GATE. RETRACTION, 2026-08-12, measured.
#
# ``MIN_TRACKING_IMPROVEMENT_X``'s docstring used to close: "every misspecification measured lands at or below
# 0.999 -- i.e. it never improves tracking AT ALL, which is the structural signature of an error no fitted
# parameter can express... the misspecified ceiling is PINNED AT ~1.0 BY CONSTRUCTION". Same shape of argument
# as the delay-blind "ceiling" retracted in ``test_sysid_delay_wedge.py`` hours earlier: a claim about what a
# fit COULD do standing where a measurement of what it DOES belongs.
#
# The original band swept ONE family -- link mass and inertia, moved together -- and that family really is
# pinned. RE-MEASURED across x0.4 / 0.6 / 0.8 / 0.9 / 1.1 / 1.15 / 1.2 / 1.3 / 1.45 / 1.6 / 2.0 / 3.0 it gives
# 0.961 / 0.939 / 1.000 / 1.000 / 1.000 / 0.998 / 0.999 / 0.995 / 0.998 / 0.999 / 0.988 / 0.996 -- never once
# above 1.000. So does a centre-of-mass error (5 / 20 / 50 mm: 1.000 / 1.000 / 1.000) and an unmodelled joint
# spring (0.02 ... 1.0 x kp: 1.000 throughout, nothing identified).
#
# BUT THE PIN IS THE GRAVITY TERM, not a property of misspecification. Mass carries gravity, a gravity error
# is a static tracking offset no dissipative or inertial parameter can remove, so it dominates `before` and
# survives untouched into `after`. Take gravity out of the error and the pin goes with it:
#
#   link ROTATIONAL INERTIA only, mass exactly right
#     x2 1.000 | x5 1.007 | x10 1.181 | x15 1.466 | x20 1.620 | x25 1.707 | x30 1.745 | x40 1.753 | x100 1.650
#   the applied torque scaled -- wrong gear ratio / torque constant / gearbox efficiency
#     x0.9 1.467 | x1.1 1.364 | x1.2 1.507 | x1.25 1.536 | x1.35 1.600 | x1.5 1.624 | x1.75 1.647 | x3.0 1.166
#
# Both cross 1.0 and both cross the 1.5x THRESHOLD; the measured maximum is 1.753x. And the other half of that
# sentence -- "the correct floor is a sample and could go lower on a smaller perturbation" -- was the honest
# half and goes lower than the band survives: the DEFAULT injection at x0.125 reads 3.564 and at x0.0625 reads
# 1.126 (16/42 identified, correctly specified, REFUSED). So the two populations OVERLAP: correct 1.126 against
# misspecified 1.753. No threshold on this quantity can separate them. The threshold is NOT moved -- see the
# constant's docstring for why raising it would be the same fiction wearing a bigger number.
# --------------------------------------------------------------------------------------------------------

#: The plant receives 1.25x the torque the log records. Chosen as the SMALLEST point on the measured curve
#: that clears the gate (x1.2 reads 1.507, x1.25 1.536), so this is the tightest form of the claim rather than
#: the most dramatic one. "gear ratio" is listed in ``fit_parameters``' own ``assumed_correct``.
TORQUE_SCALE = 1.25
#: Link rotational inertia with mass held EXACTLY right, near the peak of its curve (x30 1.745, x40 1.753).
#: In the only units that mean anything it is a +60% error (median; 32-66% across the 14 joints) in each
#: joint's own diag(M) -- the quantity armature adds to.
INERTIA_SCALE = 30.0


@pytest.fixture(scope="module")
def torque_scale_misspecified(dog, plan):
    """A fit against hardware whose joints receive more torque than the log says -- an error no fitted
    parameter can REPORT and all three can largely ABSORB."""
    from virturoid.services.sysid import fit_parameters
    from virturoid.services.sysid.synthetic_hardware import synthetic_hardware_log

    _, lg = synthetic_hardware_log(dog, perturbation={}, delay_ticks=0, plan=plan, torque_scale=TORQUE_SCALE)
    return lg, fit_parameters(dog, lg, plan=plan, n_boot=64, measure_delay=False)


@pytest.fixture(scope="module")
def inertia_misspecified(dog, plan):
    """A fit against hardware whose link INERTIA is wrong and whose MASS is right -- so there is no gravity
    signature at all and the whole error sits in the qdd term armature owns."""
    from virturoid.services.sysid import fit_parameters
    from virturoid.services.sysid.synthetic_hardware import synthetic_hardware_log

    _, lg = synthetic_hardware_log(dog, perturbation={}, delay_ticks=0, plan=plan, inertia_scale=INERTIA_SCALE)
    return lg, fit_parameters(dog, lg, plan=plan, n_boot=64, measure_delay=False)


def test_a_torque_scale_error_CLEARS_the_tracking_gate_and_reaches_the_model(dog, plan,
                                                                            torque_scale_misspecified):
    """THE RETRACTION, asserted so it cannot be quietly reinstated: a misspecification PASSES.

    A wrong gear ratio, a wrong torque constant or an unmodelled gearbox efficiency means the joint receives
    ``g`` times the torque the log records. That is algebraically the same as dividing inertia, damping and
    friction by ``g`` -- every one of which this estimator can move -- so the fit reproduces the plant almost
    exactly and the simulator really does get closer. What it cannot move is GRAVITY, which is the only reason
    the ratio saturates around 1.65 instead of running away.

    The numbers it writes are wrong in the way this section exists to catch: they are the true parameters
    scaled by ``(1 - g)/g``, their intervals exclude the true unchanged values, and at 1.25 they clear the gate
    and are applied to the customer's compiled model.
    """
    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    from virturoid.services.sysid import MIN_TRACKING_IMPROVEMENT_X, apply_calibration

    log, fit = torque_scale_misspecified
    app = fit["application"]
    assert app["improvement_x"] >= MIN_TRACKING_IMPROVEMENT_X, (
        f"a torque-scale error of {TORQUE_SCALE:g}x now scores {app['improvement_x']}x against the "
        f"{MIN_TRACKING_IMPROVEMENT_X}x gate, i.e. it is REFUSED. Measured 1.536x when this was written. If it "
        f"is genuinely under the threshold again, the retraction in the block above is stale and the "
        f"'a misspecification can never improve tracking' claim may be reinstated -- deliberately, in writing, "
        f"with the whole family re-swept, not by deleting this test")
    assert app["passed"] is True and app["provisional"] is False, app["verdict"]

    # ...and the numbers are WRONG, which is what makes the pass a defect rather than a success.
    wrong = [(n, p) for n, r in fit["joints"].items() for p in r["identified"]
             if not (r["parameters"][p]["delta_interval"][0] <= 0.0 <= r["parameters"][p]["delta_interval"][1])]
    assert len(wrong) >= 20, (
        f"only {len(wrong)} identified cells have an interval EXCLUDING the true (unchanged) value; measured "
        f"25 of 25. Without those this fit would just be a good fit and there would be nothing to report")

    # THE MECHANISM, not just the number: every delta is its own prior times (1 - g)/g.
    import numpy as np

    from virturoid.services.sysid.bench_rig import bench_model, joint_dof_map

    model, _ = bench_model(dog)
    dofs = joint_dof_map(model, dog)
    predicted = (1.0 - TORQUE_SCALE) / TORQUE_SCALE
    got = float(np.median([fit["joints"][n]["parameters"]["damping"]["delta"] / float(model.dof_damping[a])
                           for n, a in dofs.items() if n in fit["joints"]]))
    assert 0.7 <= got / predicted <= 1.8, (
        f"the fitted damping delta is meant to be the prior times (1-g)/g = {predicted:.3f}; it came back at "
        f"{got:.3f} of the prior ({got / predicted:.2f}x predicted, measured 1.25x). If that ratio has moved a"
        f" long way, the absorption mechanism in this docstring is not the one producing the pass")

    # ...and this is the part that costs a customer: it is applied, with no override asked for.
    cal = apply_calibration(dog, fit, log=log, plan=plan)
    assert compile_gene_to_mjcf(cal, include_floor=False) != compile_gene_to_mjcf(dog, include_floor=False), (
        "the gate passed but nothing reached the model, so some other gate is catching this after all -- find "
        "it and say so here, because the whole point of this test is that nothing does")


def test_a_link_inertia_error_at_correct_mass_is_absorbed_into_armature_and_also_clears_the_gate(
        dog, plan, inertia_misspecified):  # noqa: C901
    """The second family, and the one that names the mechanism the module docstring has always described.

    ``PARAMETER_ALSO_ABSORBS['armature']`` says reflected inertia and link inertia enter the joint equation
    through the same qdd term, so the experiment cannot separate them. When the link error also carries MASS
    that is a half-truth -- the gravity half is not absorbable and pins the ratio at ~1.0, which is what the
    band was sized on. Hold mass exactly right and the sentence becomes the whole truth: the fit recovers the
    inertia error as armature, on 14/14 joints, and the simulator genuinely gets closer.

    HOW REALISTIC THIS ONE IS, said next to the number rather than under it: with mass AND geometry both right
    a link's own inertia is bounded (dumbbell mL^2/4 vs uniform rod mL^2/12, so ~3x at the very most), which is
    about +4% of diag(M) and reads 1.005x. This family is what kills "a misspecification never improves
    tracking AT ALL" -- x5 already reads 1.007 -- and it only reaches the GATE at sizes that need the geometry
    to be wrong too. The sibling test above is the one whose breach needs nothing unrealistic at all.

    The assertion that pins it is not the score but the SIZE of the armature it invents: the change this
    misspecification makes to each joint's own diag(M) is computed here from the model, and the fitted armature
    has to land on it. Measured: median fitted delta 0.0105 against the 0.0094 that would mimic the error
    exactly.
    """
    import numpy as np

    from virturoid.services.sysid import MIN_TRACKING_IMPROVEMENT_X
    from virturoid.services.sysid.bench_rig import bench_model, joint_dof_map, start_pose

    log, fit = inertia_misspecified
    app = fit["application"]
    assert app["improvement_x"] >= MIN_TRACKING_IMPROVEMENT_X, (
        f"a link-inertia error at correct mass now scores {app['improvement_x']}x against the "
        f"{MIN_TRACKING_IMPROVEMENT_X}x gate; measured 1.745x. See the sibling test for what a genuine "
        f"refusal here would mean")
    assert app["passed"] is True

    arm = {n: r["parameters"]["armature"] for n, r in fit["joints"].items()}
    assert all(c["identified"] for c in arm.values()), (
        f"armature was refused on some joint, so this is not the absorption case: "
        f"{[n for n, c in arm.items() if not c['identified']]}")
    assert not any(c["delta_interval"][0] <= 0.0 <= c["delta_interval"][1] for c in arm.values()), (
        "some armature interval covers the true (unchanged) value -- the fit is no longer confidently wrong")

    # THE MECHANISM: what armature would have to be to mimic this error exactly is diag(M(hw)) - diag(M(ours)).
    import copy

    import mujoco

    model, _ = bench_model(dog)
    dofs = joint_dof_map(model, dog)
    q0 = start_pose(model, dog)

    def _diag(m):
        data = mujoco.MjData(m)
        data.qpos[:m.nv] = q0
        mujoco.mj_forward(m, data)
        full = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, full, data.qM)
        return np.diag(full).copy()

    hw = copy.deepcopy(model)
    hw.body_inertia[1:] = np.asarray(hw.body_inertia[1:], dtype=float) * INERTIA_SCALE
    d_ours, d_hw = _diag(model), _diag(hw)
    mimic = float(np.median([d_hw[a] - d_ours[a] for a in dofs.values()]))
    got = float(np.median([c["delta"] for c in arm.values()]))
    assert 0.6 <= got / mimic <= 1.8, (
        f"the fitted armature is meant to land on the joint-space inertia the misspecification added: "
        f"{got:.5f} against {mimic:.5f} ({got / mimic:.2f}x, measured 1.12x). If it does not, the pass is "
        f"coming from somewhere other than the armature/link-inertia degeneracy this test names")

    # ...and every one of those 14 misattributed numbers reaches the customer's simulator, because the gate
    # passed and nothing downstream re-asks the question.
    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    from virturoid.services.sysid import apply_calibration, calibration_report

    cal = apply_calibration(dog, fit, log=log, plan=plan)
    assert compile_gene_to_mjcf(cal, include_floor=False) != compile_gene_to_mjcf(dog, include_floor=False)
    rep = calibration_report(cal)
    assert rep["n_changes"] == fit["identified_pairs"] and rep["withheld"] is False, (
        f"the gate passed but only {rep['n_changes']} of {fit['identified_pairs']} parameters reached the "
        f"model -- something else is catching this, which would be good news and has to be named here")


# --------------------------------------------------------------------------------------------------------
# The measurement floor: a change smaller than the instrument's resolution must not reach the model.
# --------------------------------------------------------------------------------------------------------

def test_a_change_under_the_measurement_floor_is_refused_and_one_above_it_is_not(dog, plan):
    """The floor is this estimator's own BIAS, not noise -- so the estimate carries it, and comparing the
    estimate straight to the floor let an estimate clear its own bias. Measured: +0.0008 of armature, under
    every joint's floor of 0.00158-0.00221, came back identified on 14/14 joints at a median 0.00275, which is
    floor + bias almost exactly.

    The paired above-floor injection is the other half and is the point: a gate that refuses everything would
    pass the first assertion and fail the robot."""
    import numpy as np

    from virturoid.services.sysid import fit_parameters
    from virturoid.services.sysid.identifiability import FLOOR_MARGIN
    from virturoid.services.sysid.synthetic_hardware import synthetic_hardware_log

    _, sub = synthetic_hardware_log(dog, perturbation={"armature": 0.0008}, delay_ticks=0, plan=plan)
    under = fit_parameters(dog, sub, plan=plan, n_boot=32, measure_trajectory=False, measure_delay=False)
    cells = [r["parameters"]["armature"] for r in under["joints"].values()]
    floors = [c["noise_floor"] for c in cells]
    assert min(floors) > 0.0008, (
        f"the injection is no longer under the floor (floors {min(floors)}-{max(floors)}); this is not the "
        f"adversarial case any more")
    assert not any(c["identified"] for c in cells), (
        f"named armature on {sum(c['identified'] for c in cells)} joints from a change smaller than the "
        f"instrument's own resolution; margins {sorted(round(c['floor_margin_x'], 2) for c in cells)}")
    assert under["identified_pairs"] == 0
    joined = " ".join(cells[0]["reasons_not_identified"])
    assert "measurement floor" in joined and f"{FLOOR_MARGIN:g}x" in joined

    # The floor is PER JOINT and stays that way: one constant would hold a quiet joint to a noisy joint's bar.
    fl = [r["parameters"]["frictionloss"]["noise_floor"] for r in under["joints"].values()]
    assert max(fl) > 3 * min(fl), f"the frictionloss floor collapsed to a constant: {sorted(fl)}"

    _, above = synthetic_hardware_log(
        dog, perturbation={"frictionloss": 0.05, "damping": 0.20, "armature": 0.010},
        delay_ticks=0, plan=plan)
    over = fit_parameters(dog, above, plan=plan, n_boot=64, measure_delay=False)
    assert over["identified_pairs"] >= 30, (
        f"only {over['identified_pairs']}/{over['candidate_pairs']} pairs survived an above-floor injection -- "
        f"the floor gate is refusing changes the experiment can resolve")
    assert over["application"]["passed"] is True, over["application"]["verdict"]
    model_arm = {n: r["parameters"]["armature"] for n, r in over["joints"].items()}
    assert all(c["identified"] for c in model_arm.values()), "an above-floor armature change was refused"
    assert all(c["floor_margin_x"] >= FLOOR_MARGIN for c in model_arm.values())
    assert np.median([c["delta"] for c in model_arm.values()]) == pytest.approx(0.010, rel=0.6)


def test_the_estimator_identifies_nothing_when_the_hardware_IS_the_sim(dog, plan):
    """The null, and the cheapest check that the floor means anything: replay the excitation on an UNPERTURBED
    model and hand the result back as a log. There is nothing to find. Before the floor gate compared a
    bias-corrected estimate it found 8 of 42 pairs and wrote all 8 into the model."""
    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    from virturoid.services.sysid import apply_calibration, fit_parameters
    from virturoid.services.sysid.synthetic_hardware import synthetic_hardware_log

    _, clean = synthetic_hardware_log(dog, perturbation={}, delay_ticks=0, plan=plan)
    out = fit_parameters(dog, clean, plan=plan, n_boot=32, measure_delay=False)
    assert out["identified_pairs"] == 0, (
        f"found {out['identified_pairs']} parameters on a robot that is byte-identical to our model: "
        f"{[(j, p) for j, r in out['joints'].items() for p in r['identified']]}")
    cal = apply_calibration(dog, out, log=clean, plan=plan)
    assert compile_gene_to_mjcf(cal, include_floor=False) == compile_gene_to_mjcf(dog, include_floor=False)


# --------------------------------------------------------------------------------------------------------
# What the fit ASSUMED, said out loud.
# --------------------------------------------------------------------------------------------------------

def test_the_output_says_link_mass_and_inertia_were_assumed_correct(fit, calibrated, log):
    """``parameters_not_fitted`` listed the torque-speed envelope and the controller gains and stopped there.
    It did not say that LINK MASS AND INERTIA ARE HELD FIXED and that an error in them lands in ARMATURE --
    which is exactly the failure the tracking gate above exists to catch. An engineer reading "armature
    identified, 90% CI" is entitled to know what else that number could be, in the output rather than in a
    docstring."""
    from virturoid.services.sysid import calibration_report, engineer_brief

    assumptions = " ".join(f"{k} {v}" for k, v in fit["parameters_not_fitted"].items()).lower()
    assert "link mass" in assumptions and "inertia" in assumptions
    assert "armature" in assumptions, "the disclosure does not say WHERE the error lands"
    assert "link mass" in " ".join(fit["assumed_correct"]["what"]).lower()

    # On every cell, not only in the preamble -- the interval is read per parameter, so the caveat has to be.
    arm = next(iter(fit["joints"].values()))["parameters"]["armature"]
    assert "LINK INERTIA" in (arm["also_absorbs"] or ""), arm["also_absorbs"]

    brief = engineer_brief(calibrated, fit, log=log)
    assert any("assumed correct" in ln and "armature" in ln for ln in brief["lines"]), brief["lines"]
    assert brief["assumed_correct"] and brief["parameters_not_fitted"]
    assert all(row["also_absorbs"] for row in brief["pinned"])
    assert calibration_report(calibrated)["assumed_correct"]


# --------------------------------------------------------------------------------------------------------
# Applying it: the simulator must actually change, visibly and reversibly.
# --------------------------------------------------------------------------------------------------------

def test_the_calibration_changes_the_compiled_model(dog, calibrated, fit):
    """The point of the whole stage. A fit that does not reach the MJCF has changed nothing an engineer can
    train, verify or export against."""
    import re

    from virturoid.services.gene_compiler import compile_gene_to_mjcf

    before = compile_gene_to_mjcf(dog, include_floor=False)
    after = compile_gene_to_mjcf(calibrated, include_floor=False)
    assert before != after, "the calibration did not reach the compiled model"
    assert fit["application"]["passed"] is True, (
        "this fixture is the correctly-specified fit; if it stopped clearing the tracking gate the rest of "
        f"this test is vacuous: {fit['application']['verdict']}")

    # A joint whose frictionloss was actually IDENTIFIED, not simply the first one in the dict: a refused
    # parameter is correctly left on the compiler's prior, and asserting against it would be asserting that
    # the refusal failed.
    name = next(n for n, r in fit["joints"].items() if "frictionloss" in r["identified"])
    row = fit["joints"][name]
    pat = rf'<joint name="{name}_joint".*?frictionloss="([0-9.]+)"'
    got = float(re.search(pat, after).group(1))
    assert got == pytest.approx(row["parameters"]["frictionloss"]["value"], abs=5e-5)


def test_the_calibration_is_reversible_byte_for_byte(dog, calibrated):
    """A customer who does not believe the number has to be able to take it out."""
    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    from virturoid.services.sysid import revert_calibration

    back = revert_calibration(calibrated)
    assert back == dog, "revert did not return the pre-calibration gene"
    assert compile_gene_to_mjcf(back, include_floor=False) == compile_gene_to_mjcf(dog, include_floor=False)
    assert revert_calibration(back) == back, "revert is not idempotent on an uncalibrated gene"


def test_the_record_says_what_moved_from_where_and_what_was_refused(calibrated, fit):
    """Provenance is the product here. "Your friction is 0.15" is not usable; "your friction is 0.15, up from
    the 0.08 we guessed, 90% interval [0.14, 0.17], from a 35 s bench run on 2026-08-04, and here are the 3
    parameters we refused to fit" is."""
    from virturoid.services.sysid import calibration_report

    rep = calibration_report(calibrated)
    assert rep["calibrated"] is True
    assert rep["n_changes"] == fit["identified_pairs"]
    assert rep["fitted_at"] and rep["experiment"]["duration_s"]
    for row in rep["changes"]:
        assert {"joint", "parameter", "from", "to", "delta", "interval", "unit"} <= set(row)
        assert row["from"] != row["to"]
    assert rep["stale_entries"] == [], (
        "the fitted parameters were quoted against a baseline the compiler no longer produces: "
        f"{rep['stale_entries'][:4]}")
    # Same change of addressee as in the brief: the undo instruction has to name a call the customer can make,
    # not the Python function that implements it.
    assert "calibration_status" in rep["how_to_undo"] and "revert" in rep["how_to_undo"], rep["how_to_undo"]
    assert any("delay" in s for s in rep["not_applied"])


def test_the_bench_rig_measures_the_model_the_product_actually_simulates(dog):
    """Regression, and it is the defect Stage 2's staleness check found. ``bench_model`` welds a free base to
    a stand, and ``_joint_dynamics`` branched on ``base_mount`` -- so welding silently moved every leg joint
    onto the MANIPULATOR drivetrain prior. The rig was measuring the gap of a model the product never
    simulates, and every fitted parameter was quoted against a baseline that does not exist."""
    import re

    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    from virturoid.services.sysid.bench_rig import bench_model, joint_dof_map

    assert dog.base_mount == "free", "fixture no longer floating-base; this test would be vacuous"
    model, _ = bench_model(dog)
    dofs = joint_dof_map(model, dog)
    product = compile_gene_to_mjcf(dog, include_floor=False)
    for seg in dog.actuated_joints():
        m = re.search(rf'<joint name="{seg.name}_joint".*?damping="([0-9.]+)" armature="([0-9.]+)" '
                      rf'frictionloss="([0-9.]+)"', product)
        assert m, f"{seg.name} not found in the product MJCF"
        adr = dofs[seg.name]
        for i, attr in enumerate(("damping", "armature", "frictionloss")):
            assert float(getattr(model, f"dof_{attr}")[adr]) == pytest.approx(float(m.group(i + 1)), abs=1e-4), (
                f"{seg.name} {attr}: the bench rig and the product model disagree")


def test_the_bench_gain_can_break_a_high_friction_joint_away(dog):
    """Regression. An inertia-scaled gain says nothing about stiction, and Coulomb friction does not scale
    with inertia. The four distal leg joints came out at kp = 0.22 N.m/rad against 0.12 N.m of friction: they
    never moved, every parameter on them was (correctly) refused, and the report then suggested adding the
    sustained motion the plan had already commanded."""
    import numpy as np

    from virturoid.services.sysid.bench_rig import BREAKAWAY_ERR_RAD, bench_gains, bench_model, joint_dof_map

    model, _ = bench_model(dog)
    kp, _, _ = bench_gains(model)
    dofs = joint_dof_map(model, dog)
    for name, adr in dofs.items():
        static = abs(float(model.dof_frictionloss[adr]))
        assert float(kp[adr]) * BREAKAWAY_ERR_RAD >= static, (
            f"{name}: kp {kp[adr]:.3f} cannot produce its {static:.3f} N.m of Coulomb friction within "
            f"{BREAKAWAY_ERR_RAD} rad of tracking error -- it will stick")
    assert np.isfinite(kp).all()


def test_a_fitted_value_is_carried_by_evidence_at_source_calibrated(calibrated):
    """``InputSourceType.CALIBRATED`` -- "measured from logs / system identification" -- had no producer
    anywhere in the repo. The confidence is derived from the interval's own width rather than taking the
    enum's flat prior, and is capped when the log was not measured on hardware."""
    from virturoid.schemas.input_bundle import InputSourceType
    from virturoid.services.sysid import calibration_of

    ev = calibration_of(calibrated)["evidence"]
    assert ev, "no field-level evidence was emitted for the fitted parameters"
    for e in ev:
        assert e.source_type is InputSourceType.CALIBRATED
        assert e.validate().ok, e.validate()
        assert e.field_path.startswith("joints.")
        assert 0.0 < e.confidence <= 0.6, "a sim2sim fit presented with hardware-grade confidence"
    assert len({e.confidence for e in ev}) > 1, (
        "every field got the same confidence -- it is not derived from the fit")


def test_a_junk_calibration_in_metadata_cannot_break_the_compile(dog):
    """``metadata`` is a free-form dict and a customer can put anything in it, including something they
    pasted. A NaN reaching ``dof_frictionloss`` would make every rollout on that robot silently non-finite."""
    from dataclasses import replace

    from virturoid.services.gene_compiler import compile_gene_to_mjcf

    clean = compile_gene_to_mjcf(dog, include_floor=False)
    name = dog.actuated_joints()[0].name
    for junk in (
        {"artifact": "virturoid_actuator_calibration", "joints": {name: {"damping": {"to": float("nan")}}}},
        {"artifact": "virturoid_actuator_calibration", "joints": {name: {"damping": {"to": -5.0}}}},
        {"artifact": "virturoid_actuator_calibration", "joints": {name: {"damping": {"to": "0.5"}}}},
        {"artifact": "something_else", "joints": {name: {"damping": {"to": 99.0}}}},
        {"joints": "not a dict"},
        "not a dict at all",
    ):
        g = replace(dog, metadata={**dog.metadata, "calibration": junk})
        assert compile_gene_to_mjcf(g, include_floor=False) == clean, f"junk got through: {junk}"


# --------------------------------------------------------------------------------------------------------
# The L2 rung: earned, not declared -- and on this build, refused.
# --------------------------------------------------------------------------------------------------------

def test_a_sim2sim_fit_does_not_earn_the_bench_identified_rung(calibrated):
    """L2 reads "bench-identified actuator". A fit against a simulated stand-in is not a bench identification
    and must not raise the rung, however good its numbers are. This is the direction that cannot over-claim,
    and the reasons are reported rather than the rung being silently withheld."""
    from virturoid.services.certificate_v2 import actuator_fidelity_level
    from virturoid.services.sysid import l2_requirements

    l2 = l2_requirements(calibrated)
    assert l2["earned"] is False
    assert l2["n_met"] < l2["n_requirements"]
    blocked = " ".join(l2["blocked_by"])
    assert "PHYSICAL HARDWARE" in blocked
    assert "delay" in blocked
    assert "torque-speed envelope" in blocked

    act = actuator_fidelity_level(calibrated)
    assert act["level"] == 1, "a sim2sim fit was promoted to L2"
    assert act["bench_identified"]["n_joints"] > 0, "the fit is invisible on the certificate"
    assert "BENCH-IDENTIFIED" in act["note"], "the certificate does not say what the fit bought"


def test_the_two_requirements_we_cannot_meet_are_named_as_ours(calibrated):
    """If we cannot honestly reach L2 the honest output is WHICH parts are missing and WHOSE gap each is.
    Two of the three are ours: there is nowhere in the compiled model to put an identified transport delay,
    and a safe excitation cannot reach the saturation regime the torque-speed knee lives in."""
    from dataclasses import replace

    from virturoid.services.sysid import l2_requirements

    reqs = {r["requirement"]: r for r in l2_requirements(calibrated)["requirements"]}
    ours = [r for r in reqs.values() if not r["met"] and (r.get("whose_gap") or "").startswith("OURS")]
    assert len(ours) == 2, [r["requirement"] for r in ours]
    joined = " ".join(r["evidence"] for r in ours)
    assert "dyntype=none" in joined
    assert "saturation regime" in joined
    # And the torque-speed requirement is DERIVED from what the fit covered, not a hardcoded False -- a rung
    # that can only be unblocked by editing the rung is not a measurement.
    from virturoid.services.sysid.calibration import TORQUE_SPEED_PARAMS, calibration_of

    rec = dict(calibration_of(calibrated))
    assert not (set(rec["parameters_fitted"]) & TORQUE_SPEED_PARAMS)
    faked = replace(calibrated, metadata={**calibrated.metadata, "calibration": {
        **rec, "parameters_fitted": sorted(set(rec["parameters_fitted"]) | TORQUE_SPEED_PARAMS)}})
    flipped = {r["requirement"]: r["met"] for r in l2_requirements(faked)["requirements"]}
    assert flipped["the actuator's torque-speed envelope (tau_max / knee / no-load speed) is "
                   "bench-identified rather than taken from the datasheet"] is True


def test_the_model_really_has_no_actuation_delay_and_that_is_measured_not_assumed(dog):
    """The L2 verdict turns on this fact, so it is checked against the compiled model rather than trusted to
    a constant that would rot the day someone adds a lag."""
    from virturoid.services.sysid import model_represents_actuation_delay

    d = model_represents_actuation_delay(dog)
    assert d["checked"] is True
    assert d["represented"] is False
    assert d["n_actuators"] > 0


def test_the_l1_label_does_not_claim_a_latency_the_simulator_does_not_carry():
    """The rung labels are claims and are the most-quoted line on a certificate. L1 read "datasheet
    torque-speed clamp + PD + latency"; the clamp and the PD are real and the latency was never modelled."""
    from virturoid.services.certificate_v2 import ACTUATOR_LEVELS

    assert "latency NOT modelled" in ACTUATOR_LEVELS[1]
    assert ACTUATOR_LEVELS[2] == "L2 bench-identified actuator"


def test_an_uncalibrated_robot_still_reports_its_ladder_without_a_calibration(dog):
    """The level function must not need a fit to answer, and must not crash without one."""
    from virturoid.services.certificate_v2 import actuator_fidelity_level

    act = actuator_fidelity_level(dog)
    assert act["level"] in (0, 1)
    assert act["l2"]["earned"] is False
    assert act["bench_identified"]["n_joints"] == 0


def test_a_hardware_provenance_log_is_what_the_rung_is_waiting_for(dog, plan, log):
    """The positive half of the L2 gate, exercised as far as it can honestly be exercised: with a log that
    DECLARES hardware provenance, that requirement flips and the remaining blockers are ours alone. We own no
    hardware, so this proves the code path and NOT that any physical robot was ever measured -- which is the
    distinction the whole package exists to keep."""
    from virturoid.services.sysid import apply_calibration, fit_parameters, l2_requirements, log_provenance

    assert log_provenance(log)["class"] == "sim2sim"
    assert log_provenance({})["class"] == "unstated", "an unstated provenance must not read as hardware"

    claimed = {**log, "measured_on": "hardware", "provenance": "declared by the caller in this TEST ONLY"}
    # The trajectory IS measured here, and it has to be: without it the tracking gate has nothing to rule on,
    # the fit is provisional, nothing is applied, and the joint-coverage requirement fails for a reason that
    # is neither ours nor the customer's -- it is "we did not check".
    out = fit_parameters(dog, claimed, plan=plan, n_boot=32, measure_delay=False)
    assert out["application"]["passed"] is True, out["application"]["verdict"]
    cal = apply_calibration(dog, out, log=claimed, plan=plan)
    l2 = l2_requirements(cal)
    reqs = {r["requirement"]: r["met"] for r in l2["requirements"]}
    assert reqs["the fit was made against a log measured on PHYSICAL HARDWARE"] is True
    assert l2["earned"] is False, "hardware provenance alone is not L2"
    assert all((r.get("whose_gap") or "").startswith("OURS")
               for r in l2["requirements"] if not r["met"]), (
        "with a hardware log, every remaining L2 blocker should be ours")


# --------------------------------------------------------------------------------------------------------
# The engineer-facing output: the deliverable, as a function.
# --------------------------------------------------------------------------------------------------------

def test_the_brief_is_a_sentence_an_engineer_can_act_on(calibrated, fit, log):
    """Every clause has to be a measured field. The target sentence is the one no competitor gives today:
    "your sim lags your hardware by N ms; M of K parameters are pinned, this one is not -- here is the
    experiment that would pin it"."""
    from virturoid.services.sysid import engineer_brief

    brief = engineer_brief(calibrated, fit, log=log)
    assert brief["ok"]
    s = brief["sentence"]
    assert "ms" in s and "deg RMS" in s and "pinned" in s
    assert "sim2sim" in s, "the brief does not disclose that this was never measured on hardware"
    assert len(brief["pinned"]) == fit["identified_pairs"]
    assert len(brief["pinned"]) + len(brief["not_pinned"]) == fit["candidate_pairs"]
    for row in brief["pinned"]:
        assert row["interval"][0] <= row["value"] <= row["interval"][1]
    assert brief["latency"]["applied"] is False
    assert brief["actuator_ladder"]["earned"] is False


def test_the_brief_reports_a_refusal_rather_than_a_number_when_nothing_was_pinned(dog, plan):
    """The sentence has to survive the case the tool exists for."""
    from virturoid.services.sysid import apply_calibration, engineer_brief, fit_parameters
    from virturoid.services.sysid.synthetic_hardware import synthetic_hardware_log

    _, held = synthetic_hardware_log(dog, perturbation=INJECTED, delay_ticks=0, plan=plan, hold_only=True)
    out = fit_parameters(dog, held, plan=plan, n_boot=32, measure_delay=False)
    cal = apply_calibration(dog, out, log=held, plan=plan)
    brief = engineer_brief(cal, out, log=held)
    assert brief["pinned"] == []
    assert "0 of" in brief["sentence"]
    assert brief["next_experiment"] is not None
    assert brief["next_experiment"]["duration_s"] > 0


def test_stage_one_still_refuses_what_it_refused_before_the_override_hook(dog):
    """``identifiability_report`` grew an ``override`` so Stage 2's gates rule on the ACCUMULATED estimate
    instead of the last Gauss-Newton step. Without the hook the behaviour must be byte-identical, or Stage 2
    would have quietly loosened the gate Stage 1's refusals depend on."""
    import numpy as np

    from virturoid.services.sysid.identifiability import identifiability_report

    n = 600
    rng = np.random.default_rng(0)
    qd = np.linspace(0.4, 1.2, n)
    X = np.column_stack([np.ones(n), np.sign(qd), qd, rng.normal(size=n)])
    y = 0.3 + 0.08 * np.sign(qd) + 0.6 * qd + 0.01 * rng.normal(size=n)
    stats = {"velocity_sign_reversals": 0, "speed_rms_radps": 0.8, "accel_rms_radps2": 1.0, "n_samples": n}
    plain = identifiability_report("one_way", X, y, ["frictionloss", "damping", "armature"],
                                   excitation_stats=stats)
    none_override = identifiability_report("one_way", X, y, ["frictionloss", "damping", "armature"],
                                           excitation_stats=stats, override=None)
    assert plain == none_override
    assert plain["parameters"]["frictionloss"]["identified"] is False
    assert plain["parameters"]["damping"]["identified"] is True

    # And WITH the hook the gates still rule -- an override cannot buy an unexcited parameter a verdict.
    over = identifiability_report("one_way", X, y, ["frictionloss", "damping", "armature"],
                                  excitation_stats=stats,
                                  override={"frictionloss": {"estimate": 99.0, "ci": [98.0, 100.0],
                                                             "se": 0.001}})
    assert over["parameters"]["frictionloss"]["identified"] is False
    assert over["parameters"]["frictionloss"]["estimate"] == pytest.approx(99.0)
