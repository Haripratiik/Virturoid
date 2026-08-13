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


def test_a_torque_scale_error_STILL_clears_the_tracking_gate_and_is_now_stopped_by_the_second_check(
        dog, plan, torque_scale_misspecified):
    """THE RETRACTION, still asserted so it cannot be quietly reinstated -- plus the check that closes it.

    A wrong gear ratio, a wrong torque constant or an unmodelled gearbox efficiency means the joint receives
    ``g`` times the torque the log records. That is algebraically the same as dividing inertia, damping and
    friction by ``g`` -- every one of which this estimator can move -- so the fit reproduces the plant almost
    exactly and the simulator really does get closer. What it cannot move is GRAVITY, which is the only reason
    the ratio saturates around 1.65 instead of running away.

    The numbers it writes are wrong in the way this section exists to catch: they are the true parameters
    scaled by ``(1 - g)/g`` and their intervals exclude the true unchanged values. **The tracking gate is STILL
    fooled by all of that** -- ``improvement_x`` reads 1.536x at g=1.25 against a 1.5x threshold, and the first
    two thirds of this test assert exactly that, unchanged. What is new is the SECOND check: the same coherence
    that makes the absorption work is the fingerprint that identifies it, and a replay of the PRIOR model with
    one gear scalar beats the fit outright. So the last third of this test is INVERTED (2026-08-12): the
    parameters no longer reach the customer's model, and the reason is named in the verdict.

    MEASURED at g=1.25 with this code: improvement_x 1.536 (gate passed on the merits), coherence 0.954,
    implied g 1.257, the one-number rival tracks 29.8x better than the fit, 0 parameters written.
    """
    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    from virturoid.services.sysid import MIN_TRACKING_IMPROVEMENT_X, apply_calibration

    log, fit = torque_scale_misspecified
    app = fit["application"]
    assert app["improvement_x"] >= MIN_TRACKING_IMPROVEMENT_X, (
        f"a torque-scale error of {TORQUE_SCALE:g}x now scores {app['improvement_x']}x against the "
        f"{MIN_TRACKING_IMPROVEMENT_X}x gate, i.e. the TRACKING gate refuses it on its own. Measured 1.536x. "
        f"If it is genuinely under the threshold again, the retraction in the block above is stale and the "
        f"'a misspecification can never improve tracking' claim may be reinstated -- deliberately, in writing, "
        f"with the whole family re-swept, not by deleting this test")

    # ...and the numbers are WRONG, which is what makes the tracking pass a defect rather than a success.
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

    # ---- THE CLOSE. Everything above was true at 824e0fb and the fit was WRITTEN anyway. ----------------
    gs = fit["global_scale"]
    assert gs["suspected"] is True, (
        f"the second check did not fire on the case it exists for: coherence {gs.get('coherence')} "
        f"(threshold {gs.get('coherence_threshold')}), implied g {gs.get('implied_torque_scale_g')}, rival "
        f"{gs.get('rival')}. Measured coherence 0.954 / implied g 1.257 / rival explains_x 29.8")
    assert gs["rival"]["explains_x"] >= 1.0
    assert 1.05 <= gs["implied_torque_scale_g"] <= 1.6, (
        f"the check fired but named the wrong factor: it implies {gs['implied_torque_scale_g']}x against an "
        f"injected {TORQUE_SCALE}x (measured 1.257). A refusal that misnames the cause sends the customer to "
        f"the wrong part of their BOM")
    assert app["passed"] is False and app["refused_by"] == "global_scale", app["verdict"]
    v = app["verdict"].lower()
    assert "gear ratio" in v and "torque constant" in v, (
        f"the refusal must name the thing to check, not just refuse: {app['verdict']}")

    # ...and THIS is the line that was inverted. It used to assert the model DID change.
    cal = apply_calibration(dog, fit, log=log, plan=plan)
    assert compile_gene_to_mjcf(cal, include_floor=False) == compile_gene_to_mjcf(dog, include_floor=False), (
        "the second check refused the fit and parameters reached the compiled model anyway -- the refusal is "
        "not being honoured downstream, which is the whole defect wearing a different hat")
    from virturoid.services.sysid import calibration_of

    rec = calibration_of(cal)
    assert rec["withheld_from_model"] is True
    assert rec["n_parameters_moved"] == 0
    assert rec["n_parameters_withheld"] >= 20, (
        "the withheld numbers must still be REPORTED -- a fit that vanishes is indistinguishable from one "
        "that never ran")


def test_a_link_inertia_error_STILL_clears_the_tracking_gate_and_is_now_stopped_by_the_third_check(
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
        f"{MIN_TRACKING_IMPROVEMENT_X}x gate; measured 1.745x. If it is genuinely under the threshold again "
        f"the TRACKING gate is catching it unaided, which would make this case non-adversarial and the whole "
        f"section would need re-measuring rather than deleting")

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

    # ---- THE CLOSE. Everything above was true at 12b11dc and the fit was WRITTEN anyway. --------------
    # The SAME degeneracy, read the other way round: if the fitted armature on every joint is the same
    # fraction of that joint's own ROTATIONAL inertia, then ONE inertia scalar is a live rival, and it is put
    # to a replay. MEASURED at x30: inertia coherence 0.9929, implied s 35.89, the one-number rival tracks
    # 12.418x better than the fit, 0 parameters written.
    li = fit["link_inertia"]
    assert li["suspected"] is True, (
        f"the third check did not fire on the case it exists for: coherence {li.get('coherence')}, implied s "
        f"{li.get('implied_inertia_scale_s')}, rival {li.get('rival')}. Measured coherence 0.9929 / implied s "
        f"35.89 / rival explains_x 12.418")
    assert li["rival"]["explains_x"] >= 1.0
    assert 15.0 <= li["implied_inertia_scale_s"] <= 90.0, (
        f"the check fired but named the wrong scale: it implies {li['implied_inertia_scale_s']}x against an "
        f"injected {INERTIA_SCALE}x (measured 35.89). A refusal that misnames the cause sends the customer to "
        f"the wrong part of their CAD")
    assert app["passed"] is False and app["refused_by"] == "link_inertia", app["verdict"]
    v = app["verdict"].lower()
    assert "link inertia" in v and "cad" in v, (
        f"the refusal must name the thing to check, not just refuse: {app['verdict']}")
    assert "gear ratio" not in v, (
        f"the link-inertia refusal is sending the customer to their DRIVETRAIN, which this fit has no "
        f"evidence against -- that is the torque-scale remedy on the wrong verdict: {app['verdict']}")

    # ...and THIS is the line that was inverted. It used to assert the model DID change.
    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    from virturoid.services.sysid import apply_calibration, calibration_of, calibration_report

    cal = apply_calibration(dog, fit, log=log, plan=plan)
    assert compile_gene_to_mjcf(cal, include_floor=False) == compile_gene_to_mjcf(dog, include_floor=False), (
        "the third check refused the fit and parameters reached the compiled model anyway -- the refusal is "
        "not being honoured downstream, which is the whole defect wearing a different hat")
    rep = calibration_report(cal)
    assert rep["n_changes"] == 0 and rep["withheld"] is True, rep
    rec = calibration_of(cal)
    assert rec["withheld_from_model"] is True and rec["n_parameters_moved"] == 0
    assert rec["n_parameters_withheld"] >= 14, (
        "the withheld numbers must still be REPORTED -- a fit that vanishes is indistinguishable from one "
        "that never ran")

    # ...and the SENTENCE the engineer reads has to say the right cause. Falling through to the torque-scale
    # branch would tell them to re-check a drivetrain this fit has no evidence against; falling through to the
    # tracking branch would tell them the fit "does not move your simulator toward this log" about a fit that
    # moved it 1.745x, which is false and self-contradicting in one clause.
    from virturoid.services.sysid import engineer_brief

    sentence = engineer_brief(cal, fit, log=log)["sentence"].lower()
    assert "inertia tensors" in sentence and "cad" in sentence, sentence
    assert "gear ratio" not in sentence and "torque constant" not in sentence, (
        f"the brief sends the customer to their drivetrain for a CAD error: {sentence}")
    assert "does not move your simulator toward this log" not in sentence, (
        f"the brief reports the symptom instead of the cause: {sentence}")


# --------------------------------------------------------------------------------------------------------
# THE TWO REPLAY CHECKS -- and the SEPARATION each has to hold, on BOTH populations.
#
# The block above ends "no threshold on this quantity can separate them", and that stands: a correctly
# specified fit reads 1.126 while a misspecified one reads 1.753, so ``MIN_TRACKING_IMPROVEMENT_X`` is stuck at
# 1.5 whatever it is set to. What closes BOTH breaching families is a DIFFERENT question asked of the same
# data, twice, with two different one-number rivals -- put the priors back, change ONE number, replay, and
# score the position RMS neither model optimised:
#
#   global_scale   scale every actuator's GEAR by g. A torque-scale error of g drives all three fitted
#                  parameters, on all joints, by the same (1-g)/g of their priors.
#   link_inertia   scale every link's ROTATIONAL INERTIA by s, masses untouched. A link-inertia error at
#                  correct mass drives every joint's ARMATURE by the same (s-1) of its own rotational inertia,
#                  because armature and link inertia add to the same diagonal of M.
#
# MEASURED through the shipped ``fit_parameters``, composed dog, 35 s excitation, delay 0, n_boot=64 --
# 29 fits. This is the table this test file exists to keep honest, and it is the whole sweep, not the winners:
#
#                       improvement_x  passed_1.5x  TORQUE rival  INERTIA rival   VERDICT NOW
#   torque_scale x1.2       1.507         YES          30.119         0.664       REFUSED (global_scale)
#   torque_scale x1.25      1.536         YES          29.773         0.652       REFUSED (global_scale)
#   torque_scale x1.35      1.600         YES          17.861         0.625       REFUSED (global_scale)
#   torque_scale x1.5       1.624         YES          25.529         0.616       REFUSED (global_scale)
#   torque_scale x1.75      1.647         YES         230.111         0.608       REFUSED (global_scale)
#   torque_scale x2.0       1.500         YES          57.198         0.667       REFUSED (global_scale)
#   inertia_scale x20       1.620         YES           0.617        12.365       REFUSED (link_inertia)
#   inertia_scale x25       1.707         YES           0.571       181.525       REFUSED (link_inertia)
#   inertia_scale x30       1.745         YES           0.500        12.418       REFUSED (link_inertia)
#   inertia_scale x40       1.753         YES           0.420        50.690       REFUSED (link_inertia)
#   inertia_scale x50       1.730         YES           0.387        20.705       REFUSED (link_inertia)
#   inertia_scale x70       1.691         YES           0.356        64.243       REFUSED (link_inertia)
#   inertia_scale x100      1.650         YES           0.354        13.278       REFUSED (link_inertia)
#   inertia_scale x5/10/15  1.007/1.181/1.466  no       --             --         refused (tracking)
#   inertia_scale x300      1.433         no            --             --         refused (tracking)
#   link_scale x1.3         0.995         no            --             --         refused (tracking)
#   ------------------------------------------------------------------------------------- CORRECTLY SPECIFIED
#   default, no delay      16.365         YES           0.047         0.062       applied
#   default x0.5            5.269         YES           0.190         0.196       applied
#   default x0.25           4.584         YES           0.327         0.227       applied
#   +0.05/+0.20/+0.010      3.343         YES           0.371         0.305       applied
#   frictionloss only       8.330         YES           0.160         0.119       applied
#   damping only            7.248         YES           0.198         0.138       applied
#   ARMATURE only          17.910         YES           0.022         0.099       applied
#   PROPORTIONAL +25% f,d   4.025         YES           0.389         0.248       applied
#   default + 20 ms        18.497         YES           0.048         0.055       applied
#   default x0.0625         1.126         no            --             --         refused (tracking)
#
# TWO ROWS DECIDE WHETHER THIS IS A FIX OR A COIN FLIP, and both are in the CORRECT population.
#
#   PROPORTIONAL +25% f,d   a genuine dissipation change proportional to the priors. Its torque COHERENCE is
#                           0.980, higher than four of the six torque-scale errors the first check exists to
#                           catch, so a coherence-only gate would refuse a correct calibration. The rival
#                           clears it at 0.389.
#   ARMATURE only +0.03     a genuine REFLECTED-INERTIA change, which is itself an inertia-like error -- it
#                           adds to the same diagonal of M that link inertia does. Its INERTIA coherence is
#                           0.983, inside the 0.982-0.995 the caught inertia cases read, so a coherence-only
#                           gate there would refuse it too. The rival clears it at 0.099 -- and at 0.099 again
#                           when scored over a dense 36-point grid of s rather than the shipped local search,
#                           so the win is not an artefact of where we looked.
#
# Both coherences are FILTERS AT BEST and neither is a verdict; the inertia one is not even a filter, because
# its two populations overlap outright and no line fits between them. Only the replays rule.
#
# **AND THIS WHOLE SECTION PASSED WHILE THE LINK-INERTIA CHECK WAS REFUSING CORRECT CALIBRATIONS.** Read the
# CORRECT column of the table above as a SAMPLE, not as a population: every row in it improves tracking by
# 3.343x or more, i.e. more than twice the 1.5x gate. The inertia rival's ``explains_x`` is the rival's
# improvement DIVIDED BY the fit's, so it rises as the fit weakens -- and just above the gate, where a
# customer's fit is most likely to land, an ordinary armature-only calibration (+0.009, improvement 1.602) was
# REFUSED at 1.108. Nothing here sampled that region, so nothing here failed. The band is now swept densely on
# both bodies in ``tests/test_sysid_link_inertia_danger_band.py``, which is where that claim lives; these five
# rows stay because the strong-fit end still has to hold.
# --------------------------------------------------------------------------------------------------------

#: Both populations, run once. Deliberately NOT only the winning cases: a separation claim measured on the
#: catches alone is not a separation claim.
_SEPARATION_CASES = [
    # label,                     kwargs to synthetic_hardware_log,                    population
    ("torque_scale x1.2", dict(perturbation={}, torque_scale=1.2), "MISSPEC_SCALE"),
    ("torque_scale x1.5", dict(perturbation={}, torque_scale=1.5), "MISSPEC_SCALE"),
    ("torque_scale x2.0", dict(perturbation={}, torque_scale=2.0), "MISSPEC_SCALE"),
    # The OTHER family, at the two ends of its breaching range: x20 is the first point that clears the
    # tracking gate at all (1.620) and x100 the last (1.650).
    ("inertia_scale x20", dict(perturbation={}, inertia_scale=20.0), "MISSPEC_INERTIA"),
    ("inertia_scale x100", dict(perturbation={}, inertia_scale=100.0), "MISSPEC_INERTIA"),
    ("correct default", dict(perturbation=dict(INJECTED)), "CORRECT"),
    ("correct default x0.5", dict(perturbation={p: v * 0.5 for p, v in INJECTED.items()}), "CORRECT"),
    ("correct +0.05/+0.20/+0.010",
     dict(perturbation={"frictionloss": 0.05, "damping": 0.20, "armature": 0.010}), "CORRECT"),
    # THE ADVERSARIAL CORRECT CASE for the TORQUE check: a real +25% on frictionloss and damping, i.e.
    # proportional to the leg joints' own priors, so the COHERENCE half alone reads 0.980 and would refuse it.
    ("correct proportional +25% f,d",
     dict(perturbation={"frictionloss": 0.03, "damping": 0.2}), "CORRECT"),
    # THE ADVERSARIAL CORRECT CASE for the INERTIA check, and the one that would have made it worse than the
    # defect: a REAL reflected-inertia change is itself an inertia-like error on the same diagonal of M.
    ("correct armature only +0.03", dict(perturbation={"armature": 0.03}), "CORRECT"),
]


@pytest.fixture(scope="module")
def separation(dog, plan):
    """One fit per case, both populations, through the shipped ``fit_parameters``."""
    from virturoid.services.sysid import fit_parameters
    from virturoid.services.sysid.synthetic_hardware import synthetic_hardware_log

    out = {}
    for label, kw, pop in _SEPARATION_CASES:
        _, lg = synthetic_hardware_log(dog, plan=plan, delay_ticks=0, **kw)
        out[label] = (pop, fit_parameters(dog, lg, plan=plan, n_boot=64, measure_delay=False))
    return out


def test_the_second_check_refuses_every_torque_scale_error_that_breached_the_tracking_gate(separation):
    """CATCHES: the breach, at every magnitude of it that got through the first gate.

    x1.2 is the case the retraction was written on -- 1.507x against a 1.5x threshold, the smallest realistic
    gear-ratio error that clears. x2.0 is the far end. Measured rival ``explains_x``: 30.1 / 25.5 / 57.2, i.e.
    a single gear scalar tracks the log 25-57x better than the 24-29 fitted numbers do. The margin to the 1.0
    line is not marginal, and that is the claim.
    """
    caught = {}
    for label, (pop, fit) in separation.items():
        if pop != "MISSPEC_SCALE":
            continue
        gs, app = fit["global_scale"], fit["application"]
        caught[label] = (app["improvement_x"], gs.get("coherence"), gs.get("implied_torque_scale_g"),
                         (gs.get("rival") or {}).get("explains_x"), app["passed"])
        assert app["improvement_x"] >= 1.5, (
            f"{label} no longer breaches the tracking gate ({app['improvement_x']}x) -- this case is not "
            f"adversarial any more and the whole section needs re-measuring, not deleting. Table: {caught}")
        assert gs["suspected"] is True, f"{label} was not identified as a global torque scale: {caught}"
        assert app["passed"] is False and app["refused_by"] == "global_scale", (
            f"{label} was identified as a torque scale and applied anyway: {app['verdict']}")
        assert (gs["rival"] or {})["explains_x"] >= 1.0
    assert len(caught) == 3, f"the misspecified population shrank to {sorted(caught)}"


def test_the_third_check_refuses_every_link_inertia_error_that_breached_the_tracking_gate(separation):
    """CATCHES: the OTHER breach, at both ends of the range that got through the first gate.

    x20 is the first point of the ``inertia_scale`` family that clears 1.5x at all (1.620) and x100 the last
    (1.650); the shape between them was swept and is in the table above. Measured rival ``explains_x``:
    12.365 and 13.278 here, 12.4-181 across the seven breaching points. The margin to the 1.0 line is not
    marginal, and that is the claim.

    It also asserts the ATTRIBUTION, not just the refusal. A link-inertia error must NOT come back as a
    torque-scale one: the two remedies send the engineer to different documents, and a refusal that names the
    wrong one costs them a day. Measured: the torque rival on this family reads 0.354-0.617 and loses.
    """
    caught = {}
    for label, (pop, fit) in separation.items():
        if pop != "MISSPEC_INERTIA":
            continue
        li, gs, app = fit["link_inertia"], fit["global_scale"], fit["application"]
        caught[label] = (app["improvement_x"], li.get("coherence"), li.get("implied_inertia_scale_s"),
                         (li.get("rival") or {}).get("explains_x"),
                         (gs.get("rival") or {}).get("explains_x"), app["refused_by"])
        assert app["improvement_x"] >= 1.5, (
            f"{label} no longer breaches the tracking gate ({app['improvement_x']}x, measured 1.620 / 1.650) "
            f"-- this case is not adversarial any more and the whole section needs re-measuring, not "
            f"deleting. Table: {caught}")
        assert li["suspected"] is True, f"{label} was not identified as a link-inertia scale: {caught}"
        assert app["passed"] is False and app["refused_by"] == "link_inertia", (
            f"{label} was identified as a link-inertia error and applied anyway (or refused for the WRONG "
            f"reason, {app['refused_by']}): {app['verdict']}")
        assert (li["rival"] or {})["explains_x"] >= 1.0
        assert gs["suspected"] is False, (
            f"{label} was ALSO read as a global torque scale, so the customer would be sent to their gearbox "
            f"for a CAD error. Torque rival on this family measured 0.354-0.617: {caught}")
    assert len(caught) == 2, f"the link-inertia population shrank to {sorted(caught)}"


def test_neither_check_refuses_any_of_the_correctly_specified_fits(separation):
    """KEEPS: a fix that refuses correct calibrations is worse than the defect, because the defect at least
    ships with a disclosure. BOTH checks, on the same population, because each has its own way to be wrong.

    Three of these are reference points the threshold was sized on (16.365 / 5.269 / 3.343). The other two are
    the ones that decide whether these checks are fixes or coin flips, one per check:

      * PROPORTIONAL +25% f,d -- a REAL dissipation change proportional to the leg joints' priors, so the
        TORQUE coherence reads 0.980, above four of the six torque-scale errors the sibling test catches. It
        survives because the torque rival loses at 0.389.
      * ARMATURE only +0.03 -- a REAL reflected-inertia change, which is ITSELF an inertia-like error: armature
        adds to the same diagonal of M that link inertia does, so this is the case an inertia rival could most
        plausibly fail to distinguish from a misstated tensor. Its INERTIA coherence reads 0.983, inside the
        0.982-0.995 band the caught cases occupy. It survives because the inertia rival loses at 0.099 -- and
        it is the reason there is no coherence threshold on that check at all.
    """
    kept = {}
    for label, (pop, fit) in separation.items():
        if pop != "CORRECT":
            continue
        gs, li, app = fit["global_scale"], fit["link_inertia"], fit["application"]
        kept[label] = (app["improvement_x"], gs.get("coherence"), (gs.get("rival") or {}).get("explains_x"),
                       li.get("coherence"), (li.get("rival") or {}).get("explains_x"))
        assert gs["suspected"] is False, (
            f"{label} is a CORRECTLY SPECIFIED fit and the torque-scale check refused it: coherence "
            f"{gs.get('coherence')}, rival {gs.get('rival')}. Measured torque rivals over this population: "
            f"0.022-0.389. Table: {kept}")
        assert li["suspected"] is False, (
            f"{label} is a CORRECTLY SPECIFIED fit and the LINK-INERTIA check refused it: coherence "
            f"{li.get('coherence')}, rival {li.get('rival')}. Measured inertia rivals over this population: "
            f"0.055-0.305 -- but note these are all STRONG fits (3.343x and up) and this assertion has never "
            f"been able to see the failure the check actually had; that lives in "
            f"tests/test_sysid_link_inertia_danger_band.py. Table: {kept}")
        assert app["passed"] is True and app.get("refused_by") is None, (
            f"{label} lost its PASS: {app['verdict']}. Table: {kept}")
        # THE COHERENCE FILTER IS NOT A GATE ON THE REPLAY ANY MORE, and this is where that is pinned. Both
        # rivals must have actually RUN on every correct fit that passes the tracking gate -- eight of these
        # ten never reached the torque rival when the 0.85 filter still gated it, so "no false refusal" used
        # to be a claim about a test that was never taken. It now costs ~4 s per rival and is taken every time.
        assert kept[label][2] is not None and kept[label][4] is not None, (
            f"{label} passed the tracking gate and one of the two rivals was never simulated: torque "
            f"{kept[label][2]}, inertia {kept[label][4]}. A fit that reaches the model without being asked "
            f"the question is the hole this section exists to close")
    assert len(kept) == 5, f"the correct population shrank to {sorted(kept)}"

    prop = kept["correct proportional +25% f,d"]
    assert prop[1] >= 0.85, (
        f"the adversarial correct case no longer trips the torque COHERENCE (reads {prop[1]}, measured 0.980), "
        f"so it is no longer testing that the rival is load-bearing. Find another proportional perturbation "
        f"rather than dropping the case")
    assert prop[2] is not None and prop[2] < 1.0, (
        f"the torque rival did NOT clear the correct proportional case ({prop[2]}, measured 0.389) -- one "
        f"number is beating a real calibration and the check is unsafe")

    # THE ONE THAT DECIDES THE INERTIA CHECK. Armature IS reflected inertia; if the inertia rival could not
    # tell this from a misstated tensor, the check would refuse a correct calibration and be worse than the
    # breach it closes. It is asserted as a MARGIN, not a boolean, so a drift toward 1.0 fails here rather
    # than silently later.
    arm = kept["correct armature only +0.03"]
    assert arm[3] is not None and arm[3] >= 0.90, (
        f"the armature-only case no longer looks like a link-inertia error to the coherence statistic (reads "
        f"{arm[3]}, measured 0.9831 against 0.982-0.995 on the caught cases). It is the adversarial case "
        f"BECAUSE it looks like one; if it stops, find another rather than dropping it")
    assert arm[4] is not None and arm[4] <= 0.5, (
        f"the inertia rival scores {arm[4]} on a REAL armature calibration (measured 0.099, and 0.099 again "
        f"at its global optimum over a dense 36-point grid). Anything approaching 1.0 means the check cannot "
        f"separate 'your link inertia is misstated' from 'your reflected inertia really differs', which is "
        f"the failure that would make this check worse than the defect it closes")


def test_the_link_inertia_catch_is_not_a_single_body_artifact():
    """A SECOND BODY, because a number from the composed dog is not general and this file has been burned by
    that before (docs/calibration_wedge_under_delay.md section 12; the delay wedge).

    A composed HEXAPOD, 18 joints. MEASURED end to end through the shipped code, delay 0, n_boot=48:

      correct default              improvement 20.879   torque rival 0.007   inertia rival  0.051   applied
      correct ARMATURE only        improvement 18.107   torque rival 0.005   inertia rival  0.106   applied
      inertia_scale x30            improvement  2.015   torque rival 0.049   inertia rival 18.125   REFUSED
      inertia_scale x40            improvement  1.973   torque rival 0.047   inertia rival 320.55   REFUSED

    Two things this establishes and one it does not. The BREACH reproduces on a second body -- bigger here
    (2.015) than on the dog (1.745) -- and the check refuses it there, which is the first time either catch has
    been shown off the body it was built on. The torque rival correctly loses on it, so the ATTRIBUTION holds
    too. What it does NOT establish is the torque-scale catch: on this hexapod, as on the Menagerie Go2, a
    torque-scale error scores 0.989-1.005 and the tracking gate refuses it unaided, so that breach does not
    reproduce and there is nothing there for the second check to catch.
    """
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.sysid import build_excitation, fit_parameters
    from virturoid.services.sysid.synthetic_hardware import synthetic_hardware_log

    hexa = compose_robot("a six legged walking robot", llm=None)
    hplan = build_excitation(hexa, budget_s=BUDGET_S)

    _, bad = synthetic_hardware_log(hexa, perturbation={}, delay_ticks=0, plan=hplan, inertia_scale=30.0)
    mis = fit_parameters(hexa, bad, plan=hplan, n_boot=32, measure_delay=False)
    mapp, mli = mis["application"], mis["link_inertia"]
    assert mapp["improvement_x"] >= 1.5, (
        f"the link-inertia breach does not reproduce on this body ({mapp['improvement_x']}x, measured 2.015) "
        f"-- then this test proves nothing about generality and needs a body where it does, not deleting")
    assert mli["suspected"] is True and mapp["refused_by"] == "link_inertia", (
        f"the catch did not reproduce on a second body: coherence {mli.get('coherence')}, rival "
        f"{mli.get('rival')}, verdict {mapp['verdict']}")
    assert (mis["global_scale"] or {}).get("suspected") is False, (
        "the hexapod's link-inertia error came back as a TORQUE-SCALE one, which sends the customer to the "
        "wrong document. Measured torque rival on this case: 0.049")

    # ...and the false-refusal side on the same body, including the case that is itself an inertia change.
    _, good = synthetic_hardware_log(hexa, perturbation={"armature": 0.03}, delay_ticks=0, plan=hplan)
    ok = fit_parameters(hexa, good, plan=hplan, n_boot=32, measure_delay=False)
    assert ok["application"]["passed"] is True and ok["application"].get("refused_by") is None, (
        f"a REAL armature calibration on the hexapod was refused: {ok['application']['verdict']}")
    assert ok["link_inertia"]["rival"]["explains_x"] < 1.0, (
        f"the inertia rival beat a correct armature calibration on the second body: "
        f"{ok['link_inertia']['rival']} (measured 0.106)")


def test_the_disclosure_names_both_catches_and_what_is_STILL_invisible(inertia_misspecified):
    """The honest half, and it is asserted rather than described, because a partial catch stated as total is
    the failure mode this whole file is about.

    Two families breached ``MIN_TRACKING_IMPROVEMENT_X`` and both are now caught -- by different checks, with
    different remedies. The disclosure has to say BOTH of those and it still has to name what neither sees.
    """
    _log, fit = inertia_misspecified
    field = fit["application"]["what_this_gate_does_not_catch"].lower()
    assert "torque-scale" in field and "link rotational inertia" in field, (
        f"the disclosure no longer names both families that breached this gate: {field}")
    assert "gear from 1.2 to 2.0" in field or "g from 1.2 to 2.0" in field, (
        f"the torque-scale catch RANGE is not stated, so a customer cannot tell what was ruled out: {field}")
    assert "inertia_scale x15 to x300" not in field, (
        "the link-inertia catch range must be the MEASURED one (x20 to x100, the points that actually breach), "
        "not a wider claim")
    assert "x20 to x100" in field, f"the link-inertia catch range is not stated: {field}"
    # ...and the limits that remain. Each of these is a measured hole, not a hedge.
    for phrase in ("some joints", "not_measurable_because"):
        assert phrase in field, f"the disclosure stopped naming a limit that still exists ({phrase}): {field}"
    assert "still not caught" not in field, (
        "the disclosure still says the link-inertia family is not caught, which is now false -- it is refused "
        "by the link_inertia check. A stale disclosure is the same defect as a missing one")


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
