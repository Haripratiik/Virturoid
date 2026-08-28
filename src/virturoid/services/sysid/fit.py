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

**That gate is NECESSARY AND NOT SUFFICIENT, and it was written down as more.** The claim that a
misspecification can never improve tracking -- "the misspecified ceiling is pinned at ~1.0 by construction" --
was MEASURED on 2026-08-12 and is RETRACTED at ``MIN_TRACKING_IMPROVEMENT_X``. What the band was sized on is
one family, link mass and inertia, whose pin is the GRAVITY term rather than anything structural. Errors with
no gravity signature go through: link rotational inertia at correct mass reaches 1.753x and a 20% torque-scale
error reaches 1.507x, both above the 1.5x gate, both with intervals excluding the truth. A pass means the
simulator got closer to your log; it does not mean the three numbers are the three quantities they are named
after, and ``application.what_this_gate_does_not_catch`` says so in the output.

**4b. BOTH of those families are now CLOSED, by two independent REPLAY checks rather than by a bigger
threshold.** The retraction above ends "a constant cannot separate these cases; only a wider model or a second,
independent check can", and these are those checks. Each asks the same question with a different scalar: can
ONE number explain this log at least as well as the forty-two the fit wants to write? If it can, the experiment
did not distinguish them and the forty-two are withheld -- and each names a different remedy, which is why they
are two verdicts and not one.

  * ``global_scale`` -- THE DRIVETRAIN. Scaling the applied torque by ``g`` is algebraically dividing M, b and
    f by ``g``, so it moves ALL THREE fitted parameters on EVERY joint by the same ``(1-g)/g`` of their priors.
    The rival replays the PRIOR model with every actuator's gear scaled by one number. Refused as
    ``torque_scale_suspected``; the remedy is the gear ratio / torque constant. MEASURED on the composed dog:
    refuses torque_scale 1.2 (the 1.507x breach) through 2.0 at rival 17.9-230x, and refuses NONE of ten
    correctly-specified fits, including one deliberately built proportional to the priors so that a coherence
    check alone would have false-positived on it.
  * ``link_inertia`` -- THE CAD. Scaling every link's rotational inertia by ``s`` adds ``(s-1)`` times each
    joint's own rotational inertia to the same ``diag(M)`` ARMATURE adds to, which is the degeneracy
    ``PARAMETER_ALSO_ABSORBS['armature']`` has named all along. The rival replays the PRIOR model with every
    link's inertia tensor scaled by one number and every MASS untouched. Refused as ``link_inertia_suspected``;
    the remedy is the inertia tensor. MEASURED on the composed dog: refuses inertia_scale x20 (the first point
    in that family that clears 1.5x) through x100 (the last one that does) at rival 12.4-181x, and refuses none
    of the correctly-specified fits.

**The second of those DID turn out to be worse than the defect on its first shipped form, and the reason it
was missed is worth more than the fix (2026-08-13).** The legitimate ARMATURE-ONLY calibration is itself an
inertia-like change -- reflected inertia adds to the same diagonal link inertia does -- so an inertia rival
might be structurally unable to tell "your tensor is misstated" from "your reflected inertia really differs",
and a check that refuses a correct calibration is worse than one that misses a wrong one. That risk was
confronted first, measured on nine correct fits, and declared closed. **It was not closed: all nine sat at
3.34x improvement or above, and the check false-refuses between about 1.5x and 1.9x.** Armature-only +0.009
(improvement 1.602, rival 1.108) and +0.010 (1.752, 1.024) are ORDINARY calibrations against a prior of 0.010
and both were refused, sending a customer to re-check CAD tensors that are correct.

The mechanism the file gave for why it could not happen was FALSE, and its own output said so one field away:
"an inertia scalar adds a term proportional to each joint's own rotational inertia, and those span ~40x across
this body". MEASURED via ``mj_fullM``, that span is **1.547x on the dog and 1.528x on the hexapod** -- which is
what ``rotational_inertia_span_x`` has been reporting as 1.55 all along. At a span of 1.55 a single scalar CAN
imitate a uniform armature offset to within about +/-20%, which is exactly why the naive rival wins near the
gate. See ``LINK_INERTIA_RIVAL_IMPROVEMENT_X`` for the statistic that survives and the three-body sweep that
sized it -- and for the reason SPAN is not the variable that predicts how sharp this check is, which was the
next thing this file got wrong.

**4c. AND A FOURTH REFUSAL SITS IN FRONT OF ALL THREE, because all three will happily name a part of the
customer's robot off a log that cannot support any verdict (2026-08-13).** MEASURED on a composed 8-legged
spider: two CORRECT armature-only calibrations came back ``torque_scale_suspected`` -- go and check your gear
ratio -- on a log whose joints travel 298.9 rad against a 1.634 rad commanded envelope, with MuJoCo reporting
"Nan, Inf or huge value in QACC" throughout its generation. Every injection on that body reads the same way,
including the control with nothing wrong with it at all. The plant diverged; the log is a record of that. So
``log_plausibility`` runs first, on the arrays alone, and refuses as ``implausible_log`` while naming NOTHING --
not the drivetrain, not the CAD, not the parameters. Its bound is the geometric midpoint of two measured
populations (sane 0.584-6.721 over 2864 readings on 18 WHEEL-FREE bodies; degenerate 14.648-42964 on the
spider), and the sane side deliberately includes ``inertia_scale`` x40/x100 so no catch is traded for it.
**That survey has been re-taken twice and both edges moved both times.** It was first taken at ZERO actuation
delay while the harness injects 2 ticks by default, which put the line low enough to refuse real logs at 40 ms.
Re-measured across delay it gave 12.5 -- on a sane ceiling that was a WHEELED ROVER, whose wheels have no
positional ceiling, and a degenerate floor that was an artefact of a delay grid skipping 50/80/110 ms. The
ratio is now formed over LIMITED joints only, unbounded joints are named and excluded, a body with no limited
joint is reported NOT MEASURABLE, and the line is 9.9. See ``LOG_EXCURSION_RATIO_MAX``.

**5. That gate is scored at the ACTUATION DELAY, and it had to be.** Both replays used to run at delay 0
against logs that had one, so both carried a timing error no fitted parameter could remove, and the ratio's
whole dynamic RANGE collapses toward 1 as the delay grows. MEASURED on the Menagerie Go2, full 12-joint plan,
sweeping the three parameters this estimator can move over 60 points at each delay: at 30 ms of injected delay
the largest delay-blind score anything in that family reaches is 1.012x against a 1.5x threshold, and at 40 ms
it is 1.003x. Where the whole family is under the threshold the gate refuses every fit on the merits of the
LOG's latency, not of the fit. Running both replays at the identified delay fixes it, which is why
``fit_parameters`` measures the latency BEFORE the trajectory.

Two corrections to how that used to be written down, both measured, both in
``docs/calibration_wedge_under_delay.md`` and ``tests/test_sysid_delay_wedge.py``: the delay at which it bites
on THIS robot is 30 ms rather than the 20 ms once quoted (the twin used to carry 40% of Unitree's declared
damping, and the ring was the phenomenon), and the quantity was called a CEILING obtained by substituting the
TRUE hardware model for the fitted one. **It is not a ceiling.** At delay 0 against a delayed log the replay
family does not contain the data-generating process, so the true parameters are not the residual's minimiser --
a twin with too much reflected inertia mimics transport lag and scores BETTER. Full plan at 20 ms: the truth
scores 1.791x, the best model in the family reaches 2.277x (+27%), and this package's own fit reads 1.858x,
already above the figure that was being called a bound. The conclusion survives on the envelope; the word
"ceiling" does not.

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
#: **The threshold has never moved. The SCORING POINT has, once, and the band below was re-measured after it.**
#: Both replays now run at the identified actuation delay rather than at zero. The reason is in
#: ``_trajectory_improvement``: scored at zero delay against a log that has one, this ratio's RANGE is set by
#: the delay rather than by the fit -- on the Menagerie Go2 at 30 ms, no model the estimator can express scores
#: above 1.012x -- so on a robot with real latency it refused every fit, including ones that had recovered all
#: three parameters to within 13.3%. (That range used to be written here as a CEILING of 1.016x at 20 ms. Both
#: halves were wrong: the figure was measured on a twin carrying 40% of Unitree's declared damping, and it is
#: not a bound at all -- see ``_trajectory_improvement``.)
#:
#: RE-MEASURED at the new scoring point, Go2, full 12-joint 120 s plan, three injections x three delays:
#:
#:                                              0 ms      20 ms     40 ms
#:   +30% link mass and inertia                 1.000     1.059     1.033    refused at every delay
#:   hardware IS the sim (nothing to find)      n/a*      0.000     0.000    refused at every delay
#:   ------------------------------------------------------------------- the band the threshold sits in
#:   frictionloss/damping/armature (correct)   12.214     1.683     1.484
#:
#:   * no pair was identified at all, so there was nothing to gate.
#:
#: Read that honestly. The band is NARROWER than the one this threshold was originally sized in (0.999 vs
#: 3.343): the misspecified ceiling rose to 1.059 and the correct floor fell to 1.484, because at 40 ms the
#: Go2's hips ring hard enough that trajectory RMS stops being a sensitive function of the parameters. The
#: threshold is above the 40 ms correct case, so that case is REFUSED -- a false negative, at double the delay
#: Hwangbo et al. name and double this package's own default. No misspecification IN THAT TABLE reaches 1.06 at
#: any delay measured -- and that scoping matters, because the table is one family: see the retraction below,
#: where two other families clear 1.5 outright. Moving the threshold down to admit 1.484 would put it within
#: 40% of that family's ceiling, which is not a trade this package should make silently.
#:
#: The ORIGINAL calibration, taken with both replays at delay 0 on a composed 14-DOF quadruped and a 35 s
#: excitation, is kept below because it is what sized the number:
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
#:   ------------------------------------------------------------ nothing was measured in this gap -- and the
#:                                                                 gap is NOT empty; see the retraction below
#:   frictionloss/damping/armature +0.05/+0.20/+0.010      3.343
#:   the default injection + 20 ms of delay                3.536
#:   the default injection x0.25                           4.726
#:   the default injection x0.5                            5.269
#:   the default injection, no delay                      16.365
#:
#: In the engineer's units the threshold says: the position RMS has to fall by at least a THIRD before we write
#: anything into your model. A control that is 2% better is not evidence of calibration.
#:
#: RE-RUN of that table against the code as shipped (2026-08-12): 3.343 / 5.269 / 16.365 / 0.999 / 0.995 come
#: back EXACTLY, and "+ 20 ms -> 3.536" comes back exactly as ``improvement_x_at_zero_delay`` (18.497 at the
#: new scoring point). x0.25 reads 4.584 rather than 4.726. **The three rows that most made the ceiling look
#: pinned do not reproduce at all: 0.065, 0.360 and 0.000 are all 1.000 today**, because the floor gate now
#: refuses every cell in them, nothing is applied and ``before == after``. Those rows were never measurements
#: of a fit failing to help; they are measurements of a fit that no longer happens.
#:
#: **THE BAND IS NOT EMPTY, AND THE SENTENCE THAT SAID IT WAS IS RETRACTED (2026-08-12).** It read: "Every
#: misspecification measured lands at or below 0.999 -- i.e. it never improves tracking AT ALL, which is the
#: structural signature of an error no fitted parameter can express... the misspecified ceiling is pinned at
#: ~1.0 BY CONSTRUCTION and needs little margin." That is the same shape of argument as the delay-blind
#: "ceiling" retracted in ``_trajectory_improvement`` hours earlier -- a claim about what a fit COULD do,
#: standing where a measurement of what it DOES belongs -- and it fails for a related reason.
#:
#: What the table above actually swept is ONE misspecification family: link mass and inertia, moved TOGETHER.
#: That family is genuinely pinned, and it has now been swept properly -- x0.4 / 0.6 / 0.8 / 0.9 / 1.1 / 1.15 /
#: 1.2 / 1.3 / 1.45 / 1.6 / 2.0 / 3.0 gives 0.961 / 0.939 / 1.000 / 1.000 / 1.000 / 0.998 / 0.999 / 0.995 /
#: 0.998 / 0.999 / 0.988 / 0.996, i.e. never once above 1.000. So does a CENTRE-OF-MASS error (5 / 20 / 50 mm
#: on every link: 1.000 / 1.000 / 1.000). **But the pin is the GRAVITY term, not a structural property of
#: misspecification.** Mass carries gravity with it, a gravity error is a static tracking offset no dissipative
#: or inertial parameter can remove, and it therefore dominates ``before`` and survives untouched into
#: ``after``. Remove the gravity term and the pin goes with it. MEASURED on the same robot, same 35 s
#: excitation, delay 0, through the shipped ``fit_parameters``:
#:
#:   link ROTATIONAL INERTIA only, mass held exactly right (``synthetic_hardware_log(inertia_scale=)``)
#:     x2   1.000 | x5   1.007 | x10  1.181 | x15  1.466 | x20  1.620 | x25  1.707
#:     x30  1.745 | x40  1.753 | x50  1.730 | x70  1.691 | x100 1.650 | x300 1.433
#:   the applied torque scaled -- wrong gear ratio, wrong torque constant, unmodelled gearbox efficiency
#:   (``synthetic_hardware_log(torque_scale=)``; "gear ratio" is in ``assumed_correct`` below)
#:     x0.5 1.414 | x0.75 1.466 | x0.9 1.467 | x0.95 1.087 | x1.1 1.364 | x1.2  1.507
#:     x1.25 1.536 | x1.35 1.600 | x1.5 1.624 | x1.75 1.647 | x2.0 1.500 | x3.0 1.166
#:
#: **Both families cross 1.0 and both cross this threshold.** The measured maximum over everything swept is
#: 1.753x, at link inertia x40 -- and at that point armature comes back "identified" on 14/14 joints with
#: 14/14 intervals EXCLUDING the true, unchanged value, the fit PASSES, and all 14 numbers are written into the
#: compiled model. That is the exact failure this gate was built to stop, reached by a different door.
#:
#: The mechanism is not mysterious in either family, and both were predicted before they were measured:
#:
#:   * A link-inertia error with mass held right has NO gravity signature. It enters each joint's equation
#:     through the same ``qdd`` term armature does -- which ``PARAMETER_ALSO_ABSORBS`` has said all along --
#:     so armature can express most of it. At x30 the median fitted armature delta is 0.0105 against the
#:     0.0094 that would mimic the error exactly. HOW REALISTIC: with mass AND geometry both right a link's
#:     own inertia cannot be more than ~3x wrong (dumbbell mL^2/4 vs uniform rod mL^2/12), which is ~+4% of
#:     diag(M) and about 1.005x. So this family kills "never improves tracking AT ALL" at a realizable size
#:     and only reaches the GATE at sizes that need the geometry to be wrong too. It is the clean statement of
#:     the mechanism; the family that breaches the gate at an ordinary error is the next one.
#:   * Scaling the applied torque by g is ALGEBRAICALLY the same as dividing M, b and f by g, and the
#:     estimator can move all three. What it cannot move is gravity, which is why the ratio saturates around
#:     1.65 instead of running away. The fitted deltas land on the predicted ``(1 - g)/g`` times each prior:
#:     at g = 0.5 the median damping delta is +0.841 against a predicted +0.800, frictionloss +0.105 against
#:     +0.120.
#:
#: AND THE OTHER HALF OF THAT SENTENCE -- "the correct floor is a sample and could go lower on a smaller
#: perturbation" -- was the honest half, so it was measured too. It goes lower than the band can survive:
#: sweeping the DEFAULT injection down, x0.5 5.269 | x0.25 4.584 | x0.125 3.564 | x0.0625 **1.126** (16/42
#: identified, correctly specified, refused) | x0.03 1.000 (nothing identified). **So the two populations do
#: not merely touch, they OVERLAP**: a correctly-specified fit reads 1.126 while a misspecified one reads
#: 1.753. No threshold on this quantity can separate them, which is the fact that matters and is not a
#: property of where the threshold was put.
#:
#: WHAT SURVIVES, and it is narrower than what was claimed: this gate catches a misspecification whose
#: signature is mostly GRAVITY (link mass, centre of mass) and does NOT catch one the fitted parameters can
#: mimic (link inertia at correct mass, a torque-scale error). It is NECESSARY AND NOT SUFFICIENT. The
#: sentences that presented it as a general detector of "an error no fitted parameter can express" are wrong
#: and are corrected here and in ``application_gate.why_this_threshold``.
#:
#: **THE THRESHOLD IS NOT MOVED, deliberately.** Raising it above the measured 1.753 would (a) refuse
#: correctly-specified fits that already sit under it -- the Go2's correct case reads 1.484 at 40 ms and 1.121
#: on the full plan, both already false negatives at 1.5 -- and (b) be sized against a GRID MAXIMUM over four
#: families on one body, which is not a bound and would be the same fiction this paragraph just retracted.
#: A constant cannot separate these cases; only a wider model or a second, independent check can.
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
#:
#: Two rows in that older table are BODY-SPECIFIC and read as general, which is worth saying next to them: the
#: "+ 20 ms of delay -> 3.536, 42/42 identified" figures are the composed dog's. On the Go2 the same case is
#: 1.683 at the new scoring point, and the delay-blind score of the Go2's own TRUE parameters was 1.016 -- a
#: regime with no range left in it (at the comparable collapsed point on the DECLARED drivetrain, 30 ms, the
#: best model in the estimator's whole 60-point family reaches only 1.012x), so 3.536 was never reachable
#: there by any fit. The 1.016 figure itself was NOT re-measured on the declared drivetrain and is not a bound;
#: see ``_trajectory_improvement``. The composed dog's bench loop tolerates delay in a way a real quadruped's does not -- its worst
#: tracking RMS moves 0.0339 -> 0.0365 rad across 0-40 ms of delay while the Go2's hips go 0.0298 -> 0.1848 --
#: which is why every number in this package needs re-taking on an imported body before it is quoted as
#: general. See ``docs/calibration_wedge_under_delay.md`` section 3c.
MIN_TRACKING_IMPROVEMENT_X = 1.5

# ---------------------------------------------------------------------------------------------------------
# THE SECOND, INDEPENDENT CHECK -- the one that closes the breach the constant above provably cannot.
# ---------------------------------------------------------------------------------------------------------
#
# The paragraph above ends "a constant cannot separate these cases; only a wider model or a second, independent
# check can", and this is that check. It was written as a HYPOTHESIS and then measured, and the hypothesis is
# the degeneracy itself read the other way round: what makes a torque-scale error invisible to position
# tracking is exactly what makes it visible in the fit's OWN OUTPUT. Scaling the applied torque by ``g`` is
# algebraically dividing M, b and f by ``g``, so it drives ALL THREE fitted parameters, on EVERY joint, by the
# same multiplicative ``(1 - g)/g`` of their priors. A genuine per-joint dissipation change has no reason to be
# coherent like that.
#
# It is TWO parts, and both are load-bearing. Part one is free and is a filter, not a verdict; part two is the
# verdict and costs a handful of replays, so it only runs when part one has already fired.
#
#   1. COHERENCE. Take each parameter's median fitted delta as a FRACTION of its prior -- one number per
#      parameter, across all joints -- and ask how much of that 3-vector a single global scalar explains
#      (``1 - SS_res/SS_tot`` about that scalar, i.e. an R^2 about zero rather than about the mean).
#   2. THE RIVAL. Take the scalar the coherence implies, put the PRIOR parameters back, scale every actuator's
#      gear by it, and replay. That is a ONE-number explanation of the same log competing with the 42 numbers
#      the fit wants to write. It is scored on POSITION RMS -- the same quantity ``improvement_x`` uses and one
#      neither model optimised. If one number tracks the log at least as well as forty-two do, the experiment
#      did not distinguish them, and the forty-two must not be silently written into a customer's robot.
#
# **PART ONE IS NO LONGER A GATE ON PART TWO (2026-08-12), and that is a measured change, not a tidy-up.**
# The filter used to decide whether the replay ran, so a fit whose coherence was diluted got a pass without
# ever being asked the honest question. The rival now runs on EVERY fit that would otherwise be applied. Safe
# (largest torque rival over the nine correct passing fits: 0.389) and cheap (~4.0 s on a 10.7 s fit), both
# measured through the shipped code and both in the table below.
#
# **IT DID NOT CLOSE THE DILUTION HOLE, which is what it was expected to do.** The disclosed case -- a 1.35x
# torque-constant error superposed on the default injection, improvement_x 1.689, coherence 0.484 -- is still
# applied. With the filter gone the rival runs on it and LOSES, at 0.869. That is not a search failure, which
# was the obvious suspicion since the dilution also corrupts the seed (implied g 0.543 against a true 1.35):
# scored over a DENSE 33-point grid of g from 0.25 to 4.0, the best any gear scalar does on that log is
# **0.893**, still under the line. The same grid finds 5.081 on the undiluted x1.35 control, so it is not too
# coarse to see a win -- there is no win there. The right reading is that the log genuinely contains a
# dissipation change no single scalar produces, so one number IS a worse explanation than forty-two and the
# rival says so correctly. The defect in that fit is that its values carry a factor, not that a scalar explains
# it, and no rival of this shape is the instrument for that. It stays disclosed. Full family, all applied or
# already refused, none caught: kt x1.35 (1.689, rival 0.869), kt x0.7 (1.530, 0.263), plant g1.35 on top of
# the injection (1.815, 0.342), kt x1.6 (1.209, refused by tracking).
#
# MEASURED on the composed 14-DOF dog, 35 s excitation, delay 0 unless stated, n_boot=64, through the shipped
# ``fit_parameters`` -- 29 fits, ~19 s each including both rivals. ``rival`` is ``after_rms / rival_rms``, i.e.
# how many times better the ONE-number rival tracks than the fit does (>= 1.0 means the rival wins). BOTH
# rivals were scored on EVERY row, including the rows where the shipped code would not spend one, because a
# column that is never measured is a column that gets invented -- an earlier version of this table carried
# exactly that, ">1e6" and "4.04" where the shipped code measures 30.1 and 29.8:
#
#                                improvement_x   g-coherence  implied g  TORQUE rival | inertia rival
#   torque_scale x1.2                1.507          0.935       1.195       30.119    |    0.664
#   torque_scale x1.25               1.536          0.954       1.257       29.773    |    0.652
#   torque_scale x1.35               1.600          0.967       1.387       17.861    |    0.625
#   torque_scale x1.5                1.624          0.986       1.462       25.529    |    0.616
#   torque_scale x1.75               1.647          0.988       1.690      230.111    |    0.608
#   torque_scale x2.0                1.500          0.951       1.584       57.198    |    0.667
#                                                       ^ all six REFUSED as torque_scale. The inertia rival
#                                                         loses on every one, and does not even run: a g > 1
#                                                         drives armature NEGATIVE, so the implied inertia
#                                                         scale is negative (-3.53 at g=1.2) and unphysical.
#   ------------------------------------------------------------------------------ LINK INERTIA at correct mass
#                                improvement_x   i-coherence  implied s  torque rival | INERTIA rival
#   inertia_scale x20                1.620          0.9946      26.16        0.617    |   12.365   REFUSED
#   inertia_scale x25                1.707          0.9934      31.16        0.571    |  181.525   REFUSED
#   inertia_scale x30                1.745          0.9929      35.89        0.500    |   12.418   REFUSED
#   inertia_scale x40                1.753          0.9921      44.91        0.420    |   50.690   REFUSED
#   inertia_scale x50                1.730          0.9915      53.35        0.387    |   20.705   REFUSED
#   inertia_scale x70                1.691          0.9887      69.38        0.356    |   64.243   REFUSED
#   inertia_scale x100               1.650          0.9820      91.87        0.354    |   13.278   REFUSED
#   ---------------------------------------------------------------------------- already refused at 1.5x anyway
#   inertia_scale x2                 1.000            --          --           --     |     --     nothing identified
#   inertia_scale x5                 1.007          0.9837      11.83        0.874    |   15.043
#   inertia_scale x10                1.181          0.9907      16.57        0.841    |  129.019
#   inertia_scale x15                1.466          0.9920      21.50        0.671    |  188.357
#   inertia_scale x300               1.433          0.9191     215.32        0.413    |   16.865
#   link_scale x1.3 (mass+inertia)   0.995          0.8331      11.84        1.286    |    1.005
#                                                       ^ NOT caught by either rival AS SHIPPED, because the
#                                                         tracking gate refuses them first and no replay is
#                                                         spent. Every one that BREACHES is caught; the rows
#                                                         above the line are the whole family, not the winners.
#   -------------------------------------------------------------------------------------- CORRECTLY SPECIFIED
#                                improvement_x   g-coh / i-coh   implied g / s   TORQUE rival | INERTIA rival
#   default injection, no delay     16.365       0.605 / 0.9783   0.389 / 103.8      0.047    |    0.062
#   default x0.5                     5.269       0.570 / 0.9775   0.553 /  56.3      0.190    |    0.196
#   default x0.25                    4.584       0.519 / 0.9796   0.700 /  32.1      0.327    |    0.227
#   +0.05/+0.20/+0.010               3.343       0.639 / 0.9770   0.617 /  40.6      0.371    |    0.305
#   frictionloss only +0.08          8.330       0.507 / 0.9512   0.775 /   8.2      0.160    |    0.119
#   damping only +0.6                7.248       0.643 / 0.9467   0.745 /  11.7      0.198    |    0.138
#   ARMATURE only +0.03             17.910       0.319 / 0.9831   0.492 /  97.7      0.022    |    0.099
#   PROPORTIONAL +25% on f AND d     4.025       0.980 / 0.9452   0.807 /   9.1      0.389    |    0.248
#   default + 20 ms of delay        18.497       0.609 / 0.9780   0.391 / 102.1      0.048    |    0.055
#   ---------------------------------------------------------------------------- refused by TRACKING, not by us
#   default x0.0625                  1.126       0.277 / 0.9758   0.889 /  14.0      1.163    |    0.915
#
# READ THE LAST ROW, because it is the one that decides HOW FAR the unconditional rival may be taken. At
# x0.0625 the injection is barely above the measurement floor, only 16 of 42 pairs are identified, and the fit
# is so weak that a single gear scalar beats it (1.163). It is a CORRECT calibration. It is not refused by
# either rival, and the reason is structural rather than lucky: the rivals only run on a fit that would
# otherwise be WRITTEN, and this one is already refused at 1.126 against the 1.5x tracking gate. Moving the
# rivals ahead of that gate would turn a true "your experiment was too small" into a false "your gearbox is
# wrong", so they stay behind it. That is the boundary of this fix and it is stated rather than discovered.
#
# THE TWO SEPARATIONS, on the population that actually reaches a rival -- AS THEY LOOKED FROM THESE NINE ROWS:
#   torque scale   caught 17.861 ... 230.111   correct 0.022 ... 0.389   -> 46x gap, 1.0 sits inside it
#   link inertia   caught 12.365 ... 181.525   correct 0.055 ... 0.305   -> 41x gap, 1.0 sits inside it
#
# **THE SECOND OF THOSE TWO LINES IS RETRACTED (2026-08-13), and the first is narrowed.** The correct column is
# a sample from NINE fits that all improved tracking by 3.34x or more. ``explains_x`` divides the rival's
# improvement by the FIT's, so it rises as the fit weakens -- and swept densely just above the 1.5x gate, the
# link-inertia rival reaches 1.108 on an ORDINARY armature-only calibration and REFUSES it. The 41x gap is real
# for strong fits and does not exist near the gate. The repair keeps 1.0 as one of two conditions and adds a
# statistic with no fit-strength confound; see ``LINK_INERTIA_RIVAL_IMPROVEMENT_X``. On the torque side the same
# confound is present (the correct maximum in the band is higher than the 0.389 quoted here) but no correct fit
# has yet been measured above 1.0 with the tracking gate passed -- so that check is NOT changed, and the reason
# it is not is stated at ``GLOBAL_SCALE_RIVAL_X`` rather than assumed.
#
# AND THE COHERENCES DO **NOT** SEPARATE -- both of them, for different reasons, which is why both checks are
# replays. The torque coherence false-positives on a real dissipation change proportional to the priors (0.980,
# above four of the six cases it exists to catch). The inertia coherence is worse: 0.945-0.983 on correct fits
# against 0.982-0.995 on the caught ones, overlapping outright. A threshold squeezed into that overlap is the
# same mistake this file has already retracted twice, so there is none -- the inertia check has NO coherence
# filter at all and gates only on the implied scale being physical.
#
# EACH RIVAL FAILS ON THE OTHER'S FAMILY, and that half is intact: the torque rival divides the WHOLE joint
# equation -- inertia, damping AND friction -- so it can only win when the error really was multiplicative on
# torque; on a pure inertia error it has to corrupt the (correct) damping and friction to buy the inertia, and
# it loses (0.354-0.617).
#
# **THE OTHER HALF OF THIS PARAGRAPH IS RETRACTED (2026-08-13).** It read: "the inertia rival adds a term
# PROPORTIONAL to each joint's own rotational inertia, which on this body spans 40x from trunk to distal joint,
# so it CANNOT imitate the UNIFORM constant a real armature change adds". MEASURED via ``mj_fullM``, that span
# is 1.547x on the composed dog and 1.528x on the composed hexapod -- and ``link_inertia_signature`` has been
# shipping it as ``rotational_inertia_span_x: 1.55`` the whole time, one field away from the comment claiming
# 40. At 1.55x a scalar CAN imitate a uniform offset to within about +/-20%, so the inertia rival is a NEARLY
# DEGENERATE competitor to a real armature calibration, and it loses on armature-only +0.03 (0.099) for a
# different reason than the one written here: that fit is STRONG (17.910x), and ``explains_x`` is divided by the
# fit's strength. Weaken the same correct calibration to +0.009 and the identical rival reads 1.108 and REFUSES
# it. What actually separates the populations is the rival's improvement over the PRIOR -- 0.98-1.93 when the
# log came from a real armature change ON THOSE TWO BODIES, and 19.604-633.220 when it came from a real inertia
# error ON THOSE SAME TWO BODIES, because in the second case the rival's one-parameter family CONTAINS the
# data-generating process. **BOTH OF THOSE RANGES ARE THE RETIRED TWO-BODY SAMPLE and neither is the figure in
# force (this line printed the catch range as a bare "19.6-633", with no scope, until 2026-08-13).** Measured
# across three bodies the catch floor is 14.469, not 19.604, and the correct population's ceiling over every
# body now measured is 4.037 on the composed centipede, not 1.93. See ``LINK_INERTIA_RIVAL_IMPROVEMENT_X`` for
# the figures that are current and for the threshold they size.
#
# WHERE THE FALSE "~40x" CAME FROM, because how a plausible constant survives review is worth more than the
# constant: it is a REAL measured span, on the WRONG KIND OF BODY. The composed 6-joint ARM measures 43.94x and
# the composed snake 33.51x -- a shoulder against a wrist really is two decades apart in ``R_a``, and a serial
# chain is the mental image that sentence was written from. Every LEGGED body measured is an order of magnitude
# tighter (dog 1.547, hexapod 1.528, spider 2.205, cat 2.628, horse 4.040), because a leg is four short links
# repeated four to eight times rather than one long taper. The number was not invented; it was carried across
# from a morphology this check was never measured on.
#
# THE SECOND BODY, and the two checks come out of it DIFFERENTLY. ``docs/calibration_wedge_under_delay.md``
# section 12's warning stands -- a number from the composed dog is not general -- so both were re-taken on a
# composed HEXAPOD (18 joints, 45 s plan, n_boot=48, delay 0), end to end through the shipped code:
#
#                              improvement_x   torque rival   inertia rival   VERDICT
#   correct default               20.879          0.007          0.051        applied
#   correct ARMATURE only         18.107          0.005          0.106        applied
#   correct PROPORTIONAL f,d       9.384          0.107          0.108        applied   (torque coherence 0.964)
#   torque_scale x1.25             0.989            --             --         refused (TRACKING)
#   torque_scale x1.5              0.995            --             --         refused (TRACKING)
#   torque_scale x2.0              1.005            --             --         refused (TRACKING)
#   inertia_scale x30              2.015          0.049         18.125        REFUSED (link_inertia)
#   inertia_scale x40              1.973          0.047        320.550        REFUSED (link_inertia)
#
#   * THE LINK-INERTIA CHECK GENERALISES, and this is the first time either catch has been shown on a second
#     body. The breach is BIGGER here than on the dog (2.015 against 1.745) and the check refuses it on the
#     merits, with the torque rival correctly losing at 0.047-0.049 so the customer is sent to their CAD and
#     not their gearbox.
#   * THE TORQUE-SCALE CATCH STILL IS NOT SHOWN ANYWHERE BUT THE DOG. On this hexapod, as on the Menagerie Go2
#     before it, THE BREACH DOES NOT REPRODUCE -- a torque-scale error scores 0.989-1.005 and the tracking gate
#     refuses it unaided, so there is nothing left for the second check to catch. That is not evidence the
#     torque catch generalises; it is more evidence the FIRST gate's blind spot is body-specific.
#   * The FALSE-REFUSAL side is now measured on three bodies for the torque check (dog, hexapod, Go2) and two
#     for the inertia check (dog, hexapod), including on both bodies' armature-only calibration.
#   * **AND ON A SAMPLE THAT NEVER ENTERED THE REGION WHERE EITHER CHECK IS WEAKEST (added 2026-08-13).** Every
#     correct fit above improves tracking by 3.34x or more. Swept densely between 1.5x and 3.0x the inertia
#     check false-refused, which is what ``LINK_INERTIA_RIVAL_IMPROVEMENT_X`` repairs -- and the torque check's
#     correct-population maximum in that band is 0.852, not the 0.389 quoted below. It still loses, but by 1.17x
#     rather than by 2.6x, and that is a live risk rather than a comfortable margin. See ``GLOBAL_SCALE_RIVAL_X``.
#
#: How much of the per-parameter delta vector a single global torque scalar explains. **This is no longer a
#: gate on anything** -- it is reported, it names the mechanism in the refusal verdict, and it seeds the
#: rival's search. It stopped being a filter when running the rival unconditionally was measured to be safe
#: (largest torque rival over nine correct passing fits: 0.389) and cheap (~4.0 s), which is what closed the
#: dilution hole the filter had opened. Measured band, kept because it is what says a threshold here cannot
#: work: the six caught torque-scale cases run 0.935-0.988 and the ten correctly-specified fits run
#: 0.277-0.980.
GLOBAL_SCALE_COHERENCE_R2 = 0.85

#: ...and how well the one-number rival has to track before the 42-number fit is refused. 1.0 is not a tuned
#: constant, it is the question: does ONE number explain this log AT LEAST AS WELL as the numbers we are about
#: to write? Measured margin either side: caught cases 17.9 and up, correct fits that reach this stage 0.852
#: and below. (This line read "4.04 and up ... 0.38" from a stale run until 2026-08-12; the shipped code's floor
#: is 17.9. The error was in the conservative direction, which is exactly why nobody noticed it. It then read
#: "0.539 and below ... a ~33x gap with 1.0 sitting inside it, not near an edge" until 2026-08-13, which is the
#: half corrected below: the true correct-population maximum is 0.852, so 1.0 sits 1.17x from that edge and NOT
#: in the middle of anything.)
#:
#: **THE "0.539 AND BELOW" HALF WAS WRONG, AND THE CHECK IS DELIBERATELY NOT CHANGED (2026-08-13).** The
#: link-inertia check next door shipped the same statistic and false-refused correct calibrations just above the
#: 1.5x tracking gate, because ``explains_x`` divides the rival's improvement by the FIT's and the fit is weak
#: there. The same confound is in THIS number. Re-measured over a dense sweep of correctly-specified fits
#: landing in 1.5x-3.0x on the composed dog and the composed hexapod (n=63 correct fits reaching a rival), the
#: torque rival's correct-population maximum is **0.852** (dog, frictionloss-only +0.030, improvement_x 1.535,
#: reproduced through the shipped code 2026-08-13), not 0.389 and not 0.539. The margin to 1.0 is 1.17x, not
#: 2.6x, and an unnormalised statistic with 17% of headroom is not a comfortable place to be.
#:
#: **AND "IT STILL LOSES ON EVERY CORRECT FIT MEASURED" IS NO LONGER TRUE EITHER (2026-08-13).** On a composed
#: 8-legged spider, correct armature-only calibrations at +0.009 (improvement_x 1.575) and +0.05 (2.057) were
#: REFUSED as ``torque_scale_suspected`` at 1.023 and 1.218, deterministic at n_boot 64 and 96, sending the
#: customer to their gear ratio and torque constant. That is a false refusal by THIS check. What it is NOT is
#: evidence about the threshold: that body's synthetic log is physically degenerate -- max|q_meas| 298.9 rad
#: against a 1.63 rad commanded envelope, with MuJoCo QACC warnings throughout -- and both fits are now refused
#: earlier, as ``implausible_log``, by ``LOG_EXCURSION_RATIO_MAX``. On a log this estimator should have ruled on
#: at all, the correct-population maximum is still 0.852 and no false refusal has been observed.
#:
#: THE LINE IS NOT MOVED, and the reason is the absence of evidence rather than the presence of margin: there is
#: no two-population sweep behind this side. The inertia threshold next door was sized on a measured correct
#: ceiling AND a measured catch floor on three bodies; here the catch population is six torque_scale points on
#: ONE body (the dog -- the breach does not reproduce on the hexapod, the cat or the Go2), so there is nothing
#: to take a midpoint of. Moving 1.0 on this side would be a number chosen to look safe. What would decide it is
#: a correct fit measured above 1.0 here ON A SANE LOG; the nearest known point is the x0.0625 injection at
#: 1.163, which only the tracking gate keeps out. Named as a live risk in ``application_gate`` instead.
GLOBAL_SCALE_RIVAL_X = 1.0

# ---------------------------------------------------------------------------------------------------------
# THE THIRD CHECK -- the SAME trick, a different scalar, closing the OTHER family.
# ---------------------------------------------------------------------------------------------------------
#
# The block above ends by naming what it does not catch: LINK ROTATIONAL INERTIA at correct mass, 1.745x and
# 1.753x, written into the model. This is the check that closes it, and it is the same shape of argument with
# a different one-number rival.
#
# The hypothesis: a link-inertia error should be explicable by ONE number -- scale every link's rotational
# inertia by a scalar and replay -- where the 42-parameter fit needs many. Scaling ``body_inertia`` by ``s``
# adds EXACTLY ``(s-1) * R_a`` to joint a's own ``diag(M)``, where ``R_a`` is the rotational part of that
# diagonal, and armature is the parameter that adds to the same diagonal. So the fit absorbs the error into
# armature, and a single ``s`` reproduces the plant.
#
# THE RISK THAT DECIDED THIS, confronted first, declared closed -- AND IT WAS NOT (2026-08-13). The legitimate
# ARMATURE-ONLY calibration is ITSELF an inertia-like change: reflected inertia adds to the same diagonal link
# inertia does. So an inertia rival might be structurally unable to separate "your link inertia is misstated"
# from "your reflected inertia genuinely differs from the prior", and a check that refuses a correct calibration
# is WORSE than the breach. The measurement taken then was this:
#
#                                    improvement_x   inertia coherence   implied s   RIVAL explains_x
#   inertia_scale x30                    1.745            0.9929           35.89          12.418
#   inertia_scale x40                    1.753            0.9921           44.91          50.690
#   ARMATURE ONLY +0.03 (correct)       17.910            0.9831           97.73           0.099
#
# Every one of those numbers still reproduces exactly. THE INFERENCE FROM THEM DOES NOT. Nine correct fits were
# sampled and all nine sat at improvement_x 3.34 or above, so the region right above the 1.5x gate -- where a
# customer's fit is most likely to land and where the statistic misbehaves -- was never tested. MEASURED there
# (composed dog, same 35 s excitation, bit-identical at n_boot 64 and 96):
#
#   armature-only +0.008   improvement 4.009   inertia rival 0.436   applied
#   armature-only +0.009   improvement 1.602   inertia rival 1.108   REFUSED as link_inertia   <- WRONG
#   armature-only +0.010   improvement 1.752   inertia rival 1.024   REFUSED as link_inertia   <- WRONG
#   armature-only +0.011   improvement 1.925   inertia rival 0.938   applied, 6% from refusal
#
# +0.010 against a prior of 0.010 is a reflected inertia twice the prior's -- an ordinary calibration, not a
# caricature -- and the customer was told to go and check CAD tensors that are correct.
#
# AND THE MECHANISM WRITTEN DOWN FOR WHY THAT COULD NOT HAPPEN WAS FALSE. It read: "an inertia scalar adds
# ``(s-1) * R_a``, which is PROPORTIONAL to each joint's own rotational inertia... on this body ``R_a`` spans
# ~40x between the trunk and a distal leg joint, so no single ``s`` can imitate a uniform armature offset."
# MEASURED via ``mj_fullM``, ``R_a`` spans **1.547x on the composed dog and 1.528x on the composed hexapod**.
# The shipped ``rotational_inertia_span_x`` field was printing 1.55 one line below the comment claiming 40 --
# an output contradicting its own docstring, which is the exact defect this file has already had to fix twice.
# At a span of 1.55 a scalar CAN imitate a uniform offset, to within roughly +/-20% of it. That is not a small
# correction to the story; it is the reason the false refusal exists.
#
# SO WHAT DOES SEPARATE, and it is not the span. When the log really WAS generated by an inertia scale, the
# rival's one-parameter family CONTAINS the data-generating process: the rival's residual collapses and it gets
# the PRIOR model 19.6x to 633x closer to the log. When the log was generated by a real armature offset, the
# rival can only approximate it within what the 1.55x span allows, and it tops out at 1.93x closer. THAT is the
# separation -- an absolute one, in ``before_rms / rival_rms``. The old statistic ``explains_x`` divided that by
# the FIT's own improvement, which ranges over a decade inside the correct population, and the division is what
# destroyed it. See ``LINK_INERTIA_RIVAL_IMPROVEMENT_X``.
#
# AND IT IS NOT AN ARTEFACT OF A WEAK SEARCH, which is the way this measurement could have flattered itself.
# The rival is a local search seeded from the signature, so "the correct fit wins" could just mean "we did not
# look hard enough". Scored instead over a DENSE grid -- 36 values of ``s`` log-spaced from 0.1 to 300, the
# rival's global optimum on this body -- the armature-only calibration still wins by the same margin: best
# ``s`` 60.5, ``explains_x`` **0.099**, identical to the seeded search's 0.099. The other two correct controls
# behave the same (default injection 0.062 at its global optimum, proportional f,d 0.249). The seeded search is
# not leaving a win on the table; there is no win there to leave. (That cuts both ways now: a STRONGER search
# would not rescue the old statistic either -- the +0.009 rival already wins on the merits.)
#
# THE COHERENCE STATISTIC DOES NOT SEPARATE HERE, AND IS THEREFORE NOT A FILTER. Measured across the whole
# population it reads 0.945-0.983 on correct fits and 0.992 on the caught ones -- overlapping bands with no gap
# to put a line in. Squeezing a threshold between 0.983 and 0.992 is exactly the knife-edge this file has twice
# had to retract, so it is not done: the coherence is REPORTED (it is the rival's seed and it is worth seeing)
# and the RIVAL alone is the verdict. That is a deliberate difference from the global-scale check, where the
# coherence had a real 0.85 gap to sit in.

#: How well the one-number LINK-INERTIA rival has to track THE FIT before the fit may be refused. 1.0 is the
#: question, not a knob: does one number explain this log at least as well as the numbers we are about to
#: write? **It is NECESSARY AND NOT SUFFICIENT, and shipping it as sufficient was a defect (2026-08-13).** On
#: its own it refused correctly-specified armature-only calibrations at improvement_x 1.602 and 1.752 -- see
#: ``LINK_INERTIA_RIVAL_IMPROVEMENT_X``, which is the second condition and the one that separates. It is kept
#: as a conjunct because dropping it would allow refusing a fit the rival LOSES to, which no evidence supports.
LINK_INERTIA_RIVAL_X = 1.0

#: ...and how much of the LOG the one-number rival has to explain, measured against the PRIOR model rather than
#: against the fit. ``rival_improvement_x = before_rms / rival_rms``. THIS is the condition that separates the
#: two populations, and it exists because ``LINK_INERTIA_RIVAL_X`` alone provably does not.
#:
#: WHY A SECOND NUMBER RATHER THAN A BIGGER FIRST ONE. Write ``explains_x`` out:
#:
#:     explains_x = fit_rms / rival_rms = (before/rival) / (before/after) = rival_improvement / fit_improvement
#:
#: It is ALREADY a normalised ratio, and the normalisation is the bug: the denominator is the fit's own
#: strength, which ranges from 1.5x to 20.9x WITHIN the correctly-specified population on these two bodies. The
#: same rival therefore scores 0.099 against a strong correct fit and 1.108 against a weak one, and the ranking
#: is monotone in fit strength rather than in whether anything is wrong (measured, armature-only on the dog:
#: 17.91 -> 0.099, 10.86 -> 0.166, 7.81 -> 0.232, 6.09 -> 0.296, 4.01 -> 0.436, 2.55 -> 0.647, 1.75 -> 1.024,
#: 1.60 -> 1.108). Dividing by the fit is what destroys the statistic; the numerator alone does not have the
#: confound. (Two other candidates were measured on the same sweep and are worse: the "fraction of the fit's
#: gain the rival reproduces", ``(before-rival)/(before-after)``, separates only 1.16 from 1.93; and raising
#: ``LINK_INERTIA_RIVAL_X`` itself would put the line at 1.108 against a catch floor of 11.05 -- a real gap, but
#: on a quantity whose correct-population value has no ceiling as the fit gets weaker.)
#:
#: THE SWEEP THAT SIZED IT is in ``tests/test_sysid_link_inertia_danger_band.py`` and reproduced in
#: ``docs/calibration_wedge_under_delay.md`` section 15i: a DENSE sweep of correctly-specified fits, on the
#: composed dog (14 joints), the composed hexapod (18 joints) and the composed CAT (14 joints), targeted at the
#: improvement_x band from 1.5x to 3.0x that the original nine samples never entered -- armature-only at many
#: magnitudes (the natural victim, since it is itself an inertia-like change), plus frictionloss-only,
#: damping-only, mixed fits, and an armature offset built PROPORTIONAL to each joint's own ``R_a``, which is the
#: exactly-degenerate correct calibration this rival's one-parameter family contains by construction.
#:
#: **THE FIRST VERSION OF THIS BLOCK PUT THE CORRECT CEILING AT 1.930 AND THE CATCH FLOOR AT 19.604. BOTH WERE
#: HALF-SAMPLES AND BOTH MOVED, IN OPPOSITE DIRECTIONS, ON THE FIRST BODY THAT WAS NOT A DOG OR A HEXAPOD
#: (2026-08-13).** The two bodies it was sized on have nearly the same rotational-inertia span (1.547x / 1.528x).
#: The composed cat spans 2.628x and reads roughly DOUBLE on the same statistic. Measured, cat, 35 s excitation,
#: delay 0, n_boot 64, through the shipped ``fit_parameters``:
#:
#:                                          improvement_x   explains_x   rival_improvement_x
#:   armature-only +0.003                       1.542          1.989           3.089
#:   armature-only +0.0045                      2.229          1.386           3.083
#:   armature-only +0.006                       2.840          1.126           3.181
#:   armature-only +0.008                       2.400          1.312           3.142
#:   armature +30 x each joint's own R_a        1.571          2.228           3.495
#:   armature +45 x each joint's own R_a        2.285          1.610           3.679   <- the correct CEILING
#:   armature +60 x each joint's own R_a        2.939          1.237           3.626
#:   frictionloss-only +0.020                   2.108          0.475           1.003
#:   ---------------------------------------------------------------------------- and the catches on that body
#:   inertia_scale x25                          1.289           --              --      refused by TRACKING
#:   inertia_scale x30                          1.500          9.646          14.469   <- the catch FLOOR
#:   inertia_scale x35                          1.688         27.158          45.699
#:   inertia_scale x40                          1.883         15.455          29.160
#:   inertia_scale x50                          2.250          9.983          22.461
#:
#: Every one of the correct rows was APPLIED and every catch was REFUSED, so no verdict on this body is wrong.
#: What is wrong is the MARGIN the old block advertised. The two populations, over every case whose rival
#: actually RAN, now across FOUR composed bodies (dog, hexapod, cat, plus horse and humanoid probes below):
#:
#:   CORRECTLY SPECIFIED, n=88                       rival_improvement_x   0.982 ... 3.679
#:   LINK INERTIA x20..x100 -- every breach, n=15    rival_improvement_x  14.469 ... 633.220
#:
#: **AND 3.679 IS NOT THE POPULATION CEILING EITHER -- IT IS THE CEILING OVER THOSE FIVE BODIES (corrected
#: 2026-08-13).** The per-body table shipped in ``what_this_gate_does_not_catch`` records a composed CENTIPEDE
#: reaching **4.037** on the same statistic (the PROPORTIONAL family, armature offset = k x each joint's own
#: rotational inertia), and this block went on printing 3.679 as "the largest correct rival_improvement_x"
#: while the output one function away said otherwise. The correct population's ceiling over every body measured
#: is **4.037**, so the empty band is **14.469 / 4.037 = 3.58x**, not the 3.93x this block used to advertise and
#: not the 10.2x before that. Each restatement has been a SHRINKING margin discovered by adding a body, and the
#: honest reading is that the ceiling is a running maximum over the bodies sampled rather than a bound.
#:
#: The important part is still that the correct side has a
#: CEILING at all -- the statistic does not move with the fit's strength (dog, armature-only: 1.774 at
#: improvement 17.910 and 1.777 at 1.602; cat: 3.181 at 2.840 and 3.089 at 1.542, while ``explains_x`` over
#: those same cat rows moves 1.126 -> 1.989) -- but the ceiling is BODY-dependent in a way nothing here predicts.
#:
#: THE THRESHOLD IS 7.3, chosen on 2026-08-13 by the rule that picked 6.0 before it: the GEOMETRIC MIDPOINT of
#: the gap as it was then measured, sqrt(3.679 x 14.469) = 7.296. **IT IS NO LONGER THE MIDPOINT AND THIS BLOCK
#: SAYS SO RATHER THAN RE-CUTTING THE CONSTANT.** Against the ceiling that is actually in force -- the
#: centipede's 4.037 -- the line sits **1.81x above the correct ceiling and 1.98x below the catch floor**; the
#: midpoint of the current gap would be sqrt(4.037 x 14.469) = 7.643. The constant is deliberately LEFT AT 7.3:
#: no measured verdict differs between 7.3 and 7.643 (nothing correct has been observed above 4.037 and no
#: catch below 14.469), and re-cutting a threshold every time one more body nudges a running maximum is how a
#: number stops meaning anything. What is owed is the disclosure, not the digit.
#:
#: **6.0 was not wrong, it was off-centre**: on the populations as they stood it sat 1.63x above the correct
#: ceiling and 2.41x below the catch floor. And 8.5 -- the midpoint of the OLD ceiling and the OLD floor -- is
#: off-centre the other way (2.31x / 1.70x), because the third body moved the floor DOWN as well as the ceiling
#: UP; sizing on 19.604 today would be sizing on a number the same measurement retired. NO MEASURED CASE
#: CHANGES VERDICT between 6.0 and 7.3 either, so that move was a re-centring inside the gap and not a trade of
#: catches for false refusals. Every catch on every body measured is still refused, with 1.98x to spare at
#: worst, and the smallest margin on the correct side is now 1.81x rather than 1.98x.
#:
#: ON THE HEXAPOD AND THE CAT EVERY CATCH IS ITSELF INSIDE THE DANGER BAND (hexapod inertia_scale
#: x20/x30/x40/x100 read 2.005 / 2.015 / 1.973 / 1.775 on improvement_x; cat x30/x35/x40 read 1.500 / 1.688 /
#: 1.883), which is the strongest form of the claim available: in the exact region where correct fits top out at
#: 4.037, genuine inertia errors read at least 14.5.
LINK_INERTIA_RIVAL_IMPROVEMENT_X = 7.3

#: The implied inertia scale has to be physical and meaningfully away from 1.0 before a replay is spent on it.
#: This is a guard, not a threshold: a fit whose armature deltas are ~0 implies s ~ 1 (the prior model itself,
#: which cannot beat a passing fit), and a TORQUE-SCALE error with g > 1 drives armature NEGATIVE and implies a
#: NEGATIVE s (measured: -3.53 at g=1.2), which is not a robot. Both are skipped rather than replayed.
LINK_INERTIA_MIN_FRACTION = 0.05

#: Below this the implied scale is indistinguishable from 1.0 and there is no torque-scale error to name. A
#: guard against the degenerate case where nothing moved: a fit whose deltas are all ~0 is trivially "coherent"
#: with the scalar zero, and would otherwise be refused for looking like a 1.00x gear error.
GLOBAL_SCALE_MIN_FRACTION = 0.02

# ---------------------------------------------------------------------------------------------------------
# THE FOURTH CHECK -- the one that refuses to RULE, rather than ruling on which of the customer's parts is
# wrong. It is the same shape as everything above it: a confident answer on data that cannot support one.
# ---------------------------------------------------------------------------------------------------------
#
# Every check in this file so far asks "which of these explanations does the log favour?". This one asks the
# question that has to come first: **is this a log at all?** It exists because of a measured failure, and the
# failure was not in the verdict logic -- it was that the verdict logic ran.
#
# MEASURED (2026-08-13, composed 8-legged spider, 24 joints, 35 s excitation, delay 0, n_boot 64, bit-identical
# at 96, through the shipped ``fit_parameters``): correct ARMATURE-ONLY calibrations at +0.009 (improvement_x
# 1.575) and +0.05 (2.057) were REFUSED as ``torque_scale_suspected`` at torque rival ``explains_x`` 1.023 and
# 1.218, telling the customer to go and check their gear ratio and torque constant. Both refusals are wrong.
#
# But the interesting number is not the rival's. It is that on the same body a log whose command envelope is
# **1.634 rad** records **max|q_meas| = 298.9 rad** -- forty-seven revolutions of a joint that was asked to move
# a quarter turn -- with MuJoCo emitting "Nan, Inf or huge value in QACC" throughout its generation, and every
# other injection on that body doing the same (23.97 to 11245 on the same ratio; the "nothing wrong at all"
# control reads 1675). That plant is not tracking anything; the log is a record of a divergent integration. A
# tool that accepts it and issues a confident drivetrain verdict is making the file's own recurring error one
# layer further out, and the remedy is not a better rival -- it is a refusal to rule.
#
# WHY EXCURSION AND NOT THE WARNINGS. MuJoCo's QACC warnings are emitted where the log is GENERATED. A customer's
# log arrives as arrays; there is no warning counter attached to it, and there never will be. The one signature
# of a divergent plant that survives into the data is how far the joints actually went. So the guard is
# measured on ``q_meas`` against the commanded envelope, which is exactly what a customer can also check.
#
# **THE FIRST VERSION OF THIS BOUND WAS SIZED AT DELAY 0 ONLY, AND THAT WAS THE WHOLE DEFECT (2026-08-13).**
# The sane survey it quoted -- "0.874 - 1.405 over 56 logs, 7 composed bodies" -- was taken with
# ``delay_ticks=0``, while ``synthetic_hardware.DEFAULT_DELAY_TICKS`` is 2, ``tools.simulate_bench_log``
# defaults to ``delay_ms=20``, and this package validates at 0/20/40 ms precisely BECAUSE actuation delay is
# the dominant sim-to-real term (Hwangbo et al.). The one axis the guard is most exposed to is the one the
# survey held at its zero. Everything below is the re-survey.
#
# THE RE-SURVEY: 19 composed bodies x 8 injections x 11 delays = 1672 logs, plus the spider at the same 88
# points. Bodies: dog, hexapod, cat, horse, humanoid, 6-axis arm, snake, centipede, millipede, octopus,
# starfish, inchworm, biped, turtle, gecko, WHEELED ROVER, tabletop arm, crab, scorpion. Injections: the
# default, armature-only +0.009 and +0.05, frictionloss-only +0.030, damping-only +0.6, NOTHING WRONG AT ALL,
# and inertia_scale x40/x100 so the guard is still measured against the family the third check must catch.
# Delays: 0, 20, 40, 60, 100, 160, 200, 300, 400, 500, 640 ms (640 ms = 64 control ticks, the cap
# ``tools._max_ticks`` puts on the delay search, i.e. the largest delay reachable through the shipped door).
#
# WHAT MOVES WITH DELAY -- max over the 8 injections, worst bodies. **THIS TABLE IS KEPT AS THE RECORD OF THE
# RE-SURVEY AND TWO OF ITS ROWS ARE NO LONGER FIGURES OF MERIT (see the correction directly below it): the
# ROVER row is a body this statistic is not defined on, and the SPIDER row's minimum is an artefact of this
# grid's 40 ms gap between 60 and 100 ms.**
#
#                  0ms    20ms    40ms    60ms   100ms   160ms   200ms   300ms   400ms   500ms   640ms
#   millipede     1.483   5.671   6.268   6.721   6.587   6.672   6.331   6.444   6.614   6.500   6.237
#   centipede     1.078   5.322   5.692   6.028   6.139   6.001   6.513   6.029   6.217   6.042   6.017
#   rover         0.872   0.892   0.912   0.932   4.151   7.685   9.748  18.922  27.034  36.863  50.076
#   6-axis arm    0.985   1.079   1.371   1.684   2.919   5.589   6.644   8.861   9.005   9.076   9.144
#   tabletop arm  1.043   1.078   1.122   1.498   2.397   4.330   5.142   9.036   9.252  11.529  15.504
#   humanoid      1.405   1.479   1.581   2.307   3.643   3.940   4.486   4.579   4.554   4.578   4.586
#   dog           0.947   0.951   0.974   1.004   1.363   1.752   2.383   2.999   3.591   4.578   5.411
#   SPIDER (min) 23.974  47.023  28.023  20.262  16.118 125.273 411.462 633.899 2380.7   209.2   695.4
#
# **THE SHIPPED 5.80 LINE ALREADY REFUSES SANE LOGS, INSIDE THIS PACKAGE'S OWN VALIDATED BAND.** At 40 ms --
# the upper delay ``tests/test_sysid_delay_wedge.py`` validates -- FIVE composed-millipede logs cross it,
# including the control with NOTHING WRONG WITH IT AT ALL (5.859). Over 0-200 ms it refuses 52 of 1064 sane
# logs. At the package's DEFAULT 20 ms the millipede reads 5.671 against 5.80: a margin of **1.02x**, not the
# 4.1x this block used to advertise.
#
# AND THE REFUSALS ARE NOT SECRETLY CORRECT. The obvious escape -- "those logs really had diverged" -- was
# tested and closed. ``bench_rig.pd_replay``'s loop was re-run verbatim with the MjData kept so MuJoCo's own
# instability counters could be read (``data.warning[mjWARN_BADQACC|BADQPOS|BADQVEL|BADCTRL].number``; per the
# project's own hard-won lesson an ``isfinite`` check after ``mj_step`` is worthless, because ``mj_checkAcc``
# calls ``mj_resetData``). Result: **0 of 1368 non-spider logs emitted a single warning at any delay to
# 400 ms**, including the rover at 27.034 and the arm at 9.005. Those are numerically stable integrations that
# the line refuses. The spider warns on 4 of 8 injections at delay 0 and 8 of 8 at 400 ms -- which is also why
# the guard is not built on warnings: they do not even label the degenerate body cleanly, and a customer's log
# carries no counter at all.
#
# =========================================================================================================
# AND BOTH EDGES OF THAT RE-SURVEY WERE ALSO WRONG (2026-08-13, later the same day). BOTH ARE RE-TAKEN BELOW.
# =========================================================================================================
#
# **(a) THE SANE CEILING WAS A WHEELED ROVER, AND A WHEEL'S POSITION HAS NO CEILING.** 9.748 was the composed
# wheeled rover at 200 ms. The rover carries four continuous-rotation hinges (``jnt_limited = False``); a
# wheel's angle INTEGRATES, so its peak grows without limit on a perfectly sane log and an excursion RATIO on
# it is not a statistic at all. Measured through the shipped ``fit_parameters``: a composed SIX-WHEELED ROVER
# with a genuine 25% torque-scale error and nothing else wrong reads 0.812 at 0 ms -- correctly refused as
# ``global_scale``, implied g 1.2974 against a truth of 1.25 -- and 12.598 at 200 ms, INSIDE the declared
# window, where it was refused as ``implausible_log`` instead. A real drivetrain finding, deleted by this
# guard, on a log MuJoCo never warned about. So the ratio is now formed over LIMITED joints only, the excluded
# joints are NAMED, and a body with NO limited joint at all is reported NOT MEASURABLE rather than passed or
# refused. See ``log_plausibility`` and ``_bounded_channels``.
#
# **(b) THE DEGENERATE FLOOR WAS AN ARTEFACT OF A COARSE DELAY GRID.** 16.118 was the minimum over an 11-point
# grid that skipped 50/80/110 ms. Re-measured on the FINEST grid the controller can represent (10 ms = one
# control tick at 100 Hz, 21 points over 0-200 ms), the composed spider reads 14.897 / 14.834 / **14.648** at
# those three delays. The floor is 14.648, at 110 ms, on ``armature +0.050``.
#
# THE RE-TAKEN SURVEY: 19 composed bodies (18 wheel-free + the two rovers) x the SAME 8 injections x the 21-point
# delay grid, 0-200 ms; 2864 wheel-free sane readings, 672 rover readings, 168 spider readings. (The millipede
# and the centipede were run on the 11-point 20 ms grid instead -- they cost 2483 s and 1626 s respectively --
# and that is stated rather than hidden; neither body's curve has structure between 20 ms samples.)
#
#   THE TWO POPULATIONS, WHEEL-FREE, 0-200 ms, SAME 8 INJECTIONS
#   SANE       0.584 ... 6.721    n=2864, 18 bodies. Ceiling: composed MILLIPEDE / frictionloss +0.030 @ 60 ms
#   DEGENERATE 14.648 ... 42964   n=168.  Floor:   composed SPIDER / armature +0.050 @ 110 ms
#
#   per-body wheel-free ceiling: millipede 6.721, 6-axis arm 6.644, centipede 6.513, tabletop arm 6.470,
#   humanoid 4.486, biped 4.486, horse 4.353, octopus 2.845, dog 2.383, starfish 2.383, inchworm 2.383,
#   crab 2.295, scorpion 2.196, hexapod 1.894, snake 1.835, turtle 1.807, cat 1.174, gecko 1.164.
#   Both ROVERS: NOT MEASURABLE (4 of 4 and 6 of 6 joints unbounded). Pre-repair they read 0.554-9.768 on the
#   same 8 injections and 55 of their 672 readings crossed 12.5 -- every one of those a false refusal.
#
# THE LINE IS THE GEOMETRIC MIDPOINT OF THAT GAP, sqrt(6.721 x 14.648) = 9.922 -> **9.9** -- the same rule that
# sized 12.5 and that sizes ``LINK_INERTIA_RIVAL_IMPROVEMENT_X``, for the same reason: both edges are samples.
# It sits 1.473x above the sane ceiling and 1.480x below the degenerate floor.
#
# COST, MEASURED at 9.9 over 0-200 ms: **0 false refusals of 2864 wheel-free sane readings, 0 catches lost of
# 168 spider readings.** ``inertia_scale`` x40/x100 -- the family the third check exists to catch -- is inside
# the sane population and stays there, so no catch is traded for this guard.
#
# **AND A FAMILY THIS BOUND DOES NOT COVER, FOUND BY WIDENING THE INJECTION SET AND NOT SIZED AROUND.** The 8
# injections above are the ones the bound has always been sized on. Eight more were measured beside them --
# damping zeroed, frictionloss zeroed, both zeroed, torque_scale 0.8/1.25/1.5, link_scale 1.3, hold-only -- and
# the DISSIPATION-REMOVAL family breaks the separation on wheel-free bodies:
#
#   composed OCTOPUS   / damping+frictionloss zeroed @ 200 ms   22.181
#   composed TABLETOP ARM / damping+frictionloss zeroed @ 200 ms 18.778   (0 MuJoCo warnings, verified)
#   composed 6-AXIS ARM   / damping+frictionloss zeroed @ 200 ms 10.031
#
# Those are ABOVE the spider's 14.648 floor (the first two), so no threshold on this ratio separates them, and
# a customer whose robot genuinely has negligible joint damping will be false-refused. Over the full 16-family
# wheel-free population 56 of 5728 readings are refused at 9.9 and 42 at 12.5, all on those three bodies. The
# right instrument is probably a peak against each joint's OWN declared range rather than against a whole-robot
# envelope -- the tabletop arm's worst joint reaches 6.572 rad on a plan whose envelope is 0.35 rad, which its
# own ``jnt_range`` should already have made visible -- but that is a different statistic and it is NOT built
# here. Stated, not sized around.
#
# **AND THE POPULATION LABELS ARE NOT WHAT THEY LOOK LIKE.** "Spider = degenerate" is a claim about the LOGS,
# not the body. Re-run with the MjData kept so MuJoCo's counters could be read
# (``data.warning[mjWARN_BADQACC|BADQPOS|BADQVEL|BADCTRL].number``; per the project's own lesson an
# ``isfinite`` check after ``mj_step`` is worthless because ``mj_checkAcc`` calls ``mj_resetData``): of 56
# spider readings across the 8 sizing injections, **16 emit no warning at all** -- including the 182.941 case
# this whole guard was built on and the 14.648 that is now the floor. And with frictionloss zeroed the spider
# is ORDINARY: 0.977 at 0 ms, 0.992 at 100 ms, no warnings. Warnings do not label this population; the
# excursion does, and the excursion is what a customer's arrays actually carry.
#
# 0-200 ms IS STILL THE SCOPE. Past it the populations overlapped in the previous survey on the rover, which
# is now excluded -- so the window has NOT been re-derived on the wheel-free population and is carried forward
# unchanged rather than widened on an argument. That is named in ``LOG_EXCURSION_VALID_DELAY_MS``.

#: The most a LIMITED joint may travel, as a multiple of the commanded envelope over the limited joints, before
#: this package refuses to rule on the log at all. See the block above: the geometric midpoint of a wheel-free
#: sane ceiling of 6.721 (composed millipede) and a degenerate floor of 14.648 (composed spider, on the 10 ms
#: delay grid) over 0-200 ms of actuation delay, sqrt(6.721 x 14.648) = 9.922.
#:
#: **IT WAS 5.80, THEN 12.5, AND BOTH ARE RETIRED (2026-08-13).** 5.80 was sized at zero actuation delay and
#: refused sane logs at 40 ms. 12.5 was sized on two edges that were both wrong in the same direction: a sane
#: ceiling of 9.748 that was a WHEELED ROVER -- a body whose joints have no positional ceiling, now excluded
#: from the statistic rather than measured -- and a degenerate floor of 16.118 that was the minimum over a
#: delay grid skipping 50/80/110 ms, where the spider reads as low as 14.648.
LOG_EXCURSION_RATIO_MAX = 9.9

#: The delay window the bound above was measured over, and OUTSIDE WHICH IT IS NOT CLAIMED TO WORK. Reported
#: in every verdict rather than silently assumed.
#:
#: **THE ARGUMENT THAT ORIGINALLY PICKED 200 ms IS RETIRED, AND THE WINDOW IS CARRIED FORWARD ANYWAY (2026-08-13,
#: named rather than quietly re-derived.)** It used to be "past 200 ms the populations overlap, because a
#: composed wheeled rover reads 18.922 at 300 ms against the spider's 16.118". The rover is now EXCLUDED from
#: this statistic -- its joints are unbounded -- so that sentence no longer supports anything, and the two
#: figures in it are both retired (the spider's floor is 14.648, not 16.118). The window is kept at 200 ms
#: because that is the range the wheel-free population was actually re-measured over, not because a wider one
#: was tested and rejected: **beyond 200 ms nothing is measured on the wheel-free population and nothing is
#: claimed.** It comfortably contains everything this package validates (0/20/40 ms) and every actuation delay
#: Hwangbo et al. report.
LOG_EXCURSION_VALID_DELAY_MS = 200.0

#: The commanded envelope is floored at this many radians before dividing by it, because a log whose command
#: is a HOLD AT THE ZERO POSE divides by EXACTLY zero.
#:
#: **THE MEASUREMENT THIS DOCSTRING USED TO CLAIM WAS FALSE, and it is worth keeping the correction visible.**
#: It said the floor is why ``hold_only=True`` "reads exactly 1.000 on the dog, the cat and the humanoid".
#: Measured, those three hold-only commanded envelopes are 0.4175 / 0.4175 / 0.6175 rad -- 4.2x, 4.2x and 6.2x
#: ABOVE this floor, so it never binds on any of them and they read 1.000 with the floor set to 1e-9. They read
#: 1.000 because a held joint's PEAK IS ITS COMMAND, which has nothing to do with the floor.
#:
#: WHAT IS ACTUALLY LOAD-BEARING, measured over the same 19 composed bodies: two of them -- the OCTOPUS and the
#: WHEELED ROVER -- have a hold-only commanded envelope of EXACTLY 0.000 rad, because their start pose is the
#: zero pose. Without this floor the octopus's hold-only log divides 0.0147 rad by ~0 and reads 1.5e7, and the
#: rover's is 0/0. Both would be refused as ``implausible_log``: a stationary robot's log called a divergent
#: integration. That is the case the constant exists for, and ``tests/test_sysid_implausible_log_guard.py``
#: pins it with a log the guard refuses when the floor is removed.
#:
#: WHAT IT COSTS, also measured and also not in the old story: on the composed 6-axis arm and the composed
#: tabletop arm the hold-only envelope is 0.01069 rad, BELOW the floor, so the floor binds and divides by 0.1
#: instead -- reading 0.1069 where the true ratio is 1.000. There the floor makes this guard **9.4x MORE
#: PERMISSIVE**, which is the right direction for a false-refusal guard but is a real loss of sensitivity on
#: small-travel experiments and is named rather than left to be discovered. It never binds on a MOVING log in
#: the survey, though the margin is thin: the smallest full-plan envelope measured is the octopus's 0.126 rad,
#: only 1.26x above the floor.
LOG_ENVELOPE_FLOOR_RAD = 0.1

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


def global_scale_signature(model, dofs: dict, joints: dict, param_names) -> dict:
    """Part one of the second check: is this whole fit explained by ONE global torque scalar?

    Free -- it reads the fit that already exists and simulates nothing. Returns the per-parameter median delta
    as a FRACTION of its prior, the single scalar that best explains those fractions, how much of them it
    explains, and the gear/torque-constant error that scalar implies.

    The algebra it is testing: a plant that really receives ``g`` times the torque the log records is
    indistinguishable from one whose inertia, damping and friction are all divided by ``g``, so the fit lands on
    ``(1 - g)/g`` times EACH prior, on every joint and every parameter. The statistic asks whether it did.

    This is a FILTER, not a verdict, and the measured reason is in ``GLOBAL_SCALE_COHERENCE_R2``: a genuine
    dissipation change that happens to be proportional to the priors scores 0.980 here, above four of the six
    torque-scale errors this exists to catch. Only ``_global_scale_rival`` can tell those two apart.
    """
    import numpy as np

    per: dict = {}
    for p in param_names:
        vals = []
        for name, row in (joints or {}).items():
            adr = dofs.get(name)
            if adr is None:
                continue
            prior = float(getattr(model, f"dof_{p}")[adr])
            if prior <= 1e-9:
                continue
            vals.append(float(row["parameters"][p]["delta"]) / prior)
        if vals:
            per[p] = float(np.median(vals))

    base = {
        "statistic": "per-parameter median fitted delta as a fraction of its own prior, across all joints",
        "why": ("scaling the applied torque by g is algebraically dividing inertia, damping and friction by g, "
                "so a gear-ratio / torque-constant / gearbox-efficiency error drives ALL THREE fitted "
                "parameters on EVERY joint by the same (1-g)/g of their priors. A real per-joint dissipation "
                "change has no reason to be coherent like that."),
        "fraction_of_prior": {k: round(v, 4) for k, v in per.items()},
        "coherence_threshold": float(GLOBAL_SCALE_COHERENCE_R2),
    }
    if len(per) < 2:
        return {**base, "coherence": None, "implied_torque_scale_g": None, "coherent": False,
                "not_measurable_because": (
                    f"only {len(per)} of the fitted parameters has a non-zero prior on any joint, so there is "
                    f"no cross-parameter consistency to test. A torque-scale error CANNOT be ruled out on this "
                    f"robot by this check.")}

    m = np.asarray(list(per.values()), dtype=float)
    c = float(np.mean(m))
    ss_tot = float(np.sum(m ** 2))
    r2 = (1.0 - float(np.sum((m - c) ** 2)) / ss_tot) if ss_tot > 1e-18 else 0.0
    g = (1.0 / (1.0 + c)) if (1.0 + c) > 1e-6 else None
    coherent = bool(r2 >= GLOBAL_SCALE_COHERENCE_R2
                    and abs(c) >= GLOBAL_SCALE_MIN_FRACTION
                    and g is not None)
    return {**base,
            "coherence": round(r2, 4),
            "best_single_fraction": round(c, 4),
            "implied_torque_scale_g": round(g, 4) if g is not None else None,
            "coherent": coherent,
            "reading": ("the fitted deltas ARE well explained by one global torque scalar, so a wrong gear "
                        "ratio / torque constant is a live rival explanation and is put to a replay"
                        if coherent else
                        "the fitted deltas are NOT explained by one global torque scalar, so this fit is not "
                        "the signature of a gear-ratio / torque-constant error")}


def _global_scale_rival(model, aligned, *, kp, kd, q_start, ctrl_every: int, delay_ticks: int,
                        g_hint: float, after_rms: float, before_rms: float = 0.0) -> dict:
    """Part two: put the PRIOR parameters back, scale every actuator's gear by one number, and replay.

    A ONE-parameter explanation of the same log, scored on the same POSITION RMS ``improvement_x`` is scored on
    -- a quantity neither model optimised. If it tracks at least as well as the fit's forty-two numbers do, the
    experiment did not distinguish the two explanations and the forty-two are not safe to write.

    Seeded from the coherence's implied ``g`` and refined locally rather than searched over a wide grid: the
    seed is measured accurate to a few percent (1.195 for a true 1.2, 1.462 for 1.5, 1.690 for 1.75) and each
    evaluation is a full replay. Costs ~10 replays, and only runs when the coherence has already fired.
    """
    import copy

    import numpy as np

    from virturoid.services.sysid.bench_rig import pd_replay

    q_cmd, q_hw = aligned["q_cmd"], aligned["q_meas"]
    seen: dict = {}

    def rms(g: float) -> float:
        g = float(np.clip(g, 0.2, 5.0))
        key = round(g, 4)
        if key in seen:
            return seen[key]
        m = copy.deepcopy(model)
        m.actuator_gear[:, 0] = np.asarray(m.actuator_gear[:, 0], dtype=float) * key
        _, q, _, _ = pd_replay(m, q_cmd, kp=kp, kd=kd, q_start=q_start, ctrl_every=int(ctrl_every),
                               delay_ticks=int(delay_ticks))
        seen[key] = float(np.sqrt(np.mean((q - q_hw) ** 2)))
        return seen[key]

    coarse = sorted({round(float(np.clip(float(g_hint) * f, 0.2, 5.0)), 4)
                     for f in (0.75, 0.85, 0.925, 1.0, 1.075, 1.15, 1.25)})
    for g in coarse:
        rms(g)
    best = min(seen, key=seen.get)
    i = coarse.index(best) if best in coarse else len(coarse) // 2
    lo, hi = coarse[max(i - 1, 0)], coarse[min(i + 1, len(coarse) - 1)]
    for f in (0.25, 0.5, 0.75):
        rms(lo + (hi - lo) * f)
    best = min(seen, key=seen.get)
    best_rms = seen[best]
    explains = float(after_rms) / max(best_rms, 1e-12)
    return {
        "measured": ("position RMS of a replay of OUR PRIOR model with every actuator's gear scaled by one "
                     "number, against the same log, at the same actuation delay"),
        "torque_scale_g": round(float(best), 4),
        "rival_rms_rad": round(float(best_rms), 8),
        "fit_rms_rad": round(float(after_rms), 8),
        "prior_rms_rad": round(float(before_rms), 8),
        # How many times better ONE number tracks than the forty-two the fit wants to write.
        "explains_x": round(min(explains, 1e6), 3),
        "threshold_x": float(GLOBAL_SCALE_RIVAL_X),
        # REPORTED, not gated -- the fit-strength-free companion the link-inertia check now gates on. It is
        # here because the same confound is present in ``explains_x`` on this side too and a reader is entitled
        # to see the unconfounded number; see the note under ``GLOBAL_SCALE_RIVAL_X`` for why the torque check
        # does not yet gate on it (its correct population has not been shown to approach 1.0 IN the band).
        "rival_improvement_x": (round(min(float(before_rms) / max(best_rms, 1e-12), 1e6), 3)
                                if before_rms else None),
        "n_replays": len(seen),
        "beats_the_fit": bool(explains >= GLOBAL_SCALE_RIVAL_X),
    }


def link_inertia_signature(model, dofs: dict, joints: dict, q_start) -> dict:
    """The SEED for the link-inertia rival, and the diagnostic beside it. Simulates nothing.

    Scaling every link's rotational inertia by ``s`` adds exactly ``(s-1) * R_a`` to joint a's own ``diag(M)``,
    where ``R_a`` is the rotational part of that diagonal -- and ARMATURE is the parameter that adds to the same
    diagonal, which is why ``PARAMETER_ALSO_ABSORBS['armature']`` has always named link inertia. So if the fit
    is really a link-inertia error, ``armature_delta_a / R_a`` is the SAME number ``(s-1)`` on every joint.

    ``R_a`` is measured, not derived: ``diag(M)`` at twice the model's inertia minus ``diag(M)`` at the model's
    own, which is exact because the dependence is linear. Two ``mj_forward`` calls.

    **The coherence returned here is NOT a filter.** Measured on both populations it reads 0.945-0.983 on
    correctly-specified fits and 0.992 on the caught ones -- overlapping, with no gap to put a line in. It is
    reported because it is worth seeing and because it seeds the search; the RIVAL is the verdict. See the
    block at ``LINK_INERTIA_RIVAL_X`` for why a threshold is not squeezed in between.

    **``rotational_inertia_span_x`` is a WARNING, not the mechanism, and this docstring used to say the
    opposite.** It read that ``R_a`` "spans ~40x between the trunk and a distal leg joint, so no single s can
    imitate a uniform armature offset" -- while the field one line below printed 1.55. MEASURED: 1.547x on the
    composed dog, 1.528x on the composed hexapod. A span that narrow means a scalar CAN imitate a uniform
    armature offset to within roughly +/-20%, so on these bodies the rival is a NEAR-degenerate competitor to a
    real armature calibration and the naive "does the rival win?" test false-refuses. What actually separates
    the populations is how much of the log the rival explains in ABSOLUTE terms -- see
    ``LINK_INERTIA_RIVAL_IMPROVEMENT_X``.

    **AND THE PREDICTION THAT REPLACED IT IS ALSO RETRACTED (2026-08-13).** This docstring went on: "the span is
    still reported because a body with a genuinely wide one WOULD MAKE THE CHECK SHARPER", flagged in the
    disclosure as "a prediction, not a measurement". It has now been measured, on the correct-population
    ``rival_improvement_x`` (lower = the rival imitates armature less well = the check is sharper), one
    armature-only family per body:

        composed dog        span  1.547     0.982 - 1.930
        composed hexapod    span  1.528     0.982 - 1.930  (same sweep)
        composed CAT        span  2.628     3.083 - 3.679   <- nearly DOUBLE the dog at a WIDER span
        composed horse      span  4.040     1.526 - 1.549
        composed humanoid   span 14.656     1.191 - 1.231

    The humanoid and the horse fit the prediction; the cat contradicts it outright, and it is the body whose
    span sits between them. The relation is NOT MONOTONE, so span is not the controlling variable and an
    extrapolation keyed on it -- in either direction -- is unsafe. **No replacement prediction is offered**: what
    controls the correct population's ceiling on a body this package has not measured is UNKNOWN, and that is
    the honest state of it. The span is still reported because a reader is entitled to see the number the
    original story rested on, and because it is the one quantity that says how nearly degenerate the rival's
    family is with a uniform armature offset -- which is a statement about the RIVAL, not a bound on the check.
    """
    import copy

    import mujoco
    import numpy as np

    def _diag(m):
        d = mujoco.MjData(m)
        d.qpos[:m.nv] = q_start
        mujoco.mj_forward(m, d)
        full = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, full, d.qM)
        return np.diag(full).copy()

    twice = copy.deepcopy(model)
    twice.body_inertia[1:] = np.asarray(twice.body_inertia[1:], dtype=float) * 2.0
    d1, d2 = _diag(model), _diag(twice)

    per: dict = {}
    rots: list = []
    for name, row in (joints or {}).items():
        adr = dofs.get(name)
        if adr is None:
            continue
        rot = float(d2[adr] - d1[adr])
        if rot <= 1e-12:
            continue
        rots.append(rot)
        per[name] = float(row["parameters"]["armature"]["delta"]) / rot

    base = {
        "statistic": ("each joint's fitted ARMATURE delta divided by the ROTATIONAL part of its own diag(M) -- "
                      "one number per joint"),
        "why": ("scaling every link's rotational inertia by s adds exactly (s-1) x that rotational part to the "
                "same diagonal armature adds to, so a link-inertia error at correct mass drives every joint's "
                "armature by the same (s-1) of its own rotational inertia. It has NO gravity signature, which "
                "is why the tracking gate cannot see it, and it moves armature ALONE, which is why the "
                "global-scale check correctly does not fire on it."),
        # HOW CLOSE the rival's FAMILY is to a real armature change: an inertia scalar adds a term PROPORTIONAL
        # to this quantity while a real armature change adds the SAME constant everywhere, so at a narrow span
        # the two are nearly the same one-parameter family. MEASURED 1.547x on the composed dog, 1.528x on the
        # composed hexapod, 2.205x on a composed spider, 2.628x on a composed cat, 4.040x on a composed horse,
        # 14.656x on a composed humanoid. The comment here used to say "~40x", contradicting the field it
        # annotates -- that figure is an ARM's (43.94x measured) carried onto legged bodies. It does NOT predict
        # how sharp the check is: measured, the cat's correct population sits at 3.08-3.68 on
        # rival_improvement_x while the horse's, at a WIDER span, sits at 1.53-1.55. See link_inertia_signature.
        "rotational_inertia_span_x": (round(float(max(rots) / max(min(rots), 1e-12)), 2) if rots else None),
        "rotational_inertia_span_reading": (
            "how far apart the joints' own rotational inertias are. An inertia scalar adds a term proportional "
            "to each joint's, while a real armature change adds the same constant to every joint, so at a "
            "NARROW span the two are nearly the same one-parameter family. Measured 1.55 on a composed "
            "quadruped, 1.53 on a composed hexapod, 2.63 on a composed cat, 4.04 on a composed horse, 14.66 on "
            "a composed humanoid. It is reported and is NOT USED AS A BOUND in either direction, because "
            "MEASURED across those bodies it does not order how well this check separates: the cat, at 2.63, "
            "has the widest correct population of the five and the humanoid, at 14.66, the narrowest. That is "
            "why the verdict needs link_inertia.rival.rival_improvement_x and not only rival.explains_x."),
        "coherence_is_not_a_filter": (
            "measured 0.945-0.983 on correctly-specified fits and 0.992 on the caught ones -- overlapping "
            "bands. The coherence is reported and seeds the search; only the replay rules."),
    }
    if len(per) < 2:
        return {**base, "coherence": None, "implied_inertia_scale_s": None, "testable": False,
                "not_measurable_because": (
                    f"only {len(per)} joint has a non-zero rotational contribution to its own diag(M), so "
                    f"there is nothing to scale. A link-inertia error CANNOT be ruled out on this robot by "
                    f"this check.")}

    m = np.asarray(list(per.values()), dtype=float)
    c = float(np.median(m))
    ss_tot = float(np.sum(m ** 2))
    r2 = (1.0 - float(np.sum((m - c) ** 2)) / ss_tot) if ss_tot > 1e-18 else 0.0
    s = 1.0 + c
    testable = bool(s > 0.0 and abs(c) >= LINK_INERTIA_MIN_FRACTION)
    return {**base,
            "coherence": round(r2, 4),
            "fraction_of_rotational_inertia": {k: round(v, 4) for k, v in per.items()},
            "best_single_fraction": round(c, 4),
            "implied_inertia_scale_s": round(s, 4),
            "testable": testable,
            "reading": ("the fitted armature deltas imply a link-inertia scale, so a mis-stated inertia tensor "
                        "is a live rival explanation and is put to a replay"
                        if testable else
                        "the fitted armature deltas do not imply a physical link-inertia scale "
                        f"(implied s {s:.4g}), so no replay is spent on one")}


def _link_inertia_rival(model, aligned, *, kp, kd, q_start, ctrl_every: int, delay_ticks: int,
                        s_hint: float, after_rms: float, before_rms: float) -> dict:
    """Put the PRIOR parameters back, scale every link's ROTATIONAL INERTIA by one number, and replay.

    The twin of ``_global_scale_rival`` for the other degeneracy, scored the same way on the same POSITION RMS
    neither model optimised.

    Mass is NOT touched, deliberately: moving it would put a gravity term back into the rival and make it a
    weaker explanation of the very family this exists to catch (that family is defined by having correct mass).

    **TWO numbers come back and BOTH gate, and the second one is the 2026-08-13 repair.** ``explains_x`` --
    ``fit_rms / rival_rms``, "does one number track at least as well as the forty-two" -- is NOT scale-free with
    respect to how strong the fit is, because the fit's own RMS is its denominator... its numerator. Write it
    out and the reason is arithmetic: ``explains_x = (before/rival) / (before/after)``, i.e. the rival's
    improvement over the prior DIVIDED BY the fit's. Within the correctly-specified population the fit's
    improvement ranges over more than a decade (1.5x to 20.9x on these two bodies), so the same rival scores
    0.099 against a strong fit and 1.108 against a weak one. It refused armature-only +0.009 and +0.010 for
    exactly that reason. The numerator alone -- ``rival_improvement_x = before_rms / rival_rms``, how many times
    closer to the log ONE number gets the PRIOR model -- has no such confound and is flat across the same
    population. Both are reported; ``suspected`` needs both. See ``LINK_INERTIA_RIVAL_IMPROVEMENT_X``.

    Costs ~10 replays. Seeded from the signature's implied ``s`` and refined locally; the seed is measured
    accurate to ~20% (35.9 for a true 30, 44.9 for a true 40) and the local refinement closes the rest.
    """
    import copy

    import numpy as np

    from virturoid.services.sysid.bench_rig import pd_replay

    q_cmd, q_hw = aligned["q_cmd"], aligned["q_meas"]
    seen: dict = {}

    def rms(s: float) -> float:
        s = float(np.clip(s, 0.05, 400.0))
        key = round(s, 4)
        if key in seen:
            return seen[key]
        m = copy.deepcopy(model)
        m.body_inertia[1:] = np.asarray(m.body_inertia[1:], dtype=float) * key
        _, q, _, _ = pd_replay(m, q_cmd, kp=kp, kd=kd, q_start=q_start, ctrl_every=int(ctrl_every),
                               delay_ticks=int(delay_ticks))
        seen[key] = float(np.sqrt(np.mean((q - q_hw) ** 2)))
        return seen[key]

    coarse = sorted({round(float(np.clip(float(s_hint) * f, 0.05, 400.0)), 4)
                     for f in (0.4, 0.6, 0.8, 1.0, 1.25, 1.6, 2.5)})
    for s in coarse:
        rms(s)
    best = min(seen, key=seen.get)
    i = coarse.index(best) if best in coarse else len(coarse) // 2
    lo, hi = coarse[max(i - 1, 0)], coarse[min(i + 1, len(coarse) - 1)]
    for f in (0.25, 0.5, 0.75):
        rms(lo + (hi - lo) * f)
    best = min(seen, key=seen.get)
    best_rms = seen[best]
    explains = float(after_rms) / max(best_rms, 1e-12)
    rival_imp = float(before_rms) / max(best_rms, 1e-12)
    beats_fit = bool(explains >= LINK_INERTIA_RIVAL_X)
    explains_log = bool(rival_imp >= LINK_INERTIA_RIVAL_IMPROVEMENT_X)
    return {
        "measured": ("position RMS of a replay of OUR PRIOR model with every link's ROTATIONAL INERTIA scaled "
                     "by one number and its MASS untouched, against the same log, at the same actuation delay"),
        "inertia_scale_s": round(float(best), 4),
        "rival_rms_rad": round(float(best_rms), 8),
        "fit_rms_rad": round(float(after_rms), 8),
        "prior_rms_rad": round(float(before_rms), 8),
        # How many times better ONE number tracks than the forty-two the fit wants to write. NECESSARY and NOT
        # sufficient: it carries the fit's own strength in its denominator, so a weak-but-correct fit inflates
        # it. See the docstring and ``LINK_INERTIA_RIVAL_IMPROVEMENT_X``.
        "explains_x": round(min(explains, 1e6), 3),
        "threshold_x": float(LINK_INERTIA_RIVAL_X),
        # How many times closer to the log ONE number gets the PRIOR model. The fit does not appear in it, so
        # it does not move with the fit's strength -- this is the one that separates the two populations.
        "rival_improvement_x": round(min(rival_imp, 1e6), 3),
        "improvement_threshold_x": float(LINK_INERTIA_RIVAL_IMPROVEMENT_X),
        "n_replays": len(seen),
        "beats_the_fit": bool(beats_fit),
        "explains_the_log": bool(explains_log),
        # BOTH, deliberately. Either alone has a measured failure: ``beats_the_fit`` refuses correct armature
        # calibrations near the tracking gate, and ``explains_the_log`` alone would refuse a fit that the rival
        # loses to outright. The conjunction is what the sweep in LINK_INERTIA_RIVAL_IMPROVEMENT_X is over.
        "suspected": bool(beats_fit and explains_log),
        "reading": (
            f"one link-inertia number gets the PRIOR model {min(rival_imp, 1e6):.3g}x closer to your log "
            f"(threshold {LINK_INERTIA_RIVAL_IMPROVEMENT_X:g}x) and tracks {min(explains, 1e6):.3g}x better "
            f"than the fit does (threshold {LINK_INERTIA_RIVAL_X:g}x)"),
    }


def _bounded_channels(model, dofs: dict | None, ncol: int):
    """``(bounded_cols, unbounded_names, checked)`` -- which of the logged DOF columns carry a joint whose
    POSITION IS BOUNDED, read straight off the compiled model's ``jnt_limited``.

    This is the whole of the unbounded-joint repair and it is deliberately not a heuristic on the data. A
    continuous-rotation hinge and a diverging hinge look identical in ``q_meas`` -- both travel far -- so the
    only sound separator is the model's own declaration, which the compiler already carries and which
    ``bench_rig.safe_band`` has always read for exactly this reason.

    ``checked`` is False when no model was supplied (a customer's arrays arriving without one). Then nothing
    can be excluded and the guard measures every channel, which is the pre-2026-08-13 behaviour and is stated
    in the output rather than assumed.
    """
    cols = sorted({int(a) for a in (dofs or {}).values() if 0 <= int(a) < ncol}) or list(range(ncol))
    if model is None:
        return cols, None, False
    import numpy as np

    nv = int(getattr(model, "nv", ncol))
    limited = np.zeros(max(nv, ncol), dtype=bool)
    adrs = [int(a) for a in np.asarray(model.jnt_dofadr).ravel()]
    for j, adr in enumerate(adrs):
        stop = adrs[j + 1] if j + 1 < len(adrs) else nv
        if bool(np.asarray(model.jnt_limited).ravel()[j]):
            limited[adr:max(stop, adr + 1)] = True
    by_adr = {int(a): n for n, a in (dofs or {}).items()}
    bounded = [c for c in cols if limited[c]]
    unbounded = [by_adr.get(c, f"dof {c}") for c in cols if not limited[c]]
    return bounded, unbounded, True


def log_plausibility(aligned: dict, dofs: dict | None = None, model=None) -> dict:
    """Is this a LOG, or a record of a divergent integration? Reads the arrays and simulates nothing.

    The one check in this file that rules on the EXPERIMENT rather than on the model, and it has to come first:
    every other verdict here names a part of the customer's robot, and naming a part on data that cannot support
    any verdict is worse than every failure this file has already had to retract. See the block at
    ``LOG_EXCURSION_RATIO_MAX`` for the measurement that sized it and for the composed spider whose fits were
    being blamed on a gear ratio.

    Two conditions, both computable from what a customer actually sends:

      * FINITE. A NaN or Inf anywhere in the recorded position, velocity or torque means the plant that produced
        this log left the reals. Nothing downstream can mean anything.
      * BOUNDED. ``max|q_meas|`` against the commanded envelope -- **over the joints whose position is
        BOUNDED, and only those**. A joint asked for a quarter turn that records forty-seven revolutions did
        not track the command; it ran away. A WHEEL that records forty-seven revolutions is a wheel. The two
        are indistinguishable in ``q_meas`` and the only sound separator is the model's own ``jnt_limited``,
        so unbounded joints are excluded and NAMED, and a robot with no bounded joint at all is reported NOT
        MEASURABLE rather than passed or refused. MEASURED across 18 WHEEL-FREE composed bodies x 8 injection
        families x every actuation delay from 0 to 200 ms in 10 ms steps, a sane log reads 0.584-6.721 --
        INCLUDING the misspecified ``inertia_scale`` x40/x100 logs the link-inertia check must still catch.

    **THIS SURVEY HAS BEEN RE-TAKEN TWICE AND BOTH EDGES MOVED BOTH TIMES.** The range this docstring first
    quoted (0.874-1.405, 56 logs, 7 bodies) was taken at ``delay_ticks=0`` while the harness's own default is
    2 ticks; across delay the ratio rises steeply and body-specifically -- a composed millipede goes 1.483 ->
    6.268 between 0 and 40 ms -- and at the threshold that produced (5.80) real logs were refused at 40 ms.
    The range it quoted next (0.554-9.748) had a WHEELED ROVER as its ceiling, which is what this function no
    longer measures. See ``LOG_EXCURSION_RATIO_MAX``.

    **AND THE BOUND IS SCOPED TO A DELAY WINDOW, which ships with it.** Past ``LOG_EXCURSION_VALID_DELAY_MS``
    nothing is measured on the wheel-free population and nothing is claimed; that is reported rather than
    papered over. **A FAMILY INSIDE THE WINDOW IS NOT SEPARATED EITHER, and it ships in the disclosure**: with
    a robot's damping and joint friction removed entirely, a composed octopus reads 22.181 and a composed
    tabletop arm 18.778 at 200 ms, above the degenerate floor and with zero MuJoCo instability warnings.

    This deliberately does NOT try to name the cause. A divergent log can come from a broken model, an unstable
    rig, a bad timestep, a mis-scaled command or a driver fault, and this experiment cannot separate them --
    which is the whole point of refusing rather than guessing.
    """
    import numpy as np

    q = np.asarray(aligned.get("q_meas"), dtype=float)
    cmd = np.asarray(aligned.get("q_cmd"), dtype=float)
    base = {
        "measured": ("the largest joint excursion recorded in the log, as a multiple of the largest position "
                     "the log COMMANDS -- plus a finiteness check on every recorded channel"),
        "why": ("every other verdict in this report names a part of your robot. A log whose joints travel far "
                "outside anything they were asked to do is a record of a plant that diverged, not a "
                "measurement of one, and no parameter verdict taken from it means anything. This package "
                "refuses to rule rather than naming your drivetrain or your CAD off data that cannot support "
                "either."),
        "excursion_ratio_threshold_x": float(LOG_EXCURSION_RATIO_MAX),
        "valid_delay_window_ms": float(LOG_EXCURSION_VALID_DELAY_MS),
        "sampled_range_on_sane_logs": (
            "0.584 - 6.721 over 2864 readings on 18 WHEEL-FREE composed bodies (quadruped, hexapod, cat, "
            "horse, humanoid, biped, 6-axis arm, tabletop arm, snake, centipede, millipede, octopus, starfish, "
            "inchworm, turtle, gecko, crab, scorpion) x 8 injections each x every actuation delay from 0 to "
            "200 ms on a 10 ms grid, including a log with nothing wrong with it and including link-inertia "
            "errors of x40 and x100. The ceiling is a composed millipede at 60 ms. The degenerate population, "
            "measured on a composed spider over the same 8 injections and the same grid, runs 14.648 - 42964; "
            "the threshold is the geometric midpoint of that gap, sqrt(6.721 x 14.648) = 9.922, and it sits "
            "1.473x above the sane ceiling and 1.480x below the degenerate floor. "
            "WHAT IS SAMPLED, EXACTLY, because this field twice printed a range as though it were a population "
            "bound: those 8 injection families are the default perturbation, armature-only +0.009 and +0.050, "
            "frictionloss-only +0.030, damping-only +0.6, nothing wrong at all, and inertia_scale x40 and "
            "x100. EIGHT MORE FAMILIES WERE MEASURED AND ARE OUTSIDE THIS RANGE -- damping zeroed, "
            "frictionloss zeroed, both zeroed, torque_scale 0.8/1.25/1.5, link_scale 1.3, hold-only. Removing "
            "a robot's dissipation drives a composed octopus to 22.181 and a composed tabletop arm to 18.778 "
            "at 200 ms, both above the degenerate floor and both with ZERO MuJoCo instability warnings, so "
            "over that family this statistic does NOT separate and a robot with genuinely negligible joint "
            "damping will be refused here. That is a limit of the statistic, not of the threshold; no line on "
            "this ratio fixes it. "
            "TWO EARLIER FIGURES THIS FIELD PRINTED ARE RETIRED (both 2026-08-13). '0.874 - 1.405 over 56 logs "
            "on 7 bodies' was measured at ZERO actuation delay only, while this package's harness injects 2 "
            "control ticks by default and validates at 20 and 40 ms; at the threshold it produced (5.80) five "
            "composed-millipede logs at 40 ms were false-refused, one of them the control with nothing wrong "
            "with it. '0.554 - 9.748 over 1064 logs on 19 bodies' had a WHEELED ROVER as its ceiling -- a body "
            "whose joints have no positional ceiling at all -- and was paired with a degenerate floor of "
            "16.118 taken on a delay grid that skipped 50/80/110 ms, where the spider reads 14.648."),
        "outside_the_valid_delay_window": (
            f"past {LOG_EXCURSION_VALID_DELAY_MS:g} ms of actuation delay NOTHING IS MEASURED on the wheel-free "
            f"population and no verdict from this statistic should be trusted. If your log carries more delay "
            f"than that -- check application.log_plausibility against the delay this package recovers for you "
            f"-- read this guard as unmeasured rather than as a pass or a refusal. THE REASON THIS FIELD USED "
            f"TO GIVE IS RETIRED (2026-08-13): it said the populations overlap past 200 ms because a composed "
            f"wheeled rover reads 18.922 at 300 ms against the spider's 16.118. The rover is now excluded from "
            f"this statistic (its wheels are unbounded) and the spider's floor is 14.648, so both halves of "
            f"that sentence are gone. The window is kept because it is the range that was re-measured, not "
            f"because a wider one was tested and rejected."),
    }
    if q.size == 0 or cmd.size == 0:
        return {**base, "plausible": True, "measurable": True, "excursion_ratio": None, "finite": True,
                "unbounded_joints": None, "unbounded_joint_count": 0, "unbounded_joints_checked": model is not None,
                "reading": "the log carries no positions to check, so this guard does not rule"}

    finite = bool(np.all(np.isfinite(q)) and np.all(np.isfinite(cmd)))
    for key in ("qd_meas", "tau_meas"):
        arr = aligned.get(key)
        if arr is not None and np.asarray(arr).size:
            finite = finite and bool(np.all(np.isfinite(np.asarray(arr, dtype=float))))

    # WHICH CHANNELS THE RATIO MAY BE FORMED OVER. An unbounded joint's position integrates, so its excursion
    # grows without limit on a PERFECTLY SANE log and this statistic is undefined on it. It is EXCLUDED and
    # NAMED rather than measured -- see LOG_EXCURSION_RATIO_MAX for the rover this was found on.
    ncol = int(q.shape[1]) if q.ndim == 2 else 1
    cols, unbounded, checked = _bounded_channels(model, dofs, ncol)
    base = {**base,
            "unbounded_joints": list(unbounded) if unbounded else ([] if checked else None),
            "unbounded_joint_count": len(unbounded) if checked else None,
            "unbounded_joints_checked": checked,
            "measured_joint_count": len(cols),
            "excluded_because_unbounded": (
                f"{len(unbounded)} joint(s) carry jnt_limited=False in the compiled model -- their position "
                f"integrates without bound, so an excursion RATIO on them is undefined and they are excluded "
                f"from this statistic rather than measured: {', '.join(unbounded)}" if checked and unbounded else
                ("every logged joint's position is bounded (jnt_limited=True), so nothing was excluded"
                 if checked else
                 "no compiled model was supplied with this log, so jnt_limited could not be read and NO channel "
                 "could be excluded. On a robot with a wheel, a roller or any continuous-rotation joint this "
                 "guard is measuring a quantity that has no bound -- pass the model, or exclude those channels "
                 "from q_meas/q_cmd before sending the log"))}

    # THE STATISTIC CANNOT BE FORMED AT ALL when every logged joint is unbounded -- a wheeled rover is 4 of 4.
    # Reported as NOT MEASURABLE, which is neither a pass nor a refusal: `plausible` is None and every consumer
    # branches on `is False`. Calling this a pass would advertise a check that never ran; calling it a refusal
    # is the regression this repair exists to remove.
    if checked and not cols:
        return {
            **base, "plausible": None, "measurable": False,
            "commanded_envelope_rad": None, "largest_recorded_excursion_rad": None,
            "excursion_ratio": None, "worst_joint": None, "finite": finite,
            "not_measurable_because": (
                f"every one of the {len(unbounded)} logged joints on this robot is a continuous-rotation / "
                f"unbounded joint (jnt_limited=False in the compiled model): "
                f"{', '.join(unbounded)}. A peak POSITION divided by a commanded envelope has no bound on such "
                f"a joint, so there is no channel here this guard can rule on."),
            "reading": ("this guard DID NOT RUN on this log: every logged joint's position integrates without "
                        "bound, so the excursion ratio is undefined. That is neither a pass nor a refusal -- "
                        "the finiteness half still applies, and nothing about the excursions is claimed"),
        }
    sel = (lambda a: a[:, cols]) if q.ndim == 2 else (lambda a: a)
    qs, cs = sel(q), sel(cmd)
    envelope = float(np.max(np.abs(cs))) if np.all(np.isfinite(cs)) else 0.0
    denom = max(envelope, float(LOG_ENVELOPE_FLOOR_RAD))
    peak = float(np.max(np.abs(qs))) if finite else float("inf")
    ratio = peak / denom
    worst = None
    if finite and q.ndim == 2 and qs.shape[1]:
        j = int(cols[int(np.argmax(np.max(np.abs(qs), axis=0)))])
        worst = next((n for n, adr in (dofs or {}).items() if int(adr) == j), f"dof {j}")
    plausible = bool(finite and ratio <= float(LOG_EXCURSION_RATIO_MAX))
    return {
        **base,
        "commanded_envelope_rad": round(envelope, 6),
        "largest_recorded_excursion_rad": round(peak, 6) if finite else None,
        "excursion_ratio": round(ratio, 3) if finite else None,
        "worst_joint": worst,
        "finite": finite,
        "measurable": True,
        "plausible": plausible,
        "reading": (
            f"the largest joint excursion is {ratio:.3g}x the commanded envelope "
            f"(threshold {LOG_EXCURSION_RATIO_MAX:g}x); this log is usable"
            if plausible and finite else
            ("the log contains non-finite values, so no verdict can be taken from it" if not finite else
             f"the largest joint excursion is {ratio:.3g}x the commanded envelope of {envelope:.3g} rad, "
             f"against a threshold of {LOG_EXCURSION_RATIO_MAX:g}x. This is not a robot tracking a command")),
    }


def application_gate(trajectory: dict | None, n_identified: int, *,
                     global_scale: dict | None = None,
                     link_inertia: dict | None = None,
                     log_plausibility: dict | None = None,
                     threshold: float = MIN_TRACKING_IMPROVEMENT_X) -> dict:
    """May this fit be written into the model? Ruled on ``trajectory.improvement_x``, not on the fit's own
    objective.

    This exists because the tool already knew and nobody read it. ``improvement_x`` was computed here, copied
    into the calibration record, printed in the engineer brief -- and no consumer branched on it, so a fit that
    made tracking marginally WORSE (0.993x, from a hardware built with +30% link mass and inertia) wrote 15
    misattributed values into the compiled model and satisfied the L2 joint-coverage requirement on the way.

    Fail-closed in both directions that matter. A fit whose effect on tracking was never MEASURED is
    provisional too: "we did not check" is not "it helped".

    **This gate is NECESSARY AND NOT SUFFICIENT, and it used to be described as more than that.** See
    ``MIN_TRACKING_IMPROVEMENT_X``: the claim that no misspecification can improve tracking was measured and is
    RETRACTED. It holds for an error whose signature is GRAVITY (link mass, centre of mass); it does not hold
    for one the fitted parameters can mimic. Passing means the simulator got closer to the log. It does not
    mean the three numbers are the three physical quantities they are named after, which is why
    ``what_this_gate_does_not_catch`` ships in every verdict.

    **``global_scale`` is the SECOND, INDEPENDENT refusal, and it closes one of the two families that breached
    the first.** It is not another threshold on the same quantity -- the block above proves no threshold on
    ``improvement_x`` can work, because a correctly-specified fit reads 1.126 while a misspecified one reads
    1.753. It is a different question asked of the same data: can ONE global torque scalar explain this fit,
    and if so, does that one number track the log at least as well as the fit's forty-two? When it does, the
    two explanations were not distinguished by this experiment, one of them means every value is wrong by
    ``(1-g)/g``, and the fit is refused as ``torque_scale_suspected`` rather than written. Measured on the
    composed dog: it catches torque_scale x1.2 (the 1.507x case that broke the first gate) through x2.0, and
    refuses none of the ten correctly-specified fits. See ``GLOBAL_SCALE_COHERENCE_R2`` for the table -- and
    ``GLOBAL_SCALE_RIVAL_X`` for the two correct calibrations it HAS since been measured refusing, on a
    composed spider whose log is physically degenerate and which the fourth check below now stops first.

    **``link_inertia`` is the THIRD refusal, and it closes the OTHER family.** Same shape, different scalar:
    scaling every link's rotational inertia by ``s`` adds ``(s-1)`` times each joint's own rotational inertia to
    the same ``diag(M)`` armature adds to, so the fit absorbs the error into armature and one number reproduces
    the plant. It is refused as ``link_inertia_suspected``, and the remedy it names is the inertia tensor / the
    CAD, not the drivetrain. Measured on two composed bodies: it catches every point of the inertia family that
    clears the tracking gate (x20 to x100 on the dog, x20 to x100 on the hexapod).

    **AND IT TOOK TWO CONDITIONS, because on one it refused correct calibrations (2026-08-13).** This docstring
    used to end "refuses none of the twelve correctly-specified fits, INCLUDING the armature-only calibration".
    That was true of the twelve and false of the population: all twelve improved tracking by 3.34x or more, and
    just above the 1.5x gate -- where a customer's fit is most likely to land -- armature-only +0.009 (1.602x)
    and +0.010 (1.752x) were both REFUSED. The cause is that ``rival.explains_x`` is the rival's improvement
    DIVIDED BY the fit's, so a weak-but-correct fit inflates it. The verdict now also requires
    ``rival.rival_improvement_x`` -- the rival against the PRIOR model, with the fit nowhere in it -- to clear
    ``LINK_INERTIA_RIVAL_IMPROVEMENT_X``. See that constant for the three-body danger-band sweep, and
    ``what_this_gate_does_not_catch`` for exactly which region was sampled.

    **AND THE MARGIN THAT SWEEP ADVERTISED WAS ITSELF A TWO-BODY SAMPLE (2026-08-13, same day).** This
    docstring and that constant both quoted a correct ceiling of 1.930 against a catch floor of 19.604 with the
    line at 6.0. On a composed CAT the correct ceiling rose to 3.679 and the catch floor fell to 14.469, so the
    line moved to 7.3, the geometric midpoint of the gap as it stood. **AND THEN A COMPOSED CENTIPEDE READ
    4.037, so the ceiling is 4.037 and the band is 3.58x -- not the 3.93x that correction claimed, and not the
    10.2x before it.** 7.3 is therefore no longer the midpoint: it sits 1.81x above the correct ceiling and
    1.98x below the catch floor. The constant is left where it is on purpose (no measured verdict differs) and
    the off-centring is disclosed instead. Each restatement of this band has SHRUNK it, every time by adding a
    body, which is the reading to carry: these edges are running maxima over the bodies sampled, not bounds.

    **``log_plausibility`` is the FOURTH refusal, and it is the only one that names nothing.** It is checked
    FIRST, before the other three, and it exists because all three of them will happily name a part of the
    customer's robot off a log that cannot support any verdict at all. MEASURED on a composed 8-legged spider
    whose synthetic log records 298.9 rad of joint travel against a 1.634 rad commanded envelope: two CORRECT
    armature-only calibrations came back ``torque_scale_suspected``, sending the customer to a gear ratio that
    was right. The remedy is not a better rival; it is a refusal. Refused as ``implausible_log``, and the only
    thing named is the log. See ``LOG_EXCURSION_RATIO_MAX`` for the two populations that sized the bound.
    """
    base = {
        "gate": "tracking_improvement",
        # ALWAYS present, so a consumer can branch on WHICH gate refused without probing for a key. It is
        # None on every path except the three non-tracking refusals, which set it to "global_scale",
        # "link_inertia" or "implausible_log".
        "refused_by": None,
        "measured_on": "position RMS of our replay against the log, before vs after applying ONLY the "
                       "identified deltas -- a quantity the fit did not optimise (it minimised a TORQUE "
                       "residual)",
        "threshold_x": float(threshold),
        "why_this_threshold": (
            "measured, not chosen: across sim2sim cases on a composed 14-DOF quadruped, a link mass and "
            "inertia error (x0.4 to x3.0), a centre-of-mass error (5-50 mm) and a sub-floor injection improved "
            "tracking by at most 1.000x -- i.e. not at all -- while every correctly-specified perturbation "
            "improved it by at least 3.343x. The threshold sits between those. In engineer units: the position "
            "RMS must fall by at least a third before anything is written to the model."),
        # NOT a footnote. The sentence above used to end "...the threshold sits in that EMPTY band", and the
        # band is not empty: two misspecification families were measured straight through it and out the other
        # side. A customer reading a PASS is entitled to the limit in the same dict as the verdict.
        #
        # REWRITTEN when the second check landed, and NARROWED rather than deleted -- it now describes what is
        # still invisible AFTER that check, which is less than before and is not nothing.
        "what_this_gate_does_not_catch": (
            "an error the fitted parameters can MIMIC, MINUS the two that separate replay checks now catch. "
            "Two families were measured through this threshold on a composed 14-DOF quadruped and BOTH are now "
            "caught, each by its own one-number rival in this same verdict, each naming a different remedy. "
            "(1) A TORQUE-SCALE error -- wrong gear ratio, wrong torque constant, unmodelled gearbox "
            "efficiency -- cleared it from 20% (1.507x) and reached 1.647x. Scaling the applied torque by g is "
            "algebraically dividing inertia, damping and friction by g, so it moves all three fitted parameters "
            "on all joints by the same (1-g)/g, and a replay of your PRIOR model with one gear scalar is put up "
            "against the fit. Measured catch range on that robot: g from 1.2 to 2.0 (everything in that family "
            "that clears this threshold at all), rival 17.9-230x. (2) LINK ROTATIONAL INERTIA wrong at exactly "
            "correct mass reaches 1.753x. It has no gravity signature, so this gate cannot see it, and it is "
            "absorbed by ARMATURE ALONE, so the global-scale check correctly does not fire on it (coherence "
            "0.289). It is caught by the 'link_inertia' check instead: a replay of your PRIOR model with every "
            "link's rotational inertia scaled by one number and its MASS untouched. Measured catch range: "
            "inertia_scale x20 to x100 -- every point in that family that clears this threshold at all -- at "
            "rival_improvement_x 14.469-633.220, over 15 breaches on THREE composed bodies. (This field used "
            "to print '19.6-633x' here and '14.469' as the catch floor a few sentences later, as though both "
            "were current. They are not: 19.604 is the SUPERSEDED two-body floor, retired on 2026-08-13 when "
            "a third body read 14.469, and the current floor is the one printed here. Only one catch floor is "
            "in force and it is 14.469.) "
            "THAT CHECK REFUSED CORRECT CALIBRATIONS WHEN IT FIRST SHIPPED AND THE STATEMENT HERE WAS WRONG "
            "(corrected 2026-08-13). It used to read 'ZERO correctly-specified fits refused', on a sample of "
            "nine that all improved tracking by 3.34x or more. Swept densely in the 1.5x-3.0x band those nine "
            "never entered, an ordinary armature-only calibration (+0.009 and +0.010 against a prior of 0.010) "
            "was REFUSED, because the old statistic divided the rival's improvement by the FIT's and the fit "
            "is weak there. The verdict now needs the rival to explain the LOG (rival_improvement_x, which "
            "does not contain the fit) as well as to beat the fit. WHAT IS SAMPLED, and no claim is made "
            "outside it: on a composed 14-DOF quadruped, a composed 18-DOF hexapod and a composed 14-DOF CAT, "
            "a dense sweep of correctly-specified fits landing between 1.5x and 3.0x -- armature-only at many "
            "magnitudes, frictionloss-only, damping-only, mixed, and an armature offset built PROPORTIONAL to "
            "each joint's own rotational inertia (the exactly-degenerate correct calibration) -- plus the same "
            "families above and below the band, gives ZERO refusals over 88 correct fits that reach a rival. "
            "The largest correct rival_improvement_x over EVERY body measured is 4.037 (composed centipede, "
            "proportional family -- see the per-body table below), the threshold is 7.3, and the catch floor "
            "is 14.469. THOSE THREE NUMBERS WERE 1.930 / 6.0 / 19.604 UNTIL 2026-08-13 and were quoted here as "
            "the measured band; they were a two-body sample. The cat moved the correct ceiling UP to 3.679 (an "
            "armature offset proportional to each joint's own rotational inertia, improvement_x 2.285) and the "
            "catch floor DOWN to 14.469 (inertia_scale x30, improvement_x 1.500) in the same session; then the "
            "CENTIPEDE moved the ceiling again to 4.037. This field printed 3.679 as 'the largest correct' "
            "while the table two paragraphs down already said 4.037 -- corrected 2026-08-13. The empty band is "
            "therefore 3.58x, not the 3.93x that correction claimed and not the 10.2x before it, and 7.3 is no "
            "longer its geometric midpoint: it sits 1.81x above the correct ceiling and 1.98x below the catch "
            "floor. The threshold is deliberately NOT re-cut for that -- no verdict on any measured case "
            "differs between 7.3 and the current midpoint of 7.643, and every restatement of this band has "
            "shrunk it by adding one more body, so the edges are running maxima and not bounds. "
            "WHAT IS NOT SAMPLED, so nothing is claimed there: real hardware; any IMPORTED body; and DAMPING-"
            "ONLY anywhere inside the band on the dog or the hexapod -- that family steps across it "
            "discontinuously as more pairs become identified (dog 1.129x at +0.118 to 3.042x at +0.120; "
            "hexapod 1.267x at +0.060 to 3.427x at +0.070), and every straddling point is applied. "
            "AND AN EXTRAPOLATION THAT USED TO STAND HERE IS RETRACTED (2026-08-13): this field said nothing "
            "was claimed for a body whose joints' rotational inertias span much more than the 1.53-1.55x the "
            "first two do, because 'a wider span would make this check SHARPER -- a prediction, not a "
            "measurement'. Measured, span does not order them. THE RANGES BELOW ARE PER-FAMILY, NOT POPULATION "
            "BOUNDS, and this field used to print them with no scope at all while the scope lived only in "
            "link_inertia_signature's docstring (corrected 2026-08-13): each was ONE armature-only family on "
            "ONE body, and a different CORRECT family on the same body goes outside it. Both families now "
            "measured, delay 0, n_boot 64 -- armature-only (+0.009, +0.050) and the adversarial PROPORTIONAL "
            "family (armature offset = k x each joint's own rotational inertia, k = 30/45/60), which is the "
            "exactly-degenerate correct calibration the rival's one-parameter family contains by construction. "
            "Format 'span -> armature-only / proportional': "
            "dog 1.55 -> 1.777-1.777 / 1.805-1.823; "
            "cat 2.63 -> 2.219-3.118 / 3.495-3.679; "
            "horse 4.04 -> 1.526-1.562 / 1.639-1.696 (the '1.53-1.55' printed here before was the "
            "armature-only family alone, and the proportional family on the same body exceeds it); "
            "humanoid 14.66 -> 1.191-1.240 / no rival runs; "
            "snake 33.51 -> no rival runs / 1.319-1.364; "
            "6-axis arm 44.83 -> 0.832-0.940 / 1.334-1.337; "
            "CENTIPEDE 32.22 -> 1.937-2.383 / 3.130-4.037. "
            "**THE CENTIPEDE'S 4.037 IS ABOVE THE 3.679 THIS FIELD USED TO CALL THE LARGEST CORRECT "
            "rival_improvement_x**, so the correct population's ceiling over the bodies now measured is 4.037, "
            "and that is the figure this field now states everywhere rather than in this one paragraph only. "
            "No measured verdict changes -- the threshold is 7.3 and the catch floor 14.469, so the line sits "
            "1.81x above the ceiling and 1.98x below the floor. It is therefore no longer exactly centred, and "
            "that is stated rather than used as a reason to re-cut a constant on one more body. The widest "
            "correct population belongs to the centipede and the second widest to the cat, which is the second "
            "NARROWEST span. Span is "
            "reported (link_inertia.rotational_inertia_span_x) and is not a bound in either direction; what "
            "sets a body's correct ceiling is not known, and no replacement prediction is offered. "
            "WHAT REMAINS INVISIBLE, measured rather than assumed. Neither check sees an error that moves only "
            "SOME joints: both rivals are whole-robot scalars. The global-scale check cannot run at all on a "
            "robot where fewer than two of the three parameters have a non-zero prior, and the link-inertia "
            "check cannot run where fewer than two joints have a non-zero rotational inertia in their own "
            "diag(M); both say so, in global_scale.not_measurable_because and link_inertia.not_measurable_"
            "because. A link-inertia error SMALLER than about x10 of the tensor (+19% of diag(M)) is not caught "
            "-- but it is also not a breach: it scores under 1.5x and this gate refuses it. A CENTRE-OF-MASS "
            "or LINK-MASS error still reaches neither rival, because the tracking gate already refuses that "
            "family at <= 1.000x -- if you ever see one of those clear 1.5x, nothing here is watching for it. "
            "AND A RISK ON THE OTHER CHECK, NAMED RATHER THAN PATCHED: the torque-scale rival is still scored "
            "with the un-normalised statistic the link-inertia check had to abandon, so it too rises as the "
            "fit weakens. Over the same dense 1.5x-3.0x sweep its largest value on a CORRECT fit is 0.852 "
            "(against a 1.0 line) rather than the 0.389 measured on strong fits, so the margin there is 1.17x "
            "and not 2.6x. THIS FIELD USED TO ADD 'it has never been observed to refuse a correct calibration'. "
            "That is no longer true (2026-08-13): on a composed 8-legged spider, correct armature-only "
            "calibrations at +0.009 and +0.05 were refused as torque_scale at 1.023 and 1.218. Both of those "
            "logs are physically degenerate -- max|q_meas| 298.9 rad against a 1.63 rad commanded envelope -- "
            "and are now refused earlier as 'implausible_log' rather than being blamed on a drivetrain, so on a "
            "SANE log no false refusal has been observed and the 0.852 ceiling stands. The 1.0 line is "
            "deliberately NOT moved: unlike the link-inertia threshold there is no two-population sweep behind "
            "it (the torque-scale catch reproduces on ONE body only, so there is no catch floor to take a "
            "midpoint against), and a number chosen without one would be chosen to look safe. If your fit is "
            "just above the tracking threshold and is refused as torque_scale, read global_scale.rival's "
            "numbers before touching your drivetrain. "
            "AND A REFUSAL THAT NAMES NO PART OF YOUR ROBOT AT ALL: before any of the above, the LOG itself is "
            "checked for finiteness and for joint excursions wildly outside the commanded envelope. A log that "
            "fails is refused as 'implausible_log' and nothing -- not the drivetrain, not the CAD -- is named, "
            "because a divergent plant cannot support any parameter verdict. Measured sane range on that "
            "statistic: 0.584-6.721 over 2864 readings on 18 WHEEL-FREE composed bodies x 8 injection families "
            "x every actuation delay to 200 ms on a 10 ms grid; see log_plausibility in this output for the "
            "families sampled and for the ones that fall OUTSIDE that range. THAT SURVEY WAS ORIGINALLY TAKEN "
            "AT ZERO DELAY (quoted here as '0.874-1.405 over 56 logs on 7 bodies') WHILE THE HARNESS INJECTS "
            "TWO CONTROL TICKS BY DEFAULT, and the threshold it produced -- 5.80 -- refused five "
            "composed-millipede logs at 40 ms, one of them the control with nothing wrong with it. Re-measured "
            "across delay the line became 12.5; re-measured again with wheeled bodies excluded and the spider "
            "taken on a 10 ms delay grid it is 9.9, the geometric midpoint of 6.721 and 14.648. The 12.5 pair "
            "(9.748 / 16.118) is retired: the first was a wheeled rover, the second an artefact of a delay "
            "grid that skipped 50/80/110 ms. The bound is SCOPED: past 200 ms nothing is measured and the "
            "guard is not claimed to work, which log_plausibility.outside_the_valid_delay_window says in the "
            "verdict. "
            "AND THE STRUCTURAL BLIND SPOT OF THAT GUARD, WHICH WAS A LIVE FALSE REFUSAL AND IS NOW HANDLED "
            "RATHER THAN ONLY NAMED (2026-08-13): it divides a peak POSITION by a commanded envelope, so it "
            "assumes every joint's position is BOUNDED. Any joint whose position INTEGRATES WITHOUT BOUND -- a "
            "drive wheel, a roller, a spindle, a continuous-rotation or velocity-commanded joint, or a "
            "multi-turn joint logged without wrapping -- makes that ratio grow without limit on a PERFECTLY "
            "SANE log. WHAT IT COST BEFORE THE REPAIR, measured through the shipped fit_parameters: a composed "
            "SIX-WHEELED ROVER carrying a genuine 25% torque-scale error (torque_scale=1.25, nothing else "
            "wrong) read 0.812 at 0 ms of actuation delay and was correctly refused as 'global_scale' with an "
            "implied g of 1.2974 against a truth of 1.25 -- and at 200 ms, INSIDE the declared valid window, "
            "the same error read 12.598 and was refused as 'implausible_log' instead, deleting the drivetrain "
            "finding on a log with zero MuJoCo instability warnings. THE REPAIR: the ratio is now formed over "
            "LIMITED joints only, read off the compiled model's jnt_limited, and the excluded joints are NAMED "
            "in log_plausibility.unbounded_joints with their count. Where a body has NO limited joint at all -- "
            "the rover is 4 of 4 unbounded, or 6 of 6 on the six-wheeled variant, while every other composed "
            "body surveyed is 0 of 6..56 -- the statistic cannot be formed and is reported as NOT MEASURABLE "
            "(plausible=None, not_measurable_because), which is neither a pass nor a refusal. WHAT IS STILL "
            "OPEN: jnt_limited is only readable when a compiled model is available. A log supplied as bare "
            "ARRAYS with no model cannot be checked, and there the blind spot is exactly as it was -- measured, "
            "a wheel turning at a steady 5 rad/s for 4 s beside a normally-tracking arm still reads 99.95x and "
            "is still refused. It also cannot be sampled properly in this package's own sim2sim gate, because "
            "bench_rig.bench_model WELDS the base and REMOVES the floor, so nothing rolls and every joint is "
            "driven to a bounded position setpoint. If you are sending arrays without a model and your robot "
            "has a wheel, a turret or any continuous joint, exclude those channels from q_meas/q_cmd or read an "
            "'implausible_log' verdict as UNMEASURED rather than as a finding. "
            "AND THE ONE THAT IS STILL OPEN AND WAS EXPECTED TO CLOSE: a torque-scale error SUPERPOSED on a "
            "real dissipation change. Measured, a 1.35x torque-constant error on top of a genuine perturbation "
            "reads 1.689x and IS applied -- the gear rival now runs on it (the coherence filter that used to "
            "stop it is gone) and honestly LOSES at 0.869, because the log really does contain a change no "
            "single scalar produces. Neither rival is the instrument for that one; every value in such a fit "
            "is out by the same factor while the fit itself is the better explanation of the log. "
            "So a PASS says your simulator now tracks your log, and that neither a single global torque scale "
            "nor a single link-inertia scale explains it as well as the fit does. It does not say these three "
            "numbers are the three physical quantities they are named after. If your link inertia tensor is "
            "uncertain, read the per-parameter 'also_absorbs' field before acting on a value."),
    }
    gs = dict(global_scale or {})
    li = dict(link_inertia or {})
    lp = dict(log_plausibility or {})
    base["log_plausibility"] = lp or None
    x = (trajectory or {}).get("improvement_x")
    if int(n_identified) <= 0:
        return {**base, "improvement_x": x, "passed": None, "provisional": False,
                "global_scale": gs or None, "link_inertia": li or None,
                "verdict": "nothing was identified, so there is nothing to apply and nothing to gate"}
    # THE FIRST REFUSAL, and it has to be first: every verdict below it names a PART of the customer's robot,
    # and naming a part off a log that cannot support any verdict is the worst failure available here. MEASURED
    # on a composed spider whose joints travel 183x their commanded envelope: two correct calibrations were
    # refused as torque_scale, sending the customer to a gear ratio that is right. See LOG_EXCURSION_RATIO_MAX.
    # ``is False``, not ``not ...``: ``plausible`` is None when the guard could not form its statistic at all
    # (every logged joint unbounded -- a wheeled rover is 4 of 4), and NOT MEASURABLE is neither a pass nor a
    # refusal. Reading None as a refusal deleted a real drivetrain finding on a numerically stable rover log.
    if lp and lp.get("plausible") is False:
        return {
            **base, "improvement_x": x, "passed": False, "provisional": True,
            "global_scale": gs or None, "link_inertia": li or None,
            "refused_by": "implausible_log",
            "verdict": (
                f"THIS LOG CANNOT BE RULED ON -- and the problem is the LOG, not your robot. "
                + (f"It contains non-finite values (NaN or Inf), so the plant that produced it left the reals. "
                   if not lp.get("finite", True) else
                   f"Joint '{lp.get('worst_joint')}' travels {lp.get('largest_recorded_excursion_rad')} rad "
                   f"against a commanded envelope of {lp.get('commanded_envelope_rad')} rad -- "
                   f"{lp.get('excursion_ratio')}x, against a threshold of "
                   f"{lp.get('excursion_ratio_threshold_x')}x. ")
                + f"That is not a robot tracking a command; it is the record of a divergent simulation or rig. "
                  f"Every parameter in this fit is still reported so it can be read, but NOTHING is written to "
                  f"the model and NO part of your hardware is named -- deliberately. On data like this the "
                  f"estimator cannot separate a real parameter error from an unstable integration, so any "
                  f"verdict pointing at a component would be a guess wearing a measurement's clothes. The "
                  f"honest answer is that this experiment cannot tell you anything. "
                  f"WHAT TO DO: re-take the log. Check that the excitation "
                  f"amplitudes are inside the joints' travel, that the rig actually holds the base, that the "
                  f"controller gains in the plan match the ones that ran, and -- if this came out of a "
                  f"simulator -- that it did not report instability warnings while producing it. What a SANE "
                  f"log looks like on this statistic: {lp.get('sampled_range_on_sane_logs')}"),
        }
    if x is None:
        return {**base, "improvement_x": None, "passed": False, "provisional": True,
                "global_scale": gs or None, "link_inertia": li or None,
                "verdict": "the fit's effect on tracking was NOT MEASURED, so it cannot be shown to help. The "
                           "fit is PROVISIONAL and is withheld from the model. Re-fit with "
                           "measure_trajectory=True, or apply it deliberately with allow_provisional=True."}
    x = float(x)
    # THE SECOND REFUSAL, and it is checked BEFORE the tracking verdict so the customer is told the cause
    # rather than the symptom. A fit that a single gear scalar explains at least as well is not a calibration;
    # it is a reading of an instrument whose scale is wrong, and every number in it carries that factor.
    rival = gs.get("rival") or {}
    if gs.get("suspected"):
        g = gs.get("implied_torque_scale_g")
        gg = rival.get("torque_scale_g", g)
        return {
            **base, "improvement_x": x, "passed": False, "provisional": True,
            "global_scale": gs,
            "refused_by": "global_scale",
            "verdict": (
                f"TORQUE SCALE SUSPECTED -- this fit is REFUSED even though it tracks the log {x:g}x closer "
                f"(threshold {threshold:g}x). Every fitted parameter moved by nearly the same fraction of its "
                f"own prior on every joint (coherence {gs.get('coherence')}), which is the exact signature of "
                f"your joints receiving about {gg:g}x the torque this log records rather than of a real change "
                f"in friction, damping or reflected inertia. Put to the test: replaying our ORIGINAL model with "
                f"nothing changed except every actuator's gear scaled by {gg:g} tracks your log "
                f"{rival.get('explains_x')}x BETTER than this fit does. One number explains your log at least "
                f"as well as the {int(n_identified)} this fit wants to write, and the experiment cannot tell "
                f"them apart -- so the {int(n_identified)} are withheld. "
                f"WHAT TO DO: check the GEAR RATIO and TORQUE CONSTANT your driver used to compute the torque "
                f"in this log against the BOM / motor datasheet (a {gg:g}x error there produces exactly this), "
                f"and check for an unmodelled gearbox efficiency. If you can log the per-joint motor CURRENT "
                f"alongside the reported torque, re-fit with torque_constant_nm_per_a set from the datasheet: "
                f"that is the one channel that carries information this experiment does not. If you have "
                f"confirmed the drivetrain is right and believe the coherence is a coincidence, "
                f"fit_actuators {{allow_provisional: true}} will write it anyway."),
        }
    # THE THIRD REFUSAL. Same question, other scalar, and a DIFFERENT remedy -- which is why it is a separate
    # verdict rather than more wording on the one above. A fit that a single link-inertia scale explains at
    # least as well is not a measurement of the drivetrain; it is a measurement of a CAD number.
    lriv = li.get("rival") or {}
    if li.get("suspected"):
        ss = lriv.get("inertia_scale_s", li.get("implied_inertia_scale_s"))
        return {
            **base, "improvement_x": x, "passed": False, "provisional": True,
            "global_scale": gs or None, "link_inertia": li,
            "refused_by": "link_inertia",
            "verdict": (
                f"LINK INERTIA SUSPECTED -- this fit is REFUSED even though it tracks the log {x:g}x closer "
                f"(threshold {threshold:g}x). Every joint's fitted ARMATURE moved by nearly the same fraction "
                f"of its own link-side rotational inertia, which is what happens when the INERTIA TENSORS in "
                f"your model are wrong rather than when reflected inertia has really changed -- reflected and "
                f"link inertia enter each joint's equation through the same acceleration term, so this "
                f"experiment cannot separate them and the link side is held fixed. Put to the test: replaying "
                f"our ORIGINAL model with nothing changed except every link's rotational inertia scaled by "
                f"{ss:g} -- masses untouched -- gets our ORIGINAL model "
                f"{lriv.get('rival_improvement_x')}x closer to your log (we require "
                f"{lriv.get('improvement_threshold_x')}x before saying this), and tracks "
                f"{lriv.get('explains_x')}x BETTER than this fit does. One number explains your log at least "
                f"as well as the {int(n_identified)} this fit wants to write, so the {int(n_identified)} are "
                f"withheld. "
                f"WHAT TO DO: check the LINK INERTIA TENSOR in your model against your CAD -- the ixx/iyy/izz "
                f"(or the <inertial> block of your URDF) of the moving links, and the density or fill fraction "
                f"they were derived from. Note your MASSES are not implicated: a mass error would have shown "
                f"up as a gravity offset this fit could not remove, and it did not. If you have confirmed the "
                f"tensors are right, fit_actuators {{allow_provisional: true}} will write it anyway."),
        }
    if x >= float(threshold):
        return {**base, "improvement_x": x, "passed": True, "provisional": False,
                "global_scale": gs or None, "link_inertia": li or None,
                "verdict": f"applying this fit tracks the log {x:g}x closer in position RMS "
                           f"(threshold {threshold:g}x)"}
    return {
        **base, "improvement_x": x, "passed": False, "provisional": True,
        "global_scale": gs or None, "link_inertia": li or None,
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
                   delay_max_ticks: int = 8,
                   torque_constant_nm_per_a=None) -> dict:
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
    from virturoid.services.sysid.torque_channel import convert_current_to_torque

    t_wall = time.perf_counter()
    # A motor-current channel becomes the torque channel here, through a torque constant, and the conversion
    # rides on the result. It matters MORE at this stage than at Stage 1: the delay is a lag and survives a
    # wrong constant, but every parameter below comes out of a torque residual, so an error in kt is an error
    # of the same fraction in frictionloss, damping and armature. See ``torque_channel``.
    log, torque_channel = convert_current_to_torque(gene, log, explicit=torque_constant_nm_per_a)
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
                "error": "the log carries neither tau_meas nor a usable motor-current channel",
                "why": "the quantity being fitted is a TORQUE residual. Without measured torque there is "
                       "nothing to regress, and a fit against the trajectory alone would be an unidentifiable "
                       "mixture of every parameter at once.",
                "what_is_still_available": "measure_gap on this same log still reports the per-joint "
                                           "trajectory gap AND the actuation delay -- the delay is read out "
                                           "of the motion rather than out of a torque channel. Only the "
                                           "PARAMETERS need torque.",
                "how": "log the per-joint effort (ROS 2 JointState.effort / ros2_control's effort state "
                       "interface) or the per-joint motor current under i_meas, and pass "
                       "torque_constant_nm_per_a if you have the datasheet value",
                "torque_channel": torque_channel}

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
                "(0.0100 -> 0.01664, 90% CI [0.01317, 0.02082]). THAT case is caught by the "
                "tracking-improvement gate under 'application' -- because its error is mostly GRAVITY, which "
                "no fitted parameter can remove. An INERTIA error at CORRECT MASS has no gravity term, so that "
                "gate is blind to it: measured at a +60% error in each joint's own diag(M), the same 14/14 "
                "armature misattribution scores 1.745x and clears the 1.5x gate. It is caught by the "
                "'link_inertia' check instead (2026-08-12), which is the SAME degeneracy read as evidence: if "
                "the fitted armature on every joint is the same fraction of that joint's own ROTATIONAL "
                "inertia, one inertia scalar is a live rival, and our prior model replayed with that one "
                "scalar -- masses untouched -- is put up against the whole fit. Measured catch range on the "
                "composed dog: inertia_scale x20 to x100, i.e. every point in that family that breaches the "
                "tracking gate, at rival 12.4-181x. THIS SENTENCE USED TO END 'with zero of the "
                "correctly-specified fits refused' (corrected 2026-08-13): that was true of the nine fits then "
                "sampled and false of the population -- the check as first shipped refused ordinary "
                "armature-only calibrations just above the tracking gate, which is why the verdict now needs "
                "TWO conditions. See application.what_this_gate_does_not_catch for the three-body sweep, the "
                "measured correct ceiling (4.037, composed centipede -- it was quoted here as 3.679 until "
                "2026-08-13, which was the ceiling over five bodies and not over all of them) and catch floor "
                "(14.469) that size it, and the region that "
                "is still unsampled. This entry is what NAMES the mechanism; that field is what bounds the "
                "gates' reach, and until 2026-08-12 this sentence claimed more than they delivered.",
            "the gear ratio / torque constant your driver used to report torque":
                "ASSUMED CORRECT, still not FITTED -- but no longer unchecked. If your joint really receives g "
                "times the torque this log records, that is algebraically the same as dividing inertia, damping "
                "and friction by g -- all three of which ARE fitted -- so the error is absorbed almost exactly "
                "and the fit reports the true parameters scaled by (1-g)/g. MEASURED sim2sim: at g=0.5 the "
                "median damping delta comes back at +0.841 against the +0.800 that predicts, and at g=1.2 the "
                "resulting fit scores 1.507x and CLEARS the tracking gate. The sentence that used to end this "
                "entry -- 'nothing in this report can tell that apart from a real dissipation change' -- is "
                "RETRACTED (2026-08-12) and replaced by the 'global_scale' block in this same output. What "
                "tells it apart is that the absorption is COHERENT: one scalar for all three parameters on all "
                "joints, which a real dissipation change has no reason to be. When it is, our PRIOR model is "
                "replayed with nothing changed but that one gear scalar, and if one number tracks your log at "
                "least as well as the whole fit does, the fit is refused as torque_scale_suspected instead of "
                "written. Measured catch range on the composed dog: g 1.2 to 2.0, with zero of ten "
                "correctly-specified fits refused.",
            "contact / gearbox elasticity / backlash":
                "not represented in the compiled model at all, so there is no parameter to move. The bench rig "
                "measures a joint on a stand rather than a foot on the ground precisely so contact is not in "
                "the residual; backlash and drivetrain compliance still are, and they land in damping and "
                "frictionloss. MEASURED for the elasticity half: an unmodelled joint SPRING at 0.02 to 1.0 "
                "times the bench gain moves trajectory RMS by up to 0.114 rad and yields NOTHING identified at "
                "any size -- so this one costs the experiment rather than corrupting it.",
        },
        "assumed_correct": {
            "what": ["link mass", "link inertia", "centre of mass", "kinematic geometry", "gear ratio"],
            "consequence": "an error in any of these is not reported; it is absorbed by the three parameters "
                           "that ARE fitted, armature first. See parameters_not_fitted for the measured case. "
                           "FOUR gates now rule on the result and they catch different things. Before any of "
                           "them, the LOG-PLAUSIBILITY guard refuses to rule at all -- naming no component -- "
                           "when the log's joint excursions run wildly outside what it commanded or carry "
                           "non-finite values, because a divergent plant cannot support any of the verdicts "
                           "below it; see application.log_plausibility. Then: the TRACKING "
                           "gate refuses the fit when the error's signature is gravity (link mass, centre of "
                           "mass). The GLOBAL-SCALE check refuses it when the error is a single multiplicative "
                           "factor on the applied torque (gear ratio, torque constant, gearbox efficiency), "
                           "which the tracking gate provably cannot -- see application.global_scale. The "
                           "LINK-INERTIA check refuses it when the error is a single multiplicative factor on "
                           "the links' rotational inertia at correct mass, which neither of the other two can: "
                           "no gravity term for the first, and armature alone is not a global scale for the "
                           "second -- see application.link_inertia. What no check sees is an error that moves "
                           "only SOME joints; both rivals are whole-robot scalars. See "
                           "application.what_this_gate_does_not_catch.",
        },
        "joints": joints,
        "identified_pairs": len(identified_pairs),
        "candidate_pairs": len(joints) * len(param_names),
        "refused": refused,
        "wall_clock_s": None,
    }

    # LATENCY FIRST, and that ordering is load-bearing. ``_trajectory_improvement`` replays the loop, and a
    # replay run at the wrong delay measures the delay instead of the fit -- MEASURED on the Go2 at 30 ms of
    # injected delay, NOTHING the estimator can express scores above 1.012x on the delay-blind metric, against a
    # 1.5x threshold. Scoring both replays at the IDENTIFIED delay is what makes the ratio about the parameters
    # again, and that requires knowing the delay before the trajectory is measured.
    if measure_delay:
        out["latency"] = _delay_on_the_fitted_model(gene, model, aligned, deltas, dofs, kp, kd, log, plan,
                                                    joints, int(delay_max_ticks))
    ticks = 0
    if measure_trajectory:
        lat = out.get("latency") or {}
        ticks = int(lat["delay_ticks"]) if lat.get("identified") and lat.get("delay_ticks") else 0
        out["trajectory"] = _trajectory_improvement(gene, model, aligned, deltas, dofs, kp, kd, log, plan,
                                                    joints, delay_ticks=ticks,
                                                    delay_identified=bool(lat.get("identified")),
                                                    delay_source=lat.get("source"))

    # THE SECOND, INDEPENDENT CHECK. Part one is free and always runs, because "is this fit a single global
    # torque scale?" is worth reporting whatever the verdict. Part two costs ~10 replays and runs ONLY when
    # part one has fired AND the fit would otherwise be written -- there is nothing to protect a customer from
    # on a fit that is already being refused, and every correctly-specified fit measured so far stops at part
    # one (8 of 10) or loses the replay (2 of 10).
    gscale = global_scale_signature(model, dofs, joints, param_names)
    traj = out.get("trajectory") or {}
    imp = traj.get("improvement_x")
    q_hw = aligned["q_meas"]
    q0r = start_pose(model, gene) if q_hw.shape[0] == 0 else q_hw[0].copy()
    after_rms = float(traj.get("after_rms_rad") or 0.0)
    # IS THIS A LOG AT ALL? Free (it reads the arrays), always computed, and it gates the two rivals as well as
    # the verdict: on a divergent log a rival's number is a confident answer to a question the data cannot
    # answer, and shipping one is how a composed spider's correct calibration got blamed on its gear ratio.
    # THE MODEL IS PASSED, and that is the whole of the unbounded-joint repair at this call site: `jnt_limited`
    # is readable straight off the compiled model and is the only sound way to tell a continuous-rotation hinge
    # from a diverging one. See LOG_EXCURSION_RATIO_MAX.
    plaus = log_plausibility(aligned, dofs, model)
    out["log_plausibility"] = plaus
    # WOULD THIS FIT OTHERWISE BE WRITTEN? That, and not the coherence, is what decides whether a rival is
    # simulated. There is nothing to protect a customer from on a fit that is already being refused.
    would_apply = bool(identified_pairs and imp is not None
                       and float(imp) >= float(MIN_TRACKING_IMPROVEMENT_X) and after_rms > 0.0
                       # `is not False`, NOT a truthiness test: `plausible` is None when the statistic could
                       # not be formed (every logged joint unbounded), and NOT MEASURABLE must not act as a
                       # refusal. That conflation is exactly the rover regression.
                       and plaus.get("plausible") is not False)

    gscale["suspected"] = False
    # THE COHERENCE FILTER IS NO LONGER A GATE ON THE REPLAY, and that is a measured change (2026-08-12).
    # It used to read ``if gscale["coherent"] and ...``, i.e. the cheap statistic decided whether the honest
    # test was allowed to run -- which meant any fit whose coherence was diluted got a pass without ever being
    # asked the question. Removing it is SAFE and CHEAP, and both halves were measured through the shipped code
    # rather than argued: over the nine correctly-specified fits that pass the tracking gate, the rival that
    # previously never ran reads 0.022-0.389 -- every one still loses, with a factor of 2.5 to spare at the
    # worst -- and it costs ~4.0 s on a fit that costs 10.7 s on this body. (That column had been INVENTED in a
    # docstring once; this time it was taken and it is in ``GLOBAL_SCALE_COHERENCE_R2``.)
    #
    # **IT DOES NOT CLOSE THE DILUTION HOLE, and the reason is worth stating because it was expected to.** The
    # disclosed case -- a 1.35x torque-constant error SUPERPOSED on a real dissipation change, improvement_x
    # 1.689, coherence 0.484 diluted below the old filter -- is STILL applied with the filter gone. The rival
    # now runs on it and honestly LOSES, at 0.869. That is not the filter's fault and not a search failure: the
    # log really does contain a dissipation change no single gear scalar can produce, so one number really is
    # a worse explanation than the fit's forty-two, and the rival says so correctly. What is wrong with that
    # fit is that its numbers carry a factor, not that a scalar explains it -- and neither of these checks is
    # the instrument for that. It stays named in ``what_this_gate_does_not_catch``. Removing the filter is
    # still right (it removes a class of miss for free); it just does not remove THIS one.
    # The coherence is still computed, still reported, and still seeds the search -- it just no longer decides
    # who gets asked.
    if would_apply and gscale.get("implied_torque_scale_g"):
        gscale["rival"] = _global_scale_rival(
            model, aligned, kp=kp, kd=kd, q_start=q0r, ctrl_every=ctrl_every, delay_ticks=int(ticks),
            g_hint=float(gscale["implied_torque_scale_g"]), after_rms=after_rms,
            before_rms=float(traj.get("before_rms_rad") or 0.0))
        gscale["suspected"] = bool(gscale["rival"]["beats_the_fit"])
        # Named so a reader can see the alternative was actually simulated rather than argued.
        gscale["rival_model"] = ("our PRIOR parameters, unchanged, with every actuator's gear scaled by one "
                                 "number -- a ONE-parameter explanation of the same log competing with the "
                                 f"{len(identified_pairs)} this fit produced")
    out["global_scale"] = gscale

    # THE THIRD CHECK, for the family the one above provably does not fire on. Only run when the global scale
    # has NOT already claimed the fit: a torque-scale error drives armature NEGATIVE and implies a negative
    # inertia scale, so the two populations do not compete for the same verdict -- but if a fit ever did look
    # like both, the drivetrain reading is the one the customer can check first.
    linertia = link_inertia_signature(model, dofs, joints, q0r)
    linertia["suspected"] = False
    if would_apply and not gscale["suspected"] and linertia.get("testable"):
        linertia["rival"] = _link_inertia_rival(
            model, aligned, kp=kp, kd=kd, q_start=q0r, ctrl_every=ctrl_every, delay_ticks=int(ticks),
            s_hint=float(linertia["implied_inertia_scale_s"]), after_rms=after_rms,
            before_rms=float(traj.get("before_rms_rad") or 0.0))
        # BOTH conditions, and the second is the 2026-08-13 repair: ``beats_the_fit`` alone refused correct
        # armature-only calibrations whose improvement_x landed between ~1.5x and ~1.9x, because it divides by
        # the fit's own strength. See ``LINK_INERTIA_RIVAL_IMPROVEMENT_X``.
        linertia["suspected"] = bool(linertia["rival"]["suspected"])
        linertia["rival_model"] = ("our PRIOR parameters, unchanged, with every link's ROTATIONAL INERTIA "
                                   "scaled by one number and every MASS left alone -- a ONE-parameter "
                                   f"explanation of the same log competing with the {len(identified_pairs)} "
                                   f"this fit produced")
    out["link_inertia"] = linertia

    # THE CONSUMER ``improvement_x`` never had. Computed here so every caller -- apply_calibration,
    # l2_requirements, engineer_brief -- rules on the same field rather than each re-deriving it or, as before,
    # none of them reading it at all.
    out["application"] = application_gate(out.get("trajectory"), len(identified_pairs), global_scale=gscale,
                                          link_inertia=linertia, log_plausibility=plaus)
    if torque_channel.get("converted") or torque_channel.get("per_joint"):
        out["torque_channel"] = torque_channel
    out["wall_clock_s"] = round(time.perf_counter() - t_wall, 3)
    return out


def _delay_on_the_fitted_model(gene, model, aligned, deltas, dofs, kp, kd, log, plan, joints,
                               max_ticks: int) -> dict:
    """The actuation delay: open-loop off the log, with the closed-loop sweep run on the FITTED model beside it.

    The closed-loop sweep is the one this stage was built around, on the reasoning that Stage 1's delay-only
    search fails because it cannot see past a dynamics error and Stage 2 IS that correction. That reasoning is
    right and the remedy is still too weak to reach: MEASURED on the Menagerie Go2, the whole parameter gap is
    worth 0.00526 rad of trajectory RMS while the delay alone is worth 0.02444 rad, so the thing being removed
    is 4.6x smaller than the thing it was supposed to reveal. Closing it moved the argmin from 0 ms to 10 ms
    against a 20 ms injection and stopped there.

    So the sweep is no longer in charge. ``_delay_from_command_response`` measures the same delay without a
    plant model at all, is unaffected by how good the fit is, and recovers 0/20/40 ms exactly. The sweep is
    still run -- on the fitted model, which is the strongest version of it -- and reported, because a
    disagreement between the two is worth seeing.
    """
    from virturoid.services.sysid.bench_rig import start_pose
    from virturoid.services.sysid.gap_report import (
        _delay_from_command_response,
        _delay_search,
        _merge_latency,
    )

    dt = float(model.opt.timestep)
    log_hz = 1.0 / dt
    ctrl_hz = float(log.get("control_hz") or (plan or {}).get("controller", {}).get("control_hz") or 100.0)
    ctrl_every = max(1, int(round(log_hz / ctrl_hz)))
    q_cmd, q_hw = aligned["q_cmd"], aligned["q_meas"]
    q0 = start_pose(model, gene) if q_hw.shape[0] == 0 else q_hw[0].copy()
    applied = {name: {p: deltas[name][p] for p in (joints.get(name, {}).get("identified") or [])}
               for name in dofs}
    fitted = _model_with(model, applied, dofs)
    traj = _delay_search(fitted, q_cmd, q_hw, kp=kp, kd=kd, q_start=q0, ctrl_every=ctrl_every,
                         max_ticks=int(max_ticks), dt=dt)
    traj["searched_on"] = "the FITTED model (only identified deltas applied)"
    traj["why_this_model"] = ("a delay-only sweep on the uncorrected model is dominated by the parameter "
                              "error; closing that first is what makes the remaining mismatch about timing")
    cmd = _delay_from_command_response(fitted, aligned, dofs, plan, kp=kp, kd=kd, ctrl_every=ctrl_every,
                                       max_ticks=int(max_ticks), dt=dt)
    out = _merge_latency(cmd, traj, "the per-joint output phase lag in the gap report is the OBSERVABLE "
                                    "closed-loop lag and is a LOWER BOUND on this figure.")
    out["searched_on"] = traj["searched_on"]
    return out


def _trajectory_improvement(gene, model, aligned, deltas, dofs, kp, kd, log, plan, joints, *,
                            delay_ticks: int = 0, delay_identified: bool = False,
                            delay_source: str | None = None) -> dict:
    """Replay the log's own commands through the model BEFORE and AFTER the fit and report the position gap.

    This is the number the engineer recognises and the one that decides whether the fit was worth applying,
    and it is deliberately measured on an INDEPENDENT quantity: the fit minimised a torque residual, so a
    torque residual falling proves only that the optimiser worked. Only the identified deltas are applied --
    applying the refused ones here would flatter the number with parameters we are not allowed to ship.

    **Both replays run at the IDENTIFIED actuation delay**, and that is the fix to a defect that made this
    number useless on any robot with real latency. Both replays used to run at delay 0 against a log that had
    one, so both carried a timing error no fitted parameter could remove, and the ratio's whole RANGE collapsed
    toward 1 as the delay grew regardless of the fit. MEASURED on the Menagerie Go2, full 12-joint plan, by
    sweeping the three parameters this estimator can move over a 60-point grid at each delay: the largest
    delay-blind score reachable by ANY of them is 1.012x at 30 ms and 1.003x at 40 ms, against a 1.5x gate. At
    40 ms the fit recovered all three parameters to within 13.3% and scored 1.000x. Nothing about that number
    was about the fit.

    THE OLD WORDING OF THAT ARGUMENT IS RETRACTED (2026-08-12) and it is worth keeping the retraction here,
    because the reasoning was seductive and wrong. It read: "substitute the TRUE hardware model for the fitted
    one and you have the largest value the metric could return -- no fit can beat the parameters that generated
    the log." That holds only when the replay family CONTAINS the data-generating process. At delay 0 against a
    delayed log it does not: the timing error is unmodelled, so the true parameters are not the minimiser of the
    position residual, and a twin whose reflected inertia is too high partially MIMICS transport lag. Measured
    at 20 ms: the true perturbation scores 1.791x, ``armature +0.08`` (2.7x the injected 0.03) scores 2.277x,
    and this package's own fit -- whose armature lands 17-21% high -- reads 1.858x, i.e. ABOVE the figure that
    was being called a ceiling. Nothing here is a bound; the collapse of the RANGE is the finding, and it is now
    measured over the family rather than argued from one point.

    Running both sides at the same, correct delay cancels the timing term the way the ratio always assumed it
    did. The old figure is kept as ``improvement_x_at_zero_delay`` so the two can be read against each other,
    and when the delay could NOT be identified this falls back to zero delay and says so -- scoring at a delay
    we cannot measure would just move the error somewhere less visible.
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
    d = max(0, int(delay_ticks))

    def _pair(ticks):
        kw = dict(kp=kp, kd=kd, q_start=q0, ctrl_every=ctrl_every, delay_ticks=int(ticks))
        _, qb, _, _ = pd_replay(model, q_cmd, **kw)
        _, qa, _, _ = pd_replay(fitted, q_cmd, **kw)
        return qb, qa

    q_before, q_after = _pair(d)
    per_joint = {}
    for name, adr in dofs.items():
        b = float(np.sqrt(np.mean((q_before[:, adr] - q_hw[:, adr]) ** 2)))
        a = float(np.sqrt(np.mean((q_after[:, adr] - q_hw[:, adr]) ** 2)))
        per_joint[name] = {"before_rms_rad": round(b, 6), "after_rms_rad": round(a, 6),
                           "improvement_x": round(b / a, 3) if a > 1e-12 else None}
    before = float(np.sqrt(np.mean((q_before - q_hw) ** 2)))
    after = float(np.sqrt(np.mean((q_after - q_hw) ** 2)))

    at_zero = None
    if d:
        qb0, qa0 = _pair(0)
        b0 = float(np.sqrt(np.mean((qb0 - q_hw) ** 2)))
        a0 = float(np.sqrt(np.mean((qa0 - q_hw) ** 2)))
        at_zero = round(b0 / a0, 3) if a0 > 1e-12 else None

    return {
        "measured": "position RMS of our replay against the log, over all logged joints, with BOTH replays "
                    "run at the identified actuation delay",
        "independent_of_the_objective": "the fit minimised a TORQUE residual; this is a POSITION gap, so it "
                                        "is not the quantity that was optimised",
        "before_rms_rad": round(before, 6),
        "after_rms_rad": round(after, 6),
        "improvement_x": round(before / after, 3) if after > 1e-12 else None,
        "before_rms_deg": round(float(np.degrees(before)), 4),
        "after_rms_deg": round(float(np.degrees(after)), 4),
        "per_joint": per_joint,
        "scored_at_delay_ms": round(d * ctrl_every * dt * 1000.0, 3),
        "delay_identified": bool(delay_identified),
        "delay_source": delay_source,
        "improvement_x_at_zero_delay": at_zero,
        "why_scored_at_the_delay": (
            "both replays carry the same actuation delay, so the timing error cancels in the ratio and what is "
            "left is the fitted parameters. Scored at 0 ms this number's RANGE is set by the delay rather than "
            "by the fit -- measured on the Menagerie Go2 at 30 ms, no parameter set this estimator can express "
            "scores above 1.012x against a 1.5x gate, so a refusal there would be about your log's latency and "
            "not about your parameters."
            if d else
            ("the log shows no actuation delay, so 0 ms is the correct scoring point" if delay_identified else
             "the actuation delay could NOT be identified, so this falls back to 0 ms. Any timing error in the "
             "log is still inside this number and it may be an under-estimate of the fit's worth; the gate "
             "refusing on it is a refusal about the LOG, not about the parameters")),
        "caveat": "the delay is not written into the compiled model (MuJoCo has no transport delay and our "
                  "emitter sets no dyntype), so this is the delay HELD IN THE REPLAY, not a model that ships "
                  "with it. See calibration.model_represents_actuation_delay.",
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
