"""System identification, Stage 1: the GAP NUMBER.

The reality gap ranks #1 of 10 in the only relevant developer survey (Afzal et al., N=82), it recurs on every
hardware revision, and no product serves it. Hwangbo et al. (Science Robotics, ANYmal) pin the dominant residual
specifically to ACTUATOR DYNAMICS AND CONTROL-SIGNAL DELAY, not to contact or geometry -- which is what this
package measures. Before it, `docs/comprehensive_roadmap.md:148` named `sim2real/diff_sysid.py` and the file was
never written; there was no parameter fitting, no log replay, no trajectory matching anywhere in the repo.

Stage 1 MEASURES. It answers three questions and stops:

  1. ``excitation``       -- what should the engineer run on the hardware? A short, safe, information-rich
                             command sequence bounded by the gene's joint limits AND the BOM'd motor's datasheet.
  2. ``gap_report``       -- given the log that came back, how far is our sim from it? Per joint, in rad / ms /
                             N.m, naming which joints and which parameters are implicated. Never a scalar score.
  3. ``identifiability``  -- which of those parameters did the experiment actually pin, and which did it not?
                             A calibration tool that reports a confident number for an unidentifiable parameter
                             is worse than one that reports nothing.

Stage 2 FITS, and APPLIES:

  4. ``fit``              -- an iterated fit that returns a POSTERIOR per parameter: a value and an interval,
                             never a bare point estimate. ``coverage_table`` measures whether those intervals
                             actually contain the truth at the rate they claim, and ``application_gate``
                             refuses to apply a fit that does not make the simulator TRACK the log better --
                             because a confident interval on a parameter that is absorbing an error the
                             regressor cannot express is the most convincing wrong answer this tool can give.
  5. ``calibration``      -- attach the fit to the gene so the compiled model genuinely changes, with full
                             provenance and a one-call undo; report the sentence an engineer can act on; and
                             rule on the certificate's ``L2 bench-identified actuator`` rung by measuring the
                             six things it requires instead of asserting them.

On that dominant residual specifically: the CONTROL-SIGNAL DELAY half of it is measured OPEN-LOOP, off the log,
by ``gap_report._delay_from_command_response`` -- the shift that aligns the declared PD law evaluated on the
logged state with the torque the log says was applied. It requires ``tau_meas`` (motor current is enough) and
it does not involve the plant, which is the point: the closed-loop delay sweep this package shipped first is
biased toward zero whenever the plant is wrong, and on the Menagerie Go2 it read a 40 ms delay as 10 ms.
Actuation delay STILL cannot be applied to the compiled model -- MuJoCo has no transport delay and our emitter
sets no ``dyntype`` -- so it is reported, held in the replay that scores the tracking gate, and blocked from
the L2 rung by ``calibration.model_represents_actuation_delay``.

``synthetic_hardware`` is the sim2sim gate. WE OWN NO HARDWARE: every number this package has been validated
against comes from perturbing a known model and recovering the perturbation. That validates the PIPELINE and
the ESTIMATOR. It does not validate the PHYSICS -- MuJoCo's own modelling error cancels exactly when both the
"robot" and the "sim" are MuJoCo, and that is the one error a real log would expose. See
``synthetic_hardware.WHAT_SIM2SIM_DOES_NOT_PROVE``.
"""

from __future__ import annotations

from virturoid.services.sysid.calibration import (
    apply_calibration,
    calibration_of,
    calibration_report,
    engineer_brief,
    follow_up_experiment,
    l2_requirements,
    log_provenance,
    model_represents_actuation_delay,
    revert_calibration,
)
from virturoid.services.sysid.excitation import build_excitation, excitation_command_series
from virturoid.services.sysid.fit import (
    MIN_TRACKING_IMPROVEMENT_X,
    application_gate,
    coverage_table,
    fit_parameters,
)
from virturoid.services.sysid.gap_report import measure_gap
from virturoid.services.sysid.identifiability import identifiability_report
from virturoid.services.sysid.synthetic_hardware import (
    WHAT_SIM2SIM_DOES_NOT_PROVE,
    recovery_table,
    synthetic_hardware_log,
)

__all__ = [
    "MIN_TRACKING_IMPROVEMENT_X",
    "WHAT_SIM2SIM_DOES_NOT_PROVE",
    "application_gate",
    "apply_calibration",
    "build_excitation",
    "calibration_of",
    "calibration_report",
    "coverage_table",
    "engineer_brief",
    "excitation_command_series",
    "fit_parameters",
    "follow_up_experiment",
    "identifiability_report",
    "l2_requirements",
    "log_provenance",
    "measure_gap",
    "model_represents_actuation_delay",
    "recovery_table",
    "revert_calibration",
    "synthetic_hardware_log",
]
