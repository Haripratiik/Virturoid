"""System identification, Stage 1: the GAP NUMBER.

The reality gap ranks #1 of 10 in the only relevant developer survey (Afzal et al., N=82), it recurs on every
hardware revision, and no product serves it. Hwangbo et al. (Science Robotics, ANYmal) pin the dominant residual
specifically to ACTUATOR DYNAMICS AND CONTROL-SIGNAL DELAY, not to contact or geometry -- which is what this
package measures. Before it, `docs/comprehensive_roadmap.md:148` named `sim2real/diff_sysid.py` and the file was
never written; there was no parameter fitting, no log replay, no trajectory matching anywhere in the repo.

Stage 1 is deliberately NOT a fit. It answers three questions and stops:

  1. ``excitation``       -- what should the engineer run on the hardware? A short, safe, information-rich
                             command sequence bounded by the gene's joint limits AND the BOM'd motor's datasheet.
  2. ``gap_report``       -- given the log that came back, how far is our sim from it? Per joint, in rad / ms /
                             N.m, naming which joints and which parameters are implicated. Never a scalar score.
  3. ``identifiability``  -- which of those parameters did the experiment actually pin, and which did it not?
                             A calibration tool that reports a confident number for an unidentifiable parameter
                             is worse than one that reports nothing.

``synthetic_hardware`` is the sim2sim gate. WE OWN NO HARDWARE: every number this package has been validated
against comes from perturbing a known model and recovering the perturbation. That validates the PIPELINE and
the ESTIMATOR. It does not validate the PHYSICS -- MuJoCo's own modelling error cancels exactly when both the
"robot" and the "sim" are MuJoCo, and that is the one error a real log would expose. See
``synthetic_hardware.WHAT_SIM2SIM_DOES_NOT_PROVE``.
"""

from __future__ import annotations

from virturoid.services.sysid.excitation import build_excitation, excitation_command_series
from virturoid.services.sysid.gap_report import measure_gap
from virturoid.services.sysid.identifiability import identifiability_report
from virturoid.services.sysid.synthetic_hardware import (
    WHAT_SIM2SIM_DOES_NOT_PROVE,
    recovery_table,
    synthetic_hardware_log,
)

__all__ = [
    "WHAT_SIM2SIM_DOES_NOT_PROVE",
    "build_excitation",
    "excitation_command_series",
    "identifiability_report",
    "measure_gap",
    "recovery_table",
    "synthetic_hardware_log",
]
