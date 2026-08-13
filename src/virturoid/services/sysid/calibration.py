"""Apply the fit -- to the robot, with provenance, reversibly -- and say the sentence an engineer can act on.

A fit that stays in a report changes nothing. This module is the half that makes the simulator actually move
toward the machine, and it is written around three refusals:

**Only identified parameters are applied.** ``fit_parameters`` returns a posterior for every parameter it
looked at; this module ships only the ones that cleared the identifiability gates. The rest are recorded in
``refused`` with the reason and, when the experiment was the problem, the experiment that would fix it. A
calibration that silently wrote an unidentified estimate into the model would be indistinguishable, three
weeks later, from a measurement.

**And only a fit that measurably improves TRACKING is applied at all.** Identifiability is a statement about
the experiment, not about the model: a parameter can be excited, separable, above its floor and still be
absorbing an error that lives somewhere the regressor cannot reach. Measured -- a synthetic robot built with
+30% link mass and inertia produced armature "identified" on 14/14 joints, 13/14 intervals excluding the true
unchanged value, and every one of those numbers went into the compiled model while ``improvement_x`` read
0.993, i.e. the fit had made tracking marginally *worse*. ``fit['application']`` now rules on that number
before anything is written; a fit that fails it attaches in full, applies to nothing, and takes an explicit
``allow_provisional=True`` to override. ``l2_requirements`` reads the same signal, so a rung called
"bench-identified" cannot be earned by a fit that left the simulator no closer to the machine.

**The change is visible and it is reversible.** The calibration lives in one place -- ``gene.metadata`` under
a single key -- carrying the prior it replaced, the delta, the interval, the experiment it came from and the
timestamp. ``calibration_report`` prints it; ``revert_calibration`` removes it and returns a gene that is
equal to the pre-calibration one. A customer who does not believe the number has to be able to take it out,
and has to be able to see it was there.

**The ladder is not widened to fit what we built.** ``certificate_v2.ACTUATOR_LEVELS`` reserves L2 for a
*bench-identified actuator*. ``l2_requirements`` computes, rather than asserts, whether that has happened --
and on this build it has not. Three of its six requirements fail on our own best fit, and two of those three
are ours rather than the customer's: the identified actuation delay has nowhere to go (no actuation-delay
model exists in the compiled model, checked here rather than assumed), and the datasheet torque-speed
envelope cannot be identified by an excitation deliberately bounded away from saturation. The third is the
customer's and is the one we cannot supply: we own no hardware, so no log we produce was measured on a
physical robot. Those are stated as findings.

The engineer-facing output is the deliverable, and it is a function, not a docstring: ``engineer_brief``
returns the paragraph, the pinned/unpinned tables behind it, and -- when the experiment was what fell short
-- ``follow_up_experiment``, which is a real narrowed excitation plan with a real duration, not advice.
"""

from __future__ import annotations

CALIBRATION_KEY = "calibration"
CALIBRATION_ARTIFACT = "virturoid_actuator_calibration"
CALIBRATION_VERSION = 1

#: A calibration may only raise the actuator-fidelity ladder when this fraction of the actuated joints carries
#: at least one identified parameter. Below it the fit describes part of the machine, and a level is a claim
#: about the whole machine. Same 0.8 the existing L1 clamp test uses, for the same reason.
L2_JOINT_COVERAGE = 0.8

#: The four-quadrant torque-speed model's own parameters (``actuator_model.clamp_torque``). Nothing fits them
#: today; the L2 requirement that asks for them reads this set rather than a constant ``False``, so the rung
#: unblocks itself the day something does.
TORQUE_SPEED_PARAMS = {"tau_max", "qd_tau_max", "qd_max"}


def log_provenance(log: dict | None) -> dict:
    """Classify where a log came from. ``hardware`` is claimed only when the log SAYS so, in a field.

    Deliberately not inferred from prose. ``synthetic_hardware_log`` writes a paragraph that begins
    "SYNTHETIC", and a customer's exporter will write whatever their exporter writes; an actuator-fidelity
    rung must not turn on a substring search over free text. Unstated is treated as not-hardware, which is the
    direction that cannot over-claim.
    """
    log = log or {}
    stated = str(log.get("measured_on") or "").strip().lower()
    note = str(log.get("provenance") or "")
    if stated in ("hardware", "robot", "bench", "physical"):
        return {"class": "hardware", "stated": stated, "note": note,
                "why": "the log declares measured_on='%s'" % stated}
    if stated in ("sim", "sim2sim", "simulation", "synthetic"):
        return {"class": "sim2sim", "stated": stated, "note": note,
                "why": "the log declares measured_on='%s' -- a simulated stand-in, not a physical robot"
                       % stated}
    return {"class": "unstated", "stated": stated or None, "note": note,
            "why": "the log does not declare measured_on. Treated as NOT hardware: an unstated provenance "
                   "cannot raise a rung that means 'bench-identified'"}


def model_represents_actuation_delay(gene) -> dict:
    """Does the model this gene compiles to represent an actuation delay AT ALL? Measured, not declared.

    The answer today is no, and it matters: Hwangbo et al. name control-signal delay as a dominant term in the
    sim-to-real residual, Stage 1 recovers it exactly (20 ms injected -> 20 ms recovered), and there is
    nowhere in the compiled model to put it. MuJoCo can represent a first-order actuator lag via ``dyntype``;
    a pure transport delay it cannot, and our emitter sets no ``dyntype`` at all. So this checks the compiled
    model rather than trusting a constant that would rot the day someone adds one.
    """
    try:
        import mujoco

        from virturoid.services.gene_compiler import compile_gene_to_mjcf

        model = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene, include_floor=False))
        dyn = [int(v) for v in model.actuator_dyntype]
        represented = any(v != int(mujoco.mjtDyn.mjDYN_NONE) for v in dyn)
    except Exception as exc:  # noqa: BLE001
        return {"represented": False, "checked": False,
                "why": f"could not compile the gene to check: {type(exc).__name__}: {exc}"[:160]}
    return {
        "represented": bool(represented), "checked": True,
        "n_actuators": len(dyn), "dyntypes": sorted(set(dyn)),
        "why": ("every actuator carries dyntype=none, so the compiled model applies the commanded torque in "
                "the same tick it is computed. An identified actuation delay can be REPORTED and cannot be "
                "APPLIED." if not represented else
                "at least one actuator carries a non-trivial dyntype (a first-order lag). Note that a "
                "first-order lag is not a transport delay and the two are not interchangeable."),
    }


# ---------------------------------------------------------------------------------------------------------
# The calibration record: what moved, by how much, from which experiment, and what was refused.
# ---------------------------------------------------------------------------------------------------------

def build_calibration(gene, fit: dict, *, log: dict | None = None, plan: dict | None = None,
                      note: str | None = None, allow_provisional: bool = False) -> dict:
    """The record that gets attached to the gene. JSON-serializable, self-describing, and reversible.

    ``allow_provisional`` is the opt-in for a fit that did not clear ``fit['application']`` -- the
    tracking-improvement gate. Without it, a provisional fit still attaches a full record (every value, every
    interval, the gate's verdict) but the values move to ``joints_withheld`` and ``joints`` is left empty, so
    the compiler reads nothing and the simulator is unchanged. Visible, and not applied.

    ``provisional`` on the record is a fact about the FIT (it failed the gate) and not about what we then chose
    to do with it, so overriding the gate does not clear the flag -- it sets ``applied_over_the_gate``. The
    earlier version conflated the two and a forced fit recorded ``provisional: False``, i.e. the override
    erased the evidence that there had been anything to override.
    """
    import time

    prov = log_provenance(log)
    delay = model_represents_actuation_delay(gene)
    gate = dict(fit.get("application") or {})
    withheld = bool(gate.get("provisional")) and not bool(allow_provisional)
    joints: dict = {}
    evidence: list = []
    for name, row in (fit.get("joints") or {}).items():
        moved = {}
        for p in row.get("identified", []):
            cell = row["parameters"][p]
            moved[p] = {
                "from": cell["prior"], "to": cell["value"], "delta": cell["delta"],
                "value_interval": cell["value_interval"], "interval_level": cell["interval_level"],
                "unit": cell["unit"], "t_stat": cell["t_stat"], "noise_floor": cell["noise_floor"],
                "also_absorbs": cell.get("also_absorbs"),
            }
            if not withheld:
                evidence.append(_evidence(name, p, cell, prov))
        if moved:
            joints[name] = moved

    joints_withheld: dict = {}
    if withheld:
        # The numbers stay on the robot and out of the model. A withheld fit that vanished would be
        # indistinguishable from a fit that was never run, and the engineer needs to see what it wanted to do.
        joints, joints_withheld = {}, joints

    refused = list(fit.get("refused") or [])
    latency = dict(fit.get("latency") or {})
    return {
        "artifact": CALIBRATION_ARTIFACT,
        "version": CALIBRATION_VERSION,
        "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "robot": getattr(gene, "id", ""),
        "provenance": prov,
        "experiment": {
            "duration_s": ((plan or {}).get("budget") or {}).get("duration_s"),
            "n_joints_excited": ((plan or {}).get("budget") or {}).get("n_excitable"),
            "control_hz": ((plan or {}).get("controller") or {}).get("control_hz"),
            "only_joints": ((plan or {}).get("scope") or {}).get("only_joints"),
            "setup": (plan or {}).get("setup") or fit.get("setup"),
        },
        "estimator": fit.get("estimator"),
        "parameters_fitted": list(fit.get("parameters_fitted") or []),
        "parameters_not_fitted": fit.get("parameters_not_fitted"),
        "assumed_correct": fit.get("assumed_correct"),
        "application_gate": gate or None,
        # A fact about the FIT: it failed the tracking gate. True whether or not the gate was then overridden.
        "provisional": bool(gate.get("provisional")),
        "withheld_from_model": bool(withheld),
        "applied_to_model": not withheld and bool(joints),
        # The override, named. A provisional fit that reached the model anyway is the one state a reader most
        # needs to be able to see, and it used to be the one state the record could not express.
        "applied_over_the_gate": bool(gate.get("provisional")) and not withheld and bool(joints),
        "joints": joints,
        "joints_withheld": joints_withheld,
        "n_joints_moved": len(joints),
        "n_parameters_moved": sum(len(v) for v in joints.values()),
        "n_joints_withheld": len(joints_withheld),
        "n_parameters_withheld": sum(len(v) for v in joints_withheld.values()),
        "refused": refused,
        "n_refused": len(refused),
        "latency": {
            **latency,
            "applied": False,
            "why_not_applied": delay["why"],
            "model_delay_support": delay,
        },
        "not_applied": [
            "the actuation delay (reported above; the compiled model has nowhere to put it)",
            "the datasheet torque-speed envelope (the excitation never reaches saturation by design)",
        ] + ([f"EVERY FITTED PARAMETER on {len(joints_withheld)} joint(s) -- {gate.get('verdict')}"]
             if withheld else []),
        "trajectory": fit.get("trajectory"),
        "evidence": evidence,
        "note": note,
        # The UNDO an engineer can actually perform. It named a Python function until `sysid.tools` gave the
        # package a door; a customer driving this over MCP could not call that, so the record's own undo
        # instruction was unusable by the only person who would ever need it.
        "how_to_undo": "calibration_status {robot_id, revert: true} removes this block and puts every joint "
                       "back on the compiler's prior. Nothing else on the gene is touched, so the reverted "
                       "robot compiles byte-identically to the pre-calibration one.",
        "how_it_is_applied": "gene_compiler._joint_dynamics reads this block and emits the fitted ABSOLUTE "
                             "value for each listed joint/parameter instead of its structural prior. The "
                             "prior it replaced is recorded as 'from', so a prior that later moves is "
                             "detectable (calibration_report flags it as stale) rather than silent.",
    }


def _evidence(joint: str, param: str, cell: dict, prov: dict):
    """One ``InputEvidence`` per fitted field, at source ``CALIBRATED``.

    ``InputSourceType.CALIBRATED`` -- "measured from logs / system identification" -- was written with no
    producer anywhere in the repo. This is it. The confidence is NOT the enum's flat 0.9 prior: it is derived
    from the width of this parameter's own interval relative to its value, so a parameter the experiment
    barely resolved does not present with the same confidence as one it nailed, and a sim2sim fit is capped
    below a hardware one because the sim2sim gate cannot validate the physics.
    """
    from virturoid.schemas.input_bundle import InputEvidence, InputSourceType

    lo, hi = cell["value_interval"]
    value = float(cell["value"]) or 1e-9
    rel = abs(float(hi) - float(lo)) / abs(value)
    conf = max(0.3, min(0.9, 0.9 - rel))
    if prov["class"] != "hardware":
        conf = min(conf, 0.6)
    return InputEvidence(
        field_path=f"joints.{joint}.{param}",
        value=cell["value"],
        source_type=InputSourceType.CALIBRATED,
        unit=cell["unit"],
        source_artifact=f"sysid excitation log ({prov['class']})",
        confidence=round(conf, 3),
        note=(f"fitted; {int(cell['interval_level'] * 100)}% interval [{lo}, {hi}] "
              f"(relative width {rel:.1%}). Provenance {prov['class']}: {prov['why']}"),
    )


def calibration_of(gene) -> dict | None:
    """The calibration attached to ``gene``, or ``None``. Validated -- ``metadata`` is a free-form dict and
    anything could be under this key, including something a customer pasted."""
    meta = getattr(gene, "metadata", None)
    if not isinstance(meta, dict):
        return None
    rec = meta.get(CALIBRATION_KEY)
    if not isinstance(rec, dict) or rec.get("artifact") != CALIBRATION_ARTIFACT:
        return None
    return rec


def apply_calibration(gene, fit: dict, *, log: dict | None = None, plan: dict | None = None,
                      note: str | None = None, allow_provisional: bool = False):
    """Attach the fit to ``gene`` and return the CALIBRATED gene. The original is never mutated.

    Applying a fit that identified nothing is not an error and does not raise: it attaches a record with an
    empty ``joints`` block and a populated ``refused`` block, so the sim is unchanged and the reason it is
    unchanged is on the robot rather than in a log line somebody has to go and find.

    A fit that identified plenty and did NOT measurably improve tracking is treated the same way, and that is
    the point of ``fit['application']``. It is the case a calibration tool is most dangerous in: the numbers
    look like measurements, the intervals look tight, and they are absorbing an error in something the
    regressor never contained -- link mass, link inertia, a centre of mass. Measured, that fit wrote 15 values
    into the compiled model at ``improvement_x = 0.993``, having made tracking marginally worse. It is now
    withheld unless ``allow_provisional=True`` says otherwise, deliberately, at the call site.

    THE CONVERSE IS NOT TRUE and this function is where it would bite. A fit that DOES improve tracking can
    still be absorbing an error the regressor never contained: measured 2026-08-12, a wrong gear ratio or
    torque constant scores 1.507x at a 20% error and link inertia at correct mass reaches 1.753x, both with
    intervals excluding the truth, and this function applies both without an override. See
    ``fit.MIN_TRACKING_IMPROVEMENT_X`` and ``application.what_this_gate_does_not_catch``: the gate is
    necessary and not sufficient, so "applied" means "it tracked better", never "the numbers are right".
    """
    from dataclasses import replace

    rec = build_calibration(gene, fit, log=log, plan=plan, note=note, allow_provisional=allow_provisional)
    meta = dict(getattr(gene, "metadata", {}) or {})
    meta[CALIBRATION_KEY] = rec
    return replace(gene, metadata=meta)


def revert_calibration(gene):
    """Return ``gene`` with any calibration removed. Idempotent, and a no-op on an uncalibrated gene."""
    from dataclasses import replace

    meta = dict(getattr(gene, "metadata", {}) or {})
    if CALIBRATION_KEY not in meta:
        return gene
    meta.pop(CALIBRATION_KEY, None)
    return replace(gene, metadata=meta)


def calibrated_joint_dynamics(gene, seg_name: str) -> dict:
    """``{param: value}`` overrides for one segment, or ``{}``. Called from the COMPILER, so it is defensive
    and cheap: a malformed value must not take the compile down, and a NaN must never reach the model."""
    import math

    rec = calibration_of(gene)
    if not rec:
        return {}
    row = (rec.get("joints") or {}).get(seg_name)
    if not isinstance(row, dict):
        return {}
    out = {}
    for param in ("damping", "armature", "frictionloss"):
        cell = row.get(param)
        if not isinstance(cell, dict):
            continue
        v = cell.get("to")
        # An actual number, not something that parses as one. ``float("0.5")`` succeeds, and a pasted string
        # silently setting a physical parameter is the same defect as a NaN reaching it -- it just looks
        # tidier in the record. Booleans are ints in Python and are excluded for the same reason.
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        v = float(v)
        if math.isfinite(v) and v >= 0.0:
            out[param] = v
    return out


def calibration_report(gene) -> dict:
    """What an engineer sees: which parameters moved, by how much, from which experiment, what was refused,
    whether the record has gone stale against the current priors, and how to undo it."""
    rec = calibration_of(gene)
    if not rec:
        return {"calibrated": False,
                "note": "no calibration is attached; every joint carries the compiler's structural prior"}

    from virturoid.services.gene_compiler import _joint_dynamics_prior

    # ``.get`` rather than ``[]``: this block is reported and never applied, so a malformed one must degrade to
    # a row with holes in it rather than take the whole report down. ``metadata`` is a free-form dict.
    withheld_rows = [
        {"joint": name, "parameter": param, "would_have_moved_from": cell.get("from"), "to": cell.get("to"),
         "delta": cell.get("delta"), "interval": cell.get("value_interval"), "unit": cell.get("unit"),
         "also_absorbs": cell.get("also_absorbs")}
        for name, moved in (rec.get("joints_withheld") or {}).items()
        if isinstance(moved, dict)
        for param, cell in moved.items() if isinstance(cell, dict)]

    rows, stale = [], []
    for seg in gene.actuated_joints():
        moved = (rec.get("joints") or {}).get(seg.name)
        if not moved:
            continue
        # The prior the fit replaced, recomputed from the CURRENT compiler. Comparing against the recorded
        # 'from' is what turns a silently-rotting record into a visible one: if the structural prior has moved
        # since the fit, the measurement is still the measurement, but the reader deserves to know the
        # baseline it was quoted against no longer exists.
        d, a, f = _joint_dynamics_prior(gene, seg)
        now = {"damping": d, "armature": a, "frictionloss": f}
        for param, cell in moved.items():
            row = {"joint": seg.name, "parameter": param, "from": cell["from"], "to": cell["to"],
                   "delta": cell["delta"], "interval": cell["value_interval"], "unit": cell["unit"],
                   "change_pct": (round(100.0 * cell["delta"] / cell["from"], 2) if cell["from"] else None)}
            if param in now and abs(float(now[param]) - float(cell["from"])) > 1e-6:
                row["stale_against_current_prior"] = {"fitted_against": cell["from"],
                                                      "prior_is_now": round(float(now[param]), 6)}
                stale.append(row["joint"] + "." + param)
            rows.append(row)
    return {
        "calibrated": True,
        "robot": rec.get("robot"),
        "fitted_at": rec.get("fitted_at"),
        "provenance": rec.get("provenance"),
        "experiment": rec.get("experiment"),
        "changes": rows,
        "n_changes": len(rows),
        # The tracking-improvement gate, and what it held back. A withheld fit is REPORTED in full and applied
        # to nothing: the engineer sees the numbers the fit wanted to write, next to the measured reason they
        # were not written.
        "provisional": bool(rec.get("provisional")),
        "withheld": bool(rec.get("withheld_from_model")),
        "applied_to_model": bool(rec.get("applied_to_model")),
        # The override, surfaced rather than buried in the gate's verdict. A record that says "applied" next to
        # a gate that says "does not improve tracking" has to say which one won, in a field.
        "applied_over_the_gate": bool(rec.get("applied_over_the_gate")),
        "applied_over_the_gate_note": (
            "these values FAILED the tracking-improvement gate and were written to the model anyway, by an "
            "explicit allow_provisional=True at the call site. Read them as a hypothesis, not a measurement."
            if rec.get("applied_over_the_gate") else None),
        "application_gate": rec.get("application_gate"),
        "withheld_changes": withheld_rows,
        "n_withheld": len(withheld_rows),
        "assumed_correct": rec.get("assumed_correct"),
        "parameters_not_fitted": rec.get("parameters_not_fitted"),
        "refused": rec.get("refused"),
        "n_refused": rec.get("n_refused"),
        "not_applied": rec.get("not_applied"),
        "latency": rec.get("latency"),
        "trajectory": rec.get("trajectory"),
        "stale_entries": stale,
        "stale_note": ("the compiler's structural prior has moved since this fit for the entries listed; the "
                       "fitted value is still applied (a measurement outranks a prior) but the recorded "
                       "'from' no longer matches what an uncalibrated build would produce"
                       if stale else "no entry is stale against the current priors"),
        "how_to_undo": rec.get("how_to_undo"),
    }


# ---------------------------------------------------------------------------------------------------------
# The actuator-fidelity ladder: L2, computed rather than declared.
# ---------------------------------------------------------------------------------------------------------

def l2_requirements(gene) -> dict:
    """Has this robot earned ``L2 bench-identified actuator``? Each requirement measured, with its evidence.

    Written to be able to say yes. It says no on this build, and the two blocking requirements are ours.

    Six requirements, not five: the sixth is that applying the fit measurably improves how the simulator
    tracks the log, and it is here because the fifth was measured to be satisfiable without it. A synthetic
    robot built with +30% link mass and inertia produced a fit that "identified" a parameter on 14/14 joints
    -- clearing the 80% coverage requirement outright -- while ``improvement_x`` sat at 0.993, meaning the
    fit made tracking marginally WORSE. Joint coverage counts APPLIED parameters and a provisional fit applies
    none, so that route closed on its own; the explicit requirement is here so the REASON is named rather than
    surfacing as a mysterious 0/14.

    * The delay is identified and cannot be applied -- there is no actuation-delay model in the compiled
      output (checked, not assumed). Hwangbo et al. attribute the dominant sim-to-real residual to actuator
      dynamics AND control-signal delay; a rung called "bench-identified actuator" that leaves half of that
      unmodelled is not the rung.
    * The datasheet torque-speed envelope is not identified, and cannot be by this experiment. The excitation
      is bounded at a fraction of datasheet peak torque and no-load speed precisely so it is safe to run on
      somebody's robot -- and the motor's knee lives on the other side of that bound. A safe experiment
      cannot identify a saturation limit. That is a real trade, not an oversight, and it is the honest reason
      L1's "datasheet torque-speed clamp" is still datasheet after a successful fit.
    """
    from virturoid.services.sysid.excitation import SPEED_FRACTION, TORQUE_FRACTION

    rec = calibration_of(gene)
    # Derived, not hardcoded False. If a later stage ever fits the torque-speed envelope it will appear in the
    # record's `parameters_fitted` and this requirement flips on its own -- a rung that can only be unblocked
    # by editing the rung is not a measurement.
    fitted_names = set((rec or {}).get("parameters_fitted") or [])
    envelope_fitted = bool(fitted_names & TORQUE_SPEED_PARAMS)
    actuated = gene.actuated_joints()
    n_act = len(actuated)
    moved = (rec or {}).get("joints") or {}
    n_moved = len([s for s in actuated if s.name in moved])
    prov = (rec or {}).get("provenance") or {"class": "none"}
    delay = (rec or {}).get("latency") or {}
    gate = (rec or {}).get("application_gate") or {}

    reqs = [
        {"requirement": "a parameter fit is attached to this robot",
         "met": bool(rec),
         "evidence": (f"{rec['artifact']} v{rec['version']} fitted {rec['fitted_at']}" if rec
                      else "no calibration is attached")},
        {"requirement": "the fit was made against a log measured on PHYSICAL HARDWARE",
         "met": prov.get("class") == "hardware",
         "evidence": prov.get("why", "no calibration is attached"),
         "whose_gap": "the customer's (we cannot supply a hardware log; we own no hardware)"},
        {"requirement": f"at least {int(L2_JOINT_COVERAGE * 100)}% of actuated joints carry an identified, "
                        f"applied parameter",
         "met": bool(n_act) and n_moved >= max(1, int(L2_JOINT_COVERAGE * n_act)),
         # APPLIED, and it now means it. This requirement reads the record's ``joints`` block, which a
         # provisional fit leaves empty -- so a fit that did not improve tracking can no longer satisfy the
         # coverage requirement with values that never reached the model. It did: 14/14 at improvement_x 0.993.
         "evidence": (f"{n_moved}/{n_act} actuated joints carry at least one fitted parameter APPLIED to the "
                      f"model") + (f"; {(rec or {}).get('n_joints_withheld', 0)} further joint(s) were fitted "
                                   f"and withheld by the tracking-improvement gate"
                                   if (rec or {}).get("n_joints_withheld") else "")},
        {"requirement": "applying the fit measurably improves how the simulator tracks the log it was fitted "
                        "to",
         "met": bool(gate.get("passed")),
         # The signal the tool computed and nobody read. A rung called "bench-identified actuator" cannot be
         # earned by a fit that leaves the simulator no closer to the machine -- that is the definition of the
         # thing not having been identified, however confident the intervals are.
         "evidence": (gate.get("verdict")
                      or ("this calibration record carries no tracking measurement, so the fit cannot be "
                          "shown to have helped" if rec else "no calibration is attached"))
                     + (f" [threshold {gate['threshold_x']:g}x on position RMS, measured on a quantity the fit "
                        f"did not optimise]" if gate.get("threshold_x") else ""),
         **({} if gate.get("passed") or not rec else
            {"whose_gap": "OURS or the MODEL'S -- a fit that does not close the trajectory gap is absorbing an "
                          "error in something this experiment does not fit (link mass, link inertia, centre of "
                          "mass, drivetrain elasticity), not measuring the actuator"})},
        {"requirement": "the identified actuation delay is APPLIED to the model, not merely reported",
         "met": bool(delay.get("applied")),
         "evidence": delay.get("why_not_applied") or "no calibration is attached",
         "whose_gap": "OURS -- the compiled model has no actuation-delay representation to write it into"},
        {"requirement": "the actuator's torque-speed envelope (tau_max / knee / no-load speed) is "
                        "bench-identified rather than taken from the datasheet",
         "met": envelope_fitted,
         "evidence": (f"fitted: {sorted(fitted_names & TORQUE_SPEED_PARAMS)}" if envelope_fitted else
                      f"not fitted (the fit covers {sorted(fitted_names) or 'nothing'}): the excitation is "
                      f"bounded at {int(TORQUE_FRACTION * 100)}% of datasheet peak torque and "
                      f"{int(SPEED_FRACTION * 100)}% of no-load speed, so it never enters the saturation "
                      f"regime the knee lives in"),
         "whose_gap": "OURS -- a safe excitation and an identifiable saturation limit are in direct conflict, "
                      "and we chose safe"},
    ]
    met = [r for r in reqs if r["met"]]
    blocked = [r for r in reqs if not r["met"]]
    return {
        "earned": not blocked,
        "n_met": len(met), "n_requirements": len(reqs),
        "requirements": reqs,
        "blocked_by": [r["requirement"] for r in blocked],
        "verdict": ("L2 earned: every requirement is met." if not blocked else
                    "L2 NOT earned. " + "; ".join(
                        f"{r['requirement']} -> {r['evidence']}" for r in blocked)),
        "bench_identified": {
            "parameters": sorted({p for row in moved.values() for p in row}),
            "n_joints": n_moved,
            # "bench-identified" is the certificate's word and it is not free. If these values reached the model
            # over a FAILED tracking gate, the note says so here rather than leaving the reader to notice a
            # blocked requirement further up: the certificate note is what gets quoted.
            "applied_over_the_gate": bool((rec or {}).get("applied_over_the_gate")),
            "note": (("these values were applied over a FAILED tracking-improvement gate "
                      f"({gate.get('improvement_x')}x against {gate.get('threshold_x')}x) by an explicit "
                      "opt-in. They are a hypothesis written into the model, not a bench identification"
                      if (rec or {}).get("applied_over_the_gate") else
                      "joint dissipation and reflected inertia are bench-identified on the joints listed; the "
                      "torque-speed envelope and the actuation delay are not, which is why the level does not "
                      "move") if moved else "nothing is bench-identified on this robot"),
        },
    }


# ---------------------------------------------------------------------------------------------------------
# The engineer-facing output.
# ---------------------------------------------------------------------------------------------------------

#: Which segment mix loads which parameter. Used to author the follow-up experiment, so "your excitation never
#: reversed this joint's velocity" is followed by a plan that does, rather than by the same plan again.
_EMPHASIS = {
    "velocity_sign_reversals": (("triangle", 0.60), ("chirp", 0.16), ("step_train", 0.18), ("hold", 0.06)),
    "speed_rms_radps": (("triangle", 0.60), ("chirp", 0.16), ("step_train", 0.18), ("hold", 0.06)),
    "accel_rms_radps2": (("triangle", 0.12), ("chirp", 0.54), ("step_train", 0.28), ("hold", 0.06)),
}


def follow_up_experiment(gene, fit: dict, *, per_joint_s: float | None = None) -> dict | None:
    """A REAL narrowed excitation plan for the parameters the first run could not pin -- or ``None``.

    ``None`` is the right answer more often than it looks. Only parameters that failed the EXCITATION gate can
    be fixed by running something else; a parameter that WAS excited and still failed is confounded with
    another or is under the measurement floor, and re-running the robot cannot help either. Offering a plan
    there would send an engineer back to their hardware for nothing, which is the most expensive mistake this
    tool can make.

    Per-joint time defaults to the ORIGINAL experiment's, so the follow-up is the same length per joint on
    strictly fewer joints and can never come back longer than the run that failed. What changes is the SEGMENT
    MIX, which is where the information the fit was missing actually comes from.
    """
    from virturoid.services.sysid.excitation import MAX_PER_JOINT_S, build_excitation

    if per_joint_s is None:
        per_joint_s = ((fit.get("experiment") or {}).get("per_joint_s")) or MAX_PER_JOINT_S

    joints, metrics = set(), {}
    for row in (fit.get("joints") or {}).values():
        for p in row.get("not_identified", []):
            cell = row["parameters"][p]
            if not cell.get("suggested_experiment"):
                continue                       # excited already; a different experiment will not fix it
            joints.add(row["joint"])
            metrics[cell.get("excitation_metric") or p] = cell["suggested_experiment"]
    if not joints:
        return None

    # The mix is chosen by what was MISSING. When several things were missing, the friction/damping mix wins:
    # its triangle segment is the only one that makes Coulomb friction separable from the gravity offset at
    # all, and an unidentifiable parameter is worse than an imprecise one.
    keys = [k for k in _EMPHASIS if k in metrics]
    fractions = _EMPHASIS[keys[0]] if keys else None
    if "velocity_sign_reversals" in metrics or "speed_rms_radps" in metrics:
        fractions = _EMPHASIS["velocity_sign_reversals"]

    ordered = sorted(joints)
    plan = build_excitation(gene, budget_s=float(per_joint_s) * len(ordered), only_joints=ordered,
                            segment_fractions=fractions)
    return {
        "why": "these joints have a parameter the first experiment did not load; the segment mix below is "
               "weighted toward the excitation that was missing",
        "joints": ordered,
        "missing_excitation": sorted(metrics),
        "fixes": sorted(set(metrics.values())),
        "duration_s": plan["budget"]["duration_s"],
        "sentence": (f"{plan['budget']['duration_s']:.0f} more seconds on {len(ordered)} joint(s) "
                     f"({', '.join(ordered[:4])}{'...' if len(ordered) > 4 else ''}) would pin them"),
        "plan": plan,
    }


def engineer_brief(gene, fit: dict, *, gap: dict | None = None, log: dict | None = None) -> dict:
    """The paragraph an engineer reads, and the tables behind every number in it.

    This is the deliverable, not a summary of one. Every clause is a measured field: the trajectory gap before
    and after, the identified delay and whether it could be applied, how many (joint, parameter) pairs are
    pinned, which are not and why, and the follow-up experiment when one would help.

    Three of those clauses are here because they were missing and it mattered. When the fit fails the
    tracking-improvement gate the HEADLINE changes rather than gaining a footnote -- "0.993x closer" reads as
    closer to anyone skimming, and this sentence is what gets quoted to a team. A pinned parameter is
    reported next to what else it could be: the link masses and inertias behind it were held fixed, so an
    error in them is not in the interval, it is in the number. And "applied" is read off the GENE rather than
    off the fit, because ``apply_calibration(..., allow_provisional=True)`` can overrule the gate -- a brief
    that reported the gate's intention said "this fit was NOT applied" about a model already carrying every
    one of its numbers.
    """
    if not fit.get("ok"):
        return {"ok": False, "sentence": f"No fit: {fit.get('error') or fit.get('why') or 'unknown'}",
                "detail": fit}

    joints = fit.get("joints") or {}
    pinned, unpinned = [], []
    for name, row in joints.items():
        for p in row.get("identified", []):
            cell = row["parameters"][p]
            pinned.append({"joint": name, "parameter": p, "value": cell["value"],
                           "interval": cell["value_interval"], "unit": cell["unit"],
                           "moved_from": cell["prior"], "also_absorbs": cell.get("also_absorbs")})
        for p in row.get("not_identified", []):
            cell = row["parameters"][p]
            unpinned.append({"joint": name, "parameter": p, "unit": cell["unit"],
                             "reasons": cell["reasons_not_identified"],
                             "fix": cell["suggested_experiment"]})

    traj = fit.get("trajectory") or {}
    gate = dict(fit.get("application") or {})
    lat = dict(fit.get("latency") or {})
    if not lat and gap:
        lat = dict(gap.get("latency") or {})
    delay_support = model_represents_actuation_delay(gene)
    prov = log_provenance(log if log is not None else {"provenance": fit.get("log_provenance")})
    nxt = follow_up_experiment(gene, fit)
    n_total = len(pinned) + len(unpinned)
    # Whether the fit FAILED the gate is a fact about the fit; whether it reached the model is a fact about the
    # GENE, because ``apply_calibration(..., allow_provisional=True)`` overrides the gate at the call site.
    # Reading only the first told an engineer "this fit was NOT applied" about a model it had already been
    # written into -- the same class of untrue headline the gate exists to stop.
    failed_gate = bool(gate.get("provisional"))
    over_the_gate = bool((calibration_of(gene) or {}).get("applied_over_the_gate"))
    withheld = failed_gate and not over_the_gate

    lines = []
    if failed_gate and over_the_gate:
        lines.append(
            f"These values FAILED the tracking gate and were applied ANYWAY, by an explicit "
            f"allow_provisional=True at the call site: applying them moves your simulator "
            f"{gate.get('improvement_x')}x toward this log against a required {gate.get('threshold_x')}x. "
            f"Your model now carries them. Read them as a hypothesis, not a measurement, and "
            f"calibration_status {{robot_id, revert: true}} takes them back out.")
    if withheld:
        # The headline sentence has to CHANGE, not gain a footnote. "0.993x closer" read as an improvement to
        # anyone skimming, and it is the sentence an engineer quotes to their team.
        #
        # ...and the SECOND refusal needs its own sentence for exactly the same reason. A fit stopped by the
        # global-scale check CLEARED the tracking gate -- 1.536x against a required 1.5x -- so the branch below
        # would have printed "does not move your simulator toward this log (1.536x, and we require 1.5x)",
        # which is both false and self-contradicting in one clause. A refusal that misstates its own cause
        # sends the engineer to re-run an experiment instead of to their BOM.
        gs = gate.get("global_scale") or {}
        # THE LOG-PLAUSIBILITY REFUSAL NEEDS ITS OWN SENTENCE, and for a sharper version of the reason the two
        # rival refusals do. A fit stopped by that guard CLEARED the tracking gate (measured 1.575x and 2.057x
        # on the composed spider), so the improvement branch below would have printed "does not move your
        # simulator toward this log (1.575x, and we require 1.5x)" -- false, self-contradicting, and it sends
        # the engineer to re-run the very experiment whose OUTPUT is the problem. This one also has to name
        # nothing: not the drivetrain, not the CAD, not the parameters.
        if gate.get("refused_by") == "implausible_log":
            lp = gate.get("log_plausibility") or {}
            lines.append(
                f"This fit was NOT applied, and the problem is THIS LOG rather than your robot or the fit. "
                + (f"It contains non-finite values, so the plant that produced it left the reals. "
                   if not lp.get("finite", True) else
                   f"Joint '{lp.get('worst_joint')}' travels {lp.get('largest_recorded_excursion_rad')} rad "
                   f"against a commanded envelope of {lp.get('commanded_envelope_rad')} rad - "
                   f"{lp.get('excursion_ratio')}x, where we require at most "
                   f"{lp.get('excursion_ratio_threshold_x')}x. "
                   + (f"Excluded from that number: {lp.get('unbounded_joint_count')} continuous-rotation "
                      f"joint(s) whose position integrates without bound and on which this ratio is not a "
                      f"statistic - {', '.join(lp.get('unbounded_joints') or [])}. "
                      if lp.get("unbounded_joint_count") else "")
                   + f"WHAT WE COMPARED YOU AGAINST, and its scope: across 18 composed bodies with no "
                     f"continuous-rotation joints, EIGHT injection families (nothing wrong at all; our default "
                     f"perturbation; armature-only +0.009 and +0.050; frictionloss-only +0.030; damping-only "
                     f"+0.6; link inertia x40 and x100), at every actuation delay from 0 to "
                     f"{lp.get('valid_delay_window_ms')} ms in 10 ms steps, the ratio runs 0.584x to 6.721x. "
                     f"That is a range over THOSE families, not over all logs: with a robot's damping and "
                     f"joint friction removed entirely, the same bodies reach 18.8x and 22.2x at 200 ms on "
                     f"numerically stable logs, so if your machine genuinely has negligible joint damping this "
                     f"guard can refuse you wrongly. Read log_plausibility.sampled_range_on_sane_logs before "
                     f"acting on this refusal. ")
                + f"That is the signature of a simulation or a rig that diverged, not of a robot tracking a "
                  f"command, and no parameter verdict taken from it would mean anything - so we are not "
                  f"issuing one, about any component. Re-take the log: check the excitation amplitudes against "
                  f"the joints' travel, that the base is really held, and that the gains that ran are the ones "
                  f"in the plan.")
        elif gate.get("refused_by") == "global_scale":
            riv = gs.get("rival") or {}
            g = riv.get("torque_scale_g") or gs.get("implied_torque_scale_g")
            lines.append(
                f"This fit was NOT applied, and it is not because it failed to help: it moves your simulator "
                f"{gate.get('improvement_x')}x closer to this log, past the {gate.get('threshold_x')}x we "
                f"require. It was refused because ONE number explains it better than the fit does. Every "
                f"parameter moved by nearly the same fraction of its own prior on every joint, which is what "
                f"happens when your joints receive about {g}x the torque this log records rather than when "
                f"friction, damping or reflected inertia has really changed - and replaying our ORIGINAL model "
                f"with nothing altered but every actuator's gear scaled by {g} tracks your log "
                f"{riv.get('explains_x')}x better than this fit. Check the GEAR RATIO and TORQUE CONSTANT your "
                f"driver used to compute the torque in this log, and any gearbox efficiency you have not "
                f"modelled; if you can log per-joint motor current, re-fit with the datasheet torque constant.")
        elif gate.get("refused_by") == "link_inertia":
            # The THIRD refusal needs its own sentence for the same reason the second did, and it is a
            # DIFFERENT sentence because it is a different remedy: this one sends the engineer to their CAD,
            # not to their BOM. Falling through to the torque-scale branch would have told them to re-check a
            # drivetrain this fit has no evidence against.
            li = gate.get("link_inertia") or {}
            riv = li.get("rival") or {}
            ss = riv.get("inertia_scale_s") or li.get("implied_inertia_scale_s")
            lines.append(
                f"This fit was NOT applied, and it is not because it failed to help: it moves your simulator "
                f"{gate.get('improvement_x')}x closer to this log, past the {gate.get('threshold_x')}x we "
                f"require. It was refused because ONE number explains it better than the fit does. Every "
                f"joint's reflected inertia moved by nearly the same fraction of its own link-side rotational "
                f"inertia, which is what happens when the INERTIA TENSORS in your model are wrong - reflected "
                f"and link inertia enter each joint's equation through the same acceleration term, so this "
                f"experiment cannot separate them - and replaying our ORIGINAL model with nothing altered but "
                f"every link's rotational inertia scaled by {ss}, masses untouched, gets that ORIGINAL model "
                f"{riv.get('rival_improvement_x')}x closer to your log (we require "
                f"{riv.get('improvement_threshold_x')}x) and tracks "
                f"{riv.get('explains_x')}x better than this fit. Check the inertia tensors of your moving "
                f"links against your CAD, and the density or fill fraction they came from. Your MASSES are "
                f"not implicated: a mass error would have left a gravity offset this fit could not remove.")
        elif traj.get("after_rms_deg") is not None and gate.get("improvement_x") is not None:
            lines.append(
                f"This fit was NOT applied: applying it does not move your simulator toward this log. "
                f"Position RMS goes {traj['before_rms_deg']:.3f} deg -> {traj['after_rms_deg']:.3f} deg "
                f"({gate.get('improvement_x')}x, and we require {gate.get('threshold_x')}x).")
        else:
            lines.append("This fit was NOT applied: its effect on your simulator's tracking was never "
                         "measured, so it cannot be shown to help.")
        if gate.get("refused_by") == "implausible_log":
            lines.append(
                "The parameters below are reported and written to nothing. Do not read them as bounds on "
                "anything either: they are a least-squares fit to a torque residual computed from motion that "
                "the plant did not physically produce, so their magnitudes are properties of the divergence "
                "and not of your robot.")
        elif gate.get("refused_by") == "link_inertia":
            lines.append(
                "The parameters below are reported and written to nothing. The armature numbers in particular "
                "are the joint-space inertia your links are missing, expressed on the wrong side of the "
                "gearbox - read them as a bound on the tensor error, not as a gearbox finding.")
        elif gate.get("refused_by") != "global_scale":
            lines.append(
                "The parameters below are reported and written to nothing. The usual cause is an error this "
                "experiment does not fit -- link mass, link inertia, centre of mass, drivetrain elasticity -- "
                "being absorbed by the three parameters it does, armature first, because reflected inertia and "
                "link inertia enter the joint equation through the same acceleration term.")
        else:
            lines.append(
                # MULTIPLY, not divide. A torque scale g makes the fit report prior/g, so the truth comes back
                # by multiplying BY g. This line said "divide" until it was checked arithmetically against the
                # x1.2 case: reported/prior = 0.887/0.782/0.841, x g -> 1.060/0.935/1.005 against a truth of
                # 1.000, while / g -> 0.743/0.655/0.704. Dividing left the engineer ~30% FURTHER from the truth
                # than the number they had already been given.
                "The parameters below are reported and written to nothing. Multiply each of them by the scale "
                "above if you want to see what they would be with the drivetrain right - but confirm the "
                "drivetrain first, because this experiment cannot.")
    elif not failed_gate and traj.get("after_rms_deg") is not None:
        # ``not failed_gate`` guards the OVERRIDE path too: a forced fit is applied, so it is not withheld, and
        # without this it would reach the "0.993x closer" sentence -- the exact headline the gate was built to
        # stop -- one branch further down.
        lines.append(
            f"Your simulator now tracks this log to {traj['after_rms_deg']:.3f} deg RMS, "
            f"{traj.get('improvement_x')}x closer than the structural prior it started from "
            f"({traj['before_rms_deg']:.3f} deg).")
    if lat.get("identified"):
        lines.append(
            f"It lags your hardware by {lat.get('delay_ms')} ms, and we could NOT apply that: "
            f"{delay_support['why']}")
    elif lat:
        lines.append(f"The actuation delay could not be identified: {lat.get('not_identified_because')}")
    lines.append(
        f"{len(pinned)} of {n_total} (joint, parameter) pairs are pinned"
        + (f"; {len(unpinned)} are not." if unpinned else "; none was refused."))
    if pinned and not withheld:
        # GAP 2, in the engineer-facing output rather than in a docstring. Somebody reading "armature 0.017,
        # 90% CI [0.013, 0.021]" is entitled to know which quantities were held fixed to produce it.
        lines.append(
            "Read those against what we did NOT fit: your link masses, link inertias and centres of mass are "
            "assumed correct and held at the values your geometry compiled to, so an error in them is not "
            "reported - it lands in armature.")
    if unpinned:
        worst = unpinned[0]
        why = (worst["reasons"][0].split(".")[0] if worst["reasons"] else "no reason recorded")
        lines.append(f"{worst['joint']} {worst['parameter']} is not - {why}.")
    if nxt:
        lines.append("Here is the " + nxt["sentence"].lower() + ".")
    lines.append(
        f"Provenance: {prov['class']} - {prov['why']}."
        + (" This fit has not been validated against any physical robot." if prov["class"] != "hardware"
           else ""))

    # Terminate every clause before joining. Several of the explanations quoted above are written to be
    # embedded and do not end in a full stop, so the paragraph ran two findings together -- measured on a Go2:
    # "...Close the parameter gap first, then re-run 28 of 36 (joint, parameter) pairs are pinned". This is the
    # sentence an engineer quotes to their team; it has to read as sentences.
    lines = [ln if (not ln or ln.rstrip()[-1] in ".!?") else ln.rstrip() + "." for ln in lines]
    return {
        "ok": True,
        "robot": fit.get("robot"),
        "sentence": " ".join(lines),
        "lines": lines,
        "gap": {"before_rms_deg": traj.get("before_rms_deg"), "after_rms_deg": traj.get("after_rms_deg"),
                "improvement_x": traj.get("improvement_x"),
                "measured_on": traj.get("measured"), "caveat": traj.get("caveat")},
        "application": gate or None,
        "applied": not withheld,
        "applied_over_the_gate": bool(failed_gate and over_the_gate),
        "latency": {**lat, "applied": False, "why_not_applied": delay_support["why"]},
        "pinned": pinned,
        "not_pinned": unpinned,
        # What the numbers above could ALSO be. Named at the top level because a reader who takes one thing
        # from this brief should not take "armature = 0.017" without it.
        "assumed_correct": fit.get("assumed_correct"),
        "parameters_not_fitted": fit.get("parameters_not_fitted"),
        "next_experiment": nxt,
        "provenance": prov,
        "actuator_ladder": l2_requirements(gene),
    }
