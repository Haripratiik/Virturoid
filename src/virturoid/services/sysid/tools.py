"""The DOOR. Everything in this package was reachable only by ``import virturoid.services.sysid``.

Four engineers used the product and searched ``list_tools`` (67 tools), ``capabilities``, the README and the
Studio for the sim-to-real gap -- our headline differentiator -- and found nothing. There was no MCP tool, no
CLI verb and no HTTP route into any of the 3,156 lines behind this file. The one README mention sat under
``## Roadmap``, telling a prospect the feature did not exist. Their verdict on the capability once they got
inside by importing internals was "the gap report is the best artifact in the product". Good and unreachable is
the worst combination there is, and this module is the fix.

It is a DOOR, not a re-implementation: every handler is a thin wrapper over the same functions, and it exists
to solve the three things that made the surface unusable to someone who has not read the source.

**Names an engineer already has.** ``delay_ticks`` is a number of control ticks and nobody thinks in control
ticks; the door takes ``delay_ms``. ``budget_s``/``link_scale``/``hold_only``/``allow_provisional`` existed only
in Python signatures; here they are schema properties with units and defaults, which is what a strict MCP client
validates a ``tools/call`` against. And ``synthetic_hardware_log`` returns an UNLABELLED 2-tuple -- an engineer
guessed the order wrong and burned a 143 s run finding out -- so this door returns named paths and never a tuple.

**A journey, not an entry point.** Someone who owns a robot and has telemetry needs a different path from
someone who owns nothing yet, and the two must not be confusable:

    have hardware   ->  plan_bench_experiment  ->  [run it on YOUR robot]  ->  measure_sim_to_real_gap
                                                                           ->  fit_actuators {apply: true}
    have no robot   ->  simulate_bench_log     ->  (the same two tools, on a SIMULATED stand-in)

**And the synthetic path must be impossible to mistake for a measurement.** ``simulate_bench_log`` perturbs a
second MuJoCo model and hands back a log in the exact schema a bench log arrives in -- that is the point of the
harness, and it is also the hazard. So provenance is carried, not inferred: the log declares ``measured_on``, a
``provenance`` block rides on every downstream result, and the first line of every headline says which of the
two it is. A synthetic result cannot raise the actuator-fidelity rung, and it says so in the same breath.
"""

from __future__ import annotations

# The ordered journey, published in every plan so an agent can follow it without reading this file.
JOURNEY = (
    {"step": 1, "tool": "plan_bench_experiment",
     "what": "author the short, safe, information-rich experiment for THIS robot -- bounded by its own joint "
             "limits and the datasheet torque/speed of the motors its BOM sized"},
    {"step": 2, "tool": "simulate_bench_log",
     "what": "NO HARDWARE ONLY: run that experiment on a deliberately perturbed copy of the model, to see what "
             "the report looks like. The result is a SIMULATION, never a measurement of your machine"},
    {"step": 3, "tool": "measure_sim_to_real_gap",
     "what": "hand back the log: per joint, in rad / ms / N.m, how far our simulator is from it, which "
             "parameters are implicated, and which the experiment could not pin"},
    {"step": 4, "tool": "fit_actuators",
     "what": "fit each joint's damping / reflected inertia / dry friction to the log with an interval per "
             "parameter, and (apply: true) write the identified ones into the robot you hold"},
    {"step": 5, "tool": "calibration_status",
     "what": "what is attached, what moved, what was refused, whether the bench-identified rung is earned, and "
             "revert: true to take it all back out"},
)

#: Field names accepted for each log column. The left-hand name is what the estimator reads; everything on the
#: right is a name a real exporter writes. The experiment plan advertises the ``*_rad`` / ``*_nm`` spellings, so
#: an engineer who follows our own log schema to the letter MUST be able to hand that file straight back.
LOG_ALIASES: dict[str, tuple[str, ...]] = {
    "t": ("t", "t_s", "time", "time_s", "timestamp_s", "stamp_s"),
    "q_cmd": ("q_cmd", "q_cmd_rad", "cmd", "q_des", "q_desired", "q_target", "position_cmd"),
    "q_meas": ("q_meas", "q_meas_rad", "q", "q_actual", "position", "position_meas"),
    "qd_meas": ("qd_meas", "qd_meas_radps", "qd", "dq", "velocity", "velocity_meas"),
    "tau_meas": ("tau_meas", "tau_meas_nm", "tau", "torque", "torque_meas", "effort"),
    "joints": ("joints", "joint_names", "names", "joint_order"),
    "control_hz": ("control_hz", "ctrl_hz", "control_rate_hz"),
    "measured_on": ("measured_on", "provenance_class", "source"),
}

#: What the ``measured_on`` field is allowed to say, and what each one means for the rung. Kept here rather than
#: in prose because it is the field that decides whether a number may be called a measurement of a real robot.
MEASURED_ON_VALUES = {
    "hardware": "this log came off a PHYSICAL robot. Only this value may raise the bench-identified rung.",
    "sim2sim": "this log came off a simulated stand-in. Reported in full; raises no rung.",
    "unstated": "the log does not say. Treated as NOT hardware -- the direction that cannot over-claim.",
}

_MAX_LOG_BYTES = 256 * 1024 * 1024      # a bench log is a few MB; this is the ceiling before we refuse to read
_CSV_SEPARATORS = (".", "/", "::", "__", ":")


# ---------------------------------------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------------------------------------

def _held(robot_id):
    """``(gene, None)`` for a held robot, or ``(None, refusal)``. The handle every tool here is addressed by."""
    from virturoid.services import session_state as S
    rid = str(robot_id or "").strip()
    if not rid:
        return None, {"ok": False, "error": "provide robot_id: the robot to measure",
                      "how": "hold one first -- ingest_project for a robot you already own, create_robot or "
                             "submit_design for one authored here"}
    gene = S.get_robot(rid)
    if gene is None:
        return None, {"ok": False, "error": f"no robot '{rid}'",
                      "how": "hold one first -- ingest_project for a robot you already own, create_robot or "
                             "submit_design for one authored here"}
    return gene, None


def _artifact_dir(robot_id: str, out_dir=None):
    """Where this robot's plan/log files live. Confined under ``build/`` like every other agent write path."""
    from virturoid.services.agent_tools import safe_build_path
    root = safe_build_path(out_dir, "sysid")
    d = root / "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(robot_id))[:80]
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_json(path, payload) -> str:
    import json
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return str(path)


def _pick(d: dict, key: str):
    """The first alias of ``key`` present in ``d``. Case-insensitive on the caller's keys."""
    lowered = {str(k).lower(): k for k in d}
    for alias in LOG_ALIASES[key]:
        if alias in lowered:
            return d[lowered[alias]]
    return None


def _normalize_log(raw: dict, *, measured_on=None, source: str = "inline") -> tuple:
    """``(log, None)`` in the schema the estimator reads, or ``(None, refusal)`` naming what is missing.

    The estimator reads ``q_cmd``/``q_meas``/``tau_meas``; the experiment plan's own log schema advertises
    ``q_cmd_rad``/``q_meas_rad``/``tau_meas_nm``. Before this, an engineer who followed our published schema to
    the letter got "log is missing required field 'q_cmd'" back from their own correctly-formatted file. Both
    spellings are accepted here, and so are the handful a real exporter writes.
    """
    if not isinstance(raw, dict):
        return None, {"ok": False, "error": f"the log at {source} is a {type(raw).__name__}, not an object",
                      "expected": sorted(LOG_ALIASES)}
    # A bundle written by simulate_bench_log (or by anyone who saved plan+log together) carries the log inside.
    if "log" in raw and isinstance(raw.get("log"), dict):
        raw = raw["log"]

    log = {}
    for key in ("t", "q_cmd", "q_meas", "qd_meas", "tau_meas", "joints", "control_hz"):
        v = _pick(raw, key)
        if v is not None:
            log[key] = v
    missing = [k for k in ("t", "q_cmd", "q_meas", "joints") if k not in log]
    if missing:
        return None, {
            "ok": False,
            "error": f"the log at {source} is missing {missing}",
            "found_keys": sorted(str(k) for k in raw)[:40],
            "required": {k: list(LOG_ALIASES[k]) for k in ("t", "q_cmd", "q_meas", "joints")},
            "strongly_recommended": {"tau_meas": list(LOG_ALIASES["tau_meas"]),
                                     "why": "the residual that attributes a gap to a PARAMETER is a torque "
                                            "residual. Without measured torque (or motor current) only the "
                                            "trajectory gap can be reported and no parameter may be named"},
            "how": "plan_bench_experiment returns the exact schema under log_format",
        }
    stated = str(measured_on or _pick(raw, "measured_on") or "").strip().lower()
    if stated and stated not in MEASURED_ON_VALUES and stated not in (
            "sim", "simulation", "synthetic", "robot", "bench", "physical"):
        return None, {"ok": False, "error": f"measured_on={stated!r} is not a value this field takes",
                      "accepted": MEASURED_ON_VALUES}
    log["measured_on"] = stated or ""
    if raw.get("provenance"):
        log["provenance"] = raw["provenance"]
    if raw.get("log_hz"):
        log["log_hz"] = raw["log_hz"]
    return log, None


def _load_csv(path, *, measured_on=None) -> tuple:
    """A wide CSV -- one row per sample, one column per (joint, field) -- into the log schema.

    Real telemetry arrives as CSV far more often than as JSON, and it arrives one column per signal. The column
    convention is ``<joint><sep><field>`` for any of ``. / :: __ :`` and any field alias in ``LOG_ALIASES``, so
    ``FL_hip.q_meas_rad``, ``FL_hip/q``, and ``FL_hip__tau_nm`` all land in the same place.
    """
    import csv

    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))
    if not rows:
        return None, {"ok": False, "error": f"{path} has no data rows"}
    headers = [h for h in (rows[0].keys() or []) if h]

    field_of = {a: key for key, aliases in LOG_ALIASES.items()
                for a in aliases if key in ("q_cmd", "q_meas", "qd_meas", "tau_meas")}
    field_of.update({"tau_nm": "tau_meas", "q_rad": "q_meas", "qd_radps": "qd_meas"})

    t_col = next((h for h in headers if h.strip().lower() in LOG_ALIASES["t"]), None)
    cols: dict = {}                                        # field -> {joint: header}
    for h in headers:
        for sep in _CSV_SEPARATORS:
            if sep not in h:
                continue
            joint, _, tail = h.rpartition(sep)
            key = field_of.get(tail.strip().lower())
            if joint.strip() and key:
                cols.setdefault(key, {})[joint.strip()] = h
            break
    if t_col is None or "q_cmd" not in cols or "q_meas" not in cols:
        return None, {
            "ok": False, "error": f"{path} does not carry the columns a bench log needs",
            "found_columns": headers[:40],
            "expected": "a time column (t_s) plus one column per joint per signal, named "
                        "'<joint>.<field>' (separators . / :: __ : are all accepted)",
            "field_names": {k: list(v) for k, v in LOG_ALIASES.items()
                            if k in ("q_cmd", "q_meas", "qd_meas", "tau_meas")},
            "example_header": "t_s,FL_hip.q_cmd_rad,FL_hip.q_meas_rad,FL_hip.tau_meas_nm,FL_knee.q_cmd_rad,...",
        }
    joints = sorted(set(cols["q_meas"]) & set(cols["q_cmd"]))
    if not joints:
        return None, {"ok": False, "error": f"{path} has no joint with BOTH a commanded and a measured column",
                      "commanded": sorted(cols["q_cmd"]), "measured": sorted(cols["q_meas"])}

    def _col(key, joint):
        h = cols.get(key, {}).get(joint)
        return None if h is None else [float(r[h]) for r in rows]

    log = {"joints": joints, "t": [float(r[t_col]) for r in rows],
           "q_cmd": [[float(r[cols["q_cmd"][j]]) for j in joints] for r in rows],
           "q_meas": [[float(r[cols["q_meas"][j]]) for j in joints] for r in rows]}
    for key in ("qd_meas", "tau_meas"):
        if all(j in cols.get(key, {}) for j in joints):
            log[key] = [[float(r[cols[key][j]]) for j in joints] for r in rows]
    first = rows[0]
    for alias in LOG_ALIASES["measured_on"]:
        if alias in first:
            log["measured_on"] = str(first[alias])
            break
    for alias in LOG_ALIASES["control_hz"]:
        if alias in first:
            log["control_hz"] = float(first[alias])
            break
    return _normalize_log(log, measured_on=measured_on, source=str(path))


def _read_log(log_path=None, inline=None, *, measured_on=None) -> tuple:
    """``(log, None)`` or ``(None, refusal)``. Accepts an inline object, a .json bundle, or a wide .csv."""
    import json
    from pathlib import Path

    if inline is not None:
        return _normalize_log(inline, measured_on=measured_on, source="the inline 'log' argument")
    if not log_path:
        return None, {"ok": False,
                      "error": "provide log_path (a .json or .csv file) or an inline log object",
                      "no_hardware": "if you have no robot to run the experiment on, call simulate_bench_log "
                                     "-- it produces a log in this exact schema off a SIMULATED stand-in, "
                                     "clearly labelled as such",
                      "schema": "plan_bench_experiment returns it under log_format"}
    p = Path(str(log_path)).expanduser()
    if not p.exists():
        return None, {"ok": False, "error": f"no log file at {p}"}
    if p.is_dir():
        return None, {"ok": False, "error": f"{p} is a directory; point log_path at the log file itself"}
    try:
        size = p.stat().st_size
    except OSError as exc:
        return None, {"ok": False, "error": f"cannot read {p}: {exc}"}
    if size > _MAX_LOG_BYTES:
        return None, {"ok": False, "error": f"{p} is {size / 1e6:.0f} MB, over the {_MAX_LOG_BYTES / 1e6:.0f} MB "
                                            f"ceiling; trim the log to the excitation window"}
    try:
        if p.suffix.lower() == ".csv":
            return _load_csv(p, measured_on=measured_on)
        return _normalize_log(json.loads(p.read_text(encoding="utf-8")), measured_on=measured_on, source=str(p))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return None, {"ok": False, "error": f"could not parse {p}: {type(exc).__name__}: {exc}"}


def _provenance(log: dict) -> dict:
    """The honesty block that rides on every result. ``hardware`` is only ever what the log itself declares."""
    from virturoid.services.sysid.calibration import log_provenance
    prov = log_provenance(log)
    cls = prov["class"]
    hardware = cls == "hardware"
    return {
        "class": cls,
        "is_a_measurement_of_your_hardware": hardware,
        "why": prov["why"],
        "headline": ("MEASURED ON HARDWARE. Every number below is about your physical robot."
                     if hardware else
                     "NOT A MEASUREMENT OF YOUR ROBOT. This log came from a simulated stand-in, so the numbers "
                     "below describe our estimator on a model we perturbed ourselves -- not your machine. "
                     "MuJoCo's own modelling error cancels when both sides are MuJoCo, and that is exactly the "
                     "error a real log exists to expose."
                     if cls == "sim2sim" else
                     "PROVENANCE UNSTATED. The log does not declare measured_on, so it is treated as NOT "
                     "hardware. If this really did come off a physical robot, re-run with measured_on:'hardware' "
                     "-- it is the field the bench-identified rung turns on."),
        "raises_the_bench_identified_rung": hardware,
    }


def _load_plan(gene, plan_path=None, *, robot_id: str, log=None) -> tuple:
    """``(plan, source_note)``. An explicit path wins; otherwise adopt this robot's saved plan when it MATCHES
    the log, and say so. Windowing a log with a plan it did not come from would score every joint over somebody
    else's excitation, so a mismatch degrades to whole-log windows rather than to a wrong number."""
    import json
    from pathlib import Path

    if plan_path:
        p = Path(str(plan_path)).expanduser()
        if not p.exists():
            return None, f"no plan file at {p}"
        try:
            plan = json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            return None, f"could not parse {p}: {type(exc).__name__}: {exc}"
        plan = plan.get("plan", plan) if isinstance(plan, dict) else plan
        return plan, f"the plan you passed ({p})"
    saved = _artifact_dir(robot_id) / "excitation_plan.json"
    if not saved.exists():
        return None, ("no excitation plan supplied and none saved for this robot: every joint is scored over "
                      "the WHOLE log rather than over its own excitation window")
    try:
        plan = json.loads(saved.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None, "the saved plan for this robot could not be read; whole-log windows used"
    if log is not None:
        want = {j["joint"] for j in plan.get("joints", []) if j.get("excitable")}
        have = {str(n) for n in (log.get("joints") or [])}
        if not want <= have:
            return None, (f"the saved plan excites {sorted(want - have)}, which this log does not contain -- "
                          f"ignored, and every joint is scored over the whole log")
        rows = len(log.get("t") or [])
        span = (float(log["t"][-1]) - float(log["t"][0])) if rows > 1 else 0.0
        want_s = float((plan.get("budget") or {}).get("duration_s") or 0.0)
        if want_s > 0 and abs(span - want_s) > max(1.0, 0.25 * want_s):
            return None, (f"the saved plan is {want_s:.1f} s of motion and this log is {span:.1f} s -- too "
                          f"different to trust the per-joint windows, so the whole log is used for every joint")
    return plan, f"the plan saved for this robot ({saved})"


def _run_tag(injected: dict, ticks: int, ctrl_hz: float, link_scale: float, plan: dict, hold_only: bool) -> str:
    """A short, readable, deterministic name for ONE synthetic configuration.

    Readable rather than a hash because the engineer reads it off disk later and has to know which run it was;
    deterministic rather than a timestamp because re-running the same configuration should overwrite its own
    file and nothing else. Every value that changes the log is in here.
    """
    bits = [f"{ticks * 1000.0 / max(ctrl_hz, 1e-9):g}ms",
            f"f{float(injected.get('frictionloss', 0.0)):g}",
            f"b{float(injected.get('damping', 0.0)):g}",
            f"a{float(injected.get('armature', 0.0)):g}",
            f"{float((plan.get('budget') or {}).get('duration_s') or 0.0):g}s"]
    if hold_only:
        bits.append("hold")
    if link_scale != 1.0:
        bits.append(f"link{link_scale:g}")
    return "_".join(bits).replace(" ", "")


def _sentence(text: str) -> str:
    """One clause, terminated. The underlying explanations are written to be embedded and do not all end in a
    full stop, and a headline is read as one paragraph -- "...then re-run Implicated parameters: armature" is
    two findings that look like one sentence."""
    text = str(text or "").strip()
    return text if (not text or text[-1] in ".!?") else text + "."


def _max_ticks(max_delay_ms, plan) -> int:
    """A delay-search grid in CONTROL TICKS, from a bound an engineer stated in MILLISECONDS."""
    from virturoid.services.sysid.gap_report import DEFAULT_DELAY_MAX_TICKS
    if max_delay_ms is None:
        return int(DEFAULT_DELAY_MAX_TICKS)
    hz = float(((plan or {}).get("controller") or {}).get("control_hz") or 100.0)
    return max(1, min(64, int(round(float(max_delay_ms) * hz / 1000.0))))


# ---------------------------------------------------------------------------------------------------------
# 1. the experiment
# ---------------------------------------------------------------------------------------------------------

def plan_bench_experiment(args: dict) -> dict:
    """Author the bench experiment for a held robot, and say what to send back."""
    gene, refusal = _held(args.get("robot_id"))
    if refusal:
        return refusal
    rid = str(args["robot_id"])
    try:
        from virturoid.services.sysid.excitation import build_excitation
        plan = build_excitation(gene, budget_s=float(args.get("budget_s", 120.0)),
                                control_hz=float(args.get("control_hz", 100.0)),
                                only_joints=args.get("only_joints") or None)
    except Exception as exc:  # noqa: BLE001 - agent-facing: refuse with the reason
        return {"ok": False, "error": f"could not author an experiment for '{rid}': {type(exc).__name__}: {exc}"}

    path = _write_json(_artifact_dir(rid, args.get("out_dir")) / "excitation_plan.json", plan)
    budget = plan["budget"]
    schema = dict(plan["log_schema"])
    schema["accepted_field_names"] = {k: list(v) for k, v in LOG_ALIASES.items()}
    schema["measured_on_values"] = MEASURED_ON_VALUES
    schema["csv"] = ("a wide CSV works too: a t_s column plus '<joint>.<field>' columns, e.g. "
                     "t_s,FL_hip.q_cmd_rad,FL_hip.q_meas_rad,FL_hip.tau_meas_nm,...")
    not_excitable = [{"joint": j["joint"], "because": j["not_excitable_because"]}
                     for j in plan["joints"] if not j["excitable"]]
    return {
        "ok": True,
        "robot_id": rid,
        "plan_path": path,
        "headline": (f"{budget['duration_s']:.0f} s of motion across {budget['n_excitable']} of "
                     f"{budget['n_joints']} joints, one joint at a time, on a stand."),
        "setup": plan["setup"],
        "controller": plan["controller"],
        "budget": budget,
        "segments": plan["segments"],
        "safety_envelope": plan["safety_envelope"],
        "joints": plan["joints"],
        "excitation_order": plan["excitation_order"],
        "not_excitable": not_excitable,
        "log_format": schema,
        "next": {
            "if_you_have_the_robot": f"run this on the hardware, then call measure_sim_to_real_gap "
                                     f"{{robot_id:'{rid}', log_path:'<your log>', measured_on:'hardware'}}",
            "if_you_do_not": f"call simulate_bench_log {{robot_id:'{rid}'}} -- it runs this same experiment on a "
                             f"deliberately perturbed copy of the model and hands back a log in the schema "
                             f"above, labelled as a SIMULATION so no downstream number can be read as a "
                             f"measurement of a physical machine",
        },
        "journey": list(JOURNEY),
        "review": plan["safety_envelope"]["caveat"],
    }


# ---------------------------------------------------------------------------------------------------------
# 2. the no-hardware path, and it is signposted as one
# ---------------------------------------------------------------------------------------------------------

def simulate_bench_log(args: dict) -> dict:
    """Run the experiment on a perturbed copy of the model. A SIMULATION, and every field says so."""
    gene, refusal = _held(args.get("robot_id"))
    if refusal:
        return refusal
    rid = str(args["robot_id"])
    plan, plan_note = _load_plan(gene, args.get("plan_path"), robot_id=rid)
    # Read with CONSTANT keys rather than over a (name, arg) loop: `agent_tools._accepted_params` derives the
    # advertised schema from this function's own AST, and a key read through a loop variable is invisible to it
    # -- which is how a parameter ends up honoured and unadvertised, i.e. unreachable from a strict MCP client.
    offsets = {"frictionloss": args.get("friction_offset_nm"), "damping": args.get("damping_offset"),
               "armature": args.get("armature_offset_kgm2")}
    perturbation = {k: float(v) for k, v in offsets.items() if v is not None}

    try:
        from virturoid.services.sysid.excitation import build_excitation
        from virturoid.services.sysid.synthetic_hardware import (
            DEFAULT_PERTURBATION,
            WHAT_SIM2SIM_DOES_NOT_PROVE,
            synthetic_hardware_log,
        )
        budget_s = float(args.get("budget_s", 120.0))
        if plan is None:
            plan = build_excitation(gene, budget_s=budget_s)
        ctrl_hz = float((plan.get("controller") or {}).get("control_hz") or 100.0)
        delay_ms = float(args.get("delay_ms", 20.0))
        ticks = max(0, int(round(delay_ms * ctrl_hz / 1000.0)))
        injected = dict(perturbation or DEFAULT_PERTURBATION)
        # NAMED, not positional. The underlying helper returns a bare 2-tuple and an engineer guessed the order
        # wrong, burning a 143 s run to find out; nothing crosses this boundary without a key on it.
        plan, log = synthetic_hardware_log(gene, perturbation=injected, delay_ticks=ticks, plan=plan,
                                           budget_s=budget_s, hold_only=bool(args.get("hold_only", False)),
                                           link_scale=float(args.get("link_scale", 1.0)))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"could not simulate a bench log for '{rid}': {type(exc).__name__}: {exc}"}

    d = _artifact_dir(rid, args.get("out_dir"))
    link_scale = float(args.get("link_scale", 1.0))
    # The filename CARRIES THE CONFIGURATION, and it has to. This tool hands back a path, and a caller who runs
    # it twice with different injections holds a path whose meaning silently changed under them -- found here,
    # by a test that measured a 20 ms-delay log while believing it held the 0 ms one and read the resulting
    # withheld fit as a regression. Deterministic rather than a counter, so re-running the SAME configuration
    # is idempotent and reuses the file, while any different one gets its own.
    hold = bool(args.get("hold_only", False))
    tag = _run_tag(injected, ticks, ctrl_hz, link_scale, plan, hold)
    log_path = _write_json(d / f"bench_log_synthetic_{tag}.json", log)
    plan_path = _write_json(d / "excitation_plan.json", plan)
    return {
        "ok": True,
        "robot_id": rid,
        "synthetic": True,
        "about": "A SIMULATED STAND-IN, NOT YOUR ROBOT.",
        "provenance": _provenance(log),
        "log_path": log_path,                                # named paths, never a tuple
        "plan_path": plan_path,
        "plan_source": plan_note,
        "injected": {
            "frictionloss_nm": injected.get("frictionloss"),
            "damping_nm_s_per_rad": injected.get("damping"),
            "armature_kgm2": injected.get("armature"),
            "delay_ms": round(ticks * 1000.0 / ctrl_hz, 3),
            "delay_control_ticks": ticks,
            "link_mass_and_inertia_scale": link_scale,
            "hold_only": hold,
            "note": "these are the errors deliberately put IN. A working pipeline gets them back out; that is "
                    "the whole claim this mode supports",
        },
        "file_naming": "the log filename carries this configuration, so a second run with different injections "
                       "gets its own file and the path you are holding keeps meaning what it meant. Re-running "
                       "the SAME configuration overwrites its own file and nothing else.",
        "adversarial_modes": {
            "hold_only": "the robot is perturbed exactly as usual and never moves. A real gap exists and "
                         "NOTHING in the data can locate it -- a tool that returns three confident numbers "
                         "here is broken in the way that matters",
            "link_scale": "multiplies every link's mass and inertia, an error NO fitted parameter can express. "
                          "It enters the joint equation through the same acceleration term reflected inertia "
                          "does, so the estimator absorbs it into armature and reports intervals that exclude "
                          "the truth. Measured at 1.30: 14/14 joints 'identified', 13/14 intervals wrong, and "
                          "tracking marginally WORSE. This is what the tracking gate in fit_actuators rules on",
        },
        "log_summary": {
            "n_rows": len(log["t"]), "n_joints": len(log["joints"]), "joints": log["joints"],
            "duration_s": round(float(log["t"][-1]) - float(log["t"][0]), 3),
            "carries_measured_torque": "tau_meas" in log,
            "measured_on": log["measured_on"],
        },
        "what_this_does_not_prove": WHAT_SIM2SIM_DOES_NOT_PROVE,
        "next": f"measure_sim_to_real_gap {{robot_id:'{rid}', log_path:'{log_path}'}} -- and read the result as "
                f"a test of the pipeline, not as a fact about any physical robot",
        "journey": list(JOURNEY),
    }


# ---------------------------------------------------------------------------------------------------------
# 3. the gap
# ---------------------------------------------------------------------------------------------------------

def measure_sim_to_real_gap(args: dict) -> dict:
    """Per joint, in rad / ms / N.m: how far is our simulator from this log, and which parameter is implicated."""
    gene, refusal = _held(args.get("robot_id"))
    if refusal:
        return refusal
    rid = str(args["robot_id"])
    log, bad = _read_log(args.get("log_path"), args.get("log"), measured_on=args.get("measured_on"))
    if bad:
        return bad
    plan, plan_note = _load_plan(gene, args.get("plan_path"), robot_id=rid, log=log)
    try:
        from virturoid.services.sysid.gap_report import measure_gap
        gap = measure_gap(gene, log, plan=plan,
                          delay_max_ticks=_max_ticks(args.get("max_delay_ms"), plan),
                          measure_noise_floor=bool(args.get("noise_floor", True)))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"could not measure the gap for '{rid}': {type(exc).__name__}: {exc}"}
    if not gap.get("ok"):
        return {**gap, "ok": False, "robot_id": rid,
                "how": "plan_bench_experiment returns the exact log schema under log_format"}

    prov = _provenance(log)
    joints = gap.get("joints") or {}
    worst = gap.get("worst_joints_by_position_rad") or []
    lat = gap.get("latency") or {}
    lines = [prov["headline"]]
    if worst:
        w = joints[worst[0]]
        lines.append(f"Worst joint: {w['joint']} at {w['position_rms_deg']:.3f} deg RMS position error"
                     + (f" ({w['torque_rms_nm']:.3f} N.m RMS torque)." if "torque_rms_nm" in w else "."))
    if lat.get("identified"):
        lines.append(f"Our simulator leads this log by {lat.get('delay_ms')} ms of actuation delay.")
    elif lat:
        lines.append(_sentence(f"Actuation delay not identified: {lat.get('not_identified_because')}"))
    implicated = gap.get("implicated_parameters") or []
    lines.append(f"Implicated parameters: {', '.join(implicated)}." if implicated
                 else "No parameter is implicated above the estimator's own noise floor.")
    return {
        "ok": True,
        "robot_id": rid,
        "about": ("a measurement of your physical robot" if prov["is_a_measurement_of_your_hardware"]
                  else "NOT a measurement of your robot -- " + prov["class"]),
        "provenance": prov,
        "headline": " ".join(lines),
        "plan_source": plan_note,
        "setup": gap.get("setup"),
        "log": gap.get("log"),
        "joints": joints,
        "worst_joints_by_position_rad": worst,
        "latency": lat,
        "attribution": gap.get("attribution"),
        "implicated_parameters": implicated,
        "joints_with_an_identified_gap": gap.get("joints_with_an_identified_gap"),
        "measured": gap.get("measured"),
        "next": f"fit_actuators {{robot_id:'{rid}', log_path:'{args.get('log_path') or '<your log>'}'}} turns "
                f"this into a fitted value with an interval per parameter, and can write the identified ones "
                f"into the robot you hold",
        "journey": list(JOURNEY),
    }


# ---------------------------------------------------------------------------------------------------------
# 4. the fit, and (optionally) applying it
# ---------------------------------------------------------------------------------------------------------

def fit_actuators(args: dict) -> dict:
    """Fit damping / reflected inertia / dry friction per joint, with an interval, and optionally apply them."""
    gene, refusal = _held(args.get("robot_id"))
    if refusal:
        return refusal
    rid = str(args["robot_id"])
    log, bad = _read_log(args.get("log_path"), args.get("log"), measured_on=args.get("measured_on"))
    if bad:
        return bad
    plan, plan_note = _load_plan(gene, args.get("plan_path"), robot_id=rid, log=log)
    apply_it = bool(args.get("apply", False))
    provisional = bool(args.get("allow_provisional", False))
    try:
        from virturoid.services.sysid.calibration import apply_calibration, engineer_brief
        from virturoid.services.sysid.fit import MIN_TRACKING_IMPROVEMENT_X, fit_parameters
        fit = fit_parameters(gene, log, plan=plan, iterations=int(args.get("iterations", 6)),
                             seed=int(args.get("seed", 0)),
                             delay_max_ticks=_max_ticks(args.get("max_delay_ms"), plan))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"could not fit '{rid}': {type(exc).__name__}: {exc}"}
    if not fit.get("ok"):
        return {**fit, "ok": False, "robot_id": rid}

    prov = _provenance(log)
    applied = {"applied": False, "why": "apply was not requested; nothing was written to the robot you hold"}
    if apply_it:
        from virturoid.services import session_state as S
        try:
            calibrated = apply_calibration(gene, fit, log=log, plan=plan, note=args.get("note"),
                                           allow_provisional=provisional)
            S.commit_robot(rid, calibrated, label="calibration")
            gene = calibrated
            from virturoid.services.sysid.calibration import calibration_of
            rec = calibration_of(gene) or {}
            applied = {
                "applied": bool(rec.get("applied_to_model")),
                "n_joints_changed": len(rec.get("joints") or {}),
                "withheld_by_the_tracking_gate": bool(rec.get("withheld_from_model")),
                "applied_over_the_gate": bool(rec.get("applied_over_the_gate")),
                "why": (rec.get("not_applied") or
                        "the identified parameters are now in the model this robot compiles to; verify_robot "
                        "and export_held use them from here on"),
                "how_to_undo": f"calibration_status {{robot_id:'{rid}', revert:true}}",
            }
        except Exception as exc:  # noqa: BLE001
            applied = {"applied": False, "why": f"could not attach the fit: {type(exc).__name__}: {exc}"}

    brief = engineer_brief(gene, fit, log=log)
    return {
        "ok": True,
        "robot_id": rid,
        "about": ("a measurement of your physical robot" if prov["is_a_measurement_of_your_hardware"]
                  else "NOT a measurement of your robot -- " + prov["class"]),
        "provenance": prov,
        # The deliverable is the paragraph, not a summary of one; the tables behind every clause come with it.
        "headline": prov["headline"] + " " + brief["sentence"],
        "brief": brief,
        "pinned": brief["pinned"],
        "not_pinned": brief["not_pinned"],
        "next_experiment": brief.get("next_experiment"),
        "application_gate": fit.get("application"),
        "tracking_gate_threshold_x": MIN_TRACKING_IMPROVEMENT_X,
        "trajectory": fit.get("trajectory"),
        "latency": fit.get("latency"),
        "assumed_correct": fit.get("assumed_correct"),
        "parameters_not_fitted": fit.get("parameters_not_fitted"),
        "identified_pairs": fit.get("identified_pairs"),
        "candidate_pairs": fit.get("candidate_pairs"),
        "refused": fit.get("refused"),
        "estimator": fit.get("estimator"),
        "plan_source": plan_note,
        "calibration": applied,
        "next": (f"calibration_status {{robot_id:'{rid}'}} shows what is attached and whether the "
                 f"bench-identified rung is earned; revert:true takes it back out"),
        "journey": list(JOURNEY),
    }


# ---------------------------------------------------------------------------------------------------------
# 5. what is attached, and how to take it out
# ---------------------------------------------------------------------------------------------------------

def calibration_status(args: dict) -> dict:
    """What the fit changed on this robot, what was refused, whether L2 is earned -- and the one-call undo."""
    gene, refusal = _held(args.get("robot_id"))
    if refusal:
        return refusal
    rid = str(args["robot_id"])
    try:
        from virturoid.services.sysid.calibration import (
            calibration_of,
            calibration_report,
            l2_requirements,
            revert_calibration,
        )
        reverted = None
        if bool(args.get("revert", False)):
            from virturoid.services import session_state as S
            had = calibration_of(gene) is not None
            if had:
                gene = revert_calibration(gene)
                S.commit_robot(rid, gene, label="calibration reverted")
            reverted = {"reverted": bool(had),
                        "why": ("every joint is back on the compiler's structural prior; the record is gone "
                                "from the robot you hold" if had else
                                "nothing to revert -- no calibration was attached")}
        report = calibration_report(gene)
        ladder = l2_requirements(gene)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"could not read '{rid}': {type(exc).__name__}: {exc}"}

    out = {
        "ok": True,
        "robot_id": rid,
        "calibrated": bool(report.get("calibrated")),
        "headline": (ladder["verdict"] if report.get("calibrated") else
                     "No calibration is attached: every joint carries the compiler's structural prior. "
                     "plan_bench_experiment is where the journey starts."),
        "report": report,
        "actuator_ladder": ladder,
        "how_to_undo": f"calibration_status {{robot_id:'{rid}', revert:true}}",
        "journey": list(JOURNEY),
    }
    if reverted is not None:
        out["revert"] = reverted
    return out


# ---------------------------------------------------------------------------------------------------------
# 6. the OTHER unreachable sim-to-real number: a trained policy across a held-out dynamics distribution
# ---------------------------------------------------------------------------------------------------------

def sim_to_real_transfer(args: dict) -> dict:
    """How much of a trained policy's travel survives dynamics it was never trained on. Needs a real policy.

    ``sim2real.sim2real_transfer_report`` accepts ``policy=None`` and the rollout it delegates to treats that as
    a RANDOM policy, not a zero baseline -- so calling it without one returns a confident retention number for
    noise. That is refused here rather than defaulted, because the failure is silent and the number looks fine.
    """
    gene, refusal = _held(args.get("robot_id"))
    if refusal:
        return refusal
    rid = str(args["robot_id"])
    policy_path = args.get("policy_path")
    if not policy_path:
        return {"ok": False, "robot_id": rid,
                "error": "provide policy_path: a trained policy .npz",
                "why": "with no policy the rollout runs a RANDOM one -- and would report a transfer-survival "
                       "and a reality-gap figure for noise, which reads exactly like a result",
                "how": f"train_held {{robot_id:'{rid}'}} trains one and returns where it was written; "
                       f"a banked policy from the flywheel works too",
                "note": "this is the pure-sim probe (nominal vs a deliberately WIDER held-out randomization). "
                        "For the gap against a real machine, that is plan_bench_experiment -> "
                        "measure_sim_to_real_gap"}
    from pathlib import Path
    p = Path(str(policy_path)).expanduser()
    if not p.exists():
        return {"ok": False, "robot_id": rid, "error": f"no policy at {p}"}
    try:
        from virturoid.services.morph_policy import MorphPolicy
        from virturoid.services.sim2real import HELD_OUT_DR, sim2real_transfer_report
        policy = MorphPolicy.from_npz(p)
        rep = sim2real_transfer_report(gene, policy, n=int(args.get("n", 12)),
                                       steps=int(args.get("steps", 900)), seed=int(args.get("seed", 0)))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "robot_id": rid,
                "error": f"could not measure transfer for '{rid}': {type(exc).__name__}: {exc}"}
    return {
        "ok": True, "robot_id": rid, "policy": str(p),
        "about": "a PURE-SIM probe: nominal dynamics vs a held-out randomization deliberately WIDER than the "
                 "training one. It measures robustness to unmodelled dynamics, not agreement with a real robot.",
        "default_held_out_distribution": HELD_OUT_DR,
        **rep,
        "next": f"measure_sim_to_real_gap {{robot_id:'{rid}'}} is the one that needs a hardware log",
    }


# ---------------------------------------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------------------------------------

def _robot_id() -> dict:
    """A FRESH property dict each time. The schemas are mutated in place by the aggregator's repair pass, so a
    shared inner dict would let one tool's correction land on five others."""
    return {"type": "string",
            "description": "the held robot (from ingest_project / create_robot / submit_design)"}


def _log_args() -> dict:
    """Fresh copies of the three log-input properties every measuring tool takes."""
    return {k: dict(v) for k, v in _LOG_ARGS_TEMPLATE.items()}


_LOG_ARGS_TEMPLATE = {
    "log_path": {"type": "string",
                 "description": "path to the log that came back: .json in the schema plan_bench_experiment "
                                "publishes, or a wide .csv (t_s plus '<joint>.q_meas_rad'-style columns)"},
    "log": {"type": "object", "description": "the log inline instead of a file, same field names"},
    "measured_on": {"type": "string", "enum": ["hardware", "sim2sim", "unstated"],
                    "description": "where the log came from, when the file does not say. 'hardware' is the "
                                   "only value that lets a fit raise the bench-identified rung, so an "
                                   "unstated log is treated as NOT hardware"},
    "plan_path": {"type": "string",
                  "description": "the excitation plan this log was recorded under (default: the one saved for "
                                 "this robot, used only if it matches the log)"},
}

SYSID_TOOLS: dict[str, dict] = {
    "plan_bench_experiment": {
        "description": "SIM-TO-REAL, step 1 of 5. Author the short, safe, information-rich bench experiment to "
                       "run on a held robot's real hardware -- amplitudes bounded by its own declared joint "
                       "limits, frequencies bounded by the datasheet torque and no-load speed of the motors its "
                       "BOM sized, one joint at a time so each joint's residual is attributable. Returns the "
                       "setup, the PD gains, the per-joint signal, the safety envelope that produced it and the "
                       "exact log schema to send back. No hardware? simulate_bench_log runs the same experiment "
                       "on a perturbed copy of the model. Pure arithmetic + one model compile; seconds.",
        "parameters": {"type": "object", "required": ["robot_id"], "properties": {
            "robot_id": _robot_id(),
            "budget_s": {"type": "number", "default": 120.0,
                         "description": "seconds of motion for the WHOLE machine, not per joint (a 12-DOF "
                                        "quadruped comes out near 72 s). Under the floor the four segments "
                                        "stop being separable and the plan says so instead of truncating"},
            "control_hz": {"type": "number", "default": 100.0, "description": "the PD loop rate to run at"},
            "only_joints": {"type": "array", "items": {"type": "string"},
                            "description": "narrow the experiment to these joints; every other joint holds its "
                                           "start pose"},
            "out_dir": {"type": "string", "description": "where to write the plan, confined under build/ "
                                                         "(default build/sysid/<robot_id>)"}}},
        "handler": plan_bench_experiment, "heavy": False,
    },
    "simulate_bench_log": {
        "description": "SIM-TO-REAL, the NO-HARDWARE path. Runs the bench experiment on a deliberately "
                       "PERTURBED second copy of the model and returns a log in the exact schema a real bench "
                       "log arrives in -- so you can see the whole report before you touch a robot. The result "
                       "is a SIMULATION: it validates the pipeline and the estimator, never the physics (both "
                       "sides are MuJoCo, so MuJoCo's own modelling error cancels), it can raise no fidelity "
                       "rung, and every downstream result it feeds says so. Two adversarial modes: hold_only "
                       "(a real gap nothing in the data can locate) and link_scale (an error no fitted "
                       "parameter can express). Real MuJoCo over the whole experiment; ~2 min on a quadruped.",
        "parameters": {"type": "object", "required": ["robot_id"], "properties": {
            "robot_id": _robot_id(),
            "plan_path": {"type": "string", "description": "the experiment to run (default: this robot's "
                                                           "saved plan, else a fresh one)"},
            "budget_s": {"type": "number", "default": 120.0,
                         "description": "seconds of motion, if no plan exists yet"},
            "delay_ms": {"type": "number", "default": 20.0,
                         "description": "actuation delay to inject, in MILLISECONDS (rounded to whole control "
                                        "ticks). Hwangbo et al. name control-signal delay as a dominant term "
                                        "in the real residual"},
            "friction_offset_nm": {"type": "number",
                                   "description": "dry friction to add per joint, N.m (default 0.08)"},
            "damping_offset": {"type": "number",
                               "description": "viscous damping to add per joint, N.m.s/rad (default 0.6)"},
            "armature_offset_kgm2": {"type": "number",
                                     "description": "reflected inertia to add per joint, kg.m^2 (default 0.03)"},
            "hold_only": {"type": "boolean", "default": False,
                          "description": "perturb the robot exactly as usual but never move it -- the "
                                         "adversarial case for the EXPERIMENT. Confident numbers here mean the "
                                         "tool is broken"},
            "link_scale": {"type": "number", "default": 1.0,
                           "description": "multiply every link's mass and inertia -- the adversarial case for "
                                          "the FIT, an error no fitted parameter can express"},
            "out_dir": {"type": "string", "description": "where to write plan + log, confined under build/"}}},
        "handler": simulate_bench_log, "heavy": True,
    },
    "measure_sim_to_real_gap": {
        "description": "SIM-TO-REAL, the gap number. Given the log that came back, how far is our simulator "
                       "from THIS robot -- per joint, in rad, ms and N.m, never a scalar score. Replays the "
                       "same commands through our model and compares trajectories, then subtracts our model's "
                       "inverse dynamics from the measured torque and regresses what is left on the model's own "
                       "sensitivity, so it names WHICH joints and WHICH parameters are implicated. Actuation "
                       "delay is identified OPEN-LOOP, by aligning the control law the plan declares against "
                       "the applied torque in your log -- so it needs tau_meas (motor current is enough) and "
                       "it does not depend on how good our dynamics model is. Every "
                       "attribution is put past an identifiability gate first: a parameter the experiment "
                       "could not load is reported as unidentified rather than guessed. Real MuJoCo; ~1 min "
                       "on a quadruped.",
        "parameters": {"type": "object", "required": ["robot_id"], "properties": {
            "robot_id": _robot_id(), **_log_args(),
            "max_delay_ms": {"type": "number",
                             "description": "top of the actuation-delay search, in milliseconds (default 80)"},
            "noise_floor": {"type": "boolean", "default": True,
                            "description": "measure the estimator's own floor by replaying the same excitation "
                                           "on our unperturbed model; estimates under it are reported as "
                                           "indistinguishable from zero. Turning it off is faster and less "
                                           "honest"}}},
        "handler": measure_sim_to_real_gap, "heavy": True,
    },
    "fit_actuators": {
        "description": "SIM-TO-REAL, the fit. Iterated Gauss-Newton on the torque residual, returning a "
                       "POSTERIOR per joint parameter -- viscous damping, reflected inertia and dry friction, "
                       "each with a value AND an interval, never a bare point estimate. apply:true writes the "
                       "identified ones into the robot you hold, reversibly and with full provenance, so "
                       "verify_robot and export_held use them from then on. Two refusals are load-bearing: only "
                       "parameters that cleared the identifiability gate are applied, and a fit that does not "
                       "measurably improve how the simulator TRACKS the log is withheld entirely -- that is the "
                       "case where the intervals look tight and the parameters are absorbing an error the "
                       "regressor cannot express. Returns the engineer's paragraph, the pinned and unpinned "
                       "tables behind it, and a real narrowed follow-up experiment when one would help. Real "
                       "MuJoCo, iterated; a few minutes on a quadruped.",
        "parameters": {"type": "object", "required": ["robot_id"], "properties": {
            "robot_id": _robot_id(), **_log_args(),
            "apply": {"type": "boolean", "default": False,
                      "description": "attach the fit to the held robot so the compiled model genuinely changes "
                                     "(reversible via calibration_status)"},
            "allow_provisional": {"type": "boolean", "default": False,
                                  "description": "apply even when the fit FAILS the tracking-improvement gate. "
                                                 "Deliberate, and the record and the brief both say it was "
                                                 "forced -- read the values as a hypothesis, not a measurement"},
            "iterations": {"type": "integer", "default": 6,
                           "description": "Gauss-Newton passes; a joint whose residual stops falling is frozen"},
            "seed": {"type": "integer", "default": 0,
                     "description": "bootstrap seed for the intervals (set it to reproduce a run)"},
            "max_delay_ms": {"type": "number",
                             "description": "top of the actuation-delay search, in milliseconds (default 80)"},
            "note": {"type": "string", "description": "free text kept on the calibration record"}}},
        "handler": fit_actuators, "heavy": True,
    },
    "calibration_status": {
        "description": "SIM-TO-REAL, what is attached. Which parameters moved on this robot, by how much, from "
                       "which experiment, what was refused and why, whether the record has gone stale against "
                       "the current structural priors -- and whether the certificate's L2 'bench-identified "
                       "actuator' rung is earned, computed requirement by requirement rather than asserted. "
                       "revert:true removes the whole record and puts every joint back on the compiler's prior. "
                       "Fast: reads the held robot, no rollout.",
        "parameters": {"type": "object", "required": ["robot_id"], "properties": {
            "robot_id": _robot_id(),
            "revert": {"type": "boolean", "default": False,
                       "description": "remove the calibration and put every joint back on the structural "
                                      "prior (idempotent; a no-op on an uncalibrated robot)"}}},
        "handler": calibration_status, "heavy": False,
    },
    "sim_to_real_transfer": {
        "description": "The PURE-SIM transfer probe, for a trained policy: how much of its travel survives a "
                       "held-out dynamics randomization deliberately WIDER than the one it trained under. "
                       "Reports survival, forward retention and metres of travel lost. Requires a real policy "
                       "and refuses without one -- the underlying rollout treats a missing policy as a RANDOM "
                       "one, which returns a confident number for noise. This measures robustness to unmodelled "
                       "dynamics; the gap against a physical machine is measure_sim_to_real_gap. Real MuJoCo "
                       "over n rollouts; minutes.",
        "parameters": {"type": "object", "required": ["robot_id", "policy_path"], "properties": {
            "robot_id": _robot_id(),
            "policy_path": {"type": "string", "description": "a trained policy .npz (train_held writes one)"},
            "n": {"type": "integer", "default": 12, "description": "held-out randomization draws"},
            "steps": {"type": "integer", "default": 900, "description": "physics steps per rollout"},
            "seed": {"type": "integer", "default": 0, "description": "set it to reproduce a run"}}},
        "handler": sim_to_real_transfer, "heavy": True,
    },
}
